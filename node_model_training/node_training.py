#!/usr/bin/env python3
"""Multi-node training pipeline: generate data -> extract features -> train -> cleanup.

執行範例（在 E2EDT/node_model_training/ 下）：

1) 使用下方 DEFAULTS，不額外帶參：
   python node_training.py

2) 常用覆寫（CLI 會覆蓋 DEFAULTS）：
   python node_training.py \\
     --node_specs "node_2=white_square,clean;node_3=small_hello_kitty,color_grid,clean;node_4=white_grid,green_square,clean" \\
     --blended_alpha_square 1 \\
     --blended_alpha_hello_kitty 0.3 \\
     --extracted_layer 7_point \\
     --batch_size 8 --lr 0.0001 --epoch 150 \\
     --run_name my_plan_alpha1 \\
     --gpu_id cuda:0

3) 除錯：保留暫存生成資料
   python node_training.py --keep_generated --run_name debug_keep_gen

權重固定寫入 training_report/<run_name>/checkpoints/（與該次報告同目錄）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms

import node_training_report as treport
from model_architecture import backbone_multi, models

# Allow importing node_traindata_generate from sibling node_dataset/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_NODE_DATASET_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "node_dataset"))
_DEFAULT_TRAINING_NOTE = os.path.join(_SCRIPT_DIR, "training_report")
if _NODE_DATASET_DIR not in sys.path:
    sys.path.insert(0, _NODE_DATASET_DIR)

from node_traindata_generate import (  # noqa: E402
    cleanup_generated_node,
    generate_all_nodes,
    parse_node_specs,
)

torch.backends.cudnn.benchmark = True

# =============================================================================
# 預設參數（改這裡即可；命令列參數會覆寫）
# node_specs 格式: "node_2=white_square,clean;node_3=a,b,clean"
# 可用類別: clean, white_square, green_square, white_grid, color_grid,
#           big_hello_kitty, small_hello_kitty
# blended_alpha_*: 0=看不見 trigger；1=mask 區完全貼上 trigger（硬貼）
# square/grid 與 hello_kitty 分開，避免同一透明度下視覺強度差太大。
# =============================================================================
DEFAULTS = {
    "node_specs": "node_1=white_square,big_hello_kitty,clean;node_2=white_square,clean;node_3=small_hello_kitty,color_grid,clean;node_4=white_grid,green_square,clean",
    "dataset_root": _NODE_DATASET_DIR,          # 相對路徑請從本腳本目錄執行時自行改
    "trigger_dir": "",                          # 空= dataset_root/Attack_trigger_image
    "generated_root": "",                       # 空= dataset_root/generated
    "extracted_layer": "7_point",
    "blended_alpha_square": 0.7,
    "blended_alpha_hello_kitty": 0.3,
    "seed": 0,
    "keep_generated": False,
    "feature_batch_size": 32,
    "batch_size": 8,
    "lr": 0.0001,
    "epoch": 50,
    "gpu_id": "cuda:0",
    "training_note_root": _DEFAULT_TRAINING_NOTE,
    "run_name": "",                             # 空=時間戳+方案名
}

CFG = {
    "log_interval": 25,
    "l2_decay": 5e-5,
    "betas": [0.9, 0.999],
}


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction="mean"):
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction="mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


def load_image_folder(root_path, batch_size, kwargs, shuffle=False):
    transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    data = datasets.ImageFolder(root=root_path, transform=transform)
    return (
        torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=shuffle, **kwargs
        ),
        data.class_to_idx,
    )


def evaluate(model, loader, device, n_class, source_name, target_name, split_name="eval"):
    model.eval()
    test_loss = AverageMeter()
    correct_total = 0.0
    criterion = nn.CrossEntropyLoss()
    len_dataset = len(loader.dataset)
    labels_list = list(range(n_class))

    pred_matrix_total = None
    target_matrix_total = None

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            test_output = model.predict(data, test_flag=1)
            loss = criterion(test_output, target)
            test_loss.update(loss.item())

            pred = torch.max(test_output, 1)[1]
            correct_total += torch.sum(pred == target)

            if pred_matrix_total is None:
                pred_matrix_total = pred.data
                target_matrix_total = target.data
            else:
                pred_matrix_total = torch.cat((pred_matrix_total, pred.data))
                target_matrix_total = torch.cat((target_matrix_total, target.data))

    target_np = target_matrix_total.cpu().numpy()
    pred_np = pred_matrix_total.cpu().numpy()
    cm = confusion_matrix(target_np, pred_np, labels=labels_list)
    acc = 100.0 * correct_total.type(torch.float32) / len_dataset

    print(
        "{} --> {} [{}]: correct={}, accuracy={:.3f} %".format(
            source_name, target_name, split_name, int(correct_total.item()), float(acc)
        )
    )
    print(f"confusion_matrix ({n_class}x{n_class}):\n{cm}")
    if n_class == 2:
        tn, fp, fn, tp = cm.ravel()
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return {
        "accuracy": float(acc),
        "correct": int(correct_total.item()),
        "confusion_matrix": cm,
        "loss": float(test_loss.avg),
        "y_true": target_np,
        "y_pred": pred_np,
    }


def train_one_node(
    opt,
    node_name,
    class_names,
    feature_path,
    label_path,
    val_root,
    test_root,
    device,
    kwargs,
    run_dir=None,
):
    n_class = len(class_names)
    source_name = f"{node_name}[{','.join(class_names)}]"
    print(f"\n===== Training {source_name} (n_class={n_class}) =====")
    if run_dir:
        treport.append_console_log(run_dir, node_name, f"===== Training {source_name} =====")

    source_train = torch.from_numpy(np.load(feature_path)).float()
    source_train_label = torch.from_numpy(np.load(label_path)).long()
    if source_train_label.ndim > 1:
        source_train_label = source_train_label.view(-1)

    source_dataset = Data.TensorDataset(source_train, source_train_label)
    source_loader = Data.DataLoader(
        dataset=source_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        persistent_workers=True,
    )

    val_loader, val_class_to_idx = load_image_folder(
        val_root, opt.batch_size, kwargs, shuffle=False
    )
    test_loader, test_class_to_idx = load_image_folder(
        test_root, opt.batch_size, kwargs, shuffle=False
    )
    print(f"[INFO] val class_to_idx={val_class_to_idx}")
    print(f"[INFO] test class_to_idx={test_class_to_idx}")

    model = models.Transfer_Net(n_class)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.base_network.parameters(), "lr": 100 * opt.lr},
            {"params": model.base_network.avgpool.parameters(), "lr": 100 * opt.lr},
            {"params": model.bottle_layer.parameters(), "lr": 10 * opt.lr},
            {"params": model.classifier_layer.parameters(), "lr": 10 * opt.lr},
        ],
        lr=opt.lr,
        betas=CFG["betas"],
        weight_decay=CFG["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    criterion = LabelSmoothingCrossEntropy(reduction="sum")

    epoch_rows = []
    for e in range(opt.epoch):
        train_loss_clf = AverageMeter()
        model.train()
        iter_source = iter(source_loader)
        n_batch = len(source_loader)
        t_start = time.time()

        for i in range(n_batch):
            data_source, label_source = next(iter_source)
            data_source = data_source.to(device)
            if data_source.dim() == 2:
                data_source = data_source.unsqueeze(-1).unsqueeze(-1)
            label_source = torch.squeeze(label_source.to(device))

            optimizer.zero_grad()
            _, label_source_pred = model(data_source, label_source, test_flag=0)
            clf_loss = criterion(label_source_pred, label_source)
            clf_loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())

            if i % CFG["log_interval"] == 0:
                msg = "Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}".format(
                    e + 1,
                    opt.epoch,
                    int(100.0 * i / max(n_batch, 1)),
                    train_loss_clf.avg,
                )
                print(msg)
                if run_dir:
                    treport.append_console_log(run_dir, node_name, msg)

        scheduler.step()
        epoch_time = time.time() - t_start
        print(f"epoch_time={epoch_time:.2f}s")
        val_result = evaluate(
            model,
            val_loader,
            device,
            n_class,
            source_name,
            f"{node_name}_val",
            split_name="val",
        )
        row = {
            "epoch": e + 1,
            "train_loss": float(train_loss_clf.avg),
            "val_acc": float(val_result["accuracy"]),
            "val_loss": float(val_result["loss"]),
        }
        epoch_rows.append(row)
        if run_dir:
            treport.append_console_log(
                run_dir,
                node_name,
                (
                    f"epoch={e + 1} time={epoch_time:.2f}s "
                    f"train_loss={row['train_loss']:.6f} "
                    f"val_acc={row['val_acc']:.3f} val_loss={row['val_loss']:.6f}"
                ),
            )

    if run_dir:
        treport.write_epoch_metrics_csv(run_dir, node_name, epoch_rows)
        treport.plot_curves(run_dir, node_name, epoch_rows)

    test_result = evaluate(
        model,
        test_loader,
        device,
        n_class,
        source_name,
        f"{node_name}_test",
        split_name="test",
    )

    if run_dir:
        # Folder names from ImageFolder (e.g. 00_white_square) for readable reports
        folder_names = [None] * n_class
        for name, idx in test_class_to_idx.items():
            if 0 <= idx < n_class:
                folder_names[idx] = name
        report_class_names = [
            folder_names[i] if folder_names[i] is not None else class_names[i]
            for i in range(n_class)
        ]
        treport.save_final_results(
            run_dir=run_dir,
            node_name=node_name,
            class_names=report_class_names,
            y_true=test_result["y_true"],
            y_pred=test_result["y_pred"],
            confusion=test_result["confusion_matrix"],
            accuracy=test_result["accuracy"],
            test_loss=test_result["loss"],
        )

    if not run_dir:
        raise ValueError("run_dir is required to save checkpoints under training_report/")

    ckpt_root = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)
    class_tag = "_".join(class_names)
    save_name = f"{node_name}_{class_tag}.pth"
    save_path = os.path.join(ckpt_root, save_name)
    torch.save(model.state_dict(), save_path)
    print("saved model to", save_path)

    return {
        "epoch_rows": epoch_rows,
        "test": test_result,
        "checkpoint_path": save_path,
        "checkpoint_filename": save_name,
        "classes": list(class_names),
    }


def build_argparser():
    """CLI 預設值來自檔案最上方 DEFAULTS；命令列可覆寫。"""
    d = DEFAULTS
    parser = argparse.ArgumentParser(
        description="Generate node datasets, extract features, train classifiers, then cleanup. "
        "Defaults are defined in DEFAULTS at the top of this file."
    )
    parser.add_argument("--extracted_layer", type=str, default=d["extracted_layer"])
    parser.add_argument("--dataset_root", type=str, default=d["dataset_root"])
    parser.add_argument(
        "--node_specs",
        type=str,
        default=d["node_specs"],
        help='e.g. "node_2=white_square,clean;node_3=white_square,green_square,clean"',
    )
    parser.add_argument("--trigger_dir", type=str, default=d["trigger_dir"])
    parser.add_argument("--generated_root", type=str, default=d["generated_root"])
    parser.add_argument(
        "--blended_alpha_square",
        type=float,
        default=d["blended_alpha_square"],
        help="MaskBlended alpha for square/grid triggers (ABS_PATCH_CLASSES).",
    )
    parser.add_argument(
        "--blended_alpha_hello_kitty",
        type=float,
        default=d["blended_alpha_hello_kitty"],
        help="MaskBlended alpha for big/small_hello_kitty.",
    )
    parser.add_argument("--seed", type=int, default=d["seed"])
    parser.add_argument(
        "--keep_generated",
        action="store_true",
        default=d["keep_generated"],
    )
    parser.add_argument("--feature_batch_size", type=int, default=d["feature_batch_size"])

    parser.add_argument("--batch_size", type=int, default=d["batch_size"])
    parser.add_argument("--lr", type=float, default=d["lr"])
    parser.add_argument("--epoch", type=int, default=d["epoch"])
    parser.add_argument("--gpu_id", type=str, default=d["gpu_id"])

    parser.add_argument(
        "--training_note_root",
        type=str,
        default=d["training_note_root"],
        help="Root directory for per-run reports (default: ./training_report).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=d["run_name"],
        help="Optional run folder name under training_note_root "
        "(default: timestamp + node/class tags).",
    )
    return parser


if __name__ == "__main__":
    opt = build_argparser().parse_args()

    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)

    backbone_multi.extracted_layer = opt.extracted_layer
    if torch.cuda.is_available() and opt.gpu_id.startswith("cuda"):
        DEVICE = torch.device(opt.gpu_id)
        torch.cuda.set_device(DEVICE)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = os.path.abspath(opt.dataset_root)
    trigger_dir = opt.trigger_dir or os.path.join(dataset_root, "Attack_trigger_image")
    generated_root = opt.generated_root or os.path.join(dataset_root, "generated")
    training_note_root = os.path.abspath(opt.training_note_root)

    node_specs = parse_node_specs(opt.node_specs)
    opt._node_specs = node_specs
    print(f"[INFO] device={DEVICE}")
    print(f"[INFO] node_specs={node_specs}")

    run_dir = treport.make_run_dir(
        training_note_root=training_note_root,
        run_name=opt.run_name or None,
        node_specs=node_specs,
    )
    treport.save_config(run_dir, opt, node_specs)

    kwargs = {"num_workers": 2, "pin_memory": False, "persistent_workers": True}

    gen_results = None
    try:
        gen_results = generate_all_nodes(
            dataset_root=dataset_root,
            node_specs=node_specs,
            trigger_dir=trigger_dir,
            output_root=generated_root,
            blended_alpha_square=opt.blended_alpha_square,
            blended_alpha_hello_kitty=opt.blended_alpha_hello_kitty,
            extracted_layer=opt.extracted_layer,
            seed=opt.seed,
            device=DEVICE,
            feature_batch_size=opt.feature_batch_size,
            extract_features=True,
        )

        # Save class samples before generated/ is deleted.
        for node_name, info in gen_results.items():
            treport.save_data_samples(
                run_dir=run_dir,
                node_name=node_name,
                train_image_root=os.path.join(info["output_dir"], "train"),
                n_per_class=5,
            )

        ckpt_entries = {}
        for node_name, info in gen_results.items():
            train_out = train_one_node(
                opt=opt,
                node_name=node_name,
                class_names=info["classes"],
                feature_path=info["feature_path"],
                label_path=info["label_path"],
                val_root=os.path.join(info["output_dir"], "val"),
                test_root=os.path.join(info["output_dir"], "test"),
                device=DEVICE,
                kwargs=kwargs,
                run_dir=run_dir,
            )
            ckpt_entries[node_name] = {
                "path": train_out["checkpoint_path"],
                "classes": train_out["classes"],
                "filename": train_out["checkpoint_filename"],
            }
        treport.save_checkpoint_index(run_dir, ckpt_entries)
        print(f"[INFO] training report saved at: {run_dir}")
    finally:
        if not opt.keep_generated and gen_results is not None:
            for node_name in gen_results:
                cleanup_generated_node(generated_root, node_name)
            # Remove empty generated root if possible
            if os.path.isdir(generated_root) and not os.listdir(generated_root):
                os.rmdir(generated_root)
