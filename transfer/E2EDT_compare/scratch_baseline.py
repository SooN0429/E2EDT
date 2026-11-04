#!/usr/bin/env python3
"""From-scratch baseline: source new-class + target old-class (cross-node).

Builds ImageFolder data where:
  - old classes come from target_node person images (+ MaskBlended triggers)
  - new attack class comes from source_node person images (+ attack trigger)

Training aligns with node_training: extract cut-point features with
ResNet18Extractor, then train Transfer_Net_ResNet18 tail with test_flag=0.
Val/test still run on RGB images with test_flag=1.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

_COMPARE_DIR = os.path.dirname(os.path.abspath(__file__))
_TRANSFER_DIR = os.path.abspath(os.path.join(_COMPARE_DIR, ".."))
_E2EDT_ROOT = os.path.abspath(os.path.join(_TRANSFER_DIR, ".."))
_NODE_TRAINING_DIR = os.path.join(_E2EDT_ROOT, "node_model_training")
_NODE_DATASET_DIR = os.path.join(_E2EDT_ROOT, "node_dataset")

if _NODE_TRAINING_DIR not in sys.path:
    sys.path.insert(0, _NODE_TRAINING_DIR)
if _NODE_DATASET_DIR not in sys.path:
    sys.path.insert(0, _NODE_DATASET_DIR)
if _COMPARE_DIR not in sys.path:
    sys.path.insert(0, _COMPARE_DIR)
if _TRANSFER_DIR not in sys.path:
    sys.path.insert(0, _TRANSFER_DIR)

from model_architecture import backbone_multi, model_resnet18  # noqa: E402
import node_traindata_generate as ntdg  # noqa: E402
import e2edt_common as common  # noqa: E402
import compute_resources as cres  # noqa: E402

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

SPLITS = ("train", "val", "test")


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        if self.reduction == "mean":
            loss = (-log_preds.sum(dim=-1)).mean()
            nll = F.nll_loss(log_preds, target, reduction="mean")
        else:
            loss = (-log_preds.sum(dim=-1)).sum()
            nll = F.nll_loss(log_preds, target, reduction="sum")
        return self.epsilon * (loss / n) + (1 - self.epsilon) * nll


def _per_person_quota_for_node(
    dataset_root: str, node_name: str, n_classes_draw: int, split: str
) -> int:
    """k = min_person floor(n_images / n_classes_draw)."""
    if n_classes_draw < 1:
        raise ValueError("n_classes_draw must be >= 1")
    split_dir = os.path.join(dataset_root, node_name, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    quotas: List[int] = []
    for person_dir in ntdg.list_person_dirs(split_dir):
        n = len(ntdg.list_images(person_dir))
        quotas.append(n // n_classes_draw)
    if not quotas:
        raise RuntimeError(f"No images found for {node_name}/{split}")
    k = min(quotas)
    if k < 1:
        raise RuntimeError(
            f"Per-person quota is 0 for {node_name}/{split} "
            f"with n_classes_draw={n_classes_draw}"
        )
    return k


def _write_class_images_from_node(
    *,
    node_src_root: str,
    split: str,
    class_names: Sequence[str],
    class_indices: Sequence[int],
    out_split_root: str,
    triggers: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
    per_person_k: int,
    blended_alpha_square: float,
    blended_alpha_hello_kitty: float,
    rng: np.random.RandomState,
) -> Dict[str, int]:
    """Sample person images from one node and write into shared ImageFolder split."""
    split_src = os.path.join(node_src_root, split)
    counts = {c: 0 for c in class_names}
    class_dirs = []
    for idx, cname in zip(class_indices, class_names):
        cdir = os.path.join(out_split_root, ntdg.class_folder_name(idx, cname))
        os.makedirs(cdir, exist_ok=True)
        class_dirs.append(cdir)

    persons = ntdg.list_person_dirs(split_src)
    n_draw = len(class_names)
    for person_dir in persons:
        images = ntdg.list_images(person_dir)
        need = per_person_k * n_draw
        if len(images) < need:
            raise RuntimeError(
                f"{person_dir} has {len(images)} images, need at least {need}"
            )
        order = rng.permutation(len(images))
        selected = [images[i] for i in order[:need]]
        person_name = os.path.basename(person_dir).replace(" ", "_")
        for local_i, cname in enumerate(class_names):
            chunk = selected[local_i * per_person_k : (local_i + 1) * per_person_k]
            trigger_rgb, mask_rgb = triggers[cname]
            for src_path in chunk:
                img = np.array(Image.open(src_path).convert("RGB"), dtype=np.uint8)
                if cname == "clean" or trigger_rgb is None:
                    out_img = img
                else:
                    alpha = ntdg.resolve_blended_alpha(
                        cname, blended_alpha_square, blended_alpha_hello_kitty
                    )
                    out_img = ntdg.apply_maskblended(
                        img,
                        trigger_rgb,
                        alpha,
                        mask_rgb=mask_rgb,
                        class_name=cname,
                    )
                base = os.path.splitext(os.path.basename(src_path))[0]
                out_name = f"{person_name}_{base}_{cname}.jpg"
                Image.fromarray(out_img).save(
                    os.path.join(class_dirs[local_i], out_name), quality=95
                )
                counts[cname] += 1
    return counts


def _subsample_split_to_per_class_balance(
    out_split_root: str,
    eval_classes: Sequence[str],
    class_indices: Sequence[int],
    rng: np.random.RandomState,
) -> Dict[str, int]:
    """Randomly drop files so every class folder has the same count (min across classes)."""
    class_files: Dict[str, List[str]] = {}
    for idx, cname in zip(class_indices, eval_classes):
        cdir = os.path.join(out_split_root, ntdg.class_folder_name(idx, cname))
        if not os.path.isdir(cdir):
            raise FileNotFoundError(f"Missing class dir for balance: {cdir}")
        files = [
            os.path.join(cdir, name)
            for name in sorted(os.listdir(cdir))
            if os.path.isfile(os.path.join(cdir, name))
        ]
        class_files[cname] = files

    n_per_class = min(len(files) for files in class_files.values())
    if n_per_class < 1:
        raise RuntimeError(f"Cannot balance empty classes under {out_split_root}")

    counts: Dict[str, int] = {}
    for cname, files in class_files.items():
        if len(files) == n_per_class:
            counts[cname] = n_per_class
            continue
        order = rng.permutation(len(files))
        keep = {files[i] for i in order[:n_per_class]}
        for path in files:
            if path not in keep:
                os.remove(path)
        counts[cname] = n_per_class
    return counts


def generate_cross_node_dataset(
    *,
    dataset_root: str,
    target_node: str,
    source_node: str,
    old_classes: Sequence[str],
    new_class: str,
    output_root: str,
    trigger_dir: str,
    blended_alpha_square: float,
    blended_alpha_hello_kitty: float,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Cross-node ImageFolder (per-class balanced):
      old classes ← target_node people
      new_class   ← source_node people
    Shared per-person k = min(k_old_max, k_new_max); then subsample if needed.
    """
    old_classes = list(old_classes)
    eval_classes = old_classes + [new_class]
    if new_class in old_classes:
        raise ValueError(f"new_class {new_class!r} already in old_classes")

    dataset_root = os.path.abspath(dataset_root)
    trigger_dir = os.path.abspath(trigger_dir)
    output_root = os.path.abspath(output_root)
    if os.path.isdir(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    all_triggers = ntdg.load_trigger_arrays(trigger_dir, eval_classes)
    rng = np.random.RandomState(seed)
    split_counts: Dict[str, Dict[str, int]] = {}
    split_quotas: Dict[str, Dict[str, int]] = {}

    for split in SPLITS:
        k_old_max = _per_person_quota_for_node(
            dataset_root, target_node, len(old_classes), split
        )
        k_new_max = _per_person_quota_for_node(dataset_root, source_node, 1, split)
        k_used = min(k_old_max, k_new_max)
        out_split = os.path.join(output_root, split)
        os.makedirs(out_split, exist_ok=True)

        counts_old = _write_class_images_from_node(
            node_src_root=os.path.join(dataset_root, target_node),
            split=split,
            class_names=old_classes,
            class_indices=list(range(len(old_classes))),
            out_split_root=out_split,
            triggers=all_triggers,
            per_person_k=k_used,
            blended_alpha_square=blended_alpha_square,
            blended_alpha_hello_kitty=blended_alpha_hello_kitty,
            rng=rng,
        )
        counts_new = _write_class_images_from_node(
            node_src_root=os.path.join(dataset_root, source_node),
            split=split,
            class_names=[new_class],
            class_indices=[len(old_classes)],
            out_split_root=out_split,
            triggers=all_triggers,
            per_person_k=k_used,
            blended_alpha_square=blended_alpha_square,
            blended_alpha_hello_kitty=blended_alpha_hello_kitty,
            rng=rng,
        )
        merged = {**counts_old, **counts_new}
        if len(set(merged.values())) != 1:
            merged = _subsample_split_to_per_class_balance(
                out_split_root=out_split,
                eval_classes=eval_classes,
                class_indices=list(range(len(eval_classes))),
                rng=rng,
            )
        split_counts[split] = merged
        split_quotas[split] = {
            "k_old_max": int(k_old_max),
            "k_new_max": int(k_new_max),
            "k_used": int(k_used),
        }
        print(
            f"[INFO] cross-node {split}: k_old_max={k_old_max} "
            f"k_new_max={k_new_max} k_used={k_used} counts={merged}"
        )

    meta = {
        "dataset_root": dataset_root,
        "target_node": target_node,
        "source_node": source_node,
        "old_classes": old_classes,
        "new_class": new_class,
        "eval_classes": eval_classes,
        "output_root": output_root,
        "split_counts": split_counts,
        "split_quotas": split_quotas,
        "per_class_balanced": True,
        "seed": seed,
    }
    common.save_json(os.path.join(output_root, "dataset_meta.json"), meta)
    return meta


def train_scratch_classifier(
    *,
    data_root: str,
    class_names: Sequence[str],
    extracted_layer: str,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    seed: int = 0,
    out_dir: Optional[str] = None,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 1e-3,
) -> Dict[str, Any]:
    """Extract cut-point features, then train ResNet18 tail (node_training style).

    Early stopping monitors validation CrossEntropy loss (no label smoothing);
    patience<=0 disables early stop. Improvement requires best - val_loss >= min_delta.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    backbone_multi.extracted_layer = extracted_layer

    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")
    test_root = os.path.join(data_root, "test")
    n_class = len(class_names)
    patience = int(early_stop_patience)
    min_delta = float(early_stop_min_delta)

    features_dir = os.path.join(data_root, "features")
    print(
        f"[INFO] extracting train features at {extracted_layer!r} "
        f"from {train_root}"
    )
    feat_path, label_path = ntdg.extract_train_features(
        train_image_root=train_root,
        features_dir=features_dir,
        extracted_layer=extracted_layer,
        device=device,
        batch_size=max(int(batch_size), 32),
    )
    source_train = torch.from_numpy(np.load(feat_path)).float()
    source_labels = torch.from_numpy(np.load(label_path)).long()
    if source_labels.ndim > 1:
        source_labels = source_labels.view(-1)
    feature_shape = list(source_train.shape)

    train_ds = TensorDataset(source_train, source_labels)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    val_ds = datasets.ImageFolder(val_root, transform=IMAGENET_TRANSFORM)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = model_resnet18.Transfer_Net_ResNet18(n_class).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.base_network.parameters(), "lr": 100 * lr},
            {"params": model.classifier_layer.parameters(), "lr": 10 * lr},
        ],
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=5e-5,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    # Align with node_training: sum reduction (not mean).
    criterion = LabelSmoothingCrossEntropy(reduction="sum")
    val_criterion = nn.CrossEntropyLoss(reduction="mean")

    # FLOPs for one train-step forward: cut-point features + test_flag=0 (tail only)
    feat_example = source_train[:1].to(device)
    if feat_example.dim() == 2:
        feat_example = feat_example.unsqueeze(-1).unsqueeze(-1)
    try:
        flop_info = cres.measure_forward_flops(
            lambda x: model.predict(x, test_flag=0),
            feat_example,
            model_for_profile=model,
        )
        flop_info["flops_path"] = "feature_tail_test_flag_0"
        forward_flops = flop_info["forward_flops"]
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] scratch FLOPs measure failed: {e}")
        flop_info = {
            "flops_path": "feature_tail_test_flag_0",
            "input_shape": list(feat_example.shape),
            "error": str(e),
        }
        forward_flops = None

    label_counts = common.count_imagefolder_images(train_root)
    new_class_name = str(class_names[-1]) if class_names else None
    new_folder = (
        ntdg.class_folder_name(len(class_names) - 1, new_class_name)
        if new_class_name is not None
        else None
    )
    n_new_class_images = None
    if new_folder is not None:
        per = label_counts.get("per_class") or {}
        if new_folder in per:
            n_new_class_images = int(per[new_folder])
        else:
            # Fallback: match by class name suffix
            for folder, n in per.items():
                if folder.endswith(f"_{new_class_name}") or folder == new_class_name:
                    n_new_class_images = int(n)
                    new_folder = folder
                    break
    label_counts = {
        **label_counts,
        "n_new_class_images": n_new_class_images,
        "new_class_name": new_class_name,
        "new_class_folder": new_folder,
    }

    epoch_rows: List[Dict[str, Any]] = []
    epoch_times: List[float] = []
    n_steps = 0
    t_all0 = time.time()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    early_stopped = False
    epochs_ran = 0

    if patience > 0:
        print(
            f"[INFO] scratch early stopping: monitor=val_loss "
            f"patience={patience} min_delta={min_delta:g} restore_best_weights=True"
        )
    else:
        print("[INFO] scratch early stopping disabled (patience<=0)")

    for e in range(int(epochs)):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(-1).unsqueeze(-1)
            y = torch.squeeze(y.to(device))
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(x, y, test_flag=0)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            # loss is already batch-sum; average over batches for logging
            loss_sum += float(loss.item())
            n_batches += 1
            n_steps += 1
        scheduler.step()
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        epochs_ran = e + 1

        model.eval()
        correct = 0
        total = 0
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model.predict(x, test_flag=1)
                pred = logits.argmax(dim=-1)
                correct += int((pred == y).sum().item())
                total += y.numel()
                batch_loss = float(val_criterion(logits, y).item())
                val_loss_sum += batch_loss * int(y.numel())
                val_n += int(y.numel())
        val_acc = 100.0 * correct / max(total, 1)
        val_loss = val_loss_sum / max(val_n, 1)
        row = {
            "epoch": e + 1,
            "train_loss": loss_sum / max(n_batches, 1),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "epoch_time_sec": epoch_time,
        }
        epoch_rows.append(row)
        print(
            f"[scratch] epoch {e + 1}/{epochs} "
            f"loss={row['train_loss']:.6f} val_loss={val_loss:.6f} "
            f"val_acc={val_acc:.3f} time={epoch_time:.2f}s"
        )

        improved = (best_val_loss - float(val_loss)) >= min_delta
        if improved:
            best_val_loss = float(val_loss)
            best_epoch = e + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience:
            early_stopped = True
            print(
                f"[scratch] early stop at epoch={e + 1} "
                f"(no val_loss improve >= {min_delta:g} for {patience} epochs); "
                f"best_epoch={best_epoch} best_val_loss={best_val_loss:.6f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"[INFO] restored best scratch weights from epoch={best_epoch} "
            f"val_loss={best_val_loss:.6f}"
        )

    wall_sec = time.time() - t_all0
    early_stop_meta = {
        "early_stop_patience": patience,
        "early_stop_min_delta": min_delta,
        "early_stopped": bool(early_stopped),
        "best_epoch": int(best_epoch) if best_epoch else None,
        "best_val_loss": float(best_val_loss) if best_state is not None else None,
        "epochs_ran": int(epochs_ran),
        "epochs_max": int(epochs),
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ckpt_path = os.path.join(out_dir, "scratch_state_dict.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_names": list(class_names),
                "extracted_layer": extracted_layer,
                "n_class": n_class,
                "train_mode": "feature_tail",
                "feature_path": feat_path,
                "feature_shape": feature_shape,
                **early_stop_meta,
            },
            ckpt_path,
        )
        common.save_json(
            os.path.join(out_dir, "scratch_train_history.json"),
            {
                "epoch_rows": epoch_rows,
                "wall_clock_sec": wall_sec,
                **early_stop_meta,
            },
        )
    else:
        ckpt_path = None

    resources = cres.pack_resource_report(
        wall_clock_sec=wall_sec,
        n_train_steps=n_steps,
        forward_flops=forward_flops,
        epoch_times=epoch_times,
        extra={
            "label_data": label_counts,
            "forward_profile": flop_info,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "test_root": test_root,
            "train_root": train_root,
            "val_root": val_root,
            "checkpoint_path": ckpt_path,
            "train_mode": "feature_tail",
            "feature_path": feat_path,
            "label_path": label_path,
            "feature_shape": feature_shape,
            "extracted_layer": extracted_layer,
            **early_stop_meta,
        },
    )
    if out_dir:
        common.save_json(os.path.join(out_dir, "scratch_resources.json"), resources)

    return {
        "model": model,
        "resources": resources,
        "class_names": list(class_names),
        "test_root": test_root,
        "train_root": train_root,
        "checkpoint_path": ckpt_path,
        "epoch_rows": epoch_rows,
        "feature_path": feat_path,
        "label_path": label_path,
        **early_stop_meta,
    }


def run_scratch_baseline(
    *,
    dataset_root: str,
    target_node: str,
    source_node: str,
    old_classes: Sequence[str],
    new_class: str,
    trigger_dir: str,
    extracted_layer: str,
    device: torch.device,
    output_dir: str,
    blended_alpha_square: float = 0.7,
    blended_alpha_hello_kitty: float = 0.3,
    seed: int = 0,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 1e-3,
) -> Dict[str, Any]:
    data_root = os.path.join(os.path.abspath(output_dir), "scratch_data")
    meta = generate_cross_node_dataset(
        dataset_root=dataset_root,
        target_node=target_node,
        source_node=source_node,
        old_classes=old_classes,
        new_class=new_class,
        output_root=data_root,
        trigger_dir=trigger_dir,
        blended_alpha_square=blended_alpha_square,
        blended_alpha_hello_kitty=blended_alpha_hello_kitty,
        seed=seed,
    )
    train_out = os.path.join(os.path.abspath(output_dir), "scratch")
    result = train_scratch_classifier(
        data_root=data_root,
        class_names=meta["eval_classes"],
        extracted_layer=extracted_layer,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        out_dir=train_out,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )
    result["dataset_meta"] = meta
    result["data_root"] = data_root
    return result
