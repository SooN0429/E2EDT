#!/usr/bin/env python3
"""From-scratch baseline: source new-class + target old-class (cross-node).

Builds ImageFolder data where:
  - old classes come from target_node person images (+ MaskBlended triggers)
  - new attack class comes from source_node person images (+ attack trigger)

Trains Transfer_Net_ResNet18 end-to-end on images (aligned with KD target arch).
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
from torch.utils.data import DataLoader
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
    Cross-node ImageFolder:
      old classes ← target_node people
      new_class   ← source_node people
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

    for split in SPLITS:
        k_old = _per_person_quota_for_node(
            dataset_root, target_node, len(old_classes), split
        )
        k_new = _per_person_quota_for_node(dataset_root, source_node, 1, split)
        out_split = os.path.join(output_root, split)
        os.makedirs(out_split, exist_ok=True)

        counts_old = _write_class_images_from_node(
            node_src_root=os.path.join(dataset_root, target_node),
            split=split,
            class_names=old_classes,
            class_indices=list(range(len(old_classes))),
            out_split_root=out_split,
            triggers=all_triggers,
            per_person_k=k_old,
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
            per_person_k=k_new,
            blended_alpha_square=blended_alpha_square,
            blended_alpha_hello_kitty=blended_alpha_hello_kitty,
            rng=rng,
        )
        merged = {**counts_old, **counts_new}
        split_counts[split] = merged
        print(
            f"[INFO] cross-node {split}: k_old={k_old} k_new={k_new} counts={merged}"
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
) -> Dict[str, Any]:
    """Train Transfer_Net_ResNet18 from scratch on ImageFolder train/val."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    backbone_multi.extracted_layer = extracted_layer

    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")
    test_root = os.path.join(data_root, "test")
    n_class = len(class_names)

    train_ds = datasets.ImageFolder(train_root, transform=IMAGENET_TRANSFORM)
    val_ds = datasets.ImageFolder(val_root, transform=IMAGENET_TRANSFORM)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
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
    criterion = LabelSmoothingCrossEntropy(reduction="mean")

    # FLOPs on one forward
    example = next(iter(train_loader))[0][:1].to(device)
    try:
        flop_info = cres.measure_forward_flops(
            lambda x: model.predict(x, test_flag=1),
            example,
            model_for_profile=model,
        )
        forward_flops = flop_info["forward_flops"]
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] scratch FLOPs measure failed: {e}")
        flop_info = None
        forward_flops = None

    label_counts = common.count_imagefolder_images(train_root)
    epoch_rows: List[Dict[str, Any]] = []
    epoch_times: List[float] = []
    n_steps = 0
    t_all0 = time.time()

    for e in range(int(epochs)):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(x, y, test_flag=0)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            loss_sum += float(loss.item()) * bs
            n += bs
            n_steps += 1
        scheduler.step()
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model.predict(x, test_flag=1).argmax(dim=-1)
                correct += int((pred == y).sum().item())
                total += y.numel()
        val_acc = 100.0 * correct / max(total, 1)
        row = {
            "epoch": e + 1,
            "train_loss": loss_sum / max(n, 1),
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
        }
        epoch_rows.append(row)
        print(
            f"[scratch] epoch {e + 1}/{epochs} "
            f"loss={row['train_loss']:.6f} val_acc={val_acc:.3f} "
            f"time={epoch_time:.2f}s"
        )

    wall_sec = time.time() - t_all0
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ckpt_path = os.path.join(out_dir, "scratch_state_dict.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_names": list(class_names),
                "extracted_layer": extracted_layer,
                "n_class": n_class,
            },
            ckpt_path,
        )
        common.save_json(
            os.path.join(out_dir, "scratch_train_history.json"),
            {"epoch_rows": epoch_rows, "wall_clock_sec": wall_sec},
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
    )
    result["dataset_meta"] = meta
    result["data_root"] = data_root
    return result
