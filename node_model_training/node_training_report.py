#!/usr/bin/env python3
"""Training-run report helpers for training_report/ (library; imported by node_training.py)."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    plt = None
    _HAS_MPL = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _safe_name(text: str) -> str:
    text = text.strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "run"


def default_run_name(node_specs: Dict[str, List[str]]) -> str:
    """Build YYYYMMDD_HHMMSS_node2_cls__node3_cls..."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = []
    for node_name, classes in node_specs.items():
        short_node = node_name.replace("node_", "node")
        class_tag = "_".join(_safe_name(c) for c in classes)
        parts.append(f"{_safe_name(short_node)}_{class_tag}")
    return f"{stamp}_{'__'.join(parts)}"


def unique_run_dir(root: str, run_name: str) -> str:
    """Return an unused directory path under root (append _2, _3, ... if needed)."""
    base = _safe_name(run_name)
    candidate = os.path.join(root, base)
    if not os.path.exists(candidate):
        return candidate
    idx = 2
    while True:
        path = os.path.join(root, f"{base}_{idx}")
        if not os.path.exists(path):
            return path
        idx += 1


def make_run_dir(
    training_note_root: str,
    run_name: Optional[str],
    node_specs: Dict[str, List[str]],
) -> str:
    os.makedirs(training_note_root, exist_ok=True)
    name = run_name.strip() if run_name else default_run_name(node_specs)
    run_dir = unique_run_dir(training_note_root, name)
    for sub in ("data_samples", "epoch_logs", "results"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    print(f"[INFO] training report dir: {run_dir}")
    return run_dir


def save_config(run_dir: str, opt: Any, node_specs: Dict[str, List[str]]) -> str:
    cfg = {
        "node_specs": node_specs,
        "extracted_layer": getattr(opt, "extracted_layer", None),
        "dataset_root": getattr(opt, "dataset_root", None),
        "trigger_dir": getattr(opt, "trigger_dir", None),
        "generated_root": getattr(opt, "generated_root", None),
        "blended_alpha": getattr(opt, "blended_alpha", None),
        "seed": getattr(opt, "seed", None),
        "keep_generated": getattr(opt, "keep_generated", None),
        "feature_batch_size": getattr(opt, "feature_batch_size", None),
        "batch_size": getattr(opt, "batch_size", None),
        "lr": getattr(opt, "lr", None),
        "epoch": getattr(opt, "epoch", None),
        "save_parameter_path": getattr(opt, "save_parameter_path", None),
        "save_parameter_path_name": getattr(opt, "save_parameter_path_name", None),
        "gpu_id": getattr(opt, "gpu_id", None),
        "training_note_root": getattr(opt, "training_note_root", None),
        "run_name": getattr(opt, "run_name", None),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = os.path.join(run_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return path


def save_data_samples(
    run_dir: str,
    node_name: str,
    train_image_root: str,
    n_per_class: int = 5,
) -> str:
    """Copy up to n_per_class images per class folder under train_image_root."""
    out_root = os.path.join(run_dir, "data_samples", node_name)
    os.makedirs(out_root, exist_ok=True)
    if not os.path.isdir(train_image_root):
        raise FileNotFoundError(f"train_image_root not found: {train_image_root}")

    class_dirs = sorted(
        d
        for d in os.listdir(train_image_root)
        if os.path.isdir(os.path.join(train_image_root, d))
    )
    for class_name in class_dirs:
        src_dir = os.path.join(train_image_root, class_name)
        dst_dir = os.path.join(out_root, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        images = sorted(
            f
            for f in os.listdir(src_dir)
            if os.path.isfile(os.path.join(src_dir, f))
            and os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        for i, fname in enumerate(images[:n_per_class], start=1):
            ext = os.path.splitext(fname)[1].lower() or ".jpg"
            dst_name = f"sample_{i:02d}{ext}"
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, dst_name))
    print(f"[INFO] saved data samples -> {out_root}")
    return out_root


def write_epoch_metrics_csv(
    run_dir: str,
    node_name: str,
    rows: Sequence[Dict[str, float]],
) -> str:
    node_dir = os.path.join(run_dir, "epoch_logs", node_name)
    os.makedirs(node_dir, exist_ok=True)
    path = os.path.join(node_dir, "epoch_metrics.csv")
    fieldnames = ["epoch", "train_loss", "val_acc", "val_loss"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_acc": float(row["val_acc"]),
                    "val_loss": float(row["val_loss"]),
                }
            )
    return path


def append_console_log(run_dir: str, node_name: str, line: str) -> None:
    node_dir = os.path.join(run_dir, "epoch_logs", node_name)
    os.makedirs(node_dir, exist_ok=True)
    path = os.path.join(node_dir, "train_console.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def plot_curves(run_dir: str, node_name: str, rows: Sequence[Dict[str, float]]) -> Optional[str]:
    if not rows:
        return None
    node_dir = os.path.join(run_dir, "epoch_logs", node_name)
    os.makedirs(node_dir, exist_ok=True)
    path = os.path.join(node_dir, "curves_loss_acc.png")
    if not _HAS_MPL:
        print("[WARN] matplotlib not available; skip curves_loss_acc.png")
        return None

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(epochs, train_loss, label="train_loss", color="C0")
    axes[0].plot(epochs, val_loss, label="val_loss", color="C1")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"{node_name} training curves")

    axes[1].plot(epochs, val_acc, label="val_acc (%)", color="C2")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _cm_to_markdown(cm: np.ndarray, class_names: Sequence[str]) -> str:
    header = "| | " + " | ".join(class_names) + " |"
    sep = "|---|" + "|".join(["---"] * len(class_names)) + "|"
    lines = [header, sep]
    for i, row_name in enumerate(class_names):
        cells = " | ".join(str(int(v)) for v in cm[i])
        lines.append(f"| {row_name} | {cells} |")
    return "\n".join(lines)


def save_final_results(
    run_dir: str,
    node_name: str,
    class_names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confusion: np.ndarray,
    accuracy: float,
    test_loss: float,
) -> Dict[str, str]:
    out_dir = os.path.join(run_dir, "results", node_name)
    os.makedirs(out_dir, exist_ok=True)

    labels = list(range(len(class_names)))
    f1_per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1_weighted = float(
        f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )

    metrics = {
        "node": node_name,
        "classes": list(class_names),
        "accuracy": float(accuracy),
        "test_loss": float(test_loss),
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": {
            class_names[i]: float(f1_per[i]) for i in range(len(class_names))
        },
        "confusion_matrix": confusion.astype(int).tolist(),
    }
    if len(class_names) == 2:
        tn, fp, fn, tp = confusion.ravel()
        metrics["binary"] = {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    cm_npy = os.path.join(out_dir, "confusion_matrix.npy")
    np.save(cm_npy, confusion.astype(np.int64))

    cm_png = os.path.join(out_dir, "confusion_matrix.png")
    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(5 + 0.4 * len(class_names), 4 + 0.3 * len(class_names)))
        im = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=list(class_names),
            yticklabels=list(class_names),
            ylabel="True",
            xlabel="Predicted",
            title=f"{node_name} confusion matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        thresh = confusion.max() / 2.0 if confusion.size else 0
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                ax.text(
                    j,
                    i,
                    int(confusion[i, j]),
                    ha="center",
                    va="center",
                    color="white" if confusion[i, j] > thresh else "black",
                )
        fig.tight_layout()
        fig.savefig(cm_png, dpi=150)
        plt.close(fig)
    else:
        print("[WARN] matplotlib not available; skip confusion_matrix.png")
        cm_png = ""

    f1_lines = "\n".join(
        f"- `{c}`: {metrics['f1_per_class'][c]:.4f}" for c in class_names
    )
    summary = (
        f"# {node_name} test results\n\n"
        f"- classes: {', '.join(class_names)}\n"
        f"- accuracy: {accuracy:.3f}%\n"
        f"- test_loss: {test_loss:.6f}\n"
        f"- f1_macro: {f1_macro:.4f}\n"
        f"- f1_weighted: {f1_weighted:.4f}\n\n"
        f"## F1 per class\n\n{f1_lines}\n\n"
        f"## Confusion matrix\n\n{_cm_to_markdown(confusion, class_names)}\n"
    )
    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"[INFO] saved results -> {out_dir}")
    return {
        "metrics": metrics_path,
        "confusion_matrix_npy": cm_npy,
        "confusion_matrix_png": cm_png,
        "summary": summary_path,
    }
