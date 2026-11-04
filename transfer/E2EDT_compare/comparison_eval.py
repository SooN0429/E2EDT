#!/usr/bin/env python3
"""Unified per-class metrics + E2EDT vs scratch / vs original-target comparison."""

from __future__ import annotations

import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    labels = list(range(len(class_names)))
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        mask = y_true == i
        n = int(mask.sum())
        if n == 0:
            rec = 0.0
            acc = 0.0
        else:
            rec = float((y_pred[mask] == i).mean())
            acc = rec  # one-vs-rest recall on true class samples == class accuracy
        out[name] = {
            "f1": float(f1s[i]),
            "recall": rec,
            "acc": acc,
            "n": n,
        }
    return out


def _macro_from_per_class(
    per_class: Dict[str, Dict[str, float]], names: Sequence[str]
) -> Dict[str, float]:
    if not names:
        return {"f1": 0.0, "recall": 0.0, "acc": 0.0}
    return {
        "f1": float(np.mean([per_class[n]["f1"] for n in names])),
        "recall": float(np.mean([per_class[n]["recall"] for n in names])),
        "acc": float(np.mean([per_class[n]["acc"] for n in names])),
    }


def materialize_old_class_test_root(
    test_root: str,
    old_classes: Sequence[str],
    dest_root: str,
) -> str:
    """
    Build a 2-class (old-only) ImageFolder from a multi-class test root.

    Copies/hardlinks only ``{idx:02d}_{old_class}`` folders into ``dest_root``.
    """
    import node_traindata_generate as ntdg

    test_root = os.path.abspath(test_root)
    dest_root = os.path.abspath(dest_root)
    if os.path.isdir(dest_root):
        shutil.rmtree(dest_root)
    os.makedirs(dest_root, exist_ok=True)

    n_linked = 0
    for idx, cname in enumerate(old_classes):
        folder = ntdg.class_folder_name(idx, cname)
        src_dir = os.path.join(test_root, folder)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(
                f"Old-class folder missing under test_root: {src_dir}"
            )
        dst_dir = os.path.join(dest_root, folder)
        os.makedirs(dst_dir, exist_ok=True)
        for name in sorted(os.listdir(src_dir)):
            src = os.path.join(src_dir, name)
            if not os.path.isfile(src) or name.startswith("."):
                continue
            dst = os.path.join(dst_dir, name)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            n_linked += 1

    if n_linked < 1:
        raise RuntimeError(f"No images materialized into {dest_root}")
    print(
        f"[INFO] materialize_old_class_test_root: {n_linked} files -> {dest_root} "
        f"classes={list(old_classes)}"
    )
    return dest_root


@torch.no_grad()
def evaluate_predict_fn(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    test_root: str,
    eval_class_names: Sequence[str],
    device: torch.device,
    batch_size: int = 32,
    n_logits: Optional[int] = None,
    n_old_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a logits predict_fn on ImageFolder test.

    ``n_old_classes``: first N names are old classes. Default ``len-1`` (last is new).
    Set ``n_old_classes=len(eval_class_names)`` for old-only (e.g. original target @ 2-class).
    """
    import node_traindata_generate as ntdg

    n_all = len(eval_class_names)
    if n_old_classes is None:
        n_old = n_all - 1 if n_all >= 1 else 0
    else:
        n_old = int(n_old_classes)
        if n_old < 0 or n_old > n_all:
            raise ValueError(
                f"n_old_classes={n_old_classes} out of range for "
                f"{n_all} eval classes"
            )
    ds = datasets.ImageFolder(root=test_root, transform=IMAGENET_TRANSFORM)
    folder_names = [n for n, _ in sorted(ds.class_to_idx.items(), key=lambda kv: kv[1])]
    expected = [ntdg.class_folder_name(i, c) for i, c in enumerate(eval_class_names)]
    if folder_names != expected:
        name_to_eval = {
            ntdg.class_folder_name(i, c): i for i, c in enumerate(eval_class_names)
        }
        remap = {
            ds.class_to_idx[fname]: name_to_eval[fname]
            for fname in folder_names
            if fname in name_to_eval
        }
    else:
        remap = {i: i for i in range(n_all)}

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []

    for x, y_raw in loader:
        y = np.asarray([remap[int(v)] for v in y_raw.tolist()], dtype=np.int64)
        x = x.to(device)
        logits = predict_fn(x)
        if not torch.is_tensor(logits):
            raise TypeError("predict_fn must return a torch.Tensor of logits")
        n_out = logits.shape[-1]
        if n_logits is not None and n_out != n_logits:
            # still use actual n_out
            pass
        pred = logits.argmax(dim=-1).cpu().numpy()
        # If model has only old-class logits, new-class GT can never match
        # (pred in 0..n_old-1). Keep as-is.
        y_true_list.append(y)
        y_pred_list.append(pred)

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    per_class = _per_class_metrics(y_true, y_pred, eval_class_names)
    old_names = list(eval_class_names[:n_old])
    new_name = eval_class_names[n_old] if n_all > n_old else None
    overall_acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    return {
        "per_class": per_class,
        "old_macro": _macro_from_per_class(per_class, old_names),
        "new_class": per_class.get(new_name) if new_name else None,
        "new_class_name": new_name,
        "overall_acc": overall_acc,
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(n_all)),
                average="macro",
                zero_division=0,
            )
        ),
        "n_samples": int(len(y_true)),
        "eval_class_names": list(eval_class_names),
        "n_model_logits": int(n_out) if len(y_pred_list) else None,
        "n_old_classes": int(n_old),
        "test_root": os.path.abspath(test_root),
    }


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_inference_time(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    test_root: str,
    device: torch.device,
    batch_size: int = 32,
    warmup_batches: int = 5,
) -> Dict[str, Any]:
    """
    Wall-clock inference on ImageFolder test (same preprocess as evaluate_predict_fn).

    Warmup a few batches, synchronize CUDA, then time a full loader pass (forward only).
    """
    ds = datasets.ImageFolder(root=test_root, transform=IMAGENET_TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    n_images = len(ds)
    warm = max(int(warmup_batches), 0)

    # Warmup
    for i, (x, _y) in enumerate(loader):
        if i >= warm:
            break
        _ = predict_fn(x.to(device))
    _sync_device(device)

    n_timed = 0
    _sync_device(device)
    t0 = time.perf_counter()
    for x, _y in loader:
        x = x.to(device)
        _ = predict_fn(x)
        n_timed += int(x.size(0))
    _sync_device(device)
    total_sec = float(time.perf_counter() - t0)

    ms_per_image = (
        (total_sec * 1000.0 / float(n_timed)) if n_timed > 0 else None
    )
    return {
        "total_sec": total_sec,
        "ms_per_image": ms_per_image,
        "n_images": int(n_images),
        "n_timed_images": int(n_timed),
        "batch_size": int(batch_size),
        "warmup_batches": int(warm),
        "test_root": os.path.abspath(test_root),
    }


def _delta_per_class(
    after: Dict[str, Dict[str, float]],
    before: Dict[str, Dict[str, float]],
    names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        a = after.get(name, {"f1": 0.0, "recall": 0.0, "acc": 0.0})
        b = before.get(name, {"f1": 0.0, "recall": 0.0, "acc": 0.0})
        out[name] = {
            "f1": float(a["f1"] - b["f1"]),
            "recall": float(a["recall"] - b["recall"]),
            "acc": float(a["acc"] - b["acc"]),
        }
    return out


def build_comparison_report(
    *,
    e2edt_metrics: Dict[str, Any],
    scratch_metrics: Dict[str, Any],
    original_metrics: Dict[str, Any],
    distill_resources: Dict[str, Any],
    scratch_resources: Dict[str, Any],
    n_source_images: int,
    sample_ratio: Any,
    eval_class_names: Sequence[str],
    e2edt_inference: Optional[Dict[str, Any]] = None,
    scratch_inference: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    n_old = len(eval_class_names) - 1
    old_names = list(eval_class_names[:n_old])
    new_name = eval_class_names[n_old]

    # Original @ 2-class baseline; E2EDT @ 3-class (incl. new). New before=0.
    before_new = {"f1": 0.0, "recall": 0.0, "acc": 0.0, "n": 0}
    after_new = e2edt_metrics.get("new_class") or before_new
    orig_old_macro = original_metrics.get("old_macro") or _macro_from_per_class(
        original_metrics.get("per_class") or {}, old_names
    )

    vs_original = {
        "old_class_delta": _delta_per_class(
            e2edt_metrics["per_class"],
            original_metrics["per_class"],
            old_names,
        ),
        "old_macro_delta": {
            "f1": float(e2edt_metrics["old_macro"]["f1"] - orig_old_macro["f1"]),
            "recall": float(
                e2edt_metrics["old_macro"]["recall"] - orig_old_macro["recall"]
            ),
            "acc": float(e2edt_metrics["old_macro"]["acc"] - orig_old_macro["acc"]),
        },
        "new_class_gain": {
            "f1": float(after_new["f1"] - before_new["f1"]),
            "recall": float(after_new["recall"] - before_new["recall"]),
            "acc": float(after_new["acc"] - before_new["acc"]),
            "note": "original target has no new-class logit; before=0",
        },
        "new_class_name": new_name,
        "protocol_note": (
            "original_target evaluated on old-class-only (2-class) test; "
            "E2EDT/scratch on full 3-class test. "
            "Old Δ = E2EDT@3 old − original@2 old; "
            "new gain = E2EDT@3 new − 0."
        ),
    }

    scratch_label_data = scratch_resources.get("label_data") or {}
    scratch_labels = scratch_label_data.get("total")
    scratch_new = scratch_label_data.get("n_new_class_images")
    label_compare = {
        "e2edt_n_source_images": int(n_source_images),
        "e2edt_sample_ratio": sample_ratio,
        "e2edt_note": (
            "Count is original new-class images selected by SAMPLE_RATIO; "
            "patch/mask tensors are treated as augmentation and excluded."
        ),
        "scratch_n_new_class_images": scratch_new,
        "scratch_new_class_name": scratch_label_data.get("new_class_name") or new_name,
        "scratch_note": (
            "Primary compare uses new-class labeled train images only "
            "(same axis as E2EDT sample_ratio source images)."
        ),
        "scratch_n_train_images": scratch_labels,
        "scratch_per_class": scratch_label_data.get("per_class"),
        "difference_scratch_minus_e2edt": (
            None
            if scratch_new is None
            else int(scratch_new) - int(n_source_images)
        ),
        "ratio_e2edt_over_scratch": (
            None
            if not scratch_new
            else float(n_source_images) / float(scratch_new)
        ),
    }

    def _sec(res: Dict[str, Any]) -> Optional[float]:
        v = res.get("wall_clock_sec")
        return float(v) if v is not None else None

    d_sec = _sec(distill_resources)
    s_sec = _sec(scratch_resources)
    time_compare = {
        "distill_wall_clock_sec": d_sec,
        "scratch_wall_clock_sec": s_sec,
        "difference_scratch_minus_distill_sec": (
            None if d_sec is None or s_sec is None else s_sec - d_sec
        ),
        "ratio_distill_over_scratch": (
            None if not s_sec or d_sec is None else d_sec / s_sec
        ),
    }

    d_flops = (distill_resources.get("flops") or {}).get("total_train_flops")
    s_flops = (scratch_resources.get("flops") or {}).get("total_train_flops")
    flops_compare = {
        "distill_total_train_flops": d_flops,
        "scratch_total_train_flops": s_flops,
        "difference_scratch_minus_distill": (
            None if d_flops is None or s_flops is None else float(s_flops) - float(d_flops)
        ),
        "distill_detail": distill_resources.get("flops"),
        "scratch_detail": scratch_resources.get("flops"),
    }

    e_inf = e2edt_inference or {}
    s_inf = scratch_inference or {}
    e_ms = e_inf.get("ms_per_image")
    s_ms = s_inf.get("ms_per_image")
    inference_compare = {
        "e2edt": e2edt_inference,
        "scratch": scratch_inference,
        "ratio_e2edt_over_scratch": (
            None
            if e_ms is None or not s_ms
            else float(e_ms) / float(s_ms)
        ),
        "difference_scratch_minus_e2edt_ms_per_image": (
            None
            if e_ms is None or s_ms is None
            else float(s_ms) - float(e_ms)
        ),
    }

    vs_scratch_perf = {
        "e2edt_per_class": e2edt_metrics["per_class"],
        "scratch_per_class": scratch_metrics["per_class"],
        "delta_e2edt_minus_scratch": _delta_per_class(
            e2edt_metrics["per_class"],
            scratch_metrics["per_class"],
            list(eval_class_names),
        ),
        "e2edt_old_macro": e2edt_metrics["old_macro"],
        "scratch_old_macro": scratch_metrics["old_macro"],
        "e2edt_new_class": e2edt_metrics.get("new_class"),
        "scratch_new_class": scratch_metrics.get("new_class"),
    }

    return {
        "eval_class_names": list(eval_class_names),
        "models": {
            "e2edt": e2edt_metrics,
            "scratch": scratch_metrics,
            "original_target": original_metrics,
        },
        "time": time_compare,
        "inference": inference_compare,
        "label_data": label_compare,
        "flops": flops_compare,
        "vs_scratch_performance": vs_scratch_perf,
        "vs_original_target": vs_original,
    }


def comparison_report_to_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = ["# E2EDT comparison report", ""]
    names = report.get("eval_class_names") or []
    lines.append("## Resource comparison (E2EDT distill vs from-scratch)")
    t = report["time"]
    lines.append(
        f"- Wall-clock: distill={t.get('distill_wall_clock_sec'):.2f}s, "
        f"scratch={t.get('scratch_wall_clock_sec'):.2f}s"
        if t.get("distill_wall_clock_sec") is not None
        and t.get("scratch_wall_clock_sec") is not None
        else f"- Wall-clock: {t}"
    )
    ld = report["label_data"]
    lines.append(
        f"- New-class images: E2EDT sample_ratio={ld.get('e2edt_n_source_images')} "
        f"(ratio={ld.get('e2edt_sample_ratio')}); "
        f"scratch new-class={ld.get('scratch_n_new_class_images')} "
        f"({ld.get('scratch_new_class_name')}); "
        f"diff(scratch-e2edt)={ld.get('difference_scratch_minus_e2edt')}; "
        f"scratch total train images={ld.get('scratch_n_train_images')}"
    )
    fl = report["flops"]
    lines.append(
        f"- Train FLOPs: distill={fl.get('distill_total_train_flops')}, "
        f"scratch={fl.get('scratch_total_train_flops')}"
    )
    inf = report.get("inference") or {}
    e_inf = inf.get("e2edt") or {}
    s_inf = inf.get("scratch") or {}
    if e_inf.get("ms_per_image") is not None and s_inf.get("ms_per_image") is not None:
        lines.append(
            f"- Inference: E2EDT={e_inf['ms_per_image']:.3f}ms/img "
            f"({e_inf.get('total_sec'):.3f}s), "
            f"scratch={s_inf['ms_per_image']:.3f}ms/img "
            f"({s_inf.get('total_sec'):.3f}s); "
            f"batch={e_inf.get('batch_size', s_inf.get('batch_size'))}"
        )
    lines.append("")
    vo = report["vs_original_target"]
    lines.append("## Per-class performance (F1 / recall / acc)")
    lines.append(
        "- Protocol: E2EDT/scratch on 3-class test; "
        "original target on old-class-only (2-class) test. "
        "Δold = E2EDT@3 − original@2; new-class original = N/A (gain vs 0)."
    )
    if vo.get("protocol_note"):
        lines.append(f"- Note: {vo['protocol_note']}")
    lines.append("| class | E2EDT F1 | scratch F1 | original F1 | Δ (E2EDT−orig) F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    e = report["models"]["e2edt"]["per_class"]
    s = report["models"]["scratch"]["per_class"]
    o = report["models"]["original_target"]["per_class"]
    deltas = vo["old_class_delta"]
    new_gain = vo["new_class_gain"]
    new_name = vo["new_class_name"]
    for name in names:
        if name == new_name:
            o_f1 = "N/A"
            dlt = new_gain.get("f1")
        else:
            o_f1 = f"{o[name]['f1']:.4f}" if name in o else "N/A"
            dlt = deltas.get(name, {}).get("f1") if name in deltas else None
        lines.append(
            f"| {name} | {e[name]['f1']:.4f} | {s[name]['f1']:.4f} | "
            f"{o_f1} | "
            f"{'' if dlt is None else f'{dlt:.4f}'} |"
        )
    lines.append("")
    om = vo["old_macro_delta"]
    lines.append(
        f"- Old macro Δ (E2EDT@3 − original@2): "
        f"F1={om['f1']:.4f}, recall={om['recall']:.4f}, acc={om['acc']:.4f}"
    )
    lines.append(
        f"- New class gain vs original (before=0): "
        f"F1={new_gain['f1']:.4f}, recall={new_gain['recall']:.4f}, "
        f"acc={new_gain['acc']:.4f}"
    )
    lines.append("")
    return "\n".join(lines)
