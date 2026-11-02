#!/usr/bin/env python3
"""Validate localization quality for a localization_result run.

Surrogate section (per-image):
  - Weighted Test MSE on hold-out masks (mask_seed=0 by default)
  - R^2 on the same test masks (target-class logit)
  - Source Response Variance on the original fitting masks

Subnet section (when hard_masks.pt exists):
  - E_loc / E_src on hold-out patch masks (hard-masked full Ms)
  - Target retention on node test ImageFolder: original vs learned vs random×N
    (Target F1 / Recall / AUROC, Clean FPR)

Masked patch inputs are rebuilt in memory from retained patch metadata;
tensors/ need not exist after sweep cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_E2EDT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_TRANSFER_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_PATCH_GEN_DIR = os.path.join(_TRANSFER_DIR, "patch_data_generate")
_NODE_TRAINING_DIR = os.path.join(_E2EDT_ROOT, "node_model_training")
_NODE_DATASET_DIR = os.path.join(_E2EDT_ROOT, "node_dataset")

for _p in (_SCRIPT_DIR, _PATCH_GEN_DIR, _NODE_TRAINING_DIR, _NODE_DATASET_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import node_traindata_generate as ntdg  # noqa: E402
import patch_data_generate as patch_gen  # noqa: E402
from node_traindata_generate import (  # noqa: E402
    load_trigger_arrays,
    resolve_blended_alpha,
)
from specific_localziation import (  # noqa: E402
    MASK_TYPES,
    REFERENCE_MASK_ID,
    MaskedResNet18Net,
    bernoulli_locality_weights,
    group_manifest_by_image,
    keep_ratio_map_for_keys,
    load_json,
    load_manifest,
    load_source_model,
    soft_to_hard_masks,
    top_k_map_for_keys,
    _write_csv,
)

R2_DENOM_EPS = 1e-12
DEFAULT_TEST_MASK_SEED = 0
DEFAULT_RANDOM_SEEDS = 10
DEFAULT_RANDOM_BASE_SEED = 0
IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ---------------------------------------------------------------------------
# Path / config helpers
# ---------------------------------------------------------------------------


def _is_valid_patch_meta_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    return all(
        os.path.isfile(os.path.join(path, name))
        for name in ("config_resolved.json", "manifest.jsonl")
    )


def resolve_patch_run_dir(
    configured: Optional[str],
    override: Optional[str] = None,
) -> str:
    """Resolve patch metadata dir with fallbacks after path moves / cleanup."""
    candidates: List[str] = []
    if override:
        candidates.append(os.path.abspath(override))
    if configured:
        configured = os.path.abspath(configured)
        candidates.append(configured)
        base = os.path.basename(configured.rstrip("/"))
        # Historical: transfer/patch_generated/<run>
        # Current: transfer/patch_data_generate/patch_generated/<run>
        candidates.append(
            os.path.join(_PATCH_GEN_DIR, "patch_generated", base)
        )
        candidates.append(
            os.path.join(_TRANSFER_DIR, "patch_generated", base)
        )

    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if _is_valid_patch_meta_dir(c):
            return c

    tried = ", ".join(seen) if seen else "(none)"
    raise FileNotFoundError(
        "Could not resolve patch_run_dir with config_resolved.json + "
        f"manifest.jsonl. Tried: {tried}"
    )


def load_surrogates(surrogate_dir: str) -> Dict[str, np.ndarray]:
    """Load W [D,C] keyed by image_id from g_*.npz."""
    if not os.path.isdir(surrogate_dir):
        raise FileNotFoundError(surrogate_dir)
    out: Dict[str, np.ndarray] = {}
    names = sorted(
        n for n in os.listdir(surrogate_dir) if n.startswith("g_") and n.endswith(".npz")
    )
    if not names:
        raise FileNotFoundError(f"No g_*.npz under {surrogate_dir}")
    for name in names:
        path = os.path.join(surrogate_dir, name)
        z = np.load(path, allow_pickle=True)
        image_id = str(z["image_id"].item() if z["image_id"].shape == () else z["image_id"])
        W = np.asarray(z["W"], dtype=np.float64)
        if image_id in out:
            raise ValueError(f"Duplicate image_id {image_id!r} in surrogates")
        out[image_id] = W
    return out


# ---------------------------------------------------------------------------
# Mask / input rebuild
# ---------------------------------------------------------------------------


def _mask_key(b: np.ndarray) -> bytes:
    return np.asarray(b, dtype=np.int64).tobytes()


def fitting_masks_from_rows(
    rows: Sequence[Dict[str, Any]],
    grid_rows: int,
    grid_cols: int,
) -> Tuple[np.ndarray, List[str]]:
    """Return B_fit [K, RC] and mask_ids in stable order (mask_* then reference)."""
    ordered = sorted(
        rows,
        key=lambda r: (
            1 if r["mask_id"] == REFERENCE_MASK_ID else 0,
            r["mask_id"],
        ),
    )
    B_list: List[np.ndarray] = []
    ids: List[str] = []
    for r in ordered:
        b = np.asarray(r["binary_patch_mask"], dtype=np.int64)
        if b.shape != (grid_rows, grid_cols):
            raise ValueError(
                f"{r.get('image_id')}/{r['mask_id']}: mask shape {b.shape} != "
                f"({grid_rows},{grid_cols})"
            )
        B_list.append(b.reshape(-1))
        ids.append(str(r["mask_id"]))
    return np.stack(B_list, axis=0), ids


def generate_test_masks(
    image_id: str,
    grid_rows: int,
    grid_cols: int,
    num_masks: int,
    keep_prob: float,
    test_mask_seed: int,
) -> np.ndarray:
    """Bernoulli hold-out masks [K, RC]; no reference_all_ones."""
    rng = patch_gen._mask_rng_for_image(int(test_mask_seed), image_id)
    masks = patch_gen.bernoulli_patch_masks(
        grid_rows, grid_cols, int(num_masks), float(keep_prob), rng
    )
    return np.stack([m.reshape(-1).astype(np.float64) for m in masks], axis=0)


def mask_overlap_fraction(B_fit: np.ndarray, B_test: np.ndarray) -> float:
    """Fraction of test masks that also appear in fitting set."""
    if B_test.size == 0:
        return 0.0
    fit_keys = {_mask_key(b) for b in B_fit}
    hits = sum(1 for b in B_test if _mask_key(b) in fit_keys)
    return float(hits) / float(B_test.shape[0])


def rebuild_poisoned_tensor(
    source_face_path: str,
    attack_class: str,
    trigger_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    blended_alpha: float,
    image_size: int,
) -> torch.Tensor:
    transform = patch_gen._default_transform(image_size)
    return patch_gen.poisoned_tensor_from_source(
        source_face_path=source_face_path,
        trigger_rgb=trigger_rgb,
        mask_rgb=mask_rgb,
        class_name=attack_class,
        blended_alpha=float(blended_alpha),
        transform=transform,
    )


def apply_masks_to_poisoned(
    x: torch.Tensor,
    B: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    baseline: torch.Tensor,
    image_size: int,
) -> List[torch.Tensor]:
    """Apply each flattened mask in B [K, RC] to poisoned x; return list of [3,H,W]."""
    out: List[torch.Tensor] = []
    for i in range(B.shape[0]):
        grid = B[i].reshape(grid_rows, grid_cols).astype(np.int64)
        spatial = patch_gen.patch_mask_to_spatial(grid, image_size, image_size)
        out.append(patch_gen.apply_mask_after_norm(x, spatial, baseline))
    return out


@torch.no_grad()
def collect_ms_logits_from_tensors(
    model: nn.Module,
    tensors: Sequence[torch.Tensor],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    logits_list: List[np.ndarray] = []
    batch: List[torch.Tensor] = []
    for t in tensors:
        batch.append(t)
        if len(batch) >= batch_size:
            x = torch.stack(batch, dim=0).to(device)
            out = model.predict(x, test_flag=1)
            logits_list.append(out.detach().cpu().numpy())
            batch = []
    if batch:
        x = torch.stack(batch, dim=0).to(device)
        out = model.predict(x, test_flag=1)
        logits_list.append(out.detach().cpu().numpy())
    if not logits_list:
        raise ValueError("No tensors to score")
    return np.concatenate(logits_list, axis=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def weighted_test_mse_target(
    y_t: np.ndarray,
    g_t: np.ndarray,
    pi: np.ndarray,
) -> float:
    """sum_b Π_b (y-g)^2 / sum_b Π_b on target logit."""
    y_t = np.asarray(y_t, dtype=np.float64).reshape(-1)
    g_t = np.asarray(g_t, dtype=np.float64).reshape(-1)
    pi = np.asarray(pi, dtype=np.float64).reshape(-1)
    denom = float(pi.sum())
    if denom <= 0:
        return float("nan")
    return float(np.sum(pi * (y_t - g_t) ** 2) / denom)


def weighted_test_mse_multiclass(
    Y: np.ndarray,
    G: np.ndarray,
    pi: np.ndarray,
) -> float:
    """Match train g_fit mse_weighted: sum_b π ||e_b||^2 / C."""
    Y = np.asarray(Y, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64).reshape(-1)
    err = G - Y
    sq = np.sum(err * err, axis=1)
    c = max(Y.shape[1], 1)
    return float(np.sum(pi * sq) / c)


def r2_score(y: np.ndarray, yhat: np.ndarray, eps: float = R2_DENOM_EPS) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    y_bar = float(np.mean(y))
    ss_tot = float(np.sum((y - y_bar) ** 2))
    if ss_tot < eps:
        return float("nan")
    ss_res = float(np.sum((y - yhat) ** 2))
    return float(1.0 - ss_res / ss_tot)


def source_response_variance(y_t: np.ndarray) -> float:
    return float(np.var(np.asarray(y_t, dtype=np.float64).reshape(-1), ddof=0))


def _nanmean(vals: Sequence[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _nanmedian(vals: Sequence[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def _nanpercentile(vals: Sequence[float], q: float) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_localization_surrogates(
    localization_dir: str,
    patch_run_dir: Optional[str] = None,
    test_mask_seed: int = DEFAULT_TEST_MASK_SEED,
    device_str: Optional[str] = None,
    batch_size: int = 32,
    r2_eps: float = R2_DENOM_EPS,
) -> Dict[str, Any]:
    localization_dir = os.path.abspath(localization_dir)
    cfg_path = os.path.join(localization_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(cfg_path)
    loc_cfg = load_json(cfg_path)

    patch_dir = resolve_patch_run_dir(
        loc_cfg.get("patch_run_dir"), override=patch_run_dir
    )
    patch_resolved = load_json(os.path.join(patch_dir, "config_resolved.json"))
    patch_cfg = patch_resolved["patch_config"]
    selection = patch_resolved["selection"]
    training_report = patch_resolved["training_report"]

    grid_rows = int(loc_cfg.get("grid_rows") or patch_cfg["GRID_ROWS"])
    grid_cols = int(loc_cfg.get("grid_cols") or patch_cfg["GRID_COLS"])
    keep_prob = float(loc_cfg.get("keep_prob") or patch_cfg["KEEP_PROB"])
    num_masks = int(patch_cfg["NUM_MASKS_PER_IMAGE"])
    image_size = int(patch_cfg.get("IMAGE_SIZE", 224))
    train_mask_seed = int(patch_cfg["MASK_SEED"])
    include_reference = bool(loc_cfg.get("include_reference", True))

    attack_class = str(
        loc_cfg.get("resolved_attack_class") or selection["attack_class"]
    )
    node = str(loc_cfg.get("resolved_node") or selection["node"])
    class_names = list(loc_cfg.get("resolved_class_names") or [])
    if not class_names:
        class_names = list(training_report["node_specs"][node])
    target_idx = int(loc_cfg["resolved_target_class_idx"])
    extracted = str(
        loc_cfg.get("resolved_extracted_layer")
        or loc_cfg.get("extracted_layer")
        or "7_point"
    )

    ckpt = loc_cfg.get("checkpoint_path")
    if not ckpt or not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint_path missing or not found: {ckpt}")

    is_holdout = int(test_mask_seed) != int(train_mask_seed)

    device_name = device_str or loc_cfg.get("device") or "cpu"
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available; falling back to CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    W_by_image = load_surrogates(os.path.join(localization_dir, "surrogate_models"))
    rows = load_manifest(os.path.join(patch_dir, "manifest.jsonl"))
    groups = group_manifest_by_image(rows, include_reference=include_reference)

    missing = sorted(set(W_by_image) - set(groups))
    if missing:
        raise RuntimeError(
            f"{len(missing)} surrogate image_id(s) missing from manifest, "
            f"e.g. {missing[:3]}"
        )

    # Trigger / baseline for in-memory rebuild
    blended_alpha = resolve_blended_alpha(
        attack_class,
        float(training_report["blended_alpha_square"]),
        float(training_report["blended_alpha_hello_kitty"]),
    )
    triggers = load_trigger_arrays(
        training_report["trigger_dir"], training_report["node_specs"][node]
    )
    trigger_rgb, mask_rgb = triggers[attack_class]
    if trigger_rgb is None or mask_rgb is None:
        raise RuntimeError(f"Missing trigger/mask for attack class {attack_class!r}")

    baseline = patch_gen.resolve_mask_baseline(
        patch_cfg["MASK_BASELINE"],
        patch_cfg.get("MASK_BASELINE_CHANNEL_CONSTANTS"),
        image_size,
        image_size,
    )

    print(f"[INFO] localization_dir={localization_dir}")
    print(f"[INFO] patch_run_dir={patch_dir}")
    print(f"[INFO] checkpoint={ckpt}")
    print(
        f"[INFO] node={node} attack={attack_class} "
        f"target_idx={target_idx} ({class_names[target_idx]})"
    )
    print(
        f"[INFO] grid={grid_rows}x{grid_cols} keep_prob={keep_prob} "
        f"n_test_masks={num_masks} train_MASK_SEED={train_mask_seed} "
        f"test_mask_seed={test_mask_seed} holdout={is_holdout}"
    )
    print(f"[INFO] n_images={len(W_by_image)} device={device}")

    model = load_source_model(len(class_names), ckpt, extracted, device)

    per_image: List[Dict[str, Any]] = []
    image_ids = sorted(W_by_image.keys())
    for idx, image_id in enumerate(image_ids):
        fit_rows = groups[image_id]
        # source_face_path is identical across masks for one image
        source_face_path = fit_rows[0]["source_face_path"]
        if not os.path.isfile(source_face_path):
            raise FileNotFoundError(
                f"source_face_path missing for {image_id}: {source_face_path}"
            )

        B_fit, fit_mask_ids = fitting_masks_from_rows(
            fit_rows, grid_rows, grid_cols
        )
        B_test = generate_test_masks(
            image_id=image_id,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            num_masks=num_masks,
            keep_prob=keep_prob,
            test_mask_seed=test_mask_seed,
        )
        overlap = mask_overlap_fraction(B_fit, B_test)

        x = rebuild_poisoned_tensor(
            source_face_path=source_face_path,
            attack_class=attack_class,
            trigger_rgb=trigger_rgb,
            mask_rgb=mask_rgb,
            blended_alpha=blended_alpha,
            image_size=image_size,
        )

        fit_tensors = apply_masks_to_poisoned(
            x, B_fit, grid_rows, grid_cols, baseline, image_size
        )
        test_tensors = apply_masks_to_poisoned(
            x, B_test, grid_rows, grid_cols, baseline, image_size
        )

        Y_fit = collect_ms_logits_from_tensors(
            model, fit_tensors, device, batch_size=batch_size
        )
        Y_test = collect_ms_logits_from_tensors(
            model, test_tensors, device, batch_size=batch_size
        )

        W = W_by_image[image_id]
        G_test = B_test.astype(np.float64) @ W

        y_fit_t = Y_fit[:, target_idx]
        y_test_t = Y_test[:, target_idx]
        g_test_t = G_test[:, target_idx]

        pi_test = bernoulli_locality_weights(B_test, keep_prob)
        mse_t = weighted_test_mse_target(y_test_t, g_test_t, pi_test)
        mse_mc = weighted_test_mse_multiclass(Y_test, G_test, pi_test)
        r2 = r2_score(y_test_t, g_test_t, eps=r2_eps)
        v_i = source_response_variance(y_fit_t)
        low_info = bool(
            (not np.isfinite(r2))
            or (v_i < r2_eps)
            or (float(np.sum((y_test_t - float(np.mean(y_test_t))) ** 2)) < r2_eps)
        )

        row = {
            "image_idx": idx,
            "image_id": image_id,
            "n_train_masks": int(B_fit.shape[0]),
            "n_test_masks": int(B_test.shape[0]),
            "test_mse_weighted": mse_t,
            "test_mse_weighted_multiclass": mse_mc,
            "r2": r2,
            "source_response_variance": v_i,
            "mask_overlap_frac": overlap,
            "low_informativeness": int(low_info),
            "is_holdout": int(is_holdout),
            "train_mask_seed": train_mask_seed,
            "test_mask_seed": int(test_mask_seed),
            "target_class_idx": target_idx,
            "include_reference_in_V": int(include_reference),
            "n_fit_mask_ids": len(fit_mask_ids),
        }
        per_image.append(row)

        if (idx + 1) % 10 == 0 or idx + 1 == len(image_ids):
            print(
                f"[quality] {idx + 1}/{len(image_ids)} {image_id} "
                f"mse={mse_t:.6f} r2={r2 if np.isfinite(r2) else float('nan'):.4f} "
                f"V={v_i:.6f} overlap={overlap:.3f}"
            )

    out_dir = os.path.join(localization_dir, "surrogate_quality")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "surrogate_quality_per_image.csv")
    _write_csv(csv_path, per_image)

    mse_vals = [r["test_mse_weighted"] for r in per_image]
    r2_vals = [r["r2"] for r in per_image]
    v_vals = [r["source_response_variance"] for r in per_image]
    overlap_vals = [r["mask_overlap_frac"] for r in per_image]
    n_low = int(sum(int(r["low_informativeness"]) for r in per_image))

    summary: Dict[str, Any] = {
        "localization_dir": localization_dir,
        "patch_run_dir": patch_dir,
        "checkpoint_path": ckpt,
        "node": node,
        "attack_class": attack_class,
        "target_class_idx": target_idx,
        "target_class_name": class_names[target_idx],
        "n_images": len(per_image),
        "train_mask_seed": train_mask_seed,
        "test_mask_seed": int(test_mask_seed),
        "is_holdout": is_holdout,
        "include_reference_in_V": include_reference,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "keep_prob": keep_prob,
        "num_test_masks_per_image": num_masks,
        "r2_denom_eps": r2_eps,
        "n_low_informativeness": n_low,
        "warnings": [],
        "metrics": {
            "test_mse_weighted": {
                "mean": _nanmean(mse_vals),
                "median": _nanmedian(mse_vals),
                "p10": _nanpercentile(mse_vals, 10),
                "p90": _nanpercentile(mse_vals, 90),
            },
            "r2": {
                "mean": _nanmean(r2_vals),
                "median": _nanmedian(r2_vals),
                "p10": _nanpercentile(r2_vals, 10),
                "p90": _nanpercentile(r2_vals, 90),
                "n_nan": int(sum(1 for v in r2_vals if not np.isfinite(v))),
            },
            "source_response_variance": {
                "mean": _nanmean(v_vals),
                "median": _nanmedian(v_vals),
                "p10": _nanpercentile(v_vals, 10),
                "p90": _nanpercentile(v_vals, 90),
            },
            "mask_overlap_frac": {
                "mean": _nanmean(overlap_vals),
                "median": _nanmedian(overlap_vals),
                "p90": _nanpercentile(overlap_vals, 90),
            },
            "test_mse_weighted_multiclass": {
                "mean": _nanmean(
                    [r["test_mse_weighted_multiclass"] for r in per_image]
                ),
                "median": _nanmedian(
                    [r["test_mse_weighted_multiclass"] for r in per_image]
                ),
            },
        },
        "per_image_csv": csv_path,
    }
    if not is_holdout:
        summary["warnings"].append(
            "test_mask_seed equals train MASK_SEED; test set is not hold-out."
        )
    if _nanmean(overlap_vals) > 0.05:
        summary["warnings"].append(
            f"mean mask_overlap_frac={_nanmean(overlap_vals):.4f}; "
            "high overlap can make Test MSE optimistic."
        )
    if n_low > 0:
        summary["warnings"].append(
            f"{n_low}/{len(per_image)} images flagged low_informativeness "
            "(near-zero response variance or R^2 undefined)."
        )

    summary_path = os.path.join(out_dir, "surrogate_quality_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] wrote {csv_path}")
    print(f"[INFO] wrote {summary_path}")
    print(
        f"[SUMMARY] n={len(per_image)} "
        f"mse_mean={summary['metrics']['test_mse_weighted']['mean']:.6f} "
        f"r2_mean={summary['metrics']['r2']['mean']:.4f} "
        f"V_mean={summary['metrics']['source_response_variance']['mean']:.6f} "
        f"overlap_mean={summary['metrics']['mask_overlap_frac']['mean']:.4f} "
        f"low_info={n_low} holdout={is_holdout}"
    )
    for w in summary["warnings"]:
        print(f"[WARN] {w}")

    summary["summary_json"] = summary_path
    return summary


# ---------------------------------------------------------------------------
# Subnet quality: E_loc / E_src + target retention
# ---------------------------------------------------------------------------


def _safe_rmtree(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)


def load_hard_masks(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    hard = torch.load(path, map_location="cpu")
    if not isinstance(hard, dict):
        raise TypeError(f"Expected dict hard masks in {path}")
    return {k: v.float().to(device) for k, v in hard.items()}


def build_masked_ms(
    num_class: int,
    checkpoint_path: str,
    extracted_layer: str,
    enable: Dict[str, bool],
    device: torch.device,
    hard: Optional[Dict[str, torch.Tensor]] = None,
) -> MaskedResNet18Net:
    """Fresh source Ms wrapped with MaskedResNet18Net; optional hard masks."""
    base = load_source_model(num_class, checkpoint_path, extracted_layer, device)
    masked = MaskedResNet18Net(
        base=base,
        extracted_layer=extracted_layer,
        enable=enable,
        init="zeros",
    )
    masked.to(device)
    masked.eval()
    if hard is not None:
        masked.apply_hard_masks(hard)
    return masked


def make_random_hard_masks(
    block_meta: Sequence[Dict[str, Any]],
    masks_cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Uniform random soft scores -> soft_to_hard_masks (same top-k / S_O rules)."""
    enable = {
        "M_H": bool(masks_cfg.get("enable", {}).get("M_H", True)),
        "M_O": bool(masks_cfg.get("enable", {}).get("M_O", True)),
    }
    rng = np.random.default_rng(int(seed))
    soft: Dict[str, torch.Tensor] = {}
    for meta in block_meta:
        bkey = meta["key"]
        if enable["M_H"]:
            soft[f"{bkey}.M_H"] = torch.from_numpy(
                rng.random(int(meta["n_h"])).astype(np.float32)
            )
        if enable["M_O"]:
            soft[f"{bkey}.M_O"] = torch.from_numpy(
                rng.random(int(meta["n_o"])).astype(np.float32)
            )
    if not soft:
        raise RuntimeError("No mask types enabled for random hard masks")
    hard, _ = soft_to_hard_masks(
        soft,
        keep_ratio=keep_ratio_map_for_keys(list(soft.keys()), masks_cfg),
        top_k_cfg=top_k_map_for_keys(list(soft.keys()), masks_cfg),
        block_meta=block_meta,
    )
    return hard


def clean_class_index(class_names: Sequence[str]) -> int:
    for i, name in enumerate(class_names):
        if str(name).lower() == "clean":
            return i
    raise ValueError(f"No 'clean' class in {list(class_names)}")


def compute_retention_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    target_idx: int,
    clean_idx: int,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    n_class = len(class_names)
    labels = list(range(n_class))
    f1s = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    target_f1 = float(f1s[target_idx])
    mask_t = y_true == target_idx
    target_recall = (
        float((y_pred[mask_t] == target_idx).mean()) if mask_t.any() else 0.0
    )
    mask_c = y_true == clean_idx
    clean_fpr = (
        float((y_pred[mask_c] == target_idx).mean()) if mask_c.any() else 0.0
    )
    # Target AUROC: one-vs-rest on target-class probability
    y_bin = (y_true == target_idx).astype(np.int64)
    target_auroc: Optional[float]
    if y_bin.min() == y_bin.max():
        target_auroc = None
    else:
        try:
            target_auroc = float(roc_auc_score(y_bin, probs[:, target_idx]))
        except ValueError:
            target_auroc = None

    return {
        "target_f1": target_f1,
        "target_recall": target_recall,
        "target_auroc": target_auroc,
        "clean_fpr": clean_fpr,
        "n_samples": int(len(y_true)),
        "n_target": int(mask_t.sum()),
        "n_clean": int(mask_c.sum()),
        "accuracy": float((y_pred == y_true).mean()) if len(y_true) else 0.0,
        "target_class": class_names[target_idx],
        "clean_class": class_names[clean_idx],
    }


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, preds, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model.predict(x, test_flag=1)
        prob = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        ys.append(y.numpy())
        preds.append(pred.cpu().numpy())
        probs.append(prob.cpu().numpy())
    return (
        np.concatenate(ys, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(probs, axis=0),
    )


def generate_node_test_loader(
    training_report: Dict[str, Any],
    node: str,
    class_names: Sequence[str],
    extracted_layer: str,
    batch_size: int,
    work_root: str,
) -> Tuple[DataLoader, str]:
    """Generate temporary node test ImageFolder; return (loader, output_root)."""
    os.makedirs(work_root, exist_ok=True)
    dataset_root = training_report.get("dataset_root") or _NODE_DATASET_DIR
    trigger_dir = training_report["trigger_dir"]
    results = ntdg.generate_all_nodes(
        dataset_root=dataset_root,
        node_specs={node: list(class_names)},
        trigger_dir=trigger_dir,
        output_root=work_root,
        blended_alpha_square=float(training_report["blended_alpha_square"]),
        blended_alpha_hello_kitty=float(
            training_report["blended_alpha_hello_kitty"]
        ),
        extracted_layer=extracted_layer,
        seed=int(training_report.get("seed", 0)),
        extract_features=False,
    )
    test_root = os.path.join(results[node]["output_dir"], "test")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"eval test split missing: {test_root}")
    ds = datasets.ImageFolder(root=test_root, transform=IMAGENET_TRANSFORM)
    folder_names = [n.split("_", 1)[-1] for n in ds.classes]
    if folder_names != list(class_names):
        print(
            f"[WARN] ImageFolder class order {folder_names} != expected "
            f"{list(class_names)}; metrics use ImageFolder indices."
        )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return loader, work_root


def evaluate_functional_localization(
    localization_dir: str,
    loc_cfg: Dict[str, Any],
    patch_dir: str,
    patch_resolved: Dict[str, Any],
    device: torch.device,
    test_mask_seed: int,
    batch_size: int,
    hard_learned: Dict[str, torch.Tensor],
    enable: Dict[str, bool],
) -> Dict[str, Any]:
    """Compute E_loc and E_src on B_test (mask_seed) over localization images."""
    patch_cfg = patch_resolved["patch_config"]
    selection = patch_resolved["selection"]
    training_report = patch_resolved["training_report"]

    grid_rows = int(loc_cfg.get("grid_rows") or patch_cfg["GRID_ROWS"])
    grid_cols = int(loc_cfg.get("grid_cols") or patch_cfg["GRID_COLS"])
    keep_prob = float(loc_cfg.get("keep_prob") or patch_cfg["KEEP_PROB"])
    num_masks = int(patch_cfg["NUM_MASKS_PER_IMAGE"])
    image_size = int(patch_cfg.get("IMAGE_SIZE", 224))
    include_reference = bool(loc_cfg.get("include_reference", True))

    attack_class = str(
        loc_cfg.get("resolved_attack_class") or selection["attack_class"]
    )
    node = str(loc_cfg.get("resolved_node") or selection["node"])
    class_names = list(loc_cfg.get("resolved_class_names") or [])
    if not class_names:
        class_names = list(training_report["node_specs"][node])
    target_idx = int(loc_cfg["resolved_target_class_idx"])
    extracted = str(
        loc_cfg.get("resolved_extracted_layer")
        or loc_cfg.get("extracted_layer")
        or "7_point"
    )
    ckpt = loc_cfg["checkpoint_path"]

    W_by_image = load_surrogates(os.path.join(localization_dir, "surrogate_models"))
    rows = load_manifest(os.path.join(patch_dir, "manifest.jsonl"))
    groups = group_manifest_by_image(rows, include_reference=include_reference)

    blended_alpha = resolve_blended_alpha(
        attack_class,
        float(training_report["blended_alpha_square"]),
        float(training_report["blended_alpha_hello_kitty"]),
    )
    triggers = load_trigger_arrays(
        training_report["trigger_dir"], training_report["node_specs"][node]
    )
    trigger_rgb, mask_rgb = triggers[attack_class]
    if trigger_rgb is None or mask_rgb is None:
        raise RuntimeError(f"Missing trigger/mask for {attack_class!r}")
    baseline = patch_gen.resolve_mask_baseline(
        patch_cfg["MASK_BASELINE"],
        patch_cfg.get("MASK_BASELINE_CHANNEL_CONSTANTS"),
        image_size,
        image_size,
    )

    ms_src = load_source_model(len(class_names), ckpt, extracted, device)
    ms_mask = build_masked_ms(
        len(class_names), ckpt, extracted, enable, device, hard=hard_learned
    )

    per_image: List[Dict[str, Any]] = []
    sum_loc = 0.0
    sum_src = 0.0
    n_terms = 0
    image_ids = sorted(set(W_by_image.keys()) & set(groups.keys()))
    print(
        f"[subnet] E_loc/E_src over {len(image_ids)} images, "
        f"|B_test|={num_masks}, test_mask_seed={test_mask_seed}"
    )

    for idx, image_id in enumerate(image_ids):
        fit_rows = groups[image_id]
        source_face_path = fit_rows[0]["source_face_path"]
        B_test = generate_test_masks(
            image_id=image_id,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            num_masks=num_masks,
            keep_prob=keep_prob,
            test_mask_seed=test_mask_seed,
        )
        x = rebuild_poisoned_tensor(
            source_face_path=source_face_path,
            attack_class=attack_class,
            trigger_rgb=trigger_rgb,
            mask_rgb=mask_rgb,
            blended_alpha=blended_alpha,
            image_size=image_size,
        )
        test_tensors = apply_masks_to_poisoned(
            x, B_test, grid_rows, grid_cols, baseline, image_size
        )
        Y_src = collect_ms_logits_from_tensors(
            ms_src, test_tensors, device, batch_size=batch_size
        )
        Y_mask = collect_ms_logits_from_tensors(
            ms_mask, test_tensors, device, batch_size=batch_size
        )
        W = W_by_image[image_id]
        g_t = (B_test.astype(np.float64) @ W)[:, target_idx]
        y_mask_t = Y_mask[:, target_idx]
        y_src_t = Y_src[:, target_idx]
        e_loc_i = float(np.mean((y_mask_t - g_t) ** 2))
        e_src_i = float(np.mean((y_mask_t - y_src_t) ** 2))
        k = int(B_test.shape[0])
        sum_loc += float(np.sum((y_mask_t - g_t) ** 2))
        sum_src += float(np.sum((y_mask_t - y_src_t) ** 2))
        n_terms += k
        per_image.append(
            {
                "image_idx": idx,
                "image_id": image_id,
                "n_test_masks": k,
                "e_loc": e_loc_i,
                "e_src": e_src_i,
            }
        )
        if (idx + 1) % 10 == 0 or idx + 1 == len(image_ids):
            print(
                f"[subnet-loc] {idx + 1}/{len(image_ids)} {image_id} "
                f"E_loc={e_loc_i:.6f} E_src={e_src_i:.6f}"
            )

    e_loc = float(sum_loc / max(n_terms, 1))
    e_src = float(sum_src / max(n_terms, 1))
    return {
        "E_loc": e_loc,
        "E_src": e_src,
        "n_images": len(image_ids),
        "n_terms": n_terms,
        "n_test_masks_per_image": num_masks,
        "test_mask_seed": int(test_mask_seed),
        "per_image": per_image,
    }


def evaluate_subnet_quality(
    localization_dir: str,
    patch_run_dir: Optional[str] = None,
    test_mask_seed: int = DEFAULT_TEST_MASK_SEED,
    device_str: Optional[str] = None,
    batch_size: int = 32,
    random_seeds: int = DEFAULT_RANDOM_SEEDS,
    random_base_seed: int = DEFAULT_RANDOM_BASE_SEED,
) -> Dict[str, Any]:
    localization_dir = os.path.abspath(localization_dir)
    cfg_path = os.path.join(localization_dir, "config.json")
    hard_path = os.path.join(localization_dir, "hard_masks.pt")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(cfg_path)
    if not os.path.isfile(hard_path):
        raise FileNotFoundError(
            f"hard_masks.pt not found under {localization_dir}; "
            "cannot run subnet quality"
        )

    loc_cfg = load_json(cfg_path)
    patch_dir = resolve_patch_run_dir(
        loc_cfg.get("patch_run_dir"), override=patch_run_dir
    )
    patch_resolved = load_json(os.path.join(patch_dir, "config_resolved.json"))
    training_report = patch_resolved["training_report"]
    selection = patch_resolved["selection"]

    node = str(loc_cfg.get("resolved_node") or selection["node"])
    attack_class = str(
        loc_cfg.get("resolved_attack_class") or selection["attack_class"]
    )
    class_names = list(loc_cfg.get("resolved_class_names") or [])
    if not class_names:
        class_names = list(training_report["node_specs"][node])
    target_idx = int(loc_cfg["resolved_target_class_idx"])
    clean_idx = clean_class_index(class_names)
    extracted = str(
        loc_cfg.get("resolved_extracted_layer")
        or loc_cfg.get("extracted_layer")
        or "7_point"
    )
    ckpt = loc_cfg.get("checkpoint_path")
    if not ckpt or not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint_path missing: {ckpt}")

    masks_cfg = loc_cfg.get("masks") or {
        "enable": {"M_H": True, "M_O": True},
        "keep_ratio": {"M_H": 0.25, "M_O": 0.25},
        "top_k": {"M_H": None, "M_O": None},
    }
    enable = {
        "M_H": bool(masks_cfg.get("enable", {}).get("M_H", True)),
        "M_O": bool(masks_cfg.get("enable", {}).get("M_O", True)),
    }

    device_name = device_str or loc_cfg.get("device") or "cpu"
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available; falling back to CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    out_dir = os.path.join(localization_dir, "subnet_quality")
    os.makedirs(out_dir, exist_ok=True)

    hard_learned = load_hard_masks(hard_path, device)
    print(f"[INFO] subnet quality: localization_dir={localization_dir}")
    print(f"[INFO] hard_masks={hard_path} keys={sorted(hard_learned)}")
    print(
        f"[INFO] node={node} attack={attack_class} "
        f"target={class_names[target_idx]} clean={class_names[clean_idx]}"
    )

    # ---- A. E_loc / E_src ----
    func_loc = evaluate_functional_localization(
        localization_dir=localization_dir,
        loc_cfg=loc_cfg,
        patch_dir=patch_dir,
        patch_resolved=patch_resolved,
        device=device,
        test_mask_seed=test_mask_seed,
        batch_size=batch_size,
        hard_learned=hard_learned,
        enable=enable,
    )
    func_csv = os.path.join(out_dir, "functional_localization.csv")
    _write_csv(func_csv, func_loc["per_image"])
    func_json = {
        "E_loc": func_loc["E_loc"],
        "E_src": func_loc["E_src"],
        "n_images": func_loc["n_images"],
        "n_terms": func_loc["n_terms"],
        "n_test_masks_per_image": func_loc["n_test_masks_per_image"],
        "test_mask_seed": func_loc["test_mask_seed"],
        "per_image_csv": func_csv,
    }
    with open(
        os.path.join(out_dir, "functional_localization.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(func_json, f, indent=2, ensure_ascii=False)
    print(
        f"[SUMMARY] E_loc={func_loc['E_loc']:.6f} "
        f"E_src={func_loc['E_src']:.6f}"
    )

    # ---- B. Target retention on node test ----
    # Probe block_meta from a temporary masked wrapper (for random masks)
    probe = build_masked_ms(
        len(class_names), ckpt, extracted, enable, device, hard=None
    )
    block_meta = probe.block_meta
    del probe

    work_root = tempfile.mkdtemp(
        prefix="subnet_eval_", dir=out_dir
    )
    retention: Dict[str, Any] = {}
    try:
        print(f"[INFO] generating node test data under {work_root}")
        loader, _ = generate_node_test_loader(
            training_report=training_report,
            node=node,
            class_names=class_names,
            extracted_layer=extracted,
            batch_size=batch_size,
            work_root=work_root,
        )
        print(f"[INFO] node test n={len(loader.dataset)}")

        # Original Ms
        ms_orig = load_source_model(len(class_names), ckpt, extracted, device)
        y_t, y_p, probs = predict_loader(ms_orig, loader, device)
        retention["original"] = compute_retention_metrics(
            y_t, y_p, probs, target_idx, clean_idx, class_names
        )
        print(
            f"[retention] original "
            f"F1={retention['original']['target_f1']:.4f} "
            f"R={retention['original']['target_recall']:.4f} "
            f"AUROC={retention['original']['target_auroc']} "
            f"cleanFPR={retention['original']['clean_fpr']:.4f}"
        )
        del ms_orig

        # Learned hard mask
        ms_learned = build_masked_ms(
            len(class_names),
            ckpt,
            extracted,
            enable,
            device,
            hard=hard_learned,
        )
        y_t, y_p, probs = predict_loader(ms_learned, loader, device)
        retention["learned"] = compute_retention_metrics(
            y_t, y_p, probs, target_idx, clean_idx, class_names
        )
        print(
            f"[retention] learned "
            f"F1={retention['learned']['target_f1']:.4f} "
            f"R={retention['learned']['target_recall']:.4f} "
            f"AUROC={retention['learned']['target_auroc']} "
            f"cleanFPR={retention['learned']['clean_fpr']:.4f}"
        )
        del ms_learned

        # Random baselines
        random_rows: List[Dict[str, Any]] = []
        for k in range(int(random_seeds)):
            seed = int(random_base_seed) + k
            hard_r = make_random_hard_masks(block_meta, masks_cfg, seed)
            hard_r = {kk: vv.to(device) for kk, vv in hard_r.items()}
            ms_r = build_masked_ms(
                len(class_names),
                ckpt,
                extracted,
                enable,
                device,
                hard=hard_r,
            )
            y_t, y_p, probs = predict_loader(ms_r, loader, device)
            m = compute_retention_metrics(
                y_t, y_p, probs, target_idx, clean_idx, class_names
            )
            m["seed"] = seed
            random_rows.append(m)
            print(
                f"[retention] random seed={seed} "
                f"F1={m['target_f1']:.4f} R={m['target_recall']:.4f} "
                f"cleanFPR={m['clean_fpr']:.4f}"
            )
            del ms_r
        retention["random"] = random_rows
    finally:
        _safe_rmtree(work_root)
        print(f"[INFO] cleaned eval data {work_root}")

    rand_f1 = [r["target_f1"] for r in retention["random"]]
    rand_rec = [r["target_recall"] for r in retention["random"]]
    rand_fpr = [r["clean_fpr"] for r in retention["random"]]
    rand_auroc = [
        r["target_auroc"]
        for r in retention["random"]
        if r["target_auroc"] is not None
    ]
    random_summary = {
        "n_seeds": len(retention["random"]),
        "target_f1_mean": _nanmean(rand_f1),
        "target_f1_std": float(np.std(rand_f1)) if rand_f1 else float("nan"),
        "target_recall_mean": _nanmean(rand_rec),
        "target_recall_std": float(np.std(rand_rec)) if rand_rec else float("nan"),
        "clean_fpr_mean": _nanmean(rand_fpr),
        "clean_fpr_std": float(np.std(rand_fpr)) if rand_fpr else float("nan"),
        "target_auroc_mean": _nanmean(rand_auroc) if rand_auroc else None,
    }

    retention_path = os.path.join(out_dir, "retention_per_model.json")
    with open(retention_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "original": retention["original"],
                "learned": retention["learned"],
                "random": retention["random"],
                "random_summary": random_summary,
                "class_names": class_names,
                "target_class_idx": target_idx,
                "clean_class_idx": clean_idx,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    warnings: List[str] = []
    learned_f1 = float(retention["learned"]["target_f1"])
    rand_mean = float(random_summary["target_f1_mean"])
    if not (learned_f1 > rand_mean):
        warnings.append(
            f"learned target_f1={learned_f1:.4f} is not higher than "
            f"random mean={rand_mean:.4f}; localization may not beat chance "
            "at this keep_ratio."
        )

    summary: Dict[str, Any] = {
        "localization_dir": localization_dir,
        "patch_run_dir": patch_dir,
        "checkpoint_path": ckpt,
        "node": node,
        "attack_class": attack_class,
        "target_class_idx": target_idx,
        "target_class_name": class_names[target_idx],
        "clean_class_idx": clean_idx,
        "extracted_layer": extracted,
        "keep_ratio": {
            t: float(masks_cfg.get("keep_ratio", {}).get(t, 0.25))
            for t in MASK_TYPES
        },
        "enable": enable,
        "random_seeds": int(random_seeds),
        "random_base_seed": int(random_base_seed),
        "functional_localization": {
            "E_loc": func_loc["E_loc"],
            "E_src": func_loc["E_src"],
            "n_images": func_loc["n_images"],
            "n_terms": func_loc["n_terms"],
        },
        "retention": {
            "original": retention["original"],
            "learned": retention["learned"],
            "random_summary": random_summary,
        },
        "learned_vs_random": {
            "learned_target_f1": learned_f1,
            "random_target_f1_mean": rand_mean,
            "random_target_f1_std": random_summary["target_f1_std"],
            "delta_f1_learned_minus_random_mean": learned_f1 - rand_mean,
        },
        "warnings": warnings,
        "paths": {
            "functional_localization_csv": func_csv,
            "functional_localization_json": os.path.join(
                out_dir, "functional_localization.json"
            ),
            "retention_per_model_json": retention_path,
        },
    }
    summary_path = os.path.join(out_dir, "subnet_quality_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] wrote {retention_path}")
    print(f"[INFO] wrote {summary_path}")
    print(
        f"[SUMMARY] learned_F1={learned_f1:.4f} "
        f"random_F1={rand_mean:.4f}±{random_summary['target_f1_std']:.4f} "
        f"delta={learned_f1 - rand_mean:.4f}"
    )
    for w in warnings:
        print(f"[WARN] {w}")

    summary["summary_json"] = summary_path
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate localization quality: local-surrogate metrics and/or "
            "subnet metrics (E_loc/E_src + target retention vs random)."
        )
    )
    p.add_argument(
        "--localization_dir",
        required=True,
        help="Path to a localization_result/<run>/ directory",
    )
    p.add_argument(
        "--patch_run_dir",
        default=None,
        help=(
            "Override patch metadata directory (config_resolved.json + "
            "manifest.jsonl). Default: resolve from localization config "
            "with path fallbacks."
        ),
    )
    p.add_argument(
        "--test_mask_seed",
        type=int,
        default=DEFAULT_TEST_MASK_SEED,
        help="Fixed Bernoulli seed for hold-out test masks (default: 0)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda:0 / cpu; default from localization config.json",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Forward batch size",
    )
    p.add_argument(
        "--r2_eps",
        type=float,
        default=R2_DENOM_EPS,
        help="R^2 total-sum-of-squares floor; below -> nan",
    )
    p.add_argument(
        "--skip_surrogate",
        action="store_true",
        help="Skip local-surrogate quality section",
    )
    p.add_argument(
        "--skip_subnet",
        action="store_true",
        help="Skip subnet quality section",
    )
    p.add_argument(
        "--random_seeds",
        type=int,
        default=DEFAULT_RANDOM_SEEDS,
        help="Number of random hard-mask baselines (default: 10)",
    )
    p.add_argument(
        "--random_base_seed",
        type=int,
        default=DEFAULT_RANDOM_BASE_SEED,
        help="Base seed for random masks; actual seed = base + k",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    loc_dir = os.path.abspath(args.localization_dir)
    hard_path = os.path.join(loc_dir, "hard_masks.pt")

    if not args.skip_surrogate:
        evaluate_localization_surrogates(
            localization_dir=loc_dir,
            patch_run_dir=args.patch_run_dir,
            test_mask_seed=int(args.test_mask_seed),
            device_str=args.device,
            batch_size=int(args.batch_size),
            r2_eps=float(args.r2_eps),
        )
    else:
        print("[INFO] skip_surrogate: surrogate quality not run")

    run_subnet = not args.skip_subnet
    if run_subnet and not os.path.isfile(hard_path):
        print(
            f"[WARN] hard_masks.pt missing at {hard_path}; "
            "skipping subnet quality"
        )
        run_subnet = False
    if run_subnet:
        evaluate_subnet_quality(
            localization_dir=loc_dir,
            patch_run_dir=args.patch_run_dir,
            test_mask_seed=int(args.test_mask_seed),
            device_str=args.device,
            batch_size=int(args.batch_size),
            random_seeds=int(args.random_seeds),
            random_base_seed=int(args.random_base_seed),
        )
    elif args.skip_subnet:
        print("[INFO] skip_subnet: subnet quality not run")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
