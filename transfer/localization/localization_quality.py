#!/usr/bin/env python3
"""Validate localization quality for a localization_result run.

Surrogate section (per-image):
  - Weighted Test MSE / R^2 on hold-out masks (attack, clean, attack-clean margin)
  - Bootstrap CI for mean/median; absolute_level vs comparative blocks
  - Source Response Variance on fitting masks (valid only if MASK_SEED consistent)

Subnet section (when hard_masks.pt exists):
  - E_loc / E_src on hold-out patch masks
  - Primary: logit margin retention, FPR@TPR=95%, necessity ablate, keep-ratio curve
  - Linear probe (target vs clean) on subnet features
  - Reference: argmax F1/cleanFPR retention (may saturate)

Compare mode:
  --compare_localization_dirs dir1,dir2,...  → soft Spearman + hard Jaccard;
  paired R^2 diffs when per-image CSVs exist.

Masked patch inputs are rebuilt in memory from retained patch metadata;
tensors/ need not exist after sweep cleanup.

Examples:
  python localization_quality.py --localization_dir /path/to/loc --device cuda:0
  python localization_quality.py --compare_localization_dirs d1,d2,d3
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


def sha256_file(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_mean_median_ci(
    vals: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Bootstrap CI for mean/median of a 1-d metric vector (nan-aware)."""
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    out: Dict[str, Any] = {
        "se_mean": float("nan"),
        "mean_ci95": [float("nan"), float("nan")],
        "median_ci95": [float("nan"), float("nan")],
        "n_boot": int(n_boot),
        "n_effective": int(arr.size),
    }
    if arr.size == 0:
        return out
    if arr.size == 1:
        v = float(arr[0])
        out["se_mean"] = 0.0
        out["mean_ci95"] = [v, v]
        out["median_ci95"] = [v, v]
        return out
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    medians = np.empty(int(n_boot), dtype=np.float64)
    n = arr.size
    for i in range(int(n_boot)):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = float(np.mean(sample))
        medians[i] = float(np.median(sample))
    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)
    out["se_mean"] = float(np.std(means, ddof=1))
    out["mean_ci95"] = [
        float(np.percentile(means, lo)),
        float(np.percentile(means, hi)),
    ]
    out["median_ci95"] = [
        float(np.percentile(medians, lo)),
        float(np.percentile(medians, hi)),
    ]
    return out


def aggregate_metric_block(
    vals: Sequence[float],
    n_boot: int = 1000,
    boot_seed: int = 0,
) -> Dict[str, Any]:
    block = {
        "mean": _nanmean(vals),
        "median": _nanmedian(vals),
        "p10": _nanpercentile(vals, 10),
        "p90": _nanpercentile(vals, 90),
        "n_nan": int(sum(1 for v in vals if not np.isfinite(v))),
    }
    block.update(bootstrap_mean_median_ci(vals, n_boot=n_boot, seed=boot_seed))
    return block


def logit_margin(logits: np.ndarray, target_idx: int) -> np.ndarray:
    """Per-row margin: z_target - max_{c!=target} z_c."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    n, c = z.shape
    if not (0 <= int(target_idx) < c):
        raise ValueError(f"target_idx={target_idx} out of range for C={c}")
    t = z[:, int(target_idx)]
    mask = np.ones(c, dtype=bool)
    mask[int(target_idx)] = False
    other = np.max(z[:, mask], axis=1)
    return t - other


def fpr_at_tpr(
    scores: np.ndarray,
    y_true_bin: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """FPR at the lowest threshold achieving TPR >= target_tpr (higher score = more positive)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true_bin, dtype=np.int64).reshape(-1)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Sort thresholds from high to low
    order = np.argsort(-scores)
    y_s = y[order]
    tp = 0
    fp = 0
    best_fpr = 1.0
    for label in y_s:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        if tpr >= float(target_tpr):
            best_fpr = fp / n_neg
            break
    return float(best_fpr)


def margin_retention_rate(
    m_full: float,
    m_sub: float,
    m_rand: float,
    eps: float = 1e-6,
) -> float:
    denom = float(m_full) - float(m_rand)
    if abs(denom) < float(eps):
        return float("nan")
    return float((float(m_sub) - float(m_rand)) / denom)


def complement_hard_masks(
    hard: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Invert 0/1 hard gates (ablate the kept subnet channels)."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in hard.items():
        t = v.detach().float().cpu()
        out[k] = (1.0 - t).clamp(0.0, 1.0)
    return out


def random_ablate_hard_masks(
    hard_template: Dict[str, torch.Tensor],
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Randomly ablate the same number of channels as learned keeps per key."""
    rng = np.random.default_rng(int(seed))
    out: Dict[str, torch.Tensor] = {}
    for k, v in hard_template.items():
        t = v.detach().float().cpu().reshape(-1)
        n = int(t.numel())
        k_keep = int((t > 0.5).sum().item())
        # Ablate = turn off k_keep channels from a full-ones mask
        full = torch.ones(n, dtype=torch.float32)
        if k_keep > 0:
            idx = rng.choice(n, size=min(k_keep, n), replace=False)
            full[torch.as_tensor(idx, dtype=torch.long)] = 0.0
        out[k] = full
    return out


def hard_jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().float().cpu().reshape(-1) > 0.5
    bb = b.detach().float().cpu().reshape(-1) > 0.5
    inter = int((aa & bb).sum().item())
    union = int((aa | bb).sum().item())
    if union == 0:
        return float("nan")
    return float(inter / union)


def soft_spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    from scipy.stats import spearmanr

    aa = a.detach().float().cpu().numpy().reshape(-1)
    bb = b.detach().float().cpu().numpy().reshape(-1)
    if aa.size < 2:
        return float("nan")
    rho, _ = spearmanr(aa, bb)
    return float(rho) if np.isfinite(rho) else float("nan")


def trapezoid_auc(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    if x.size < 2:
        return float("nan")
    order = np.argsort(x)
    xx, yy = x[order], y[order]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(yy, xx))


def resolve_expected_mask_seed(loc_cfg: Dict[str, Any], patch_cfg: Dict[str, Any]) -> int:
    for key in ("MASK_SEED", "train_mask_seed"):
        if loc_cfg.get(key) is not None:
            return int(loc_cfg[key])
    return int(patch_cfg["MASK_SEED"])


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
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
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
    loc_has_mask_seed = loc_cfg.get("MASK_SEED") is not None or (
        loc_cfg.get("train_mask_seed") is not None
    )
    expected_mask_seed = resolve_expected_mask_seed(loc_cfg, patch_cfg)
    if loc_has_mask_seed:
        mask_seed_consistent = int(expected_mask_seed) == int(train_mask_seed)
    else:
        # Legacy loc configs did not snapshot MASK_SEED; cannot verify.
        mask_seed_consistent = False
    include_reference = bool(loc_cfg.get("include_reference", True))
    manifest_path = os.path.join(patch_dir, "manifest.jsonl")
    patch_manifest_sha256 = sha256_file(manifest_path)

    attack_class = str(
        loc_cfg.get("resolved_attack_class") or selection["attack_class"]
    )
    node = str(loc_cfg.get("resolved_node") or selection["node"])
    class_names = list(loc_cfg.get("resolved_class_names") or [])
    if not class_names:
        class_names = list(training_report["node_specs"][node])
    target_idx = int(loc_cfg["resolved_target_class_idx"])
    try:
        clean_idx = clean_class_index(class_names)
    except ValueError:
        clean_idx = None
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

        r2_clean = float("nan")
        mse_clean = float("nan")
        r2_margin = float("nan")
        mse_margin = float("nan")
        if clean_idx is not None:
            y_test_c = Y_test[:, clean_idx]
            g_test_c = G_test[:, clean_idx]
            r2_clean = r2_score(y_test_c, g_test_c, eps=r2_eps)
            mse_clean = weighted_test_mse_target(y_test_c, g_test_c, pi_test)
            y_m = y_test_t - y_test_c
            g_m = g_test_t - g_test_c
            r2_margin = r2_score(y_m, g_m, eps=r2_eps)
            mse_margin = weighted_test_mse_target(y_m, g_m, pi_test)

        row = {
            "image_idx": idx,
            "image_id": image_id,
            "n_train_masks": int(B_fit.shape[0]),
            "n_test_masks": int(B_test.shape[0]),
            "test_mse_weighted": mse_t,
            "test_mse_weighted_multiclass": mse_mc,
            "r2": r2,
            "r2_clean": r2_clean,
            "test_mse_clean": mse_clean,
            "r2_margin": r2_margin,
            "test_mse_margin": mse_margin,
            "source_response_variance": v_i,
            "mask_overlap_frac": overlap,
            "low_informativeness": int(low_info),
            "is_holdout": int(is_holdout),
            "train_mask_seed": train_mask_seed,
            "test_mask_seed": int(test_mask_seed),
            "target_class_idx": target_idx,
            "clean_class_idx": clean_idx if clean_idx is not None else -1,
            "include_reference_in_V": int(include_reference),
            "n_fit_mask_ids": len(fit_mask_ids),
            "mask_seed_consistent": int(mask_seed_consistent),
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
    r2_clean_vals = [r["r2_clean"] for r in per_image]
    mse_clean_vals = [r["test_mse_clean"] for r in per_image]
    r2_margin_vals = [r["r2_margin"] for r in per_image]
    mse_margin_vals = [r["test_mse_margin"] for r in per_image]
    n_low = int(sum(int(r["low_informativeness"]) for r in per_image))
    n_boot = int(bootstrap_samples)

    metrics = {
        "test_mse_weighted": aggregate_metric_block(
            mse_vals, n_boot=n_boot, boot_seed=bootstrap_seed
        ),
        "r2": aggregate_metric_block(
            r2_vals, n_boot=n_boot, boot_seed=bootstrap_seed + 1
        ),
        "source_response_variance": {
            "mean": _nanmean(v_vals),
            "median": _nanmedian(v_vals),
            "p10": _nanpercentile(v_vals, 10),
            "p90": _nanpercentile(v_vals, 90),
            "valid_for_ranking": bool(mask_seed_consistent),
        },
        "mask_overlap_frac": {
            "mean": _nanmean(overlap_vals),
            "median": _nanmedian(overlap_vals),
            "p90": _nanpercentile(overlap_vals, 90),
            "valid_for_ranking": bool(mask_seed_consistent),
        },
        "test_mse_weighted_multiclass": {
            "mean": _nanmean(
                [r["test_mse_weighted_multiclass"] for r in per_image]
            ),
            "median": _nanmedian(
                [r["test_mse_weighted_multiclass"] for r in per_image]
            ),
        },
        "r2_clean": aggregate_metric_block(
            r2_clean_vals, n_boot=n_boot, boot_seed=bootstrap_seed + 2
        ),
        "test_mse_clean": aggregate_metric_block(
            mse_clean_vals, n_boot=n_boot, boot_seed=bootstrap_seed + 3
        ),
        "r2_margin": aggregate_metric_block(
            r2_margin_vals, n_boot=n_boot, boot_seed=bootstrap_seed + 4
        ),
        "test_mse_margin": aggregate_metric_block(
            mse_margin_vals, n_boot=n_boot, boot_seed=bootstrap_seed + 5
        ),
    }

    warnings: List[str] = []
    if not is_holdout:
        warnings.append(
            "test_mask_seed equals train MASK_SEED; test set is not hold-out."
        )
    if _nanmean(overlap_vals) > 0.05:
        warnings.append(
            f"mean mask_overlap_frac={_nanmean(overlap_vals):.4f}; "
            "high overlap can make Test MSE optimistic."
        )
    if n_low > 0:
        warnings.append(
            f"{n_low}/{len(per_image)} images flagged low_informativeness "
            "(near-zero response variance or R^2 undefined)."
        )
    if not loc_has_mask_seed:
        warnings.append(
            "localization config.json has no MASK_SEED/train_mask_seed; "
            "cannot verify patch fit masks match this run "
            f"(patch MASK_SEED={train_mask_seed}). "
            "SRV/mask_overlap marked invalid for ranking."
        )
    elif not mask_seed_consistent:
        warnings.append(
            f"mask_seed_consistent=False: loc expected MASK_SEED="
            f"{expected_mask_seed} but patch config has {train_mask_seed}. "
            "source_response_variance / mask_overlap_frac may not reflect "
            "this localization run's original fit masks."
        )
    if clean_idx is None:
        warnings.append(
            "No 'clean' class; skipped r2_clean / r2_margin metrics."
        )

    summary: Dict[str, Any] = {
        "localization_dir": localization_dir,
        "patch_run_dir": patch_dir,
        "patch_manifest_sha256": patch_manifest_sha256,
        "checkpoint_path": ckpt,
        "node": node,
        "attack_class": attack_class,
        "target_class_idx": target_idx,
        "target_class_name": class_names[target_idx],
        "clean_class_idx": clean_idx,
        "n_images": len(per_image),
        "train_mask_seed": train_mask_seed,
        "expected_mask_seed": int(expected_mask_seed),
        "mask_seed_consistent": bool(mask_seed_consistent),
        "test_mask_seed": int(test_mask_seed),
        "is_holdout": is_holdout,
        "include_reference_in_V": include_reference,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "keep_prob": keep_prob,
        "num_test_masks_per_image": num_masks,
        "r2_denom_eps": r2_eps,
        "bootstrap_samples": n_boot,
        "n_low_informativeness": n_low,
        "warnings": warnings,
        "metrics": metrics,
        "absolute_level": {
            "r2": metrics["r2"],
            "test_mse_weighted": metrics["test_mse_weighted"],
            "r2_clean": metrics["r2_clean"],
            "test_mse_clean": metrics["test_mse_clean"],
            "r2_margin": metrics["r2_margin"],
            "test_mse_margin": metrics["test_mse_margin"],
            "source_response_variance": metrics["source_response_variance"],
            "note": (
                "Use absolute_level for single-run quality. "
                "Do not rank MASK_SEED trials from one summary alone."
            ),
        },
        "comparative": None,
        "comparative_note": (
            "comparative is null for a single run; use "
            "--compare_localization_dirs after per-image CSVs exist."
        ),
        "per_image_csv": csv_path,
    }

    summary_path = os.path.join(out_dir, "surrogate_quality_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] wrote {csv_path}")
    print(f"[INFO] wrote {summary_path}")
    print(
        f"[SUMMARY] n={len(per_image)} "
        f"mse_mean={metrics['test_mse_weighted']['mean']:.6f} "
        f"r2_mean={metrics['r2']['mean']:.4f} "
        f"r2_margin_mean={metrics['r2_margin']['mean']:.4f} "
        f"V_mean={metrics['source_response_variance']['mean']:.6f} "
        f"overlap_mean={metrics['mask_overlap_frac']['mean']:.4f} "
        f"low_info={n_low} holdout={is_holdout} "
        f"mask_seed_ok={mask_seed_consistent}"
    )
    for w in warnings:
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


@torch.no_grad()
def collect_logits_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, logits [N,C])."""
    ys, logits = [], []
    for x, y in loader:
        x = x.to(device)
        out = model.predict(x, test_flag=1)
        ys.append(y.numpy())
        logits.append(out.detach().cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(logits, axis=0)


@torch.no_grad()
def collect_features_loader(
    model: MaskedResNet18Net,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Features from base_network (pre-classifier), shape [N, D]."""
    ys, feats = [], []
    for x, y in loader:
        x = x.to(device)
        model.base.eval()
        f = model.base.base_network(x, test_flag=1)
        ys.append(y.numpy())
        feats.append(f.detach().cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(feats, axis=0)


def summarize_margins(
    y_true: np.ndarray,
    logits: np.ndarray,
    target_idx: int,
    clean_idx: int,
) -> Dict[str, float]:
    m = logit_margin(logits, target_idx)
    y_true = np.asarray(y_true).reshape(-1)
    mask_t = y_true == int(target_idx)
    mask_c = y_true == int(clean_idx)
    y_bin = (y_true == int(target_idx)).astype(np.int64)
    return {
        "margin_target_mean": float(m[mask_t].mean()) if mask_t.any() else float("nan"),
        "margin_clean_mean": float(m[mask_c].mean()) if mask_c.any() else float("nan"),
        "margin_all_mean": float(m.mean()) if m.size else float("nan"),
        "fpr_at_tpr95": fpr_at_tpr(m, y_bin, target_tpr=0.95),
        "n_target": int(mask_t.sum()),
        "n_clean": int(mask_c.sum()),
    }


def run_linear_probe_target_clean(
    feats: np.ndarray,
    y_true: np.ndarray,
    target_idx: int,
    clean_idx: int,
) -> Dict[str, Any]:
    """L2 logistic regression on target vs clean features only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    y_true = np.asarray(y_true).reshape(-1)
    keep = (y_true == int(target_idx)) | (y_true == int(clean_idx))
    if int(keep.sum()) < 8:
        return {"ok": False, "error": "too few target/clean samples"}
    X = np.asarray(feats, dtype=np.float64)[keep]
    y = (y_true[keep] == int(target_idx)).astype(np.int64)
    if y.min() == y.max():
        return {"ok": False, "error": "single-class after filter"}
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    out: Dict[str, Any] = {
        "ok": True,
        "accuracy": float(accuracy_score(yte, pred)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
    }
    try:
        out["auroc"] = float(roc_auc_score(yte, proba))
    except ValueError:
        out["auroc"] = None
    return out


def soft_to_hard_at_ratio(
    soft: Dict[str, torch.Tensor],
    block_meta: Sequence[Dict[str, Any]],
    masks_cfg: Dict[str, Any],
    keep_ratio: float,
) -> Dict[str, torch.Tensor]:
    cfg = {
        "enable": dict(masks_cfg.get("enable") or {"M_H": True, "M_O": True}),
        "keep_ratio": {"M_H": float(keep_ratio), "M_O": float(keep_ratio)},
        "top_k": masks_cfg.get("top_k") or {"M_H": None, "M_O": None},
    }
    hard, _ = soft_to_hard_masks(
        soft,
        keep_ratio=keep_ratio_map_for_keys(list(soft.keys()), cfg),
        top_k_cfg=top_k_map_for_keys(list(soft.keys()), cfg),
        block_meta=block_meta,
    )
    return hard


def load_soft_masks_blob(path: str) -> Dict[str, torch.Tensor]:
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "soft" in blob:
        soft = blob["soft"]
    elif isinstance(blob, dict):
        soft = blob
    else:
        raise TypeError(f"Unexpected soft_masks format in {path}")
    return {k: v.float().cpu() for k, v in soft.items()}


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
    keep_ratios: Optional[Sequence[float]] = None,
    skip_necessity: bool = False,
    skip_keep_ratio_curve: bool = False,
    skip_linear_probe: bool = False,
) -> Dict[str, Any]:
    localization_dir = os.path.abspath(localization_dir)
    cfg_path = os.path.join(localization_dir, "config.json")
    hard_path = os.path.join(localization_dir, "hard_masks.pt")
    soft_path = os.path.join(localization_dir, "soft_masks.pt")
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
    patch_cfg = patch_resolved["patch_config"]
    training_report = patch_resolved["training_report"]
    selection = patch_resolved["selection"]
    train_mask_seed = int(patch_cfg["MASK_SEED"])
    loc_has_mask_seed = loc_cfg.get("MASK_SEED") is not None or (
        loc_cfg.get("train_mask_seed") is not None
    )
    expected_mask_seed = resolve_expected_mask_seed(loc_cfg, patch_cfg)
    if loc_has_mask_seed:
        mask_seed_consistent = int(expected_mask_seed) == int(train_mask_seed)
    else:
        mask_seed_consistent = False
    manifest_path = os.path.join(patch_dir, "manifest.jsonl")
    patch_manifest_sha256 = sha256_file(manifest_path)

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
    if keep_ratios is None:
        keep_ratios = [0.02, 0.05, 0.1, 0.25]

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

    probe = build_masked_ms(
        len(class_names), ckpt, extracted, enable, device, hard=None
    )
    block_meta = probe.block_meta
    del probe

    work_root = tempfile.mkdtemp(prefix="subnet_eval_", dir=out_dir)
    retention: Dict[str, Any] = {}
    margin_block: Dict[str, Any] = {}
    necessity_block: Optional[Dict[str, Any]] = None
    keep_ratio_block: Optional[Dict[str, Any]] = None
    linear_probe_block: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    ret_rate = float("nan")

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

        # Original Ms — logits for margin + reference argmax
        ms_orig = load_source_model(len(class_names), ckpt, extracted, device)
        y_true, logits_full = collect_logits_loader(ms_orig, loader, device)
        y_pred_full = logits_full.argmax(axis=1)
        probs_full = np.exp(logits_full - logits_full.max(axis=1, keepdims=True))
        probs_full = probs_full / probs_full.sum(axis=1, keepdims=True)
        retention["original"] = compute_retention_metrics(
            y_true, y_pred_full, probs_full, target_idx, clean_idx, class_names
        )
        margin_full = summarize_margins(y_true, logits_full, target_idx, clean_idx)
        del ms_orig

        # Learned
        ms_learned = build_masked_ms(
            len(class_names), ckpt, extracted, enable, device, hard=hard_learned
        )
        y_true, logits_learned = collect_logits_loader(ms_learned, loader, device)
        y_pred_l = logits_learned.argmax(axis=1)
        probs_l = np.exp(logits_learned - logits_learned.max(axis=1, keepdims=True))
        probs_l = probs_l / probs_l.sum(axis=1, keepdims=True)
        retention["learned"] = compute_retention_metrics(
            y_true, y_pred_l, probs_l, target_idx, clean_idx, class_names
        )
        margin_learned = summarize_margins(
            y_true, logits_learned, target_idx, clean_idx
        )
        feats_learned = None
        if not skip_linear_probe:
            _, feats_learned = collect_features_loader(ms_learned, loader, device)
        del ms_learned

        # Random baselines (argmax reference + margin)
        random_rows: List[Dict[str, Any]] = []
        random_margin_rows: List[Dict[str, Any]] = []
        random_probe_rows: List[Dict[str, Any]] = []
        for k in range(int(random_seeds)):
            seed = int(random_base_seed) + k
            hard_r = make_random_hard_masks(block_meta, masks_cfg, seed)
            hard_r_dev = {kk: vv.to(device) for kk, vv in hard_r.items()}
            ms_r = build_masked_ms(
                len(class_names), ckpt, extracted, enable, device, hard=hard_r_dev
            )
            y_true, logits_r = collect_logits_loader(ms_r, loader, device)
            y_pred_r = logits_r.argmax(axis=1)
            probs_r = np.exp(logits_r - logits_r.max(axis=1, keepdims=True))
            probs_r = probs_r / probs_r.sum(axis=1, keepdims=True)
            m = compute_retention_metrics(
                y_true, y_pred_r, probs_r, target_idx, clean_idx, class_names
            )
            m["seed"] = seed
            random_rows.append(m)
            mm = summarize_margins(y_true, logits_r, target_idx, clean_idx)
            mm["seed"] = seed
            random_margin_rows.append(mm)
            if not skip_linear_probe:
                _, feats_r = collect_features_loader(ms_r, loader, device)
                pr = run_linear_probe_target_clean(
                    feats_r, y_true, target_idx, clean_idx
                )
                pr["seed"] = seed
                random_probe_rows.append(pr)
            print(
                f"[retention] random seed={seed} "
                f"F1={m['target_f1']:.4f} margin_t={mm['margin_target_mean']:.4f}"
            )
            del ms_r
        retention["random"] = random_rows

        m_rand_t = _nanmean([r["margin_target_mean"] for r in random_margin_rows])
        m_rand_c = _nanmean([r["margin_clean_mean"] for r in random_margin_rows])
        fpr_rand = _nanmean([r["fpr_at_tpr95"] for r in random_margin_rows])
        ret_rate = margin_retention_rate(
            margin_full["margin_target_mean"],
            margin_learned["margin_target_mean"],
            m_rand_t,
        )
        margin_block = {
            "original": margin_full,
            "learned": margin_learned,
            "random": random_margin_rows,
            "random_summary": {
                "margin_target_mean": m_rand_t,
                "margin_clean_mean": m_rand_c,
                "fpr_at_tpr95_mean": fpr_rand,
                "n_seeds": len(random_margin_rows),
            },
            "margin_retention": ret_rate,
            "delta_fpr_at_tpr95_learned_minus_random": (
                float(margin_learned["fpr_at_tpr95"]) - float(fpr_rand)
                if np.isfinite(margin_learned["fpr_at_tpr95"]) and np.isfinite(fpr_rand)
                else float("nan")
            ),
        }
        with open(
            os.path.join(out_dir, "margin_metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(margin_block, f, indent=2, ensure_ascii=False)
        print(
            f"[SUMMARY] margin_retention={ret_rate} "
            f"learned_target_m={margin_learned['margin_target_mean']:.4f} "
            f"rand_target_m={m_rand_t:.4f} "
            f"fpr95_learned={margin_learned['fpr_at_tpr95']:.4f}"
        )

        # ---- Necessity ----
        if not skip_necessity:
            hard_comp = {
                k: v.to(device) for k, v in complement_hard_masks(hard_learned).items()
            }
            ms_ab = build_masked_ms(
                len(class_names), ckpt, extracted, enable, device, hard=hard_comp
            )
            _, logits_ab = collect_logits_loader(ms_ab, loader, device)
            margin_ab = summarize_margins(y_true, logits_ab, target_idx, clean_idx)
            del ms_ab
            delta_learned = float(
                margin_full["margin_target_mean"] - margin_ab["margin_target_mean"]
            )
            delta_rand_list: List[float] = []
            for k in range(int(random_seeds)):
                seed = int(random_base_seed) + 1000 + k
                hard_ra = random_ablate_hard_masks(hard_learned, seed)
                hard_ra = {kk: vv.to(device) for kk, vv in hard_ra.items()}
                ms_ra = build_masked_ms(
                    len(class_names), ckpt, extracted, enable, device, hard=hard_ra
                )
                _, logits_ra = collect_logits_loader(ms_ra, loader, device)
                mm_ra = summarize_margins(y_true, logits_ra, target_idx, clean_idx)
                delta_rand_list.append(
                    float(
                        margin_full["margin_target_mean"] - mm_ra["margin_target_mean"]
                    )
                )
                del ms_ra
            delta_rand_mean = _nanmean(delta_rand_list)
            necessity_block = {
                "delta_learned": delta_learned,
                "delta_random_mean": delta_rand_mean,
                "delta_random_std": (
                    float(np.std(delta_rand_list)) if delta_rand_list else float("nan")
                ),
                "necessity_gap": float(delta_learned - delta_rand_mean)
                if np.isfinite(delta_rand_mean)
                else float("nan"),
                "ablate_learned_margin_target_mean": margin_ab["margin_target_mean"],
                "n_random": len(delta_rand_list),
            }
            with open(
                os.path.join(out_dir, "necessity_metrics.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(necessity_block, f, indent=2, ensure_ascii=False)
            print(
                f"[SUMMARY] necessity_gap={necessity_block['necessity_gap']:.4f} "
                f"delta_learned={delta_learned:.4f} "
                f"delta_rand={delta_rand_mean:.4f}"
            )

        # ---- Keep-ratio curve ----
        if not skip_keep_ratio_curve and os.path.isfile(soft_path):
            soft = load_soft_masks_blob(soft_path)
            curve_rows: List[Dict[str, Any]] = []
            for ratio in keep_ratios:
                hard_kr = soft_to_hard_at_ratio(
                    soft, block_meta, masks_cfg, float(ratio)
                )
                hard_kr_dev = {k: v.to(device) for k, v in hard_kr.items()}
                ms_kr = build_masked_ms(
                    len(class_names),
                    ckpt,
                    extracted,
                    enable,
                    device,
                    hard=hard_kr_dev,
                )
                _, logits_kr = collect_logits_loader(ms_kr, loader, device)
                m_kr = summarize_margins(y_true, logits_kr, target_idx, clean_idx)
                del ms_kr
                # one random baseline per ratio (seed = base)
                hard_rr = make_random_hard_masks(
                    block_meta,
                    {
                        **masks_cfg,
                        "keep_ratio": {
                            "M_H": float(ratio),
                            "M_O": float(ratio),
                        },
                    },
                    int(random_base_seed),
                )
                # Force keep ratio via soft_to_hard_at_ratio on random soft
                rng = np.random.default_rng(int(random_base_seed) + int(ratio * 1000))
                soft_r: Dict[str, torch.Tensor] = {}
                for meta in block_meta:
                    bkey = meta["key"]
                    if enable["M_H"]:
                        soft_r[f"{bkey}.M_H"] = torch.from_numpy(
                            rng.random(int(meta["n_h"])).astype(np.float32)
                        )
                    if enable["M_O"]:
                        soft_r[f"{bkey}.M_O"] = torch.from_numpy(
                            rng.random(int(meta["n_o"])).astype(np.float32)
                        )
                hard_rr = soft_to_hard_at_ratio(
                    soft_r, block_meta, masks_cfg, float(ratio)
                )
                hard_rr_dev = {k: v.to(device) for k, v in hard_rr.items()}
                ms_rr = build_masked_ms(
                    len(class_names),
                    ckpt,
                    extracted,
                    enable,
                    device,
                    hard=hard_rr_dev,
                )
                _, logits_rr = collect_logits_loader(ms_rr, loader, device)
                m_rr = summarize_margins(y_true, logits_rr, target_idx, clean_idx)
                del ms_rr
                ret_r = margin_retention_rate(
                    margin_full["margin_target_mean"],
                    m_kr["margin_target_mean"],
                    m_rr["margin_target_mean"],
                )
                row = {
                    "keep_ratio": float(ratio),
                    "learned_margin_target_mean": m_kr["margin_target_mean"],
                    "random_margin_target_mean": m_rr["margin_target_mean"],
                    "margin_retention": ret_r,
                    "learned_fpr_at_tpr95": m_kr["fpr_at_tpr95"],
                    "random_fpr_at_tpr95": m_rr["fpr_at_tpr95"],
                }
                curve_rows.append(row)
                print(
                    f"[keep_ratio] r={ratio:.3f} retention={ret_r} "
                    f"m_l={m_kr['margin_target_mean']:.3f} "
                    f"m_r={m_rr['margin_target_mean']:.3f}"
                )
            ratios = [r["keep_ratio"] for r in curve_rows]
            auc_l = trapezoid_auc(
                ratios, [r["margin_retention"] for r in curve_rows]
            )
            # AUC of learned-random margin gap
            gap = [
                float(r["learned_margin_target_mean"])
                - float(r["random_margin_target_mean"])
                for r in curve_rows
            ]
            auc_gap = trapezoid_auc(ratios, gap)
            keep_ratio_block = {
                "ratios": list(keep_ratios),
                "curve": curve_rows,
                "keep_ratio_auc_margin_retention": auc_l,
                "keep_ratio_auc_margin_gap_learned_minus_random": auc_gap,
            }
            kr_csv = os.path.join(out_dir, "keep_ratio_curve.csv")
            _write_csv(kr_csv, curve_rows)
            with open(
                os.path.join(out_dir, "keep_ratio_curve.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(keep_ratio_block, f, indent=2, ensure_ascii=False)
            keep_ratio_block["csv"] = kr_csv
        elif not skip_keep_ratio_curve:
            warnings.append(f"soft_masks.pt missing at {soft_path}; skip keep-ratio curve")

        # ---- Linear probe ----
        if not skip_linear_probe and feats_learned is not None:
            learned_probe = run_linear_probe_target_clean(
                feats_learned, y_true, target_idx, clean_idx
            )
            ok_rand = [r for r in random_probe_rows if r.get("ok")]
            linear_probe_block = {
                "learned": learned_probe,
                "random": random_probe_rows,
                "random_summary": {
                    "accuracy_mean": _nanmean([r["accuracy"] for r in ok_rand]),
                    "accuracy_std": float(np.std([r["accuracy"] for r in ok_rand]))
                    if ok_rand
                    else float("nan"),
                    "auroc_mean": _nanmean(
                        [r["auroc"] for r in ok_rand if r.get("auroc") is not None]
                    ),
                    "n_ok": len(ok_rand),
                },
            }
            with open(
                os.path.join(out_dir, "linear_probe.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(linear_probe_block, f, indent=2, ensure_ascii=False)
            print(
                f"[SUMMARY] linear_probe learned_acc="
                f"{learned_probe.get('accuracy')} "
                f"rand_acc={linear_probe_block['random_summary']['accuracy_mean']}"
            )

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
                "note": "reference_argmax_retention (may saturate; prefer margin)",
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

    learned_f1 = float(retention["learned"]["target_f1"])
    rand_mean = float(random_summary["target_f1_mean"])
    if not (learned_f1 > rand_mean):
        warnings.append(
            f"[reference] learned target_f1={learned_f1:.4f} is not higher than "
            f"random mean={rand_mean:.4f}."
        )
    if not loc_has_mask_seed:
        warnings.append(
            "localization config.json has no MASK_SEED/train_mask_seed; "
            f"cannot verify against patch MASK_SEED={train_mask_seed}."
        )
    elif not mask_seed_consistent:
        warnings.append(
            f"mask_seed_consistent=False: loc expected {expected_mask_seed}, "
            f"patch has {train_mask_seed}."
        )
    if not np.isfinite(ret_rate):
        warnings.append(
            "margin_retention is nan (near-zero full-vs-random denominator)."
        )

    summary: Dict[str, Any] = {
        "localization_dir": localization_dir,
        "patch_run_dir": patch_dir,
        "patch_manifest_sha256": patch_manifest_sha256,
        "checkpoint_path": ckpt,
        "node": node,
        "attack_class": attack_class,
        "target_class_idx": target_idx,
        "target_class_name": class_names[target_idx],
        "clean_class_idx": clean_idx,
        "extracted_layer": extracted,
        "train_mask_seed": train_mask_seed,
        "expected_mask_seed": int(expected_mask_seed),
        "mask_seed_consistent": bool(mask_seed_consistent),
        "test_mask_seed": int(test_mask_seed),
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
        "margin": margin_block,
        "necessity": necessity_block,
        "keep_ratio_curve": keep_ratio_block,
        "linear_probe": linear_probe_block,
        "reference_argmax_retention": {
            "original": retention["original"],
            "learned": retention["learned"],
            "random_summary": random_summary,
            "learned_vs_random": {
                "learned_target_f1": learned_f1,
                "random_target_f1_mean": rand_mean,
                "random_target_f1_std": random_summary["target_f1_std"],
                "delta_f1_learned_minus_random_mean": learned_f1 - rand_mean,
            },
            "note": "Argmax F1/cleanFPR may saturate; primary metrics are margin/necessity.",
        },
        # Backward-compatible aliases
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
            "margin_metrics_json": os.path.join(out_dir, "margin_metrics.json"),
        },
    }
    summary_path = os.path.join(out_dir, "subnet_quality_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] wrote {retention_path}")
    print(f"[INFO] wrote {summary_path}")
    print(
        f"[SUMMARY] PRIMARY margin_retention={ret_rate} "
        f"necessity_gap="
        f"{None if necessity_block is None else necessity_block.get('necessity_gap')} "
        f"| reference F1 learned={learned_f1:.4f} random={rand_mean:.4f}"
    )
    for w in warnings:
        print(f"[WARN] {w}")

    summary["summary_json"] = summary_path
    return summary


def compare_subnet_stability(
    localization_dirs: Sequence[str],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Pairwise hard Jaccard + soft Spearman across localization runs."""
    dirs = [os.path.abspath(d) for d in localization_dirs]
    if len(dirs) < 2:
        raise ValueError("Need >=2 localization dirs for stability compare")
    softs: List[Dict[str, torch.Tensor]] = []
    hards: List[Dict[str, torch.Tensor]] = []
    for d in dirs:
        sp = os.path.join(d, "soft_masks.pt")
        hp = os.path.join(d, "hard_masks.pt")
        if not os.path.isfile(sp) or not os.path.isfile(hp):
            raise FileNotFoundError(f"Missing soft/hard masks under {d}")
        softs.append(load_soft_masks_blob(sp))
        hards.append(torch.load(hp, map_location="cpu"))
        hards[-1] = {k: v.float().cpu() for k, v in hards[-1].items()}

    keys = sorted(set(hards[0].keys()) & set(softs[0].keys()))
    for i, d in enumerate(dirs):
        missing_h = set(keys) - set(hards[i].keys())
        missing_s = set(keys) - set(softs[i].keys())
        if missing_h or missing_s:
            raise KeyError(f"{d} missing keys hard={missing_h} soft={missing_s}")

    pairs: List[Dict[str, Any]] = []
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            per_key = {}
            jac_vals = []
            rho_vals = []
            for k in keys:
                jac = hard_jaccard(hards[i][k], hards[j][k])
                rho = soft_spearman(softs[i][k], softs[j][k])
                per_key[k] = {"hard_jaccard": jac, "soft_spearman": rho}
                jac_vals.append(jac)
                rho_vals.append(rho)
            pairs.append(
                {
                    "dir_a": dirs[i],
                    "dir_b": dirs[j],
                    "mean_hard_jaccard": _nanmean(jac_vals),
                    "mean_soft_spearman": _nanmean(rho_vals),
                    "per_key": per_key,
                }
            )

    out_root = os.path.abspath(output_dir) if output_dir else dirs[0]
    os.makedirs(out_root, exist_ok=True)
    result = {
        "localization_dirs": dirs,
        "n_dirs": len(dirs),
        "keys": keys,
        "pairs": pairs,
        "mean_pairwise_hard_jaccard": _nanmean(
            [p["mean_hard_jaccard"] for p in pairs]
        ),
        "mean_pairwise_soft_spearman": _nanmean(
            [p["mean_soft_spearman"] for p in pairs]
        ),
    }
    out_path = os.path.join(out_root, "subnet_stability_compare.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] wrote {out_path}")
    print(
        f"[SUMMARY] mean hard Jaccard={result['mean_pairwise_hard_jaccard']:.4f} "
        f"mean soft Spearman={result['mean_pairwise_soft_spearman']:.4f}"
    )
    result["output_json"] = out_path
    return result


def compare_surrogate_absolute_levels(
    localization_dirs: Sequence[str],
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Paired per-image R^2 differences across runs (requires quality CSVs)."""
    import csv as _csv

    dirs = [os.path.abspath(d) for d in localization_dirs]
    if len(dirs) < 2:
        raise ValueError("Need >=2 dirs for comparative surrogate")

    by_dir: Dict[str, Dict[str, float]] = {}
    for d in dirs:
        csv_path = os.path.join(
            d, "surrogate_quality", "surrogate_quality_per_image.csv"
        )
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
        by_dir[d] = {r["image_id"]: float(r["r2"]) for r in rows if r.get("r2")}

    pairs: List[Dict[str, Any]] = []
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            a, b = dirs[i], dirs[j]
            common = sorted(set(by_dir[a]) & set(by_dir[b]))
            diffs = [by_dir[b][iid] - by_dir[a][iid] for iid in common]
            boot = bootstrap_mean_median_ci(
                diffs, n_boot=bootstrap_samples, seed=bootstrap_seed
            )
            pairs.append(
                {
                    "dir_a": a,
                    "dir_b": b,
                    "n_common_images": len(common),
                    "mean_r2_diff_b_minus_a": _nanmean(diffs),
                    "frac_b_better": float(np.mean(np.asarray(diffs) > 0))
                    if diffs
                    else float("nan"),
                    "bootstrap": boot,
                }
            )

    out_root = os.path.abspath(output_dir) if output_dir else dirs[0]
    os.makedirs(
        os.path.join(out_root, "surrogate_quality"), exist_ok=True
    )
    result = {
        "localization_dirs": dirs,
        "pairs": pairs,
        "note": (
            "Paired R^2 diffs; if mean CI crosses 0, trials are not "
            "distinguishable on surrogate fit quality."
        ),
    }
    out_path = os.path.join(
        out_root, "surrogate_quality", "surrogate_comparative.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    # Also patch first dir summary if present
    summary_path = os.path.join(
        dirs[0], "surrogate_quality", "surrogate_quality_summary.json"
    )
    if os.path.isfile(summary_path):
        summary = load_json(summary_path)
        summary["comparative"] = result
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] wrote {out_path}")
    result["output_json"] = out_path
    return result


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate localization quality: surrogate metrics (R2/MSE/margin) "
            "and subnet metrics (margin/necessity/keep-ratio/probe)."
        )
    )
    p.add_argument(
        "--localization_dir",
        default=None,
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
        "--bootstrap_samples",
        type=int,
        default=1000,
        help="Bootstrap resamples for mean/median CI (default: 1000)",
    )
    p.add_argument(
        "--keep_ratios",
        type=str,
        default="0.02,0.05,0.1,0.25",
        help="Comma-separated keep ratios for subnet curve",
    )
    p.add_argument(
        "--compare_localization_dirs",
        type=str,
        default=None,
        help=(
            "Comma-separated localization dirs for soft/hard stability "
            "compare and paired R2 diffs (optional; can run alone)."
        ),
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
        "--skip_necessity",
        action="store_true",
        help="Skip necessity (ablate) metrics",
    )
    p.add_argument(
        "--skip_keep_ratio_curve",
        action="store_true",
        help="Skip multi keep-ratio curve",
    )
    p.add_argument(
        "--skip_linear_probe",
        action="store_true",
        help="Skip linear probe on subnet features",
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


def _parse_keep_ratios(s: str) -> List[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        return [0.02, 0.05, 0.1, 0.25]
    return [float(p) for p in parts]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    compare_dirs = None
    if args.compare_localization_dirs:
        compare_dirs = [
            os.path.abspath(x.strip())
            for x in args.compare_localization_dirs.split(",")
            if x.strip()
        ]

    if compare_dirs and not args.localization_dir:
        compare_subnet_stability(compare_dirs)
        # Comparative R2 only if CSVs already exist
        try:
            compare_surrogate_absolute_levels(
                compare_dirs,
                bootstrap_samples=int(args.bootstrap_samples),
            )
        except FileNotFoundError as e:
            print(f"[WARN] skip surrogate comparative: {e}")
        return 0

    if not args.localization_dir:
        raise SystemExit(
            "error: --localization_dir is required unless "
            "only --compare_localization_dirs is used"
        )

    loc_dir = os.path.abspath(args.localization_dir)
    hard_path = os.path.join(loc_dir, "hard_masks.pt")
    keep_ratios = _parse_keep_ratios(args.keep_ratios)

    if not args.skip_surrogate:
        evaluate_localization_surrogates(
            localization_dir=loc_dir,
            patch_run_dir=args.patch_run_dir,
            test_mask_seed=int(args.test_mask_seed),
            device_str=args.device,
            batch_size=int(args.batch_size),
            r2_eps=float(args.r2_eps),
            bootstrap_samples=int(args.bootstrap_samples),
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
            keep_ratios=keep_ratios,
            skip_necessity=bool(args.skip_necessity),
            skip_keep_ratio_curve=bool(args.skip_keep_ratio_curve),
            skip_linear_probe=bool(args.skip_linear_probe),
        )
    elif args.skip_subnet:
        print("[INFO] skip_subnet: subnet quality not run")

    if compare_dirs:
        # Ensure primary dir is included
        all_dirs = list(dict.fromkeys([loc_dir] + compare_dirs))
        if len(all_dirs) >= 2:
            compare_subnet_stability(all_dirs, output_dir=loc_dir)
            try:
                compare_surrogate_absolute_levels(
                    all_dirs,
                    bootstrap_samples=int(args.bootstrap_samples),
                    output_dir=loc_dir,
                )
            except FileNotFoundError as e:
                print(f"[WARN] skip surrogate comparative: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
