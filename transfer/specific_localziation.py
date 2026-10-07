#!/usr/bin/env python3
"""Category-specific localization for a node attack class.

Phase 1: fit per-image linear surrogates g_i (locality-weighted lstsq, no bias)
         that map flattened binary patch masks -> full Ms logits.
Phase 2: freeze Ms, optimize network learnable masks M_L / M_P / M_B so that
         f_t(Ms^M(x̃)) ≈ f_t(g_i(b)), with budget penalty only when activated
         mass exceeds keep_ratio * n_channels; then exact top-k -> hard masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_E2EDT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_NODE_TRAINING_DIR = os.path.join(_E2EDT_ROOT, "node_model_training")
_DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "localization_config.json")
_DEFAULT_OUTPUT_ROOT = os.path.join(_SCRIPT_DIR, "localization_result")

if _NODE_TRAINING_DIR not in sys.path:
    sys.path.insert(0, _NODE_TRAINING_DIR)

from model_architecture import backbone_multi, models  # noqa: E402

REFERENCE_MASK_ID = "reference_all_ones"
MASK_LOCATIONS = ("M_L", "M_P", "M_B")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in overrides.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_localization_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path = os.path.abspath(config_path or _DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = path
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_manifest_by_image(
    rows: Sequence[Dict[str, Any]],
    include_reference: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mid = row["mask_id"]
        if mid == REFERENCE_MASK_ID and not include_reference:
            continue
        groups[row["image_id"]].append(row)
    return dict(groups)


# ---------------------------------------------------------------------------
# Source model
# ---------------------------------------------------------------------------


def resolve_checkpoint(
    patch_cfg_resolved: Dict[str, Any],
    node: str,
    checkpoint_path: Optional[str],
    training_report_dir: Optional[str],
) -> Tuple[str, List[str]]:
    """Return (checkpoint_path, class_names)."""
    if checkpoint_path:
        ckpt = os.path.abspath(checkpoint_path)
        # Prefer classes from patch node_specs
        class_names = list(
            patch_cfg_resolved["training_report"]["node_specs"][node]
        )
        return ckpt, class_names

    run_dir = training_report_dir or patch_cfg_resolved["training_report"]["run_dir"]
    run_dir = os.path.abspath(run_dir)
    index_path = os.path.join(run_dir, "checkpoints", "index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"checkpoints/index.json not found under {run_dir}; "
            "pass --checkpoint_path explicitly"
        )
    index = load_json(index_path)
    if node not in index.get("nodes", {}):
        raise KeyError(f"node {node!r} missing in {index_path}")
    entry = index["nodes"][node]
    return os.path.abspath(entry["path"]), list(entry["classes"])


def load_source_model(
    num_class: int,
    checkpoint_path: str,
    extracted_layer: str,
    device: torch.device,
) -> models.Transfer_Net:
    backbone_multi.extracted_layer = extracted_layer
    model = models.Transfer_Net(num_class)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# Masked Transfer Net (network learnable masks)
# ---------------------------------------------------------------------------


class MaskedTransferNet(nn.Module):
    """Frozen Transfer_Net with optional channel gates M_L / M_P / M_B."""

    def __init__(
        self,
        base: models.Transfer_Net,
        enable: Dict[str, bool],
        dims: Dict[str, int],
        init: str = "zeros",
    ):
        super().__init__()
        self.base = base
        self.enable = {k: bool(enable.get(k, False)) for k in MASK_LOCATIONS}
        self.dims = {k: int(dims[k]) for k in MASK_LOCATIONS}

        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

        self.raw_masks = nn.ParameterDict()
        for name in MASK_LOCATIONS:
            if not self.enable[name]:
                continue
            n = self.dims[name]
            if init == "zeros":
                tensor = torch.zeros(n)
            elif init == "ones":
                tensor = torch.ones(n)
            else:
                raise ValueError(f"Unknown mask init {init!r}")
            self.raw_masks[name] = nn.Parameter(tensor)

        self._hooks: List[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self._clear_hooks()
        bn = self.base.base_network
        if self.enable["M_L"]:
            self._hooks.append(
                bn.convm2_layer.register_forward_hook(
                    self._make_channel_hook("M_L")
                )
            )
        if self.enable["M_P"]:
            self._hooks.append(
                bn.linear_test.register_forward_hook(
                    self._make_channel_hook("M_P")
                )
            )
        if self.enable["M_B"]:
            self._hooks.append(
                self.base.bottle_layer[1].register_forward_hook(
                    self._make_channel_hook("M_B")
                )
            )

    def _clear_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _make_channel_hook(self, name: str):
        def hook(_module, _inp, out):
            gate = torch.sigmoid(self.raw_masks[name])
            if out.dim() == 4:
                return out * gate.view(1, -1, 1, 1)
            if out.dim() == 2:
                return out * gate.view(1, -1)
            raise RuntimeError(
                f"Unsupported activation rank {out.dim()} for mask {name}"
            )

        return hook

    def soft_masks(self) -> Dict[str, torch.Tensor]:
        return {k: torch.sigmoid(v) for k, v in self.raw_masks.items()}

    def mask_masses(self) -> Dict[str, torch.Tensor]:
        return {k: m.sum() for k, m in self.soft_masks().items()}

    def predict(self, x: torch.Tensor, test_flag: int = 1) -> torch.Tensor:
        self.base.eval()
        return self.base.predict(x, test_flag=test_flag)

    def train(self, mode: bool = True):  # type: ignore[override]
        # Keep base frozen in eval; ParameterDict still receives grads.
        super().train(mode)
        self.base.eval()
        return self


# ---------------------------------------------------------------------------
# Surrogate g_i
# ---------------------------------------------------------------------------


def bernoulli_locality_weights(
    binary_masks: np.ndarray,
    keep_prob: float,
) -> np.ndarray:
    """Π_raw(b)=p^{||b||0}(1-p)^{RC-||b||0}, then renormalize to sum=1."""
    p = float(keep_prob)
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    b = binary_masks.astype(np.float64)
    if b.ndim != 2:
        raise ValueError(f"binary_masks must be [K, RC]; got {b.shape}")
    k, rc = b.shape
    ones = b.sum(axis=1)
    zeros = rc - ones
    log_pi = ones * np.log(p) + zeros * np.log(1.0 - p)
    log_pi = log_pi - log_pi.max()
    pi = np.exp(log_pi)
    s = pi.sum()
    if s <= 0:
        return np.full(k, 1.0 / k, dtype=np.float64)
    return (pi / s).astype(np.float64)


def fit_linear_surrogate(
    B: np.ndarray,
    Y: np.ndarray,
    pi: np.ndarray,
    ridge_alpha: float = 0.0,
    ridge_fallback_alpha: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Weighted least squares: min Σ π_b ||y_b - b W||^2, W [D,C], no bias."""
    if B.ndim != 2 or Y.ndim != 2:
        raise ValueError("B and Y must be 2-D")
    k, d = B.shape
    if Y.shape[0] != k:
        raise ValueError("B/Y row mismatch")
    if pi.shape != (k,):
        raise ValueError("pi shape mismatch")

    w = np.sqrt(np.maximum(pi, 0.0)).astype(np.float64)
    Bw = B.astype(np.float64) * w[:, None]
    Yw = Y.astype(np.float64) * w[:, None]

    meta: Dict[str, Any] = {"solver": "lstsq", "ridge_used": 0.0}
    W: Optional[np.ndarray] = None

    alpha = float(ridge_alpha)
    if alpha > 0:
        # Explicit ridge via augmented system
        eye = np.sqrt(alpha) * np.eye(d, dtype=np.float64)
        Bw_aug = np.vstack([Bw, eye])
        Yw_aug = np.vstack([Yw, np.zeros((d, Y.shape[1]), dtype=np.float64)])
        W, residuals, rank, singular = np.linalg.lstsq(Bw_aug, Yw_aug, rcond=None)
        meta.update(
            {
                "solver": "lstsq_ridge",
                "ridge_used": alpha,
                "rank": int(rank),
                "singular_min": float(singular.min()) if len(singular) else float("nan"),
                "singular_max": float(singular.max()) if len(singular) else float("nan"),
            }
        )
    else:
        try:
            W, residuals, rank, singular = np.linalg.lstsq(Bw, Yw, rcond=None)
            meta.update(
                {
                    "rank": int(rank),
                    "singular_min": float(singular.min()) if len(singular) else float("nan"),
                    "singular_max": float(singular.max()) if len(singular) else float("nan"),
                }
            )
            if not np.isfinite(W).all():
                raise np.linalg.LinAlgError("non-finite W from lstsq")
        except Exception as exc:  # noqa: BLE001
            fb = float(ridge_fallback_alpha)
            if fb <= 0:
                raise RuntimeError(
                    f"lstsq failed ({exc}) and ridge_fallback_alpha<=0"
                ) from exc
            eye = np.sqrt(fb) * np.eye(d, dtype=np.float64)
            Bw_aug = np.vstack([Bw, eye])
            Yw_aug = np.vstack(
                [Yw, np.zeros((d, Y.shape[1]), dtype=np.float64)]
            )
            W, residuals, rank, singular = np.linalg.lstsq(
                Bw_aug, Yw_aug, rcond=None
            )
            meta.update(
                {
                    "solver": "lstsq_ridge_fallback",
                    "ridge_used": fb,
                    "rank": int(rank),
                    "singular_min": float(singular.min()) if len(singular) else float("nan"),
                    "singular_max": float(singular.max()) if len(singular) else float("nan"),
                    "fallback_reason": str(exc),
                }
            )

    assert W is not None
    pred = B.astype(np.float64) @ W
    err = pred - Y.astype(np.float64)
    sq = np.sum(err * err, axis=1)
    mse_weighted = float(np.sum(pi * sq) / max(Y.shape[1], 1))
    mse_unweighted = float(np.mean(sq) / max(Y.shape[1], 1))
    max_abs_err = float(np.max(np.abs(err)))
    cond = float(meta["singular_max"] / max(meta["singular_min"], 1e-30))
    meta.update(
        {
            "mse_weighted": mse_weighted,
            "mse_unweighted": mse_unweighted,
            "max_abs_err": max_abs_err,
            "cond": cond,
            "n_masks": int(k),
            "feat_dim": int(d),
            "num_class": int(Y.shape[1]),
        }
    )
    return W.astype(np.float64), meta


@torch.no_grad()
def collect_ms_logits(
    model: models.Transfer_Net,
    tensor_paths: Sequence[str],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    logits_list: List[np.ndarray] = []
    batch: List[torch.Tensor] = []
    for path in tensor_paths:
        t = torch.load(path, map_location="cpu")
        if not torch.is_tensor(t):
            raise TypeError(f"Expected tensor in {path}")
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


def fit_all_surrogates(
    groups: Dict[str, List[Dict[str, Any]]],
    model: models.Transfer_Net,
    keep_prob: float,
    grid_rows: int,
    grid_cols: int,
    device: torch.device,
    out_dir: str,
    ridge_alpha: float,
    ridge_fallback_alpha: float,
    g_batch_size: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    os.makedirs(os.path.join(out_dir, "surrogate_models"), exist_ok=True)
    W_by_image: Dict[str, np.ndarray] = {}
    metrics_rows: List[Dict[str, Any]] = []
    feat_dim = grid_rows * grid_cols

    image_ids = sorted(groups.keys())
    for img_idx, image_id in enumerate(image_ids):
        rows = groups[image_id]
        # Stable order: mask_* then reference last if present
        rows = sorted(
            rows,
            key=lambda r: (
                1 if r["mask_id"] == REFERENCE_MASK_ID else 0,
                r["mask_id"],
            ),
        )
        B_list = []
        paths = []
        for r in rows:
            b = np.asarray(r["binary_patch_mask"], dtype=np.float64)
            if b.shape != (grid_rows, grid_cols):
                raise ValueError(
                    f"{image_id}/{r['mask_id']}: mask shape {b.shape} != "
                    f"({grid_rows},{grid_cols})"
                )
            B_list.append(b.reshape(-1))
            paths.append(r["masked_tensor_path"])
        B = np.stack(B_list, axis=0)
        if B.shape[1] != feat_dim:
            raise ValueError(f"Unexpected feat dim {B.shape[1]} vs {feat_dim}")

        Y = collect_ms_logits(model, paths, device, batch_size=g_batch_size)
        pi = bernoulli_locality_weights(B, keep_prob)
        W, meta = fit_linear_surrogate(
            B,
            Y,
            pi,
            ridge_alpha=ridge_alpha,
            ridge_fallback_alpha=ridge_fallback_alpha,
        )
        W_by_image[image_id] = W

        fname = f"g_{img_idx:06d}.npz"
        npz_path = os.path.join(out_dir, "surrogate_models", fname)
        np.savez_compressed(
            npz_path,
            W=W,
            image_id=np.asarray(image_id),
            grid_rows=np.int64(grid_rows),
            grid_cols=np.int64(grid_cols),
            num_class=np.int64(Y.shape[1]),
            pi=pi,
            mask_ids=np.asarray([r["mask_id"] for r in rows]),
            mse_weighted=np.float64(meta["mse_weighted"]),
            mse_unweighted=np.float64(meta["mse_unweighted"]),
        )
        metrics_rows.append(
            {
                "image_idx": img_idx,
                "image_id": image_id,
                "npz_path": npz_path,
                "n_masks": meta["n_masks"],
                "mse_weighted": meta["mse_weighted"],
                "mse_unweighted": meta["mse_unweighted"],
                "max_abs_err": meta["max_abs_err"],
                "rank": meta.get("rank"),
                "cond": meta.get("cond"),
                "solver": meta.get("solver"),
                "ridge_used": meta.get("ridge_used"),
            }
        )
        print(
            f"[g-fit] {img_idx + 1}/{len(image_ids)} {image_id} "
            f"mse_w={meta['mse_weighted']:.6f} "
            f"mse_u={meta['mse_unweighted']:.6f} "
            f"max_abs={meta['max_abs_err']:.6f}"
        )

    metrics_csv = os.path.join(out_dir, "g_fit_metrics.csv")
    _write_csv(metrics_csv, metrics_rows)
    print(f"[INFO] wrote {metrics_csv}")
    return W_by_image, metrics_rows


# ---------------------------------------------------------------------------
# Localization dataset / training
# ---------------------------------------------------------------------------


class LocalizationDataset(Dataset):
    def __init__(
        self,
        groups: Dict[str, List[Dict[str, Any]]],
        W_by_image: Dict[str, np.ndarray],
        image_ids: Sequence[str],
        grid_rows: int,
        grid_cols: int,
        target_class_idx: int,
    ):
        self.samples: List[Dict[str, Any]] = []
        self.target_class_idx = int(target_class_idx)
        feat_dim = grid_rows * grid_cols
        for image_id in image_ids:
            W = W_by_image[image_id]
            for r in groups[image_id]:
                b = np.asarray(r["binary_patch_mask"], dtype=np.float32).reshape(-1)
                if b.shape[0] != feat_dim:
                    raise ValueError("mask flatten dim mismatch")
                self.samples.append(
                    {
                        "image_id": image_id,
                        "mask_id": r["mask_id"],
                        "tensor_path": r["masked_tensor_path"],
                        "b": b,
                        "W": W.astype(np.float32),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = torch.load(s["tensor_path"], map_location="cpu")
        b = torch.from_numpy(s["b"])
        W = torch.from_numpy(s["W"])
        # g(b) = b @ W -> [C]; target logit
        g_logits = b @ W
        g_t = g_logits[self.target_class_idx]
        return x, b, g_t, s["image_id"], s["mask_id"]


def budget_penalty(
    soft: Dict[str, torch.Tensor],
    keep_ratio: Dict[str, float],
    budget_lambda: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Penalty only when mass > keep_ratio * n."""
    device = next(iter(soft.values())).device
    total = torch.zeros((), device=device)
    info: Dict[str, float] = {}
    for name, m in soft.items():
        n = m.numel()
        budget = float(keep_ratio[name]) * n
        mass = m.sum()
        over = F.relu(mass - budget)
        pen = float(budget_lambda) * (over ** 2)
        total = total + pen
        info[f"mass_{name}"] = float(mass.detach().item())
        info[f"budget_{name}"] = budget
        info[f"over_{name}"] = float(over.detach().item())
    return total, info


def resolve_top_k(n: int, keep_ratio: float, top_k: Optional[int]) -> int:
    if top_k is not None:
        k = int(top_k)
    else:
        k = int(round(float(keep_ratio) * n))
    return max(1, min(k, n))


def soft_to_hard_masks(
    soft: Dict[str, torch.Tensor],
    keep_ratio: Dict[str, float],
    top_k_cfg: Dict[str, Optional[int]],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    hard: Dict[str, torch.Tensor] = {}
    selected: Dict[str, List[int]] = {}
    meta_k: Dict[str, int] = {}
    for name, m in soft.items():
        n = m.numel()
        k = resolve_top_k(n, keep_ratio[name], top_k_cfg.get(name))
        _, idx = torch.topk(m.detach().cpu(), k)
        h = torch.zeros(n, dtype=torch.float32)
        h[idx] = 1.0
        hard[name] = h
        selected[name] = sorted(int(i) for i in idx.tolist())
        meta_k[name] = k
    return hard, {"indices": selected, "top_k": meta_k}


def train_localization(
    masked_model: MaskedTransferNet,
    dataset: LocalizationDataset,
    cfg: Dict[str, Any],
    device: torch.device,
    target_class_idx: int,
    out_dir: str,
) -> List[Dict[str, Any]]:
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    params = list(masked_model.raw_masks.parameters())
    if not params:
        raise RuntimeError("No network masks enabled")
    optimizer = torch.optim.Adam(params, lr=float(cfg["lr"]))
    keep_ratio = {
        k: float(cfg["masks"]["keep_ratio"][k])
        for k in masked_model.raw_masks.keys()
    }
    budget_lambda = float(cfg["masks"]["budget_lambda"])
    epochs = int(cfg["epochs"])
    log_every = max(1, int(cfg.get("log_every", 10)))

    history: List[Dict[str, Any]] = []
    global_step = 0
    masked_model.train()

    for epoch in range(epochs):
        epoch_loc = 0.0
        epoch_bud = 0.0
        n_batches = 0
        for batch in loader:
            x, _b, g_t, _ids, _mids = batch
            x = x.to(device)
            g_t = g_t.to(device).float()

            optimizer.zero_grad()
            logits = masked_model.predict(x, test_flag=1)
            y_t = logits[:, target_class_idx]
            loc = F.mse_loss(y_t, g_t)

            soft = masked_model.soft_masks()
            bud, mass_info = budget_penalty(soft, keep_ratio, budget_lambda)
            loss = loc + bud
            loss.backward()
            optimizer.step()

            global_step += 1
            n_batches += 1
            epoch_loc += float(loc.item())
            epoch_bud += float(bud.item())

            if global_step % log_every == 0:
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "L_loc": float(loc.item()),
                    "L_budget": float(bud.item()),
                    "L_total": float(loss.item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    **mass_info,
                }
                history.append(row)
                print(
                    f"[loc] step={global_step} epoch={epoch} "
                    f"L_loc={row['L_loc']:.6f} L_bud={row['L_budget']:.6f} "
                    + " ".join(f"{k}={v:.2f}" for k, v in mass_info.items() if k.startswith("mass_"))
                )

        if n_batches:
            print(
                f"[loc] epoch {epoch + 1}/{epochs} "
                f"mean_L_loc={epoch_loc / n_batches:.6f} "
                f"mean_L_bud={epoch_bud / n_batches:.6f}"
            )

    hist_path = os.path.join(out_dir, "localization_history.csv")
    _write_csv(hist_path, history)
    print(f"[INFO] wrote {hist_path}")
    return history


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    # Union of keys preserving first-row order then extras
    fieldnames: List[str] = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def export_selected_samples_csv(
    selected_json_path: str, out_csv: str
) -> None:
    payload = load_json(selected_json_path)
    samples = payload.get("selected_samples", [])
    rows = []
    for i, s in enumerate(samples):
        rows.append(
            {
                "idx": i,
                "image_id": s.get("image_id"),
                "original_image_path": s.get("original_image_path"),
                "source_face_path": s.get("source_face_path"),
                "poisoned_basename": s.get("poisoned_basename"),
                "person_name": s.get("person_name"),
                "base_name": s.get("base_name"),
            }
        )
    _write_csv(out_csv, rows)


def resolve_target_class_idx(
    cfg: Dict[str, Any],
    selection: Dict[str, Any],
    class_names: Sequence[str],
) -> int:
    tc = cfg.get("target_class")
    if tc is None or tc == "":
        return int(selection["class_idx"])
    if isinstance(tc, int) or (isinstance(tc, str) and tc.isdigit()):
        idx = int(tc)
        if idx < 0 or idx >= len(class_names):
            raise ValueError(f"target_class idx {idx} out of range")
        return idx
    name = str(tc)
    if name not in class_names:
        raise ValueError(
            f"target_class {name!r} not in class_names {list(class_names)}"
        )
    return int(class_names.index(name))


def default_output_dir(
    output_root: Optional[str],
    patch_run_dir: str,
    node: str,
    attack_class: str,
) -> str:
    root = output_root or _DEFAULT_OUTPUT_ROOT
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patch_name = os.path.basename(os.path.abspath(patch_run_dir).rstrip("/"))
    name = f"{stamp}_{patch_name}_{node}_{attack_class}"
    return os.path.join(root, name)


def save_run_config(cfg: Dict[str, Any], out_dir: str) -> None:
    path = os.path.join(out_dir, "config.json")
    dump = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_localization(cfg: Dict[str, Any]) -> str:
    patch_run_dir = cfg.get("patch_run_dir")
    if not patch_run_dir:
        raise ValueError("patch_run_dir is required (config or --patch_run_dir)")
    patch_run_dir = os.path.abspath(patch_run_dir)
    resolved_path = os.path.join(patch_run_dir, "config_resolved.json")
    manifest_path = os.path.join(patch_run_dir, "manifest.jsonl")
    selected_path = os.path.join(patch_run_dir, "selected_samples.json")
    for p in (resolved_path, manifest_path, selected_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    patch_resolved = load_json(resolved_path)
    patch_cfg = patch_resolved["patch_config"]
    selection = patch_resolved["selection"]
    node = selection["node"]
    attack_class = selection["attack_class"]

    grid_rows = int(patch_cfg["GRID_ROWS"])
    grid_cols = int(patch_cfg["GRID_COLS"])
    keep_prob = float(patch_cfg["KEEP_PROB"])
    include_reference = bool(cfg.get("include_reference", True))

    ckpt_path, class_names = resolve_checkpoint(
        patch_resolved,
        node,
        cfg.get("checkpoint_path"),
        cfg.get("training_report_dir"),
    )
    num_class = len(class_names)
    target_idx = resolve_target_class_idx(cfg, selection, class_names)

    # Prefer extracted_layer from training report config if present
    extracted = cfg.get("extracted_layer") or "7_point"
    tr_cfg_path = patch_resolved["training_report"].get("config_path")
    if tr_cfg_path and os.path.isfile(tr_cfg_path):
        tr_cfg = load_json(tr_cfg_path)
        if tr_cfg.get("extracted_layer"):
            extracted = tr_cfg["extracted_layer"]
    if cfg.get("extracted_layer"):
        extracted = cfg["extracted_layer"]

    device_str = cfg.get("device") or "cpu"
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available; falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = cfg.get("output_dir")
    if not out_dir:
        out_dir = default_output_dir(
            cfg.get("output_root"), patch_run_dir, node, attack_class
        )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg_snapshot = deepcopy(cfg)
    cfg_snapshot.update(
        {
            "patch_run_dir": patch_run_dir,
            "checkpoint_path": ckpt_path,
            "resolved_node": node,
            "resolved_attack_class": attack_class,
            "resolved_class_names": class_names,
            "resolved_target_class_idx": target_idx,
            "resolved_extracted_layer": extracted,
            "output_dir": out_dir,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "keep_prob": keep_prob,
        }
    )
    save_run_config(cfg_snapshot, out_dir)
    export_selected_samples_csv(
        selected_path, os.path.join(out_dir, "selected_samples.csv")
    )

    print(f"[INFO] patch_run_dir={patch_run_dir}")
    print(f"[INFO] checkpoint={ckpt_path}")
    print(f"[INFO] node={node} attack={attack_class} C={num_class}")
    print(f"[INFO] target_class_idx={target_idx} ({class_names[target_idx]})")
    print(f"[INFO] grid={grid_rows}x{grid_cols} keep_prob={keep_prob}")
    print(f"[INFO] output_dir={out_dir}")

    model = load_source_model(num_class, ckpt_path, extracted, device)

    rows = load_manifest(manifest_path)
    groups = group_manifest_by_image(rows, include_reference=include_reference)
    if not groups:
        raise RuntimeError("No samples in manifest after filtering")

    W_by_image, _metrics = fit_all_surrogates(
        groups=groups,
        model=model,
        keep_prob=keep_prob,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        device=device,
        out_dir=out_dir,
        ridge_alpha=float(cfg.get("ridge_alpha", 0.0)),
        ridge_fallback_alpha=float(cfg.get("ridge_fallback_alpha", 1e-6)),
        g_batch_size=int(cfg.get("g_batch_size", 32)),
    )

    image_ids = sorted(W_by_image.keys())
    dataset = LocalizationDataset(
        groups=groups,
        W_by_image=W_by_image,
        image_ids=image_ids,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        target_class_idx=target_idx,
    )

    masks_cfg = cfg["masks"]
    masked = MaskedTransferNet(
        base=model,
        enable=masks_cfg["enable"],
        dims=masks_cfg["dims"],
        init=str(masks_cfg.get("init", "zeros")),
    )
    masked.to(device)

    train_localization(
        masked_model=masked,
        dataset=dataset,
        cfg=cfg,
        device=device,
        target_class_idx=target_idx,
        out_dir=out_dir,
    )

    soft = {k: v.detach().cpu() for k, v in masked.soft_masks().items()}
    raw = {k: v.detach().cpu() for k, v in masked.raw_masks.items()}
    torch.save({"soft": soft, "raw": raw}, os.path.join(out_dir, "soft_masks.pt"))

    hard, sel = soft_to_hard_masks(
        soft,
        keep_ratio={k: float(masks_cfg["keep_ratio"][k]) for k in soft},
        top_k_cfg={
            k: masks_cfg.get("top_k", {}).get(k) for k in soft
        },
    )
    torch.save(hard, os.path.join(out_dir, "hard_masks.pt"))
    selected_payload = {
        "target_class_idx": target_idx,
        "target_class": class_names[target_idx],
        "keep_ratio": {k: float(masks_cfg["keep_ratio"][k]) for k in soft},
        "top_k": sel["top_k"],
        "indices": sel["indices"],
    }
    with open(
        os.path.join(out_dir, "selected_indices.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(selected_payload, f, indent=2)

    print(f"[INFO] localization done -> {out_dir}")
    for name, idx_list in sel["indices"].items():
        print(f"[INFO] {name} top-{sel['top_k'][name]}: {len(idx_list)} channels")
    return out_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Specific node/class localization (g_i + network masks)"
    )
    p.add_argument("--config", type=str, default=_DEFAULT_CONFIG)
    p.add_argument("--patch_run_dir", type=str, default=None)
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--training_report_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--extracted_layer", type=str, default=None)
    p.add_argument("--target_class", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--budget_lambda", type=float, default=None)
    p.add_argument("--ridge_alpha", type=float, default=None)
    p.add_argument(
        "--keep_ratio",
        type=float,
        default=None,
        help="Uniform keep_ratio for all enabled masks (overrides per-mask)",
    )
    p.add_argument(
        "--disable_mask",
        type=str,
        nargs="*",
        default=None,
        help="Mask names to disable, e.g. M_B",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    overrides: Dict[str, Any] = {
        "patch_run_dir": args.patch_run_dir,
        "checkpoint_path": args.checkpoint_path,
        "training_report_dir": args.training_report_dir,
        "output_dir": args.output_dir,
        "output_root": args.output_root,
        "device": args.device,
        "extracted_layer": args.extracted_layer,
        "target_class": args.target_class,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "ridge_alpha": args.ridge_alpha,
    }
    cfg = load_localization_config(args.config, overrides=overrides)

    if args.budget_lambda is not None:
        cfg.setdefault("masks", {})
        cfg["masks"]["budget_lambda"] = args.budget_lambda
    if args.keep_ratio is not None:
        cfg.setdefault("masks", {})
        cfg["masks"].setdefault("keep_ratio", {})
        for name in MASK_LOCATIONS:
            cfg["masks"]["keep_ratio"][name] = float(args.keep_ratio)
    if args.disable_mask:
        cfg.setdefault("masks", {})
        cfg["masks"].setdefault("enable", {})
        for name in args.disable_mask:
            if name not in MASK_LOCATIONS:
                raise ValueError(f"Unknown mask {name!r}")
            cfg["masks"]["enable"][name] = False

    run_localization(cfg)


if __name__ == "__main__":
    main()
