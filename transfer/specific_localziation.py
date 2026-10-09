#!/usr/bin/env python3
"""Category-specific localization for a node attack class.

Phase 1: fit per-image linear surrogates g_i (locality-weighted lstsq, no bias)
         that map flattened binary patch masks -> full Ms logits.
Phase 2: freeze Ms (Transfer_Net_ResNet18), optimize BasicBlock soft masks
         M_H / M_O after extracted_layer so f_t(Ms^M(x̃)) ≈ f_t(g_i(b));
         then top-k hard masks, compact subnetwork slice, consistency check.
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

from model_architecture import backbone_multi, model_resnet18  # noqa: E402

REFERENCE_MASK_ID = "reference_all_ones"
MASK_TYPES = ("M_H", "M_O")

# (layer_attr, block_index) after extracted_layer cut
_LOCALIZATION_BLOCKS: Dict[str, List[Tuple[str, int]]] = {
    "5_point": [("layer3", 0), ("layer3", 1), ("layer4", 0), ("layer4", 1)],
    "6_point": [("layer3", 1), ("layer4", 0), ("layer4", 1)],
    "7_point": [("layer4", 0), ("layer4", 1)],
    "8_point": [("layer4", 1)],
}


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


def localization_block_specs(extracted_layer: str) -> List[Tuple[str, int]]:
    if extracted_layer not in _LOCALIZATION_BLOCKS:
        raise ValueError(
            f"Unsupported extracted_layer={extracted_layer!r}; "
            f"expected one of {sorted(_LOCALIZATION_BLOCKS)}"
        )
    return list(_LOCALIZATION_BLOCKS[extracted_layer])


def block_key(layer_attr: str, block_idx: int) -> str:
    return f"{layer_attr}.{block_idx}"


def mask_key(layer_attr: str, block_idx: int, mask_type: str) -> str:
    return f"{block_key(layer_attr, block_idx)}.{mask_type}"


def safe_block_key(layer_attr: str, block_idx: int) -> str:
    """ModuleDict/ParameterDict-safe block id: layer4_0."""
    return f"{layer_attr}_{block_idx}"


def safe_block_key_from_logical(bkey: str) -> str:
    """layer4.0 -> layer4_0."""
    return bkey.replace(".", "_")


def param_dict_key(logical_key: str) -> str:
    """ParameterDict forbids '.'; map layer4.0.M_H -> layer4_0__M_H."""
    if logical_key.endswith(".M_H"):
        return logical_key[: -len(".M_H")].replace(".", "_") + "__M_H"
    if logical_key.endswith(".M_O"):
        return logical_key[: -len(".M_O")].replace(".", "_") + "__M_O"
    raise ValueError(f"Unexpected mask key {logical_key!r}")


def logical_mask_key(param_key: str) -> str:
    if param_key.endswith("__M_H"):
        return param_key[: -len("__M_H")].replace("_", ".", 1) + ".M_H"
    if param_key.endswith("__M_O"):
        return param_key[: -len("__M_O")].replace("_", ".", 1) + ".M_O"
    raise ValueError(f"Unexpected param key {param_key!r}")


def get_basic_block(base_network: nn.Module, layer_attr: str, block_idx: int) -> nn.Module:
    layer = getattr(base_network, layer_attr)
    return layer[block_idx]


def load_source_model(
    num_class: int,
    checkpoint_path: str,
    extracted_layer: str,
    device: torch.device,
) -> model_resnet18.Transfer_Net_ResNet18:
    backbone_multi.extracted_layer = extracted_layer
    model = model_resnet18.Transfer_Net_ResNet18(num_class)
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
# Masked ResNet18 (BasicBlock M_H / M_O)
# ---------------------------------------------------------------------------


class MaskedBasicBlock(nn.Module):
    """Wrap a frozen BasicBlock; apply soft or hard M_H / M_O around residual."""

    def __init__(
        self,
        block: nn.Module,
        soft_h_fn: Optional[Any],
        soft_o_fn: Optional[Any],
        enable_h: bool,
        enable_o: bool,
    ):
        super().__init__()
        self.block = block
        self.enable_h = bool(enable_h)
        self.enable_o = bool(enable_o)
        self._soft_h_fn = soft_h_fn
        self._soft_o_fn = soft_o_fn
        self._hard_h: Optional[torch.Tensor] = None
        self._hard_o: Optional[torch.Tensor] = None
        self.use_hard = False

    def set_hard_masks(
        self, hard_h: Optional[torch.Tensor], hard_o: Optional[torch.Tensor]
    ) -> None:
        self.use_hard = True
        self._hard_h = None if hard_h is None else hard_h.detach().float()
        self._hard_o = None if hard_o is None else hard_o.detach().float()

    def _gate_h(self, ref: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enable_h:
            return None
        if self.use_hard:
            assert self._hard_h is not None
            return self._hard_h.to(device=ref.device, dtype=ref.dtype)
        assert self._soft_h_fn is not None
        return self._soft_h_fn()

    def _gate_o(self, ref: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enable_o:
            return None
        if self.use_hard:
            assert self._hard_o is not None
            return self._hard_o.to(device=ref.device, dtype=ref.dtype)
        assert self._soft_o_fn is not None
        return self._soft_o_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)
        g_h = self._gate_h(out)
        if g_h is not None:
            out = out * g_h.view(1, -1, 1, 1)

        out = self.block.conv2(out)
        out = self.block.bn2(out)
        g_o = self._gate_o(out)
        if g_o is not None:
            out = out * g_o.view(1, -1, 1, 1)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)
        if g_o is not None:
            identity = identity * g_o.view(1, -1, 1, 1)

        out = out + identity
        out = self.block.relu(out)
        return out


class MaskedResNet18Net(nn.Module):
    """Frozen Transfer_Net_ResNet18 with learnable M_H/M_O on localization blocks."""

    def __init__(
        self,
        base: model_resnet18.Transfer_Net_ResNet18,
        extracted_layer: str,
        enable: Dict[str, bool],
        init: str = "zeros",
    ):
        super().__init__()
        self.base = base
        self.extracted_layer = extracted_layer
        self.enable = {
            "M_H": bool(enable.get("M_H", True)),
            "M_O": bool(enable.get("M_O", True)),
        }
        self.block_specs = localization_block_specs(extracted_layer)

        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

        self.raw_masks = nn.ParameterDict()
        # Plain dict to avoid double-registering blocks already under self.base
        self.masked_blocks: Dict[str, MaskedBasicBlock] = {}
        self.block_meta: List[Dict[str, Any]] = []

        bn = self.base.base_network
        for layer_attr, block_idx in self.block_specs:
            blk = get_basic_block(bn, layer_attr, block_idx)
            # Unwrap if already wrapped
            while isinstance(blk, MaskedBasicBlock):
                blk = blk.block
            bkey = block_key(layer_attr, block_idx)
            n_h = int(blk.conv1.out_channels)
            n_o = int(blk.conv2.out_channels)
            has_downsample = blk.downsample is not None
            in_channels = int(blk.conv1.in_channels)

            kh = mask_key(layer_attr, block_idx, "M_H")
            ko = mask_key(layer_attr, block_idx, "M_O")
            soft_h_fn = None
            soft_o_fn = None
            if self.enable["M_H"]:
                pk = param_dict_key(kh)
                self.raw_masks[pk] = self._init_raw(n_h, init)
                soft_h_fn = lambda k=pk: torch.sigmoid(self.raw_masks[k])
            if self.enable["M_O"]:
                pk = param_dict_key(ko)
                self.raw_masks[pk] = self._init_raw(n_o, init)
                soft_o_fn = lambda k=pk: torch.sigmoid(self.raw_masks[k])

            wrapped = MaskedBasicBlock(
                blk,
                soft_h_fn,
                soft_o_fn,
                enable_h=self.enable["M_H"],
                enable_o=self.enable["M_O"],
            )
            layer = getattr(bn, layer_attr)
            layer[block_idx] = wrapped
            self.masked_blocks[bkey] = wrapped
            self.block_meta.append(
                {
                    "key": bkey,
                    "layer_attr": layer_attr,
                    "block_idx": block_idx,
                    "n_h": n_h,
                    "n_o": n_o,
                    "has_downsample": has_downsample,
                    "in_channels": in_channels,
                }
            )

    @staticmethod
    def _init_raw(n: int, init: str) -> nn.Parameter:
        if init == "zeros":
            tensor = torch.zeros(n)
        elif init == "ones":
            tensor = torch.ones(n)
        else:
            raise ValueError(f"Unknown mask init {init!r}")
        return nn.Parameter(tensor)

    def soft_masks(self) -> Dict[str, torch.Tensor]:
        return {
            logical_mask_key(k): torch.sigmoid(v) for k, v in self.raw_masks.items()
        }

    def mask_masses(self) -> Dict[str, torch.Tensor]:
        return {k: m.sum() for k, m in self.soft_masks().items()}

    def apply_hard_masks(self, hard: Dict[str, torch.Tensor]) -> None:
        for meta in self.block_meta:
            bkey = meta["key"]
            wrapped: MaskedBasicBlock = self.masked_blocks[bkey]
            wrapped.set_hard_masks(hard.get(f"{bkey}.M_H"), hard.get(f"{bkey}.M_O"))

    def predict(self, x: torch.Tensor, test_flag: int = 1) -> torch.Tensor:
        self.base.eval()
        return self.base.predict(x, test_flag=test_flag)

    def train(self, mode: bool = True):  # type: ignore[override]
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
    model: nn.Module,
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
    model: nn.Module,
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


def mask_type_of(name: str) -> str:
    typ = name.rsplit(".", 1)[-1]
    if typ not in MASK_TYPES:
        raise ValueError(f"Not a typed mask key: {name!r}")
    return typ


def keep_ratio_map_for_keys(
    keys: Sequence[str], masks_cfg: Dict[str, Any]
) -> Dict[str, float]:
    kr = masks_cfg["keep_ratio"]
    return {k: float(kr[mask_type_of(k)]) for k in keys}


def top_k_map_for_keys(
    keys: Sequence[str], masks_cfg: Dict[str, Any]
) -> Dict[str, Optional[int]]:
    tk = masks_cfg.get("top_k") or {}
    out: Dict[str, Optional[int]] = {}
    for k in keys:
        typ = mask_type_of(k)
        v = tk.get(typ, tk.get(k))
        out[k] = None if v is None else int(v)
    return out


def soft_to_hard_masks(
    soft: Dict[str, torch.Tensor],
    keep_ratio: Dict[str, float],
    top_k_cfg: Dict[str, Optional[int]],
    block_meta: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Exact top-k hard masks; identity blocks enforce S_O ⊆ previous S_O."""
    hard: Dict[str, torch.Tensor] = {}
    selected: Dict[str, List[int]] = {}
    meta_k: Dict[str, int] = {}
    prev_s_o: Optional[List[int]] = None

    for meta in block_meta:
        bkey = meta["key"]
        kh = f"{bkey}.M_H"
        ko = f"{bkey}.M_O"
        n_o = int(meta["n_o"])

        if kh in soft:
            m = soft[kh]
            n = m.numel()
            k = resolve_top_k(n, keep_ratio[kh], top_k_cfg.get(kh))
            _, idx = torch.topk(m.detach().cpu(), k)
            h = torch.zeros(n, dtype=torch.float32)
            h[idx] = 1.0
            hard[kh] = h
            selected[kh] = sorted(int(i) for i in idx.tolist())
            meta_k[kh] = k

        if ko in soft:
            m = soft[ko]
            n = m.numel()
            k = resolve_top_k(n, keep_ratio[ko], top_k_cfg.get(ko))
            scores = m.detach().cpu()
            if (not meta["has_downsample"]) and prev_s_o is not None:
                allowed = set(prev_s_o)
                k_eff = min(k, len(allowed))
                masked_scores = torch.full_like(scores, float("-inf"))
                for i in allowed:
                    masked_scores[i] = scores[i]
                _, idx = torch.topk(masked_scores, k_eff)
                k = k_eff
            else:
                _, idx = torch.topk(scores, k)
            h = torch.zeros(n, dtype=torch.float32)
            h[idx] = 1.0
            hard[ko] = h
            selected[ko] = sorted(int(i) for i in idx.tolist())
            meta_k[ko] = k
            prev_s_o = selected[ko]
        else:
            # M_O disabled: keep all output channels for chaining
            prev_s_o = list(range(n_o))

    return hard, {"indices": selected, "top_k": meta_k}


def train_localization(
    masked_model: MaskedResNet18Net,
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
    keep_ratio = keep_ratio_map_for_keys(
        list(masked_model.raw_masks.keys()), cfg["masks"]
    )
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
# Compact subnetwork slicing + consistency
# ---------------------------------------------------------------------------


def _indices_from_hard(
    hard: Dict[str, torch.Tensor], key: str, n: int, enabled: bool
) -> List[int]:
    if not enabled or key not in hard:
        return list(range(n))
    h = hard[key]
    return sorted(int(i) for i in torch.nonzero(h > 0.5, as_tuple=False).view(-1).tolist())


def slice_bn(bn_src: nn.BatchNorm2d, indices: Sequence[int]) -> nn.BatchNorm2d:
    idx = torch.as_tensor(list(indices), dtype=torch.long)
    bn = nn.BatchNorm2d(len(indices), eps=bn_src.eps, momentum=bn_src.momentum)
    with torch.no_grad():
        if bn_src.affine:
            bn.weight.copy_(bn_src.weight.data[idx])
            bn.bias.copy_(bn_src.bias.data[idx])
        bn.running_mean.copy_(bn_src.running_mean.data[idx])
        bn.running_var.copy_(bn_src.running_var.data[idx])
        bn.num_batches_tracked.copy_(bn_src.num_batches_tracked)
    return bn


def slice_conv2d(
    conv_src: nn.Conv2d,
    out_idx: Optional[Sequence[int]],
    in_idx: Optional[Sequence[int]],
) -> nn.Conv2d:
    w = conv_src.weight.data
    if out_idx is not None:
        w = w[torch.as_tensor(list(out_idx), dtype=torch.long)]
    if in_idx is not None:
        w = w[:, torch.as_tensor(list(in_idx), dtype=torch.long)]
    out_c, in_c = w.shape[0], w.shape[1]
    conv = nn.Conv2d(
        in_c,
        out_c,
        kernel_size=conv_src.kernel_size,
        stride=conv_src.stride,
        padding=conv_src.padding,
        dilation=conv_src.dilation,
        groups=1,
        bias=conv_src.bias is not None,
    )
    with torch.no_grad():
        conv.weight.copy_(w)
        if conv_src.bias is not None:
            b = conv_src.bias.data
            if out_idx is not None:
                b = b[torch.as_tensor(list(out_idx), dtype=torch.long)]
            conv.bias.copy_(b)
    return conv


class CompactBasicBlock(nn.Module):
    """Sliced BasicBlock; optional channel gather for identity when |S_O| < |in|."""

    def __init__(
        self,
        conv1: nn.Conv2d,
        bn1: nn.BatchNorm2d,
        conv2: nn.Conv2d,
        bn2: nn.BatchNorm2d,
        downsample: Optional[nn.Module] = None,
        identity_gather_idx: Optional[List[int]] = None,
    ):
        super().__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample
        self.identity_gather_idx = identity_gather_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        elif self.identity_gather_idx is not None:
            gather = torch.as_tensor(
                self.identity_gather_idx, device=x.device, dtype=torch.long
            )
            identity = x.index_select(1, gather)
        out = self.relu(out + identity)
        return out


class CompactTransferNet(nn.Module):
    """Prefix shared from source + sliced localization blocks + sliced FC."""

    def __init__(
        self,
        source: model_resnet18.Transfer_Net_ResNet18,
        extracted_layer: str,
        compact_blocks: Dict[str, CompactBasicBlock],
        classifier: nn.Linear,
    ):
        super().__init__()
        self.extracted_layer = extracted_layer
        self.compact_blocks = nn.ModuleDict(compact_blocks)
        self.classifier_layer = classifier
        # Keep a frozen copy of stem/prefix modules by reference from a shallow structure
        src_bn = source.base_network
        self.conv1 = src_bn.conv1
        self.bn1 = src_bn.bn1
        self.relu = src_bn.relu
        self.maxpool = src_bn.maxpool
        self.layer1 = src_bn.layer1
        self.layer2 = src_bn.layer2
        self.layer3 = src_bn.layer3
        self.layer4 = src_bn.layer4
        self.avgpool = src_bn.avgpool
        self._block_specs = localization_block_specs(extracted_layer)

    def _run_block(self, layer_attr: str, block_idx: int, x: torch.Tensor) -> torch.Tensor:
        skey = safe_block_key(layer_attr, block_idx)
        if skey in self.compact_blocks:
            return self.compact_blocks[skey](x)
        layer = getattr(self, layer_attr)
        blk = layer[block_idx]
        if isinstance(blk, MaskedBasicBlock):
            return blk.block(x)
        return blk(x)

    def forward_features(self, x: torch.Tensor, test_flag: int = 1) -> torch.Tensor:
        extracted = self.extracted_layer
        if test_flag:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            if extracted == "6_point":
                x = self._run_block("layer3", 0, x)
            elif extracted == "7_point":
                x = self.layer3(x)
            elif extracted == "8_point":
                x = self.layer3(x)
                x = self._run_block("layer4", 0, x)

        if extracted == "5_point":
            x = self._run_block("layer3", 0, x)
            x = self._run_block("layer3", 1, x)
            x = self._run_block("layer4", 0, x)
            x = self._run_block("layer4", 1, x)
        elif extracted == "6_point":
            x = self._run_block("layer3", 1, x)
            x = self._run_block("layer4", 0, x)
            x = self._run_block("layer4", 1, x)
        elif extracted == "7_point":
            x = self._run_block("layer4", 0, x)
            x = self._run_block("layer4", 1, x)
        elif extracted == "8_point":
            x = self._run_block("layer4", 1, x)
        else:
            raise ValueError(f"Unsupported extracted_layer={extracted!r}")

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def predict(self, x: torch.Tensor, test_flag: int = 1) -> torch.Tensor:
        return self.classifier_layer(self.forward_features(x, test_flag=test_flag))


def _unwrap_block(blk: nn.Module) -> nn.Module:
    while isinstance(blk, MaskedBasicBlock):
        blk = blk.block
    return blk


def build_compact_subnetwork(
    masked_model: MaskedResNet18Net,
    hard: Dict[str, torch.Tensor],
) -> Tuple[CompactTransferNet, Dict[str, Any]]:
    """Physically slice localization blocks (+ FC) according to S_H / S_O."""
    enable = masked_model.enable
    source = masked_model.base
    bn = source.base_network
    compact_blocks: Dict[str, CompactBasicBlock] = {}
    meta_out: Dict[str, Any] = {"blocks": {}, "classifier_in": None}
    in_idx: Optional[List[int]] = None  # None => full original input channels

    for meta in masked_model.block_meta:
        layer_attr = meta["layer_attr"]
        block_idx = meta["block_idx"]
        bkey = meta["key"]
        src = _unwrap_block(get_basic_block(bn, layer_attr, block_idx))
        n_h, n_o = int(meta["n_h"]), int(meta["n_o"])
        in_channels = int(meta["in_channels"])

        s_h = _indices_from_hard(hard, f"{bkey}.M_H", n_h, enable["M_H"])
        s_o = _indices_from_hard(hard, f"{bkey}.M_O", n_o, enable["M_O"])

        if in_idx is None:
            in_idx_list = list(range(in_channels))
        else:
            in_idx_list = list(in_idx)

        conv1 = slice_conv2d(src.conv1, out_idx=s_h, in_idx=in_idx_list)
        bn1 = slice_bn(src.bn1, s_h)
        conv2 = slice_conv2d(src.conv2, out_idx=s_o, in_idx=s_h)
        bn2 = slice_bn(src.bn2, s_o)

        downsample = None
        identity_gather: Optional[List[int]] = None
        if src.downsample is not None:
            ds_conv = slice_conv2d(src.downsample[0], out_idx=s_o, in_idx=in_idx_list)
            ds_bn = slice_bn(src.downsample[1], s_o)
            downsample = nn.Sequential(ds_conv, ds_bn)
        else:
            # Map S_O (original ids) -> positions within current compact input (in_idx_list)
            pos = {orig: i for i, orig in enumerate(in_idx_list)}
            try:
                identity_gather = [pos[i] for i in s_o]
            except KeyError as e:
                raise RuntimeError(
                    f"{bkey}: S_O index {e} not in previous S_O / input set "
                    f"(identity ⊆ prev violated)"
                ) from e
            # If gather is identity permutation of full input and same width, skip
            if identity_gather == list(range(len(in_idx_list))) and len(s_o) == len(
                in_idx_list
            ):
                identity_gather = None

        skey = safe_block_key_from_logical(bkey)
        compact_blocks[skey] = CompactBasicBlock(
            conv1, bn1, conv2, bn2, downsample, identity_gather
        )
        meta_out["blocks"][bkey] = {
            "S_H": s_h,
            "S_O": s_o,
            "in_idx": in_idx_list,
            "has_downsample": src.downsample is not None,
            "module_key": skey,
        }
        in_idx = list(s_o)

    assert in_idx is not None
    # Slice classifier: weight [num_class, 512]
    fc_src = source.classifier_layer
    idx = torch.as_tensor(in_idx, dtype=torch.long)
    fc = nn.Linear(len(in_idx), fc_src.out_features, bias=fc_src.bias is not None)
    with torch.no_grad():
        fc.weight.copy_(fc_src.weight.data[:, idx])
        if fc_src.bias is not None:
            fc.bias.copy_(fc_src.bias.data)
    meta_out["classifier_in"] = in_idx

    compact = CompactTransferNet(
        source=source,
        extracted_layer=masked_model.extracted_layer,
        compact_blocks=compact_blocks,
        classifier=fc,
    )
    compact.eval()
    for p in compact.parameters():
        p.requires_grad = False
    return compact, meta_out


def verify_hard_vs_compact(
    masked_model: MaskedResNet18Net,
    compact: CompactTransferNet,
    sample_x: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> Dict[str, Any]:
    device = sample_x.device
    masked_model.eval()
    compact.eval()
    compact.to(device)
    with torch.no_grad():
        y_full = masked_model.predict(sample_x, test_flag=1)
        y_compact = compact.predict(sample_x, test_flag=1)
        diff = (y_full - y_compact).abs()
        max_abs = float(diff.max().item())
        denom = y_full.abs().max().clamp(min=1e-8)
        max_rel = float((diff / denom).max().item())
    ok = (max_abs <= atol) or (max_rel <= rtol)
    report = {
        "ok": ok,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "atol": atol,
        "rtol": rtol,
        "batch_size": int(sample_x.size(0)),
        "logits_shape": list(y_full.shape),
    }
    if not ok:
        raise RuntimeError(
            "Consistency check failed: hard-masked full vs compact "
            f"max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
        )
    return report


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
    enable = {
        "M_H": bool(masks_cfg.get("enable", {}).get("M_H", True)),
        "M_O": bool(masks_cfg.get("enable", {}).get("M_O", True)),
    }
    masked = MaskedResNet18Net(
        base=model,
        extracted_layer=extracted,
        enable=enable,
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
    raw = {
        logical_mask_key(k): v.detach().cpu() for k, v in masked.raw_masks.items()
    }
    torch.save({"soft": soft, "raw": raw}, os.path.join(out_dir, "soft_masks.pt"))

    hard, sel = soft_to_hard_masks(
        soft,
        keep_ratio=keep_ratio_map_for_keys(list(soft.keys()), masks_cfg),
        top_k_cfg=top_k_map_for_keys(list(soft.keys()), masks_cfg),
        block_meta=masked.block_meta,
    )
    torch.save(hard, os.path.join(out_dir, "hard_masks.pt"))
    selected_payload = {
        "target_class_idx": target_idx,
        "target_class": class_names[target_idx],
        "extracted_layer": extracted,
        "block_meta": masked.block_meta,
        "keep_ratio": {t: float(masks_cfg["keep_ratio"][t]) for t in MASK_TYPES},
        "top_k": sel["top_k"],
        "indices": sel["indices"],
    }
    with open(
        os.path.join(out_dir, "selected_indices.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(selected_payload, f, indent=2)

    masked.apply_hard_masks(
        {k: v.to(device) for k, v in hard.items()}
    )
    compact, compact_meta = build_compact_subnetwork(masked, hard)
    compact.to(device)
    torch.save(
        {
            "extracted_layer": extracted,
            "meta": compact_meta,
            "compact_blocks": compact.compact_blocks.state_dict(),
            "classifier_layer": compact.classifier_layer.state_dict(),
            # Prefix (stem..pre-cut) from frozen source for reload/rebuild
            "source_base_network": {
                k: v.cpu()
                for k, v in masked.base.base_network.state_dict().items()
            },
            "source_classifier_full": {
                k: v.cpu()
                for k, v in masked.base.classifier_layer.state_dict().items()
            },
        },
        os.path.join(out_dir, "compact_state_dict.pt"),
    )
    with open(
        os.path.join(out_dir, "compact_meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(compact_meta, f, indent=2)

    # Consistency batch from localization dataset
    n_cons = min(8, len(dataset))
    xs = []
    for i in range(n_cons):
        x_i, *_rest = dataset[i]
        xs.append(x_i)
    sample_x = torch.stack(xs, dim=0).to(device)
    cons = verify_hard_vs_compact(masked, compact, sample_x)
    with open(
        os.path.join(out_dir, "consistency_report.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(cons, f, indent=2)
    print(
        f"[INFO] consistency ok max_abs={cons['max_abs_diff']:.6g} "
        f"max_rel={cons['max_rel_diff']:.6g}"
    )

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
        help="Uniform keep_ratio for M_H and M_O (overrides per-type)",
    )
    p.add_argument(
        "--disable_mask",
        type=str,
        nargs="*",
        default=None,
        help="Mask types to disable: M_H and/or M_O",
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
        for name in MASK_TYPES:
            cfg["masks"]["keep_ratio"][name] = float(args.keep_ratio)
    if args.disable_mask:
        cfg.setdefault("masks", {})
        cfg["masks"].setdefault("enable", {})
        for name in args.disable_mask:
            if name not in MASK_TYPES:
                raise ValueError(f"Unknown mask type {name!r}; expected M_H or M_O")
            cfg["masks"]["enable"][name] = False

    run_localization(cfg)


if __name__ == "__main__":
    main()
