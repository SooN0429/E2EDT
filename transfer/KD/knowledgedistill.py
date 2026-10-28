#!/usr/bin/env python3
"""Parallel-insert compact subnet into a target model and distill adapter + expanded FC.

Pipeline:
  resolve → build ParallelInsertStudent → epoch-0 sanity → train on patches
  → generate temporary eval set (old+new classes) → three-way metrics → cleanup.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_E2EDT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_NODE_TRAINING_DIR = os.path.join(_E2EDT_ROOT, "node_model_training")
_NODE_DATASET_DIR = os.path.join(_E2EDT_ROOT, "node_dataset")
_DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "distill_config.json")
_DEFAULT_OUTPUT_ROOT = os.path.join(_SCRIPT_DIR, "distill_result")
_DEFAULT_TRAINING_REPORT_ROOT = os.path.join(_NODE_TRAINING_DIR, "training_report")

if _NODE_TRAINING_DIR not in sys.path:
    sys.path.insert(0, _NODE_TRAINING_DIR)
if _NODE_DATASET_DIR not in sys.path:
    sys.path.insert(0, _NODE_DATASET_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from model_architecture import backbone_multi, model_resnet18  # noqa: E402
import node_traindata_generate as ntdg  # noqa: E402
import specific_localziation as loc  # noqa: E402


# ---------------------------------------------------------------------------
# Config / IO helpers
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


def load_distill_config(
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


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Resolve localization + target
# ---------------------------------------------------------------------------


def find_latest_training_report(training_note_root: str) -> str:
    root = os.path.abspath(training_note_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"training_report root not found: {root}")
    candidates: List[Tuple[str, str]] = []
    for name in os.listdir(root):
        run_dir = os.path.join(root, name)
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.isdir(run_dir) or not os.path.isfile(cfg_path):
            continue
        cfg = load_json(cfg_path)
        created = cfg.get("created_at") or ""
        if not created:
            created = datetime.fromtimestamp(os.path.getmtime(run_dir)).isoformat(
                timespec="seconds"
            )
        candidates.append((created, run_dir))
    if not candidates:
        raise FileNotFoundError(f"No training runs with config.json under {root}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _checkpoint_looks_like_tail(state: Dict[str, Any]) -> bool:
    keys = list(state.keys())
    if any(k.startswith("bottle_layer") for k in keys):
        return False
    if any(k.startswith("classifier_layer.1.") for k in keys):
        return False
    return any(k.startswith("base_network.") for k in keys) and any(
        k.startswith("classifier_layer.weight") or k == "classifier_layer.weight"
        for k in keys
    )


def assert_resnet18_tail_architecture(
    report_cfg: Dict[str, Any],
    checkpoint_path: str,
) -> str:
    """Require Transfer_Net_ResNet18 / resnet18_tail. Return resolved arch label."""
    arch = report_cfg.get("model_arch")
    if arch == "resnet18_multi":
        raise ValueError(
            f"target model_arch={arch!r} is incompatible; need resnet18_tail "
            f"(Transfer_Net_ResNet18). Pick another training_report run."
        )
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint format: {checkpoint_path}")
    if arch == "resnet18_tail":
        if not _checkpoint_looks_like_tail(state):
            raise ValueError(
                f"config says resnet18_tail but checkpoint keys look incompatible: "
                f"{checkpoint_path}"
            )
        return "resnet18_tail"
    # arch missing / unknown → gate by checkpoint structure
    if not _checkpoint_looks_like_tail(state):
        raise ValueError(
            f"Cannot confirm resnet18_tail architecture for {checkpoint_path} "
            f"(model_arch={arch!r}). Refusing non-tail / bottle Transfer_Net."
        )
    return arch or "resnet18_tail(inferred)"


def resolve_target(
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve target checkpoint, classes, training report, dataset paths."""
    target_node = cfg.get("target_node")
    if not target_node:
        raise ValueError("target_node is required")

    report_root = os.path.abspath(
        cfg.get("training_report_root") or _DEFAULT_TRAINING_REPORT_ROOT
    )
    report_dir = cfg.get("target_training_report_dir")
    ckpt_path = cfg.get("target_checkpoint_path")

    if report_dir:
        report_dir = os.path.abspath(report_dir)
    elif ckpt_path:
        # Infer report dir from checkpoint parent if under checkpoints/
        ckpt_abs = os.path.abspath(ckpt_path)
        parent = os.path.dirname(ckpt_abs)
        if os.path.basename(parent) == "checkpoints":
            report_dir = os.path.dirname(parent)
        else:
            report_dir = find_latest_training_report(report_root)
    else:
        report_dir = find_latest_training_report(report_root)

    report_cfg = load_json(os.path.join(report_dir, "config.json"))
    index_path = os.path.join(report_dir, "checkpoints", "index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"checkpoints/index.json not found under {report_dir}")
    index = load_json(index_path)
    if target_node not in index.get("nodes", {}):
        raise KeyError(f"node {target_node!r} missing in {index_path}")
    entry = index["nodes"][target_node]
    class_names = list(entry["classes"])
    if not ckpt_path:
        ckpt_path = os.path.abspath(entry["path"])
    else:
        ckpt_path = os.path.abspath(ckpt_path)

    arch_label = assert_resnet18_tail_architecture(report_cfg, ckpt_path)

    dataset_root = cfg.get("dataset_root") or report_cfg.get("dataset_root") or _NODE_DATASET_DIR
    dataset_root = os.path.abspath(dataset_root)
    trigger_dir = cfg.get("trigger_dir") or report_cfg.get("trigger_dir") or ""
    trigger_dir = trigger_dir or os.path.join(dataset_root, "Attack_trigger_image")
    trigger_dir = os.path.abspath(trigger_dir)

    legacy = report_cfg.get("blended_alpha")
    fallback = float(legacy) if legacy is not None else 0.2
    alpha_sq = cfg.get("blended_alpha_square")
    alpha_hk = cfg.get("blended_alpha_hello_kitty")
    if alpha_sq is None:
        alpha_sq = report_cfg.get("blended_alpha_square", fallback)
    if alpha_hk is None:
        alpha_hk = report_cfg.get("blended_alpha_hello_kitty", fallback)

    return {
        "target_node": target_node,
        "checkpoint_path": ckpt_path,
        "class_names": class_names,
        "report_dir": report_dir,
        "report_cfg": report_cfg,
        "arch_label": arch_label,
        "dataset_root": dataset_root,
        "trigger_dir": trigger_dir,
        "blended_alpha_square": float(alpha_sq),
        "blended_alpha_hello_kitty": float(alpha_hk),
        "extracted_layer_report": report_cfg.get("extracted_layer"),
    }


def resolve_localization(localization_run_dir: str) -> Dict[str, Any]:
    run_dir = os.path.abspath(localization_run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"localization_run_dir not found: {run_dir}")

    compact_pt = os.path.join(run_dir, "compact_state_dict.pt")
    compact_meta_path = os.path.join(run_dir, "compact_meta.json")
    loc_cfg_path = os.path.join(run_dir, "config.json")
    sel_path = os.path.join(run_dir, "selected_indices.json")
    if not os.path.isfile(compact_pt):
        raise FileNotFoundError(f"missing compact_state_dict.pt under {run_dir}")
    if not os.path.isfile(compact_meta_path):
        raise FileNotFoundError(f"missing compact_meta.json under {run_dir}")
    if not os.path.isfile(loc_cfg_path):
        raise FileNotFoundError(f"missing config.json under {run_dir}")

    loc_cfg = load_json(loc_cfg_path)
    compact_meta = load_json(compact_meta_path)
    compact_blob = torch.load(compact_pt, map_location="cpu")
    extracted = (
        compact_blob.get("extracted_layer")
        or loc_cfg.get("resolved_extracted_layer")
        or loc_cfg.get("extracted_layer")
        or "7_point"
    )
    source_classes = list(
        loc_cfg.get("resolved_class_names")
        or (load_json(sel_path).get("class_names") if os.path.isfile(sel_path) else None)
        or []
    )
    sel = load_json(sel_path) if os.path.isfile(sel_path) else {}
    if not source_classes:
        source_classes = list(sel.get("class_names") or [])

    patch_run_dir = loc_cfg.get("patch_run_dir")
    if not patch_run_dir:
        raise ValueError(f"localization config missing patch_run_dir: {loc_cfg_path}")
    patch_run_dir = os.path.abspath(patch_run_dir)

    # Prefer patch config node_specs if source classes still missing
    if not source_classes:
        patch_cfg_path = os.path.join(patch_run_dir, "config_resolved.json")
        if os.path.isfile(patch_cfg_path):
            pcfg = load_json(patch_cfg_path)
            node = loc_cfg.get("resolved_node") or pcfg.get("node") or pcfg.get("node_name")
            specs = (pcfg.get("training_report") or {}).get("node_specs") or {}
            if node and node in specs:
                source_classes = list(specs[node])

    attack_class = (
        loc_cfg.get("resolved_attack_class")
        or loc_cfg.get("target_class")
        or sel.get("target_class")
    )
    # g_i index: localization target class (may equal attack)
    attack_idx: Optional[int] = None
    if loc_cfg.get("resolved_target_class_idx") is not None:
        attack_idx = int(loc_cfg["resolved_target_class_idx"])
    elif sel.get("target_class_idx") is not None:
        attack_idx = int(sel["target_class_idx"])
    elif attack_class and attack_class in source_classes:
        attack_idx = int(source_classes.index(attack_class))

    if attack_idx is None:
        raise ValueError(
            f"Cannot resolve attack/new class index from {run_dir} "
            f"(attack_class={attack_class!r}, source_classes={source_classes})"
        )
    if not attack_class:
        if 0 <= attack_idx < len(source_classes):
            attack_class = source_classes[attack_idx]
        else:
            raise ValueError(f"Cannot resolve attack/new class name from {run_dir}")

    classifier_in = compact_meta.get("classifier_in")
    if not classifier_in:
        raise ValueError("compact_meta missing classifier_in (C_f)")
    c_f = len(classifier_in)

    return {
        "run_dir": run_dir,
        "compact_blob": compact_blob,
        "compact_meta": compact_meta,
        "loc_cfg": loc_cfg,
        "extracted_layer": extracted,
        "source_class_names": source_classes,
        "attack_class": attack_class,
        "attack_idx_src": int(attack_idx),
        "patch_run_dir": patch_run_dir,
        "c_f": c_f,
    }


# ---------------------------------------------------------------------------
# Compact Mf reload
# ---------------------------------------------------------------------------


def _rebuild_one_compact_block(
    skey: str,
    bmeta: Dict[str, Any],
    sub_sd: Dict[str, torch.Tensor],
) -> loc.CompactBasicBlock:
    w1 = sub_sd["conv1.weight"]
    out_h, in_c, kH, kW = int(w1.shape[0]), int(w1.shape[1]), int(w1.shape[2]), int(w1.shape[3])
    has_ds = bool(bmeta.get("has_downsample"))
    stride = 2 if has_ds else 1
    bias1 = "conv1.bias" in sub_sd
    conv1 = nn.Conv2d(in_c, out_h, (kH, kW), stride=stride, padding=1, bias=bias1)
    bn1 = nn.BatchNorm2d(out_h)

    w2 = sub_sd["conv2.weight"]
    out_o, in_h = int(w2.shape[0]), int(w2.shape[1])
    bias2 = "conv2.bias" in sub_sd
    conv2 = nn.Conv2d(in_h, out_o, tuple(w2.shape[2:]), stride=1, padding=1, bias=bias2)
    bn2 = nn.BatchNorm2d(out_o)

    downsample = None
    if has_ds:
        dw = sub_sd["downsample.0.weight"]
        d_out, d_in = int(dw.shape[0]), int(dw.shape[1])
        ds_conv = nn.Conv2d(d_in, d_out, 1, stride=stride, bias="downsample.0.bias" in sub_sd)
        ds_bn = nn.BatchNorm2d(d_out)
        downsample = nn.Sequential(ds_conv, ds_bn)

    in_idx = list(bmeta["in_idx"])
    s_o = list(bmeta["S_O"])
    identity_gather: Optional[List[int]] = None
    if not has_ds:
        pos = {orig: i for i, orig in enumerate(in_idx)}
        try:
            identity_gather = [pos[i] for i in s_o]
        except KeyError as e:
            raise RuntimeError(f"{skey}: S_O index {e} not in in_idx") from e
        if identity_gather == list(range(len(in_idx))) and len(s_o) == len(in_idx):
            identity_gather = None

    block = loc.CompactBasicBlock(
        conv1, bn1, conv2, bn2, downsample, identity_gather
    )
    # load with module-relative keys
    block.load_state_dict(sub_sd, strict=True)
    return block


def load_compact_mf(
    compact_blob: Dict[str, Any],
    compact_meta: Dict[str, Any],
) -> Tuple[nn.ModuleDict, Optional[List[int]]]:
    """Rebuild Mf ModuleDict; return (blocks, first_block_in_idx_or_None)."""
    blocks_sd = compact_blob["compact_blocks"]
    # keys like layer4_0.conv1.weight
    by_block: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in blocks_sd.items():
        skey, rest = k.split(".", 1)
        by_block.setdefault(skey, {})[rest] = v

    modules: Dict[str, loc.CompactBasicBlock] = {}
    first_in_idx: Optional[List[int]] = None
    for bkey, bmeta in compact_meta["blocks"].items():
        skey = bmeta.get("module_key") or loc.safe_block_key_from_logical(bkey)
        if skey not in by_block:
            raise KeyError(f"compact block {skey} missing in state_dict")
        modules[skey] = _rebuild_one_compact_block(skey, bmeta, by_block[skey])
        if first_in_idx is None:
            first_in_idx = list(bmeta["in_idx"])

    return nn.ModuleDict(modules), first_in_idx


# ---------------------------------------------------------------------------
# ParallelInsertStudent
# ---------------------------------------------------------------------------


class ParallelInsertStudent(nn.Module):
    """Shared target prefix h_R → parallel {target_tail, Mf→adapter} → concat → FC."""

    def __init__(
        self,
        target: model_resnet18.Transfer_Net_ResNet18,
        extracted_layer: str,
        mf_blocks: nn.ModuleDict,
        c_f: int,
        n_old: int,
        n_new: int = 1,
        first_in_idx: Optional[List[int]] = None,
        adapter_out_channels: int = 512,
    ):
        super().__init__()
        self.extracted_layer = extracted_layer
        self.n_old = int(n_old)
        self.n_new = int(n_new)
        self.adapter_out_channels = int(adapter_out_channels)
        self.first_in_idx = first_in_idx
        self._block_specs = loc.localization_block_specs(extracted_layer)

        # Frozen target backbone pieces (same module refs as `target`)
        bn = target.base_network
        self.conv1 = bn.conv1
        self.bn1 = bn.bn1
        self.relu = bn.relu
        self.maxpool = bn.maxpool
        self.layer1 = bn.layer1
        self.layer2 = bn.layer2
        self.layer3 = bn.layer3
        self.layer4 = bn.layer4
        self.avgpool = bn.avgpool

        # Original classifier kept for detach recovery
        self.orig_classifier = target.classifier_layer

        self.mf = mf_blocks
        self.adapter = nn.Sequential(
            nn.Conv2d(c_f, adapter_out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.classifier_layer = nn.Linear(
            512 + adapter_out_channels, n_old + n_new, bias=True
        )
        self._init_expanded_classifier(target.classifier_layer)

        # freeze everything except adapter + expanded FC
        for p in self.parameters():
            p.requires_grad = False
        for p in self.adapter.parameters():
            p.requires_grad = True
        for p in self.classifier_layer.parameters():
            p.requires_grad = True

    def _init_expanded_classifier(self, src_fc: nn.Linear) -> None:
        with torch.no_grad():
            self.classifier_layer.weight.zero_()
            self.classifier_layer.bias.zero_()
            # old-class rows: [W_target | 0]
            w = src_fc.weight.data  # [C_old, 512]
            self.classifier_layer.weight[: self.n_old, :512].copy_(w)
            if src_fc.bias is not None:
                self.classifier_layer.bias[: self.n_old].copy_(src_fc.bias.data)
            # new-class rows: small random on adapter half only
            nn.init.normal_(
                self.classifier_layer.weight[self.n_old :, 512:], mean=0.0, std=0.01
            )

    def forward_prefix(self, x: torch.Tensor) -> torch.Tensor:
        """Run shared target prefix up to insertion point R → h_R."""
        extracted = self.extracted_layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if extracted == "5_point":
            return x
        if extracted == "6_point":
            return self.layer3[0](x)
        if extracted == "7_point":
            return self.layer3(x)
        if extracted == "8_point":
            x = self.layer3(x)
            return self.layer4[0](x)
        raise ValueError(f"Unsupported extracted_layer={extracted!r}")

    def forward_tail(self, h_r: torch.Tensor) -> torch.Tensor:
        """Target original tail after R (spatial feature map, no avgpool)."""
        extracted = self.extracted_layer
        x = h_r
        if extracted == "5_point":
            x = self.layer3(x)
            x = self.layer4(x)
        elif extracted == "6_point":
            x = self.layer3[1](x)
            x = self.layer4(x)
        elif extracted == "7_point":
            x = self.layer4(x)
        elif extracted == "8_point":
            x = self.layer4[1](x)
        else:
            raise ValueError(f"Unsupported extracted_layer={extracted!r}")
        return x

    def forward_mf(self, h_r: torch.Tensor) -> torch.Tensor:
        x = h_r
        if self.first_in_idx is not None:
            # If first block expects a channel subset, gather; full range → no-op
            n_in = int(x.shape[1])
            if list(self.first_in_idx) != list(range(n_in)):
                gather = torch.as_tensor(
                    self.first_in_idx, device=x.device, dtype=torch.long
                )
                x = x.index_select(1, gather)
        for layer_attr, block_idx in self._block_specs:
            skey = loc.safe_block_key(layer_attr, block_idx)
            x = self.mf[skey](x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        detach: bool = False,
    ) -> torch.Tensor:
        h_r = self.forward_prefix(x)
        z_t = self.forward_tail(h_r)
        if detach:
            feat = torch.flatten(self.avgpool(z_t), 1)
            return self.orig_classifier(feat)

        z_f = self.adapter(self.forward_mf(h_r))
        fused = torch.cat([z_t, z_f], dim=1)
        feat = torch.flatten(self.avgpool(fused), 1)
        return self.classifier_layer(feat)

    def predict(self, x: torch.Tensor, detach: bool = False) -> torch.Tensor:
        return self.forward(x, detach=detach)

    def trainable_parameters(self):
        for p in self.adapter.parameters():
            yield p
        for p in self.classifier_layer.parameters():
            yield p


def build_student_and_teacher(
    target_ckpt: str,
    n_old: int,
    extracted_layer: str,
    loc_info: Dict[str, Any],
    device: torch.device,
) -> Tuple[ParallelInsertStudent, model_resnet18.Transfer_Net_ResNet18]:
    backbone_multi.extracted_layer = extracted_layer
    teacher = model_resnet18.Transfer_Net_ResNet18(n_old)
    state = torch.load(target_ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    teacher.load_state_dict(state, strict=True)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Independent copy for student backbone / orig classifier
    student_base = model_resnet18.Transfer_Net_ResNet18(n_old)
    student_base.load_state_dict(deepcopy(teacher.state_dict()), strict=True)

    mf, first_in_idx = load_compact_mf(
        loc_info["compact_blob"], loc_info["compact_meta"]
    )
    student = ParallelInsertStudent(
        target=student_base,
        extracted_layer=extracted_layer,
        mf_blocks=mf,
        c_f=loc_info["c_f"],
        n_old=n_old,
        n_new=1,
        first_in_idx=first_in_idx,
    )
    # freeze mf explicitly
    for p in student.mf.parameters():
        p.requires_grad = False
    student.to(device)
    student.train()  # adapter/BN in adapter only; backbone frozen
    # Keep BatchNorm in frozen backbone in eval mode
    student.conv1.eval()
    student.bn1.eval()
    student.layer1.eval()
    student.layer2.eval()
    student.layer3.eval()
    student.layer4.eval()
    student.mf.eval()
    return student, teacher


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_sanity_check(
    student: ParallelInsertStudent,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    atol: float,
    rtol: float,
    max_batches: int,
) -> Dict[str, Any]:
    student.eval()
    teacher.eval()
    max_abs = 0.0
    max_rel = 0.0
    mean_abs_acc = 0.0
    n = 0
    batches = 0
    for batch in loader:
        x = batch[0].to(device)
        s_old = student.predict(x, detach=False)[:, : student.n_old]
        t_logits = teacher.predict(x, test_flag=1)
        diff = (s_old - t_logits).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        denom = t_logits.abs().clamp_min(1e-6)
        max_rel = max(max_rel, float((diff / denom).max().item()))
        mean_abs_acc += float(diff.mean().item()) * x.size(0)
        n += x.size(0)
        batches += 1
        if batches >= max_batches:
            break

    mean_abs = mean_abs_acc / max(n, 1)
    ok = (max_abs <= atol) or (max_rel <= rtol)
    report = {
        "ok": bool(ok),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "mean_abs_diff": mean_abs,
        "n_samples": n,
        "atol": atol,
        "rtol": rtol,
        "diagnostics_if_fail": [
            "classifier init: old rows must be [W_target | 0]",
            "feature order: concat must be [z_t, z_f] matching FC halves",
            "fusion / adapter path error",
            "target branch accidentally modified vs teacher",
        ],
    }
    if not ok:
        # Extra: zero-adapter path via detach should match teacher
        det_max = 0.0
        for batch in loader:
            x = batch[0].to(device)
            d_logits = student.predict(x, detach=True)
            t_logits = teacher.predict(x, test_flag=1)
            det_max = max(det_max, float((d_logits - t_logits).abs().max().item()))
            break
        report["detach_vs_teacher_max_abs"] = det_max
        raise RuntimeError(
            "Epoch-0 sanity check failed: student_old(x) !~= teacher(x). "
            f"max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
            f"(atol={atol}, rtol={rtol}). "
            "Check: FC init [W_t|0], concat feature order, fusion path, "
            "or mutated target branch. "
            f"detach_vs_teacher_max_abs={det_max:.6g}"
        )
    return report


# ---------------------------------------------------------------------------
# Distill dataset / losses
# ---------------------------------------------------------------------------


def load_surrogate_W_by_image(loc_run_dir: str) -> Dict[str, np.ndarray]:
    metrics_csv = os.path.join(loc_run_dir, "g_fit_metrics.csv")
    W_by: Dict[str, np.ndarray] = {}
    if os.path.isfile(metrics_csv):
        with open(metrics_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row["image_id"]
                npz_path = row["npz_path"]
                if not os.path.isabs(npz_path):
                    npz_path = os.path.join(loc_run_dir, npz_path)
                blob = np.load(npz_path, allow_pickle=True)
                W_by[image_id] = np.asarray(blob["W"], dtype=np.float32)
        return W_by

    # Fallback: scan surrogate_models/
    sdir = os.path.join(loc_run_dir, "surrogate_models")
    if not os.path.isdir(sdir):
        raise FileNotFoundError(f"No g_fit_metrics.csv or surrogate_models under {loc_run_dir}")
    for name in sorted(os.listdir(sdir)):
        if not name.endswith(".npz"):
            continue
        path = os.path.join(sdir, name)
        blob = np.load(path, allow_pickle=True)
        image_id = str(blob["image_id"])
        W_by[image_id] = np.asarray(blob["W"], dtype=np.float32)
    if not W_by:
        raise FileNotFoundError(f"No surrogate g_i found under {sdir}")
    return W_by


class DistillPatchDataset(Dataset):
    """Patch tensors + binary mask → g_i(b) full source logits."""

    def __init__(
        self,
        patch_run_dir: str,
        loc_run_dir: str,
        include_reference: bool = True,
    ):
        manifest_path = os.path.join(patch_run_dir, "manifest.jsonl")
        if not os.path.isfile(manifest_path):
            # some generators may use manifest.json
            alt = os.path.join(patch_run_dir, "manifest.json")
            if os.path.isfile(alt):
                manifest_path = alt
            else:
                raise FileNotFoundError(f"manifest not found under {patch_run_dir}")

        if manifest_path.endswith(".jsonl"):
            rows = loc.load_manifest(manifest_path)
        else:
            rows = load_json(manifest_path)
            if not isinstance(rows, list):
                raise TypeError("manifest.json must be a list")

        groups = loc.group_manifest_by_image(rows, include_reference=include_reference)
        W_by = load_surrogate_W_by_image(loc_run_dir)

        # grid from patch config if present
        patch_cfg_path = os.path.join(patch_run_dir, "config_resolved.json")
        if os.path.isfile(patch_cfg_path):
            pcfg = load_json(patch_cfg_path)
            grid_rows = int(pcfg.get("GRID_ROWS") or pcfg.get("grid_rows") or 0)
            grid_cols = int(pcfg.get("GRID_COLS") or pcfg.get("grid_cols") or 0)
        else:
            grid_rows = grid_cols = 0

        self.samples: List[Dict[str, Any]] = []
        for image_id, glist in groups.items():
            if image_id not in W_by:
                raise KeyError(f"No g_i for image_id={image_id}")
            W = W_by[image_id]
            for r in glist:
                b = np.asarray(r["binary_patch_mask"], dtype=np.float32).reshape(-1)
                if grid_rows and grid_cols and b.size != grid_rows * grid_cols:
                    raise ValueError(
                        f"mask size {b.size} != {grid_rows}*{grid_cols} "
                        f"for {image_id}/{r['mask_id']}"
                    )
                tensor_path = r.get("masked_tensor_path") or r.get("tensor_path")
                if not tensor_path:
                    raise KeyError(f"missing masked_tensor_path for {image_id}")
                if not os.path.isabs(tensor_path):
                    tensor_path = os.path.join(patch_run_dir, tensor_path)
                self.samples.append(
                    {
                        "tensor_path": tensor_path,
                        "b": b,
                        "W": W,
                        "image_id": image_id,
                        "mask_id": r["mask_id"],
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = torch.load(s["tensor_path"], map_location="cpu")
        if not torch.is_tensor(x):
            raise TypeError(f"Expected tensor in {s['tensor_path']}")
        b = torch.from_numpy(s["b"])
        W = torch.from_numpy(s["W"])
        g_logits = b @ W  # [C_src]
        return x, g_logits, s["image_id"], s["mask_id"]


def kd_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    T = float(temperature)
    log_p = F.log_softmax(student_logits / T, dim=-1)
    q = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean") * (T * T)


def compute_distill_losses(
    student_logits: torch.Tensor,
    teacher_old: torch.Tensor,
    g_logits: torch.Tensor,
    mode: str,
    attack_idx_src: int,
    clean_idx_student: Optional[int],
    clean_idx_src: Optional[int],
    n_old: int,
    temperature: float,
    lambda_new: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_old = student_logits[:, :n_old]
    logits_new = student_logits[:, n_old:]
    l_old = kd_kl_loss(logits_old, teacher_old, temperature)

    if mode == "mse":
        target_new = g_logits[:, attack_idx_src]
        l_new = F.mse_loss(logits_new[:, 0], target_new)
    elif mode == "kl_clean_attack":
        if clean_idx_student is None or clean_idx_src is None:
            raise ValueError("Mode B requires clean class on both student and source")
        s_pair = torch.stack(
            [student_logits[:, clean_idx_student], logits_new[:, 0]], dim=-1
        )
        t_pair = torch.stack(
            [g_logits[:, clean_idx_src], g_logits[:, attack_idx_src]], dim=-1
        )
        l_new = kd_kl_loss(s_pair, t_pair, temperature)
    else:
        raise ValueError(f"Unknown new_class_distill mode={mode!r}")

    total = l_old + float(lambda_new) * l_new
    return total, l_old, l_new


def train_distill(
    student: ParallelInsertStudent,
    teacher: nn.Module,
    loader: DataLoader,
    cfg: Dict[str, Any],
    attack_idx_src: int,
    clean_idx_student: Optional[int],
    clean_idx_src: Optional[int],
    device: torch.device,
) -> List[Dict[str, Any]]:
    opt = torch.optim.Adam(list(student.trainable_parameters()), lr=float(cfg["lr"]))
    mode = cfg["new_class_distill"]
    history: List[Dict[str, Any]] = []
    teacher.eval()

    for epoch in range(int(cfg["epochs"])):
        student.train()
        student.mf.eval()
        student.bn1.eval()
        for m in (student.layer1, student.layer2, student.layer3, student.layer4):
            m.eval()

        sum_tot = sum_old = sum_new = 0.0
        sum_old_abs = 0.0
        n = 0
        for x, g_logits, _iid, _mid in loader:
            x = x.to(device)
            g_logits = g_logits.to(device)
            with torch.no_grad():
                t_old = teacher.predict(x, test_flag=1)
            s_logits = student.predict(x, detach=False)
            loss, l_old, l_new = compute_distill_losses(
                s_logits,
                t_old,
                g_logits,
                mode=mode,
                attack_idx_src=attack_idx_src,
                clean_idx_student=clean_idx_student,
                clean_idx_src=clean_idx_src,
                n_old=student.n_old,
                temperature=float(cfg["temperature"]),
                lambda_new=float(cfg["lambda_new"]),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.size(0)
            sum_tot += float(loss.item()) * bs
            sum_old += float(l_old.item()) * bs
            sum_new += float(l_new.item()) * bs
            with torch.no_grad():
                sum_old_abs += float(
                    (s_logits[:, : student.n_old] - t_old).abs().mean().item()
                ) * bs
            n += bs

        row = {
            "epoch": epoch,
            "loss": sum_tot / max(n, 1),
            "loss_old": sum_old / max(n, 1),
            "loss_new": sum_new / max(n, 1),
            "old_logit_mean_abs_vs_teacher": sum_old_abs / max(n, 1),
        }
        history.append(row)
        print(
            f"[distill] epoch {epoch + 1}/{cfg['epochs']} "
            f"loss={row['loss']:.6f} old={row['loss_old']:.6f} "
            f"new={row['loss_new']:.6f} "
            f"|Δold|={row['old_logit_mean_abs_vs_teacher']:.6f}"
        )
    return history


# ---------------------------------------------------------------------------
# Eval: temporary dataset + three checks
# ---------------------------------------------------------------------------


IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _safe_rmtree(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)


def generate_eval_dataset(
    dataset_root: str,
    target_node: str,
    eval_classes: Sequence[str],
    output_root: str,
    trigger_dir: str,
    blended_alpha_square: float,
    blended_alpha_hello_kitty: float,
    extracted_layer: str,
    seed: int,
) -> str:
    """Generate temporary ImageFolder data; return path to test/ split."""
    os.makedirs(output_root, exist_ok=True)
    results = ntdg.generate_all_nodes(
        dataset_root=dataset_root,
        node_specs={target_node: list(eval_classes)},
        trigger_dir=trigger_dir,
        output_root=output_root,
        blended_alpha_square=blended_alpha_square,
        blended_alpha_hello_kitty=blended_alpha_hello_kitty,
        extracted_layer=extracted_layer,
        seed=seed,
        extract_features=False,
    )
    test_root = os.path.join(results[target_node]["output_dir"], "test")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"eval test split missing: {test_root}")
    return test_root


def _per_class_f1_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    labels = list(range(len(class_names)))
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    # recall per class
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() == 0:
            rec = 0.0
        else:
            rec = float((y_pred[mask] == i).mean())
        out[name] = {"f1": float(f1s[i]), "recall": rec}
    return out


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_class: int) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(n_class)),
            average="macro",
            zero_division=0,
        )
    )


def _multiclass_auroc(
    y_true: np.ndarray, probs: np.ndarray, n_class: int
) -> Optional[float]:
    if n_class < 2:
        return None
    # need at least 2 classes present
    if len(np.unique(y_true)) < 2:
        return None
    try:
        if n_class == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(
            roc_auc_score(
                y_true,
                probs,
                multi_class="ovr",
                average="macro",
                labels=list(range(n_class)),
            )
        )
    except ValueError:
        return None


@torch.no_grad()
def collect_predictions(
    model_fn,
    loader: DataLoader,
    device: torch.device,
    n_logits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns y_true, y_pred, probs."""
    ys, preds, prob_list = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model_fn(x)
        if logits.shape[-1] != n_logits:
            # allow slicing by caller; here expect matching
            pass
        prob = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        ys.append(y.numpy())
        preds.append(pred.cpu().numpy())
        prob_list.append(prob.cpu().numpy())
    return (
        np.concatenate(ys, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(prob_list, axis=0),
    )


@torch.no_grad()
def evaluate_distill(
    student: ParallelInsertStudent,
    teacher: nn.Module,
    test_root: str,
    eval_class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    student.eval()
    teacher.eval()
    n_old = student.n_old
    n_all = n_old + student.n_new

    ds = datasets.ImageFolder(root=test_root, transform=IMAGENET_TRANSFORM)
    # ImageFolder sorts folder names; folders are 00_*, 01_* so order matches eval_classes
    folder_names = [n for n, _ in sorted(ds.class_to_idx.items(), key=lambda kv: kv[1])]
    expected = [ntdg.class_folder_name(i, c) for i, c in enumerate(eval_class_names)]
    if folder_names != expected:
        print(
            f"[WARN] ImageFolder class order {folder_names} != expected {expected}; "
            "remapping labels"
        )
        # build remap from folder idx → eval idx
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

    # Collect student full / detach / teacher
    y_true_list, s_pred, s_prob = [], [], []
    t_pred, t_logits_list = [], []
    d_pred, d_logits_list = [], []
    s_old_logits_list = []

    for x, y_raw in loader:
        y = torch.tensor([remap[int(v)] for v in y_raw.tolist()], dtype=torch.long)
        x = x.to(device)
        s_logits = student.predict(x, detach=False)
        d_logits = student.predict(x, detach=True)
        t_logits = teacher.predict(x, test_flag=1)

        y_true_list.append(y.numpy())
        s_pred.append(s_logits.argmax(dim=-1).cpu().numpy())
        s_prob.append(F.softmax(s_logits, dim=-1).cpu().numpy())
        s_old_logits_list.append(s_logits[:, :n_old].cpu().numpy())
        t_pred.append(t_logits.argmax(dim=-1).cpu().numpy())
        t_logits_list.append(t_logits.cpu().numpy())
        d_pred.append(d_logits.argmax(dim=-1).cpu().numpy())
        d_logits_list.append(d_logits.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    s_pred_a = np.concatenate(s_pred)
    s_prob_a = np.concatenate(s_prob)
    t_pred_a = np.concatenate(t_pred)
    t_logits_a = np.concatenate(t_logits_list)
    d_pred_a = np.concatenate(d_pred)
    d_logits_a = np.concatenate(d_logits_list)
    s_old_a = np.concatenate(s_old_logits_list)

    old_mask = y_true < n_old
    new_mask = y_true == n_old

    # --- New-class performance ---
    new_metrics: Dict[str, Any] = {"n": int(new_mask.sum())}
    if new_mask.any():
        y_n = y_true[new_mask]
        p_n = s_pred_a[new_mask]
        new_metrics["accuracy"] = float((p_n == y_n).mean())
        new_metrics["recall"] = float((p_n == n_old).mean())
        new_metrics["f1"] = float(
            f1_score(y_n, p_n, labels=[n_old], average="macro", zero_division=0)
        )
    else:
        new_metrics["accuracy"] = None
        new_metrics["recall"] = None
        new_metrics["f1"] = None

    # --- Old-class retention ---
    old_metrics: Dict[str, Any] = {"n": int(old_mask.sum())}
    if old_mask.any():
        y_o = y_true[old_mask]
        # student pred among all classes
        p_s = s_pred_a[old_mask]
        p_t = t_pred_a[old_mask]
        old_metrics["student_accuracy"] = float((p_s == y_o).mean())
        old_metrics["teacher_accuracy"] = float((p_t == y_o).mean())
        old_metrics["student_macro_f1"] = _macro_f1(y_o, p_s, n_old)
        old_metrics["teacher_macro_f1"] = _macro_f1(y_o, p_t, n_old)
        # restrict student pred to old dims for fair old-class F1
        p_s_oldspace = s_old_a[old_mask].argmax(axis=-1)
        old_metrics["student_oldspace_accuracy"] = float((p_s_oldspace == y_o).mean())
        old_metrics["student_oldspace_macro_f1"] = _macro_f1(y_o, p_s_oldspace, n_old)
        old_metrics["per_class"] = _per_class_f1_recall(
            y_o, p_s_oldspace, list(eval_class_names[:n_old])
        )
        old_metrics["mean_abs_logit_vs_teacher"] = float(
            np.abs(s_old_a[old_mask] - t_logits_a[old_mask]).mean()
        )
    else:
        old_metrics["student_accuracy"] = None

    # --- Detach recovery (all inputs; detach outputs C_old == teacher) ---
    detach_metrics: Dict[str, Any] = {
        "max_abs_logit_diff": float(np.abs(d_logits_a - t_logits_a).max()),
        "mean_abs_logit_diff": float(np.abs(d_logits_a - t_logits_a).mean()),
        "pred_match_rate": float((d_pred_a == t_pred_a).mean()),
    }
    detach_metrics["ok"] = bool(detach_metrics["max_abs_logit_diff"] < 1e-3)

    # Overall + attack-setting metrics on full student
    overall = {
        "accuracy": float((s_pred_a == y_true).mean()),
        "macro_f1": _macro_f1(y_true, s_pred_a, n_all),
        "auroc": _multiclass_auroc(y_true, s_prob_a, n_all),
        "per_class": _per_class_f1_recall(y_true, s_pred_a, list(eval_class_names)),
    }
    attack_names = [c for c in eval_class_names if c != "clean"]
    attack_per_class = {
        name: overall["per_class"][name]
        for name in attack_names
        if name in overall["per_class"]
    }

    return {
        "new_class_performance": new_metrics,
        "old_class_retention": old_metrics,
        "detach_recovery": detach_metrics,
        "overall": overall,
        "attack_class_metrics": attack_per_class,
        "eval_class_names": list(eval_class_names),
        "n_samples": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def make_output_dir(cfg: Dict[str, Any], target_node: str, attack_class: str) -> str:
    if cfg.get("output_dir"):
        out = os.path.abspath(cfg["output_dir"])
    else:
        root = os.path.abspath(cfg.get("output_root") or _DEFAULT_OUTPUT_ROOT)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_atk = str(attack_class).replace("/", "_").replace(" ", "_")
        out = os.path.join(root, f"{stamp}_{target_node}_{safe_atk}")
    os.makedirs(out, exist_ok=True)
    return out


def run_distill(cfg: Dict[str, Any]) -> str:
    if not cfg.get("localization_run_dir"):
        raise ValueError("localization_run_dir is required")
    if not cfg.get("target_node"):
        raise ValueError("target_node is required")

    set_seed(int(cfg.get("seed", 42)))
    device_s = cfg.get("device") or "cpu"
    if str(device_s).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available; falling back to cpu")
        device_s = "cpu"
    device = torch.device(device_s)

    loc_info = resolve_localization(cfg["localization_run_dir"])
    target_info = resolve_target(cfg)

    extracted = loc_info["extracted_layer"]
    if (
        target_info.get("extracted_layer_report")
        and target_info["extracted_layer_report"] != extracted
    ):
        print(
            f"[WARN] target report extracted_layer="
            f"{target_info['extracted_layer_report']!r} != localization R="
            f"{extracted!r}; inserting at localization R"
        )

    target_classes = list(target_info["class_names"])
    new_class = loc_info["attack_class"]
    if new_class in target_classes:
        raise ValueError(
            f"new attack class {new_class!r} already in target classes {target_classes}"
        )
    eval_classes = target_classes + [new_class]
    n_old = len(target_classes)

    clean_idx_student = target_classes.index("clean") if "clean" in target_classes else None
    src_classes = loc_info["source_class_names"]
    clean_idx_src = src_classes.index("clean") if "clean" in src_classes else None
    mode = cfg.get("new_class_distill") or "mse"
    if mode == "kl_clean_attack":
        if clean_idx_student is None:
            raise ValueError("Mode B requires target class named 'clean'")
        if clean_idx_src is None:
            raise ValueError("Mode B requires source/localization class named 'clean'")

    out_dir = make_output_dir(cfg, target_info["target_node"], new_class)
    snapshot = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "resolved": {
            "localization_run_dir": loc_info["run_dir"],
            "patch_run_dir": loc_info["patch_run_dir"],
            "extracted_layer": extracted,
            "attack_class": new_class,
            "attack_idx_src": loc_info["attack_idx_src"],
            "source_class_names": src_classes,
            "target_node": target_info["target_node"],
            "target_checkpoint": target_info["checkpoint_path"],
            "target_classes": target_classes,
            "eval_classes": eval_classes,
            "target_report_dir": target_info["report_dir"],
            "arch_label": target_info["arch_label"],
            "dataset_root": target_info["dataset_root"],
            "c_f": loc_info["c_f"],
            "new_class_distill": mode,
        },
    }
    save_json(os.path.join(out_dir, "config.json"), snapshot)

    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] target={target_info['target_node']} classes={target_classes}")
    print(f"[INFO] new_class={new_class} R={extracted} mode={mode}")
    print(f"[INFO] arch={target_info['arch_label']}")

    student, teacher = build_student_and_teacher(
        target_ckpt=target_info["checkpoint_path"],
        n_old=n_old,
        extracted_layer=extracted,
        loc_info=loc_info,
        device=device,
    )

    distill_ds = DistillPatchDataset(
        patch_run_dir=loc_info["patch_run_dir"],
        loc_run_dir=loc_info["run_dir"],
        include_reference=bool(cfg.get("include_reference", True)),
    )
    distill_loader = DataLoader(
        distill_ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=0,
    )
    sanity_loader = DataLoader(
        distill_ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=0,
    )

    print("[INFO] running epoch-0 sanity check...")
    sanity = run_sanity_check(
        student,
        teacher,
        sanity_loader,
        device=device,
        atol=float(cfg.get("sanity_atol", 1e-3)),
        rtol=float(cfg.get("sanity_rtol", 1e-3)),
        max_batches=int(cfg.get("sanity_max_batches", 4)),
    )
    save_json(os.path.join(out_dir, "sanity_check.json"), sanity)
    print(
        f"[INFO] sanity ok max_abs={sanity['max_abs_diff']:.6g} "
        f"mean_abs={sanity['mean_abs_diff']:.6g}"
    )

    history = train_distill(
        student,
        teacher,
        distill_loader,
        cfg=cfg,
        attack_idx_src=loc_info["attack_idx_src"],
        clean_idx_student=clean_idx_student,
        clean_idx_src=clean_idx_src,
        device=device,
    )
    _write_csv(os.path.join(out_dir, "train_history.csv"), history)

    # Save student (trainable + structure meta)
    torch.save(
        {
            "extracted_layer": extracted,
            "n_old": n_old,
            "n_new": 1,
            "eval_classes": eval_classes,
            "target_classes": target_classes,
            "new_class": new_class,
            "adapter": student.adapter.state_dict(),
            "classifier_layer": student.classifier_layer.state_dict(),
            "mf": student.mf.state_dict(),
            "target_base": {
                k: v.cpu()
                for k, v in teacher.base_network.state_dict().items()
            },
            "orig_classifier": teacher.classifier_layer.state_dict(),
            "compact_meta": loc_info["compact_meta"],
        },
        os.path.join(out_dir, "student_state_dict.pt"),
    )

    # Temporary eval data → metrics → cleanup
    eval_gen_root = os.path.join(out_dir, "eval_generated")
    metrics: Dict[str, Any] = {}
    try:
        print("[INFO] generating temporary eval dataset (old+new classes)...")
        test_root = generate_eval_dataset(
            dataset_root=target_info["dataset_root"],
            target_node=target_info["target_node"],
            eval_classes=eval_classes,
            output_root=eval_gen_root,
            trigger_dir=target_info["trigger_dir"],
            blended_alpha_square=target_info["blended_alpha_square"],
            blended_alpha_hello_kitty=target_info["blended_alpha_hello_kitty"],
            extracted_layer=extracted,
            seed=int(cfg.get("eval_seed", 0)),
        )
        print(f"[INFO] eval test_root={test_root}")
        metrics = evaluate_distill(
            student,
            teacher,
            test_root=test_root,
            eval_class_names=eval_classes,
            device=device,
            batch_size=int(cfg.get("eval_batch_size", 32)),
        )
        save_json(os.path.join(out_dir, "metrics.json"), metrics)
        print(
            "[INFO] metrics: "
            f"new_recall={metrics['new_class_performance'].get('recall')} "
            f"old_student_acc={metrics['old_class_retention'].get('student_oldspace_accuracy')} "
            f"detach_ok={metrics['detach_recovery'].get('ok')} "
            f"macro_f1={metrics['overall'].get('macro_f1')} "
            f"auroc={metrics['overall'].get('auroc')}"
        )
    finally:
        print(f"[INFO] cleaning temporary eval data under {eval_gen_root}")
        _safe_rmtree(eval_gen_root)

    print(f"[INFO] distill done -> {out_dir}")
    return out_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parallel-insert compact subnet + distill adapter/expanded FC."
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--localization_run_dir", type=str, default=None)
    p.add_argument("--target_node", type=str, default=None)
    p.add_argument("--target_training_report_dir", type=str, default=None)
    p.add_argument("--target_checkpoint_path", type=str, default=None)
    p.add_argument("--training_report_root", type=str, default=None)
    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--trigger_dir", type=str, default=None)
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument(
        "--new_class_distill",
        type=str,
        choices=["mse", "kl_clean_attack"],
        default=None,
    )
    p.add_argument("--lambda_new", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    overrides = {
        "localization_run_dir": args.localization_run_dir,
        "target_node": args.target_node,
        "target_training_report_dir": args.target_training_report_dir,
        "target_checkpoint_path": args.target_checkpoint_path,
        "training_report_root": args.training_report_root,
        "dataset_root": args.dataset_root,
        "trigger_dir": args.trigger_dir,
        "output_root": args.output_root,
        "output_dir": args.output_dir,
        "new_class_distill": args.new_class_distill,
        "lambda_new": args.lambda_new,
        "temperature": args.temperature,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
    }
    cfg = load_distill_config(args.config, overrides=overrides)
    run_distill(cfg)


if __name__ == "__main__":
    main()
