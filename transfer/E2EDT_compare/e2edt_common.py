#!/usr/bin/env python3
"""Shared helpers for source_preparation / E2EDT Stage2 / param_sweep."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

_COMPARE_DIR = os.path.dirname(os.path.abspath(__file__))
_TRANSFER_DIR = os.path.abspath(os.path.join(_COMPARE_DIR, ".."))
_E2EDT_ROOT = os.path.abspath(os.path.join(_TRANSFER_DIR, ".."))
_MATCHING_DIR = os.path.join(_E2EDT_ROOT, "matching")
_DEFAULT_TRAINING_NOTE = os.path.join(
    _E2EDT_ROOT, "node_model_training", "training_report"
)
_DEFAULT_DISTILL_ROOT = os.path.join(_COMPARE_DIR, "compare_result")
_DEFAULT_SOURCE_PREP_ROOT = os.path.join(_TRANSFER_DIR, "source_prep")

# Back-compat aliases used by callers
_SCRIPT_DIR = _COMPARE_DIR

if _COMPARE_DIR not in sys.path:
    sys.path.insert(0, _COMPARE_DIR)
if _TRANSFER_DIR not in sys.path:
    sys.path.insert(0, _TRANSFER_DIR)
if _MATCHING_DIR not in sys.path:
    sys.path.insert(0, _MATCHING_DIR)
for _sub in ("patch_data_generate", "localization", "KD"):
    _p = os.path.join(_TRANSFER_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import patch_data_generate as patch_gen  # noqa: E402
from diffvector_data_generate import load_latest_training_plan  # noqa: E402

REFERENCE_MASK_ID = "reference_all_ones"
KEEP_N_MASKS = 9
KEEP_TOTAL = 1 + KEEP_N_MASKS
MANIFEST_NAME = "source_prep_manifest.json"

_DAY_SEQ_RE = re.compile(r"^(\d{4})_(\d+)_")


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_training_reports(training_note_root: str) -> List[Tuple[str, str]]:
    """Return [(created_at, run_dir), ...] ascending by created_at."""
    root = os.path.abspath(training_note_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"training_report root not found: {root}")
    candidates: List[Tuple[str, str]] = []
    for name in os.listdir(root):
        run_dir = os.path.join(root, name)
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.isdir(run_dir) or not os.path.isfile(cfg_path):
            continue
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        created = cfg.get("created_at") or ""
        if not created:
            created = datetime.fromtimestamp(os.path.getmtime(run_dir)).isoformat(
                timespec="seconds"
            )
        candidates.append((created, run_dir))
    candidates.sort(key=lambda x: x[0])
    return candidates


def load_checkpoint_index(run_dir: str) -> Dict[str, Any]:
    index_path = os.path.join(os.path.abspath(run_dir), "checkpoints", "index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"checkpoints/index.json not found under {run_dir}")
    return load_json(index_path)


def attack_classes_for_node(node_specs: Dict[str, List[str]], node: str) -> List[str]:
    if node not in node_specs:
        raise KeyError(f"node {node!r} not in node_specs")
    return [c for c in node_specs[node] if c != "clean"]


def prompt_choice(prompt: str, options: Sequence[str], default_idx: int = 0) -> int:
    if not options:
        raise ValueError("No options to choose from")
    print(prompt)
    for i, label in enumerate(options, start=1):
        mark = " (default)" if (i - 1) == default_idx else ""
        print(f"  [{i}] {label}{mark}")
    while True:
        raw = input(f"Select [1-{len(options)}] (Enter={default_idx + 1}): ").strip()
        if raw == "":
            return default_idx
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        print(f"[WARN] invalid choice: {raw!r}")


def _resolve_run_dir(
    training_note_root: str, run_dir: Optional[str]
) -> Tuple[str, Dict[str, Any], Dict[str, Any], List[str]]:
    reports = list_training_reports(training_note_root)
    if not reports:
        raise FileNotFoundError(f"No training runs under {training_note_root}")

    if run_dir:
        chosen_dir = os.path.abspath(run_dir)
        if not os.path.isfile(os.path.join(chosen_dir, "config.json")):
            raise FileNotFoundError(f"Invalid --run_dir (no config.json): {chosen_dir}")
    else:
        default_idx = len(reports) - 1
        labels = [f"{os.path.basename(d)}  (created_at={c})" for c, d in reports]
        idx = prompt_choice(
            "Select training_report scheme (default = latest):",
            labels,
            default_idx=default_idx,
        )
        chosen_dir = reports[idx][1]

    plan = load_latest_training_plan(training_note_root, run_dir=chosen_dir)
    index = load_checkpoint_index(chosen_dir)
    nodes_with_ckpt = sorted(index.get("nodes", {}).keys())
    node_specs: Dict[str, List[str]] = plan["node_specs"]
    available = [n for n in nodes_with_ckpt if n in node_specs]
    if len(available) < 2:
        raise RuntimeError(
            f"Need at least 2 nodes with checkpoints under {chosen_dir}; "
            f"got {available}"
        )
    return chosen_dir, plan, index, available


def _pick_node(
    label: str,
    current: Optional[str],
    available: Sequence[str],
    node_specs: Dict[str, List[str]],
    exclude: Optional[str],
) -> str:
    opts = [n for n in available if n != exclude]
    if current:
        if current not in available:
            raise KeyError(f"{label} {current!r} not in {list(available)}")
        if exclude and current == exclude:
            raise ValueError(f"{label} must differ from {exclude}")
        return current
    labels = [f"{n}: classes={node_specs[n]}" for n in opts]
    i = prompt_choice(f"Select {label}:", labels, default_idx=0)
    return opts[i]


def resolve_source_prep_selections(
    training_note_root: str,
    run_dir: Optional[str],
    source_node: Optional[str],
    attack_class: Optional[str],
) -> Dict[str, Any]:
    """Stage1: scheme + source + attack (no target)."""
    chosen_dir, plan, index, available = _resolve_run_dir(training_note_root, run_dir)
    node_specs: Dict[str, List[str]] = plan["node_specs"]

    src = _pick_node("source_node", source_node, available, node_specs, exclude=None)
    attacks = attack_classes_for_node(node_specs, src)
    if not attacks:
        raise RuntimeError(f"source {src} has no attack classes (only clean?)")

    if attack_class:
        _, atk_name, _ = patch_gen.resolve_attack_class_index(
            node_specs[src], attack_class
        )
    else:
        i = prompt_choice(
            f"Select source attack class for {src}:",
            attacks,
            default_idx=0,
        )
        atk_name = attacks[i]

    print(
        f"[INFO] scheme={plan['run_name']}\n"
        f"[INFO] source={src} attack={atk_name} classes={node_specs[src]}"
    )
    return {
        "run_dir": chosen_dir,
        "run_name": plan["run_name"],
        "plan": plan,
        "index": index,
        "source_node": src,
        "attack_class": atk_name,
        "source_classes": list(node_specs[src]),
        "available_nodes": list(available),
        "node_specs": node_specs,
    }


def resolve_target_selection(
    training_note_root: str,
    run_dir: str,
    target_node: Optional[str],
    source_node: str,
    attack_class: str,
) -> Dict[str, Any]:
    """Stage2: pick target given fixed training report / source / attack."""
    chosen_dir, plan, index, available = _resolve_run_dir(training_note_root, run_dir)
    node_specs: Dict[str, List[str]] = plan["node_specs"]
    tgt = _pick_node(
        "target_node", target_node, available, node_specs, exclude=source_node
    )
    if tgt == source_node:
        raise ValueError("target_node and source_node must differ")
    target_classes = list(index["nodes"][tgt]["classes"])
    if attack_class in target_classes:
        raise ValueError(
            f"attack class {attack_class!r} already in target {tgt} classes "
            f"{target_classes}; pick another attack or target"
        )
    print(
        f"[INFO] scheme={plan['run_name']}\n"
        f"[INFO] target={tgt} classes={target_classes}\n"
        f"[INFO] source={source_node} attack={attack_class}"
    )
    return {
        "run_dir": chosen_dir,
        "run_name": plan["run_name"],
        "plan": plan,
        "index": index,
        "target_node": tgt,
        "target_classes": target_classes,
        "node_specs": node_specs,
    }


def resolve_selections(
    training_note_root: str,
    run_dir: Optional[str],
    target_node: Optional[str],
    source_node: Optional[str],
    attack_class: Optional[str],
) -> Dict[str, Any]:
    """Legacy full selection (scheme + target + source + attack)."""
    src_sel = resolve_source_prep_selections(
        training_note_root, run_dir, source_node, attack_class
    )
    tgt_sel = resolve_target_selection(
        training_note_root,
        src_sel["run_dir"],
        target_node,
        src_sel["source_node"],
        src_sel["attack_class"],
    )
    return {
        "run_dir": src_sel["run_dir"],
        "run_name": src_sel["run_name"],
        "plan": src_sel["plan"],
        "target_node": tgt_sel["target_node"],
        "source_node": src_sel["source_node"],
        "attack_class": src_sel["attack_class"],
        "target_classes": tgt_sel["target_classes"],
        "source_classes": src_sel["source_classes"],
    }


def next_day_seq(output_root: str, mmdd: str) -> int:
    root = os.path.abspath(output_root)
    if not os.path.isdir(root):
        return 1
    max_seq = 0
    prefix = f"{mmdd}_"
    for name in os.listdir(root):
        if not name.startswith(prefix):
            continue
        m = _DAY_SEQ_RE.match(name)
        if not m or m.group(1) != mmdd:
            continue
        try:
            max_seq = max(max_seq, int(m.group(2)))
        except ValueError:
            continue
    return max_seq + 1


def make_dated_output_dir(
    output_root: str,
    name_parts: Sequence[str],
) -> str:
    mmdd = datetime.now().strftime("%m%d")
    seq = next_day_seq(output_root, mmdd)
    safe_parts = [
        str(p).replace("/", "_").replace(" ", "_") for p in name_parts if p is not None
    ]
    folder = "_".join([mmdd, str(seq)] + safe_parts)
    out = os.path.join(os.path.abspath(output_root), folder)
    os.makedirs(out, exist_ok=True)
    return out


def make_e2edt_output_dir(
    output_root: str,
    target_node: str,
    source_node: str,
    attack_class: str,
    run_name: str,
) -> str:
    return make_dated_output_dir(
        output_root, [target_node, source_node, attack_class, run_name]
    )


def make_source_prep_output_dir(
    output_root: str,
    source_node: str,
    attack_class: str,
    run_name: str,
) -> str:
    return make_dated_output_dir(
        output_root, [source_node, attack_class, run_name]
    )


def _safe_image_id_dirname(image_id: str) -> str:
    return image_id.replace("/", "__").replace("\\", "__")


def count_sample_ratio_source_images(patch_run_dir: str) -> Dict[str, Any]:
    """Count original images selected via SAMPLE_RATIO (not patch/mask count)."""
    patch_run_dir = os.path.abspath(patch_run_dir)
    selected_path = os.path.join(patch_run_dir, "selected_samples.json")
    cfg_path = os.path.join(patch_run_dir, "config_resolved.json")
    sample_ratio = None
    if os.path.isfile(cfg_path):
        cfg = load_json(cfg_path)
        sample_ratio = cfg.get("SAMPLE_RATIO")

    image_ids: List[str] = []
    if os.path.isfile(selected_path):
        sel = load_json(selected_path)
        for s in sel.get("selected_samples") or []:
            if s.get("image_id"):
                image_ids.append(str(s["image_id"]))
        if sample_ratio is None:
            sample_ratio = sel.get("SAMPLE_RATIO")

    unique_ids = sorted(set(image_ids))
    return {
        "sample_ratio": sample_ratio,
        "n_source_images": len(unique_ids),
        "image_ids": unique_ids,
        "selected_samples_path": selected_path if os.path.isfile(selected_path) else None,
    }


def count_imagefolder_images(root: str) -> Dict[str, Any]:
    """Count labeled images under an ImageFolder root (train/ or similar)."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    per_class: Dict[str, int] = {}
    total = 0
    for name in sorted(os.listdir(root)):
        cdir = os.path.join(root, name)
        if not os.path.isdir(cdir):
            continue
        n = sum(
            1
            for f in os.listdir(cdir)
            if os.path.isfile(os.path.join(cdir, f))
            and not f.startswith(".")
        )
        per_class[name] = n
        total += n
    return {"total": total, "per_class": per_class, "root": root}


def copy_confirm_patch_samples(
    patch_run_dir: str,
    dest_root: str,
    n_masks: int = KEEP_N_MASKS,
) -> Dict[str, Any]:
    """
    Copy 1 reference_all_ones + n_masks other mask previews from a single
    image_id into dest_root/confirm_patch_samples/<safe_image_id>/.
    """
    patch_run_dir = os.path.abspath(patch_run_dir)
    previews_root = os.path.join(patch_run_dir, "previews")
    selected_path = os.path.join(patch_run_dir, "selected_samples.json")
    info: Dict[str, Any] = {
        "image_id": None,
        "copied": [],
        "warnings": [],
    }

    if not os.path.isdir(previews_root):
        info["warnings"].append(f"previews missing: {previews_root}")
        print(f"[WARN] {info['warnings'][-1]}")
        return info

    image_ids: List[str] = []
    if os.path.isfile(selected_path):
        sel = load_json(selected_path)
        for s in sel.get("selected_samples") or []:
            if s.get("image_id"):
                image_ids.append(s["image_id"])

    if not image_ids:
        for dirpath, _dirnames, filenames in os.walk(previews_root):
            if f"{REFERENCE_MASK_ID}.png" in filenames:
                rel = os.path.relpath(dirpath, previews_root)
                if rel != ".":
                    image_ids.append(rel.replace(os.sep, "/"))

    chosen_id: Optional[str] = None
    chosen_dir: Optional[str] = None
    for iid in image_ids:
        pdir = os.path.join(previews_root, iid.replace("/", os.sep))
        ref = os.path.join(pdir, f"{REFERENCE_MASK_ID}.png")
        if os.path.isfile(ref):
            chosen_id = iid
            chosen_dir = pdir
            break

    if not chosen_id or not chosen_dir:
        info["warnings"].append("no image_id with reference_all_ones.png found")
        print(f"[WARN] {info['warnings'][-1]}")
        return info

    info["image_id"] = chosen_id
    out_dir = os.path.join(
        os.path.abspath(dest_root),
        "confirm_patch_samples",
        _safe_image_id_dirname(chosen_id),
    )
    os.makedirs(out_dir, exist_ok=True)

    ref_src = os.path.join(chosen_dir, f"{REFERENCE_MASK_ID}.png")
    ref_dst = os.path.join(out_dir, f"{REFERENCE_MASK_ID}.png")
    shutil.copy2(ref_src, ref_dst)
    info["copied"].append(ref_dst)

    mask_files = sorted(
        f
        for f in os.listdir(chosen_dir)
        if f.startswith("mask_") and f.endswith(".png")
    )
    for name in mask_files[:n_masks]:
        src = os.path.join(chosen_dir, name)
        dst = os.path.join(out_dir, name)
        shutil.copy2(src, dst)
        info["copied"].append(dst)

    if len(info["copied"]) < KEEP_TOTAL:
        msg = (
            f"only copied {len(info['copied'])}/{KEEP_TOTAL} previews "
            f"for image_id={chosen_id}"
        )
        info["warnings"].append(msg)
        print(f"[WARN] {msg}")
    else:
        print(
            f"[INFO] confirm_patch_samples: {len(info['copied'])} PNGs "
            f"from image_id={chosen_id} -> {out_dir}"
        )

    meta_path = os.path.join(
        os.path.abspath(dest_root), "confirm_patch_samples", "meta.json"
    )
    save_json(meta_path, info)
    return info


def load_source_prep_manifest(prep_dir: str) -> Dict[str, Any]:
    path = os.path.join(os.path.abspath(prep_dir), MANIFEST_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {MANIFEST_NAME} under {prep_dir}")
    return load_json(path)
