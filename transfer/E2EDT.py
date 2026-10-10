#!/usr/bin/env python3
"""End-to-end transfer orchestration: patch → localization → distill.

Interactive (or CLI) selection of a training_report scheme, target/source
nodes, and source attack class; then runs the three transfer scripts in-process,
names the distill result as
  {MMDD}_{day_seq}_{target}_{source}_{attack}_{full_run_name}/
keeps 10 preview PNGs (1 reference_all_ones + 9 masks from one image_id),
and cleans up this run's patch_generated heavy artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_E2EDT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_MATCHING_DIR = os.path.join(_E2EDT_ROOT, "matching")
_DEFAULT_TRAINING_NOTE = os.path.join(
    _E2EDT_ROOT, "node_model_training", "training_report"
)
_DEFAULT_DISTILL_ROOT = os.path.join(_SCRIPT_DIR, "distill_result")

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _MATCHING_DIR not in sys.path:
    sys.path.insert(0, _MATCHING_DIR)

import patch_data_generate as patch_gen  # noqa: E402
import specific_localziation as loc  # noqa: E402
import knowledgedistill as distill  # noqa: E402
from diffvector_data_generate import load_latest_training_plan  # noqa: E402

REFERENCE_MASK_ID = "reference_all_ones"
KEEP_N_MASKS = 9
KEEP_TOTAL = 1 + KEEP_N_MASKS  # 10


# ---------------------------------------------------------------------------
# Training report listing / interactive select
# ---------------------------------------------------------------------------


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
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def attack_classes_for_node(node_specs: Dict[str, List[str]], node: str) -> List[str]:
    if node not in node_specs:
        raise KeyError(f"node {node!r} not in node_specs")
    return [c for c in node_specs[node] if c != "clean"]


def _prompt_choice(prompt: str, options: Sequence[str], default_idx: int = 0) -> int:
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


def resolve_selections(
    training_note_root: str,
    run_dir: Optional[str],
    target_node: Optional[str],
    source_node: Optional[str],
    attack_class: Optional[str],
) -> Dict[str, Any]:
    reports = list_training_reports(training_note_root)
    if not reports:
        raise FileNotFoundError(f"No training runs under {training_note_root}")

    if run_dir:
        chosen_dir = os.path.abspath(run_dir)
        if not os.path.isfile(os.path.join(chosen_dir, "config.json")):
            raise FileNotFoundError(f"Invalid --run_dir (no config.json): {chosen_dir}")
    else:
        # Newest last in reports; Enter accepts latest
        default_idx = len(reports) - 1
        labels = [
            f"{os.path.basename(d)}  (created_at={c})" for c, d in reports
        ]
        idx = _prompt_choice(
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

    def _pick_node(label: str, current: Optional[str], exclude: Optional[str]) -> str:
        opts = [n for n in available if n != exclude]
        if current:
            if current not in available:
                raise KeyError(f"{label} {current!r} not in {available}")
            if exclude and current == exclude:
                raise ValueError(f"{label} must differ from {exclude}")
            return current
        labels = [
            f"{n}: classes={node_specs[n]}" for n in opts
        ]
        i = _prompt_choice(f"Select {label}:", labels, default_idx=0)
        return opts[i]

    tgt = _pick_node("target_node", target_node, exclude=None)
    src = _pick_node("source_node", source_node, exclude=tgt)
    if tgt == src:
        raise ValueError("target_node and source_node must differ")

    attacks = attack_classes_for_node(node_specs, src)
    if not attacks:
        raise RuntimeError(f"source {src} has no attack classes (only clean?)")

    if attack_class:
        _, atk_name, _ = patch_gen.resolve_attack_class_index(
            node_specs[src], attack_class
        )
    else:
        i = _prompt_choice(
            f"Select source attack class for {src}:",
            attacks,
            default_idx=0,
        )
        atk_name = attacks[i]

    target_classes = list(index["nodes"][tgt]["classes"])
    if atk_name in target_classes:
        raise ValueError(
            f"attack class {atk_name!r} already in target {tgt} classes "
            f"{target_classes}; pick another attack or target"
        )

    print(
        f"[INFO] scheme={plan['run_name']}\n"
        f"[INFO] target={tgt} classes={target_classes}\n"
        f"[INFO] source={src} attack={atk_name} "
        f"classes={node_specs[src]}"
    )
    return {
        "run_dir": chosen_dir,
        "run_name": plan["run_name"],
        "plan": plan,
        "target_node": tgt,
        "source_node": src,
        "attack_class": atk_name,
        "target_classes": target_classes,
        "source_classes": list(node_specs[src]),
    }


# ---------------------------------------------------------------------------
# Distill result naming: {MMDD}_{seq}_{target}_{source}_{attack}_{run_name}
# ---------------------------------------------------------------------------


_DAY_SEQ_RE = re.compile(r"^(\d{4})_(\d+)_")


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


def make_e2edt_output_dir(
    output_root: str,
    target_node: str,
    source_node: str,
    attack_class: str,
    run_name: str,
) -> str:
    mmdd = datetime.now().strftime("%m%d")
    seq = next_day_seq(output_root, mmdd)
    safe_atk = str(attack_class).replace("/", "_").replace(" ", "_")
    folder = f"{mmdd}_{seq}_{target_node}_{source_node}_{safe_atk}_{run_name}"
    out = os.path.join(os.path.abspath(output_root), folder)
    os.makedirs(out, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Keep 10 previews from one image_id, then cleanup patch
# ---------------------------------------------------------------------------


def _safe_image_id_dirname(image_id: str) -> str:
    return image_id.replace("/", "__").replace("\\", "__")


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
        with open(selected_path, "r", encoding="utf-8") as f:
            sel = json.load(f)
        for s in sel.get("selected_samples") or []:
            if s.get("image_id"):
                image_ids.append(s["image_id"])

    if not image_ids:
        # Fallback: walk previews tree for dirs that contain reference
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
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_e2edt(cfg: Dict[str, Any]) -> str:
    training_note_root = os.path.abspath(
        cfg.get("training_note_root") or _DEFAULT_TRAINING_NOTE
    )
    sel = resolve_selections(
        training_note_root=training_note_root,
        run_dir=cfg.get("run_dir"),
        target_node=cfg.get("target_node"),
        source_node=cfg.get("source_node"),
        attack_class=cfg.get("attack_class"),
    )

    device = cfg.get("device")
    distill_root = os.path.abspath(cfg.get("output_root") or _DEFAULT_DISTILL_ROOT)
    os.makedirs(distill_root, exist_ok=True)

    out_dir = make_e2edt_output_dir(
        output_root=distill_root,
        target_node=sel["target_node"],
        source_node=sel["source_node"],
        attack_class=sel["attack_class"],
        run_name=sel["run_name"],
    )
    print(f"[INFO] distill output_dir={out_dir}")

    # --- 1) Patch data (force previews) ---
    patch_overrides: Dict[str, Any] = {"WRITE_PREVIEWS": True}
    patch_cfg = patch_gen.load_patch_config(
        cfg.get("patch_config"), overrides=patch_overrides
    )
    print("[INFO] === patch_data_generate ===")
    patch_result = patch_gen.generate_patch_dataset(
        node_name=sel["source_node"],
        attack_class=sel["attack_class"],
        patch_cfg=patch_cfg,
        training_note_root=training_note_root,
        run_dir=sel["run_dir"],
        output_root=cfg.get("patch_output_root") or patch_gen._DEFAULT_OUTPUT_ROOT,
        dry_run=False,
    )
    patch_run_dir = patch_result["output_dir"]

    # --- 2) Localization ---
    print("[INFO] === specific_localization ===")
    loc_overrides: Dict[str, Any] = {
        "patch_run_dir": patch_run_dir,
        "training_report_dir": sel["run_dir"],
    }
    if device:
        loc_overrides["device"] = device
    loc_cfg = loc.load_localization_config(
        cfg.get("localization_config"), overrides=loc_overrides
    )
    localization_run_dir = loc.run_localization(loc_cfg)

    # --- 3) Distill ---
    print("[INFO] === knowledgedistill ===")
    distill_overrides: Dict[str, Any] = {
        "localization_run_dir": localization_run_dir,
        "target_node": sel["target_node"],
        "target_training_report_dir": sel["run_dir"],
        "training_report_root": training_note_root,
        "output_dir": out_dir,
    }
    if device:
        distill_overrides["device"] = device
    distill_cfg = distill.load_distill_config(
        cfg.get("distill_config"), overrides=distill_overrides
    )
    final_dir = distill.run_distill(distill_cfg)

    # --- 4) Keep 10 previews, cleanup patch ---
    print("[INFO] === confirm_patch_samples + cleanup patch ===")
    copy_confirm_patch_samples(patch_run_dir, final_dir)
    patch_gen.cleanup_patch_generated(patch_run_dir, keep_manifest=True)

    # Orchestration snapshot alongside distill config
    orch_path = os.path.join(final_dir, "e2edt_pipeline.json")
    with open(orch_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_report_dir": sel["run_dir"],
                "run_name": sel["run_name"],
                "target_node": sel["target_node"],
                "source_node": sel["source_node"],
                "attack_class": sel["attack_class"],
                "patch_run_dir": patch_run_dir,
                "localization_run_dir": localization_run_dir,
                "output_dir": final_dir,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[INFO] E2EDT done -> {final_dir}")
    return final_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "E2E transfer: patch_data_generate → specific_localization → "
            "knowledgedistill, with interactive scheme/node/attack selection."
        )
    )
    p.add_argument(
        "--training_note_root",
        type=str,
        default=_DEFAULT_TRAINING_NOTE,
        help="Root of training_report runs.",
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Specific training_report run (default: interactive / latest).",
    )
    p.add_argument("--target_node", type=str, default=None)
    p.add_argument("--source_node", type=str, default=None)
    p.add_argument(
        "--attack_class",
        type=str,
        default=None,
        help="Source attack class (logical name or NN_name folder).",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--output_root",
        type=str,
        default=_DEFAULT_DISTILL_ROOT,
        help="Root for distill_result/ (E2EDT naming).",
    )
    p.add_argument("--patch_config", type=str, default=None)
    p.add_argument("--localization_config", type=str, default=None)
    p.add_argument("--distill_config", type=str, default=None)
    p.add_argument(
        "--patch_output_root",
        type=str,
        default=None,
        help="Override patch_generated root.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = {
        "training_note_root": args.training_note_root,
        "run_dir": args.run_dir,
        "target_node": args.target_node,
        "source_node": args.source_node,
        "attack_class": args.attack_class,
        "device": args.device,
        "output_root": args.output_root,
        "patch_config": args.patch_config,
        "localization_config": args.localization_config,
        "distill_config": args.distill_config,
        "patch_output_root": args.patch_output_root,
    }
    run_e2edt(cfg)


if __name__ == "__main__":
    main()
