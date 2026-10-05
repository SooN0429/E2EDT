#!/usr/bin/env python3
"""Generate paired attack/clean images for differential-vector calculation.

Reads the latest training_report config, reproduces the same train-split image
selection as node_traindata_generate (same seed / C_max / RNG order), and writes
only attack-class pairs:

  attack/  — images with MaskBlended trigger (as used in training)
  clean/   — the same source images before trigger implant (identical filenames)

Temporary data under matching/diffvector_generated/; call cleanup_diffvector_data
after diffvector_cal finishes (mirrors node_training keep_generated=False).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_E2EDT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_NODE_DATASET_DIR = os.path.join(_E2EDT_ROOT, "node_dataset")
_DEFAULT_TRAINING_NOTE = os.path.join(
    _E2EDT_ROOT, "node_model_training", "training_report"
)
_DEFAULT_OUTPUT_ROOT = os.path.join(_SCRIPT_DIR, "diffvector_generated")

if _NODE_DATASET_DIR not in sys.path:
    sys.path.insert(0, _NODE_DATASET_DIR)

from node_traindata_generate import (  # noqa: E402
    SPLITS,
    apply_maskblended,
    class_folder_name,
    compute_per_person_quota,
    list_images,
    list_person_dirs,
    load_trigger_arrays,
)

CONFIRM_PAIRED_SAMPLES_DIR = "confirm_paired_samples"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _safe_rmtree(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)


def find_latest_training_report(training_note_root: str) -> str:
    """Return path to the newest run dir that contains config.json."""
    root = os.path.abspath(training_note_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"training_note_root not found: {root}")

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

    if not candidates:
        raise FileNotFoundError(f"No training runs with config.json under {root}")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _attack_classes(class_names: Sequence[str]) -> List[str]:
    return [c for c in class_names if c != "clean"]


def _resolve_paths(cfg: Dict[str, Any]) -> Tuple[str, str]:
    dataset_root = cfg.get("dataset_root") or _NODE_DATASET_DIR
    dataset_root = os.path.abspath(dataset_root)
    trigger_dir = cfg.get("trigger_dir") or ""
    trigger_dir = trigger_dir or os.path.join(dataset_root, "Attack_trigger_image")
    return dataset_root, os.path.abspath(trigger_dir)


def _expected_train_counts(
    dataset_root: str,
    node_specs: Dict[str, List[str]],
    per_person_k: int,
) -> Dict[str, Dict[str, int]]:
    """expected train images per attack class = k * n_persons."""
    out: Dict[str, Dict[str, int]] = {}
    for node_name, classes in node_specs.items():
        split_dir = os.path.join(dataset_root, node_name, "train")
        n_persons = len(list_person_dirs(split_dir))
        per_class = per_person_k * n_persons
        out[node_name] = {c: per_class for c in _attack_classes(classes)}
    return out


def load_latest_training_plan(
    training_note_root: str = _DEFAULT_TRAINING_NOTE,
    run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load training config + recompute quotas / expected attack counts.

    Returns a plan dict consumed by generate_diffvector_paired_data.
    """
    run_dir = os.path.abspath(run_dir) if run_dir else find_latest_training_report(
        training_note_root
    )
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    node_specs: Dict[str, List[str]] = cfg["node_specs"]
    if not isinstance(node_specs, dict) or not node_specs:
        raise ValueError(f"Invalid node_specs in {cfg_path}")

    dataset_root, trigger_dir = _resolve_paths(cfg)
    node_names = list(node_specs.keys())
    c_max = max(len(classes) for classes in node_specs.values())
    quotas = {
        split: compute_per_person_quota(dataset_root, node_names, c_max, split)
        for split in SPLITS
    }
    attack_classes = {n: _attack_classes(cs) for n, cs in node_specs.items()}
    expected_counts = _expected_train_counts(dataset_root, node_specs, quotas["train"])

    plan: Dict[str, Any] = {
        "run_dir": run_dir,
        "run_name": os.path.basename(run_dir),
        "config_path": cfg_path,
        "node_specs": node_specs,
        "attack_classes": attack_classes,
        "seed": int(cfg.get("seed", 0)),
        "blended_alpha": float(cfg.get("blended_alpha", 0.2)),
        "extracted_layer": str(cfg.get("extracted_layer") or "7_point"),
        "dataset_root": dataset_root,
        "trigger_dir": trigger_dir,
        "c_max": c_max,
        "quotas": quotas,
        "expected_counts": expected_counts,
        "created_at": cfg.get("created_at"),
    }
    return plan


def print_plan_summary(plan: Dict[str, Any]) -> None:
    print(f"[INFO] latest report: {plan['run_dir']}")
    print(f"[INFO] seed={plan['seed']} blended_alpha={plan['blended_alpha']} C_max={plan['c_max']}")
    print(f"[INFO] quotas (per-person per-class): {plan['quotas']}")
    print(f"[INFO] dataset_root={plan['dataset_root']}")
    print(f"[INFO] trigger_dir={plan['trigger_dir']}")
    for node_name, classes in plan["node_specs"].items():
        attacks = plan["attack_classes"][node_name]
        counts = plan["expected_counts"][node_name]
        print(
            f"[INFO] {node_name}: classes={classes} | "
            f"attack={attacks} | expected_train_per_attack={counts}"
        )


def _advance_rng_for_split(
    node_src_root: str,
    split: str,
    class_names: Sequence[str],
    per_person_k: int,
    rng: np.random.RandomState,
) -> None:
    """Consume RNG the same way generate_split_for_node does (no I/O)."""
    split_src = os.path.join(node_src_root, split)
    for person_dir in list_person_dirs(split_src):
        images = list_images(person_dir)
        need = per_person_k * len(class_names)
        if len(images) < need:
            raise RuntimeError(
                f"{person_dir} has {len(images)} images, need at least {need}"
            )
        rng.permutation(len(images))


def _write_attack_clean_pairs_for_train(
    node_src_root: str,
    node_out_root: str,
    class_names: Sequence[str],
    triggers: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
    per_person_k: int,
    blended_alpha: float,
    rng: np.random.RandomState,
) -> Dict[str, int]:
    """
    Select images identically to generate_split_for_node(train), but write only
    attack classes as paired attack/ + clean/ trees with the same filenames.
    """
    split = "train"
    split_src = os.path.join(node_src_root, split)
    attack_root = os.path.join(node_out_root, split, "attack")
    clean_root = os.path.join(node_out_root, split, "clean")
    _safe_rmtree(os.path.join(node_out_root, split))

    attack_dirs: Dict[str, str] = {}
    clean_dirs: Dict[str, str] = {}
    for idx, cname in enumerate(class_names):
        if cname == "clean":
            continue
        folder = class_folder_name(idx, cname)
        a_dir = os.path.join(attack_root, folder)
        c_dir = os.path.join(clean_root, folder)
        os.makedirs(a_dir, exist_ok=True)
        os.makedirs(c_dir, exist_ok=True)
        attack_dirs[cname] = a_dir
        clean_dirs[cname] = c_dir

    counts = {c: 0 for c in _attack_classes(class_names)}
    persons = list_person_dirs(split_src)

    for person_dir in persons:
        images = list_images(person_dir)
        need = per_person_k * len(class_names)
        if len(images) < need:
            raise RuntimeError(
                f"{person_dir} has {len(images)} images, need at least {need}"
            )
        order = rng.permutation(len(images))
        selected = [images[i] for i in order[:need]]

        person_name = os.path.basename(person_dir).replace(" ", "_")
        for class_idx, cname in enumerate(class_names):
            chunk = selected[class_idx * per_person_k : (class_idx + 1) * per_person_k]
            if cname == "clean":
                continue

            trigger_rgb, mask_rgb = triggers[cname]
            if trigger_rgb is None or mask_rgb is None:
                raise RuntimeError(f"Missing trigger/mask for attack class '{cname}'")

            for src_path in chunk:
                img = np.array(Image.open(src_path).convert("RGB"), dtype=np.uint8)
                attack_img = apply_maskblended(
                    img, trigger_rgb, blended_alpha, mask_rgb=mask_rgb
                )
                base = os.path.splitext(os.path.basename(src_path))[0]
                out_name = f"{person_name}_{base}_{cname}.jpg"
                Image.fromarray(attack_img).save(
                    os.path.join(attack_dirs[cname], out_name), quality=95
                )
                Image.fromarray(img).save(
                    os.path.join(clean_dirs[cname], out_name), quality=95
                )
                counts[cname] += 1

    return counts


def generate_diffvector_paired_data(
    plan: Optional[Dict[str, Any]] = None,
    output_root: str = _DEFAULT_OUTPUT_ROOT,
    training_note_root: str = _DEFAULT_TRAINING_NOTE,
    run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate paired attack/clean train images for all nodes in the training plan.

    Advances RNG through val/test (and non-persisted selections) in the same order
    as generate_all_nodes so train attack images match the original training run.
    """
    if plan is None:
        plan = load_latest_training_plan(training_note_root, run_dir=run_dir)

    print_plan_summary(plan)

    output_root = os.path.abspath(output_root)
    run_out = os.path.join(output_root, plan["run_name"])
    _safe_rmtree(run_out)
    os.makedirs(run_out, exist_ok=True)

    dataset_root = plan["dataset_root"]
    trigger_dir = plan["trigger_dir"]
    node_specs: Dict[str, List[str]] = plan["node_specs"]
    quotas: Dict[str, int] = plan["quotas"]
    blended_alpha = float(plan["blended_alpha"])
    rng = np.random.RandomState(int(plan["seed"]))

    results: Dict[str, Any] = {
        "output_dir": run_out,
        "run_name": plan["run_name"],
        "source_report": plan["run_dir"],
        "nodes": {},
    }

    for node_name, class_names in node_specs.items():
        node_src = os.path.join(dataset_root, node_name)
        if not os.path.isdir(node_src):
            raise FileNotFoundError(f"Node source not found: {node_src}")

        node_out = os.path.join(run_out, node_name)
        os.makedirs(node_out, exist_ok=True)
        triggers = load_trigger_arrays(trigger_dir, class_names)

        train_counts: Dict[str, int] = {}
        for split in SPLITS:
            if split == "train":
                train_counts = _write_attack_clean_pairs_for_train(
                    node_src_root=node_src,
                    node_out_root=node_out,
                    class_names=class_names,
                    triggers=triggers,
                    per_person_k=quotas[split],
                    blended_alpha=blended_alpha,
                    rng=rng,
                )
                print(f"[INFO] {node_name}/train attack counts={train_counts}")
            else:
                # Keep RNG in sync with generate_all_nodes without writing files.
                _advance_rng_for_split(
                    node_src_root=node_src,
                    split=split,
                    class_names=class_names,
                    per_person_k=quotas[split],
                    rng=rng,
                )

        attack_dir = os.path.join(node_out, "train", "attack")
        clean_dir = os.path.join(node_out, "train", "clean")
        results["nodes"][node_name] = {
            "classes": list(class_names),
            "attack_classes": plan["attack_classes"][node_name],
            "attack_dir": attack_dir,
            "clean_dir": clean_dir,
            "counts": train_counts,
        }

    meta = {
        "source_report": plan["run_dir"],
        "run_name": plan["run_name"],
        "node_specs": plan["node_specs"],
        "attack_classes": plan["attack_classes"],
        "seed": plan["seed"],
        "blended_alpha": plan["blended_alpha"],
        "dataset_root": plan["dataset_root"],
        "trigger_dir": plan["trigger_dir"],
        "c_max": plan["c_max"],
        "quotas": plan["quotas"],
        "expected_counts": plan["expected_counts"],
        "actual_counts": {
            n: info["counts"] for n, info in results["nodes"].items()
        },
        "nodes": {
            n: {
                "attack_dir": info["attack_dir"],
                "clean_dir": info["clean_dir"],
                "counts": info["counts"],
                "attack_classes": info["attack_classes"],
            }
            for n, info in results["nodes"].items()
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = os.path.join(run_out, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    results["meta_path"] = meta_path
    print(f"[INFO] wrote meta: {meta_path}")
    print(f"[INFO] diffvector paired data at: {run_out}")
    return results


def save_confirm_paired_samples(
    paired_results: Dict[str, Any],
    dest_run_dir: str,
    n_pairs: int = 1,
) -> Dict[str, Any]:
    """
    Copy n_pairs attack/clean image pairs per attack class into
    dest_run_dir/confirm_paired_samples/ for visual verification.
    """
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1")

    dest_run_dir = os.path.abspath(dest_run_dir)
    root = os.path.join(dest_run_dir, CONFIRM_PAIRED_SAMPLES_DIR)
    _safe_rmtree(root)
    os.makedirs(root, exist_ok=True)

    summary: Dict[str, Any] = {"root": root, "nodes": {}}
    nodes = paired_results.get("nodes") or {}
    if not nodes:
        raise ValueError("paired_results has no nodes")

    for node_name, info in nodes.items():
        attack_dir = info.get("attack_dir")
        clean_dir = info.get("clean_dir")
        if not attack_dir or not clean_dir:
            raise ValueError(f"Missing attack_dir/clean_dir for {node_name}")

        node_out = os.path.join(root, node_name)
        os.makedirs(node_out, exist_ok=True)
        summary["nodes"][node_name] = {}

        class_folders = sorted(
            d
            for d in os.listdir(attack_dir)
            if os.path.isdir(os.path.join(attack_dir, d))
        )
        if not class_folders:
            raise RuntimeError(f"No attack class folders under {attack_dir}")

        for class_folder in class_folders:
            attack_cls = os.path.join(attack_dir, class_folder)
            clean_cls = os.path.join(clean_dir, class_folder)
            images = sorted(
                f
                for f in os.listdir(attack_cls)
                if os.path.isfile(os.path.join(attack_cls, f))
                and os.path.splitext(f)[1].lower() in IMAGE_EXTS
            )
            if not images:
                raise RuntimeError(f"No images under {attack_cls}")

            dest_cls = os.path.join(node_out, class_folder)
            os.makedirs(dest_cls, exist_ok=True)

            class_entry: Dict[str, Any] = {"pairs": []}
            for fname in images[:n_pairs]:
                ext = os.path.splitext(fname)[1].lower() or ".jpg"
                attack_src = os.path.join(attack_cls, fname)
                clean_src = os.path.join(clean_cls, fname)
                if not os.path.isfile(clean_src):
                    raise FileNotFoundError(
                        f"Missing clean pair for {attack_src}: {clean_src}"
                    )

                attack_dst = os.path.join(dest_cls, f"attack{ext}")
                clean_dst = os.path.join(dest_cls, f"clean{ext}")
                shutil.copy2(attack_src, attack_dst)
                shutil.copy2(clean_src, clean_dst)

                pair_info = {
                    "node": node_name,
                    "class_folder": class_folder,
                    "source_filename": fname,
                    "attack_src": attack_src,
                    "clean_src": clean_src,
                    "attack": attack_dst,
                    "clean": clean_dst,
                }
                pair_info_path = os.path.join(dest_cls, "pair_info.json")
                with open(pair_info_path, "w", encoding="utf-8") as f:
                    json.dump(pair_info, f, indent=2, ensure_ascii=False)

                class_entry["pairs"].append(
                    {
                        "attack": attack_dst,
                        "clean": clean_dst,
                        "source_filename": fname,
                        "pair_info": pair_info_path,
                    }
                )

            # Flatten first pair for meta.json (n_pairs=1 typical).
            if class_entry["pairs"]:
                first = class_entry["pairs"][0]
                summary["nodes"][node_name][class_folder] = {
                    "attack": first["attack"],
                    "clean": first["clean"],
                    "source_filename": first["source_filename"],
                    "pair_info": first["pair_info"],
                }

    print(f"[INFO] saved confirm paired samples -> {root}")
    return summary


def cleanup_diffvector_data(
    output_root: str = _DEFAULT_OUTPUT_ROOT,
    run_name: Optional[str] = None,
) -> None:
    """Remove generated paired data. If run_name is None, remove entire output_root."""
    output_root = os.path.abspath(output_root)
    if run_name:
        path = os.path.join(output_root, run_name)
        _safe_rmtree(path)
        print(f"[INFO] Removed diffvector data: {path}")
        if os.path.isdir(output_root) and not os.listdir(output_root):
            os.rmdir(output_root)
            print(f"[INFO] Removed empty output root: {output_root}")
    else:
        _safe_rmtree(output_root)
        print(f"[INFO] Removed diffvector output root: {output_root}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired attack/clean train images from the latest "
            "training_report for differential-vector calculation."
        )
    )
    parser.add_argument(
        "--training_note_root",
        type=str,
        default=_DEFAULT_TRAINING_NOTE,
        help="Root of training_report directories.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Optional specific training_report run dir (default: latest).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=_DEFAULT_OUTPUT_ROOT,
        help="Root for diffvector_generated/<run_name>/.",
    )
    parser.add_argument(
        "--no_keep_generated",
        action="store_true",
        help="Delete generated data immediately after generation (default: keep for diffvector_cal).",
    )
    parser.add_argument(
        "--cleanup_only",
        action="store_true",
        help="Only delete output under --output_root (optionally --run_name).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="With --cleanup_only: delete only this run folder under output_root.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print plan summary; do not generate files.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_root = os.path.abspath(args.output_root)

    if args.cleanup_only:
        cleanup_diffvector_data(
            output_root=output_root,
            run_name=args.run_name or None,
        )
        return

    plan = load_latest_training_plan(
        training_note_root=os.path.abspath(args.training_note_root),
        run_dir=args.run_dir or None,
    )
    if args.dry_run:
        print_plan_summary(plan)
        print("[INFO] dry_run: skip generation")
        return

    results = generate_diffvector_paired_data(
        plan=plan,
        output_root=output_root,
    )

    if args.no_keep_generated:
        cleanup_diffvector_data(
            output_root=output_root,
            run_name=results["run_name"],
        )


if __name__ == "__main__":
    main()
