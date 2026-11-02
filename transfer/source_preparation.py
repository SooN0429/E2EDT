#!/usr/bin/env python3
"""Stage 1 — Source preparation: patch → localization → persist compact subnet.

選擇 training_report（未指定則互動／最新）、來源節點與攻擊類，產生 patch 資料並
做功能定位與子網路建構。輸出目錄含 source_prep_manifest.json，供 Stage2 E2EDT 使用。

預設保留 patch tensors（蒸餾需要）；可用 --cleanup_previews_only 只清 preview。

範例：
  python source_preparation.py --source_node node_1 --attack_class big_hello_kitty
  python source_preparation.py --run_dir /path/to/training_report/<run> \\
      --source_node node_4 --attack_class green_square --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_E2EDT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_COMPARE_DIR = os.path.join(_SCRIPT_DIR, "E2EDT_compare")

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _COMPARE_DIR not in sys.path:
    sys.path.insert(0, _COMPARE_DIR)
for _sub in ("patch_data_generate", "localization", "KD"):
    _p = os.path.join(_SCRIPT_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e2edt_common as common  # noqa: E402
import patch_data_generate as patch_gen  # noqa: E402
import specific_localziation as loc  # noqa: E402

_DEFAULT_TRAINING_NOTE = common._DEFAULT_TRAINING_NOTE
_DEFAULT_SOURCE_PREP_ROOT = common._DEFAULT_SOURCE_PREP_ROOT


def run_source_preparation(cfg: Dict[str, Any]) -> str:
    training_note_root = os.path.abspath(
        cfg.get("training_note_root") or _DEFAULT_TRAINING_NOTE
    )
    sel = common.resolve_source_prep_selections(
        training_note_root=training_note_root,
        run_dir=cfg.get("run_dir"),
        source_node=cfg.get("source_node"),
        attack_class=cfg.get("attack_class"),
    )

    device = cfg.get("device")
    prep_root = os.path.abspath(cfg.get("output_root") or _DEFAULT_SOURCE_PREP_ROOT)
    os.makedirs(prep_root, exist_ok=True)
    prep_dir = common.make_source_prep_output_dir(
        output_root=prep_root,
        source_node=sel["source_node"],
        attack_class=sel["attack_class"],
        run_name=sel["run_name"],
    )
    print(f"[INFO] source_prep output_dir={prep_dir}")

    # --- 1) Patch ---
    patch_overrides: Dict[str, Any] = {"WRITE_PREVIEWS": True}
    if cfg.get("sample_ratio") is not None:
        patch_overrides["SAMPLE_RATIO"] = float(cfg["sample_ratio"])
    if cfg.get("num_masks_per_image") is not None:
        patch_overrides["NUM_MASKS_PER_IMAGE"] = int(cfg["num_masks_per_image"])
    if cfg.get("keep_prob") is not None:
        patch_overrides["KEEP_PROB"] = float(cfg["keep_prob"])
    if cfg.get("mask_seed") is not None:
        patch_overrides["MASK_SEED"] = int(cfg["mask_seed"])
    if cfg.get("random_seed") is not None:
        patch_overrides["RANDOM_SEED"] = int(cfg["random_seed"])

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
    src_img_info = common.count_sample_ratio_source_images(patch_run_dir)

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

    # --- 3) Confirm previews into prep dir ---
    common.copy_confirm_patch_samples(patch_run_dir, prep_dir)

    if cfg.get("cleanup_previews_only"):
        previews = os.path.join(patch_run_dir, "previews")
        if os.path.isdir(previews):
            import shutil

            shutil.rmtree(previews)
            print(f"[INFO] cleaned previews only: {previews}")
    # Default: keep patch tensors for Stage2 distill.

    compact_pt = os.path.join(localization_run_dir, "compact_state_dict.pt")
    compact_meta = os.path.join(localization_run_dir, "compact_meta.json")
    if not os.path.isfile(compact_pt):
        raise FileNotFoundError(f"compact subnet missing: {compact_pt}")

    manifest = {
        "stage": "source_preparation",
        "training_report_dir": sel["run_dir"],
        "run_name": sel["run_name"],
        "source_node": sel["source_node"],
        "attack_class": sel["attack_class"],
        "source_classes": sel["source_classes"],
        "patch_run_dir": patch_run_dir,
        "localization_run_dir": localization_run_dir,
        "compact_state_dict": compact_pt,
        "compact_meta": compact_meta,
        "sample_ratio": src_img_info.get("sample_ratio"),
        "n_source_images": src_img_info.get("n_source_images"),
        "source_image_ids": src_img_info.get("image_ids"),
        "prep_dir": prep_dir,
        "device": device,
    }
    common.save_json(os.path.join(prep_dir, common.MANIFEST_NAME), manifest)
    print(
        f"[INFO] source preparation done -> {prep_dir}\n"
        f"[INFO] n_source_images (sample_ratio)={manifest['n_source_images']} "
        f"sample_ratio={manifest['sample_ratio']}\n"
        f"[INFO] localization_run_dir={localization_run_dir}"
    )
    return prep_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Stage1 source preparation: patch_data_generate → "
            "specific_localization → persist compact subnet + manifest."
        )
    )
    p.add_argument(
        "--training_note_root",
        type=str,
        default=_DEFAULT_TRAINING_NOTE,
    )
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--source_node", type=str, default=None)
    p.add_argument("--attack_class", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--output_root",
        type=str,
        default=_DEFAULT_SOURCE_PREP_ROOT,
        help="Root for source_prep/ outputs.",
    )
    p.add_argument("--patch_config", type=str, default=None)
    p.add_argument("--localization_config", type=str, default=None)
    p.add_argument("--patch_output_root", type=str, default=None)
    p.add_argument("--sample_ratio", type=float, default=None)
    p.add_argument("--num_masks_per_image", type=int, default=None)
    p.add_argument("--keep_prob", type=float, default=None)
    p.add_argument("--mask_seed", type=int, default=None)
    p.add_argument("--random_seed", type=int, default=None)
    p.add_argument(
        "--cleanup_previews_only",
        action="store_true",
        help="Delete patch previews/ only; keep tensors for Stage2.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = {
        "training_note_root": args.training_note_root,
        "run_dir": args.run_dir,
        "source_node": args.source_node,
        "attack_class": args.attack_class,
        "device": args.device,
        "output_root": args.output_root,
        "patch_config": args.patch_config,
        "localization_config": args.localization_config,
        "patch_output_root": args.patch_output_root,
        "sample_ratio": args.sample_ratio,
        "num_masks_per_image": args.num_masks_per_image,
        "keep_prob": args.keep_prob,
        "mask_seed": args.mask_seed,
        "random_seed": args.random_seed,
        "cleanup_previews_only": bool(args.cleanup_previews_only),
    }
    run_source_preparation(cfg)


if __name__ == "__main__":
    main()
