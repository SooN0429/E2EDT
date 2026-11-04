#!/usr/bin/env python3
"""Stage 2 — E2EDT: insert subnet into target, distill, compare baselines.

Requires Stage1 ``source_preparation.py`` output (``--prep_dir``) or an existing
``--localization_run_dir``. Compares the distilled student against:
  1) from-scratch model (source new-class images + target old-class images)
  2) original target model (pre-transfer)

Reports wall-clock, SAMPLE_RATIO original-image count vs scratch labeled images,
training FLOPs, and per-class F1/recall/acc (with old-class drop / new-class gain).

範例：
  python E2EDT.py --prep_dir ./source_prep/<run> --target_node node_2 --device cuda:0
  python E2EDT.py --localization_run_dir ./localization/localization_result/<run> \\
      --target_node node_2 --run_dir /path/to/training_report/<run>
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

import torch

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
import compute_resources as cres  # noqa: E402
import comparison_eval as ceval  # noqa: E402
import knowledgedistill as distill  # noqa: E402
import scratch_baseline as scratch  # noqa: E402
import patch_data_generate as patch_gen  # noqa: E402

_DEFAULT_TRAINING_NOTE = common._DEFAULT_TRAINING_NOTE
_DEFAULT_DISTILL_ROOT = common._DEFAULT_DISTILL_ROOT


def _resolve_prep(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg.get("prep_dir"):
        manifest = common.load_source_prep_manifest(cfg["prep_dir"])
        return manifest

    loc_dir = cfg.get("localization_run_dir")
    if not loc_dir:
        raise ValueError("Provide --prep_dir or --localization_run_dir")
    loc_dir = os.path.abspath(loc_dir)
    loc_cfg = common.load_json(os.path.join(loc_dir, "config.json"))
    patch_run_dir = os.path.abspath(loc_cfg["patch_run_dir"])
    src_img = common.count_sample_ratio_source_images(patch_run_dir)
    run_dir = cfg.get("run_dir") or loc_cfg.get("training_report_dir")
    if not run_dir:
        raise ValueError(
            "When using --localization_run_dir without --prep_dir, "
            "pass --run_dir (training_report)."
        )
    source_node = (
        loc_cfg.get("resolved_node")
        or loc_cfg.get("resolved_node_name")
        or loc_cfg.get("node_name")
    )
    attack = (
        loc_cfg.get("resolved_attack_class")
        or loc_cfg.get("target_class")
        or loc_cfg.get("attack_class")
    )
    if not source_node or not attack:
        raise ValueError(
            "Cannot infer source_node/attack_class from localization config; "
            "use --prep_dir or ensure loc config has resolved fields."
        )
    plan_name = os.path.basename(os.path.abspath(run_dir))
    return {
        "training_report_dir": os.path.abspath(run_dir),
        "run_name": plan_name,
        "source_node": source_node,
        "attack_class": attack,
        "patch_run_dir": patch_run_dir,
        "localization_run_dir": loc_dir,
        "sample_ratio": src_img.get("sample_ratio"),
        "n_source_images": src_img.get("n_source_images"),
        "source_image_ids": src_img.get("image_ids"),
        "prep_dir": None,
    }


def _reload_student_teacher(
    distill_out: str,
    localization_run_dir: str,
    target_node: str,
    run_dir: str,
    training_note_root: str,
    device: torch.device,
):
    loc_info = distill.resolve_localization(localization_run_dir)
    target_info = distill.resolve_target(
        {
            "target_node": target_node,
            "target_training_report_dir": run_dir,
            "training_report_root": training_note_root,
        }
    )
    n_old = len(target_info["class_names"])
    student, teacher = distill.build_student_and_teacher(
        target_ckpt=target_info["checkpoint_path"],
        n_old=n_old,
        extracted_layer=loc_info["extracted_layer"],
        loc_info=loc_info,
        device=device,
    )
    blob = torch.load(
        os.path.join(distill_out, "student_state_dict.pt"), map_location="cpu"
    )
    student.adapter.load_state_dict(blob["adapter"])
    student.classifier_layer.load_state_dict(blob["classifier_layer"])
    student.eval()
    teacher.eval()
    return student, teacher, loc_info, target_info


def run_e2edt_stage2(cfg: Dict[str, Any]) -> str:
    training_note_root = os.path.abspath(
        cfg.get("training_note_root") or _DEFAULT_TRAINING_NOTE
    )
    prep = _resolve_prep(cfg)
    run_dir = os.path.abspath(prep["training_report_dir"])
    source_node = prep["source_node"]
    attack_class = prep["attack_class"]
    localization_run_dir = os.path.abspath(prep["localization_run_dir"])
    patch_run_dir = os.path.abspath(prep["patch_run_dir"])
    n_source_images = int(prep.get("n_source_images") or 0)
    sample_ratio = prep.get("sample_ratio")

    tgt = common.resolve_target_selection(
        training_note_root=training_note_root,
        run_dir=run_dir,
        target_node=cfg.get("target_node"),
        source_node=source_node,
        attack_class=attack_class,
    )
    target_node = tgt["target_node"]
    target_classes = list(tgt["target_classes"])
    run_name = tgt["run_name"]

    device_s = cfg.get("device") or "cuda:0"
    if str(device_s).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available; falling back to cpu")
        device_s = "cpu"
    device = torch.device(device_s)

    distill_root = os.path.abspath(cfg.get("output_root") or _DEFAULT_DISTILL_ROOT)
    os.makedirs(distill_root, exist_ok=True)
    out_dir = common.make_e2edt_output_dir(
        output_root=distill_root,
        target_node=target_node,
        source_node=source_node,
        attack_class=attack_class,
        run_name=run_name,
    )
    print(f"[INFO] E2EDT Stage2 output_dir={out_dir}")

    # --- Distill (skip built-in target-only eval; we use 2B test) ---
    print("[INFO] === knowledgedistill ===")
    distill_overrides: Dict[str, Any] = {
        "localization_run_dir": localization_run_dir,
        "target_node": target_node,
        "target_training_report_dir": run_dir,
        "training_report_root": training_note_root,
        "output_dir": out_dir,
        "skip_eval": True,
        "device": device_s,
    }
    if cfg.get("epochs") is not None:
        distill_overrides["epochs"] = int(cfg["epochs"])
    if cfg.get("distill_lr") is not None:
        distill_overrides["lr"] = float(cfg["distill_lr"])
    if cfg.get("batch_size") is not None:
        distill_overrides["batch_size"] = int(cfg["batch_size"])
    if cfg.get("seed") is not None:
        distill_overrides["seed"] = int(cfg["seed"])
    if cfg.get("distill_early_stop_patience") is not None:
        distill_overrides["early_stop_patience"] = int(
            cfg["distill_early_stop_patience"]
        )
    if cfg.get("distill_early_stop_min_delta") is not None:
        distill_overrides["early_stop_min_delta"] = float(
            cfg["distill_early_stop_min_delta"]
        )

    distill_cfg = distill.load_distill_config(
        cfg.get("distill_config"), overrides=distill_overrides
    )
    distill_out = distill.run_distill(distill_cfg)

    student, teacher, loc_info, target_info = _reload_student_teacher(
        distill_out,
        localization_run_dir,
        target_node,
        run_dir,
        training_note_root,
        device,
    )
    extracted = loc_info["extracted_layer"]
    eval_classes = list(target_classes) + [attack_class]

    # Distill FLOPs (one forward of student on a patch-sized or ImageNet-sized tensor)
    distill_stats = common.load_json(
        os.path.join(distill_out, "distill_train_stats.json")
    )
    try:
        # Use ImageNet-size input consistent with eval / typical image forward
        example = torch.randn(1, 3, 224, 224, device=device)
        flop_info = cres.measure_forward_flops(
            lambda x: student.predict(x, detach=False),
            example,
            model_for_profile=student,
        )
        distill_resources = cres.pack_resource_report(
            wall_clock_sec=float(distill_stats["wall_clock_sec"]),
            n_train_steps=int(distill_stats["n_train_steps"]),
            forward_flops=float(flop_info["forward_flops"]),
            epoch_times=list(distill_stats.get("epoch_times_sec") or []),
            extra={"forward_profile": flop_info, "label_data_note": "see comparison"},
        )
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] distill FLOPs measure failed: {e}")
        distill_resources = cres.pack_resource_report(
            wall_clock_sec=float(distill_stats["wall_clock_sec"]),
            n_train_steps=int(distill_stats["n_train_steps"]),
            forward_flops=None,
            epoch_times=list(distill_stats.get("epoch_times_sec") or []),
        )
    common.save_json(
        os.path.join(out_dir, "distill_resources.json"), distill_resources
    )

    # --- From-scratch baseline (2B) ---
    print("[INFO] === scratch baseline (cross-node) ===")
    scratch_epochs = int(cfg.get("scratch_epochs") or 50)
    scratch_result = scratch.run_scratch_baseline(
        dataset_root=target_info["dataset_root"],
        target_node=target_node,
        source_node=source_node,
        old_classes=target_classes,
        new_class=attack_class,
        trigger_dir=target_info["trigger_dir"],
        extracted_layer=extracted,
        device=device,
        output_dir=out_dir,
        blended_alpha_square=target_info["blended_alpha_square"],
        blended_alpha_hello_kitty=target_info["blended_alpha_hello_kitty"],
        seed=int(cfg.get("seed") or 42),
        epochs=scratch_epochs,
        batch_size=int(cfg.get("scratch_batch_size") or 8),
        lr=float(cfg.get("scratch_lr") or 1e-4),
        early_stop_patience=int(
            cfg["scratch_early_stop_patience"]
            if cfg.get("scratch_early_stop_patience") is not None
            else 5
        ),
        early_stop_min_delta=float(
            cfg["scratch_early_stop_min_delta"]
            if cfg.get("scratch_early_stop_min_delta") is not None
            else 1e-3
        ),
    )
    scratch_model = scratch_result["model"]
    scratch_resources = scratch_result["resources"]
    test_root = scratch_result["test_root"]

    # --- Unified eval: E2EDT/scratch @ 3-class; original target @ old-only 2-class ---
    print("[INFO] === unified comparison eval ===")
    bs = int(cfg.get("eval_batch_size") or 32)
    e2edt_metrics = ceval.evaluate_predict_fn(
        lambda x: student.predict(x, detach=False),
        test_root=test_root,
        eval_class_names=eval_classes,
        device=device,
        batch_size=bs,
    )
    scratch_metrics = ceval.evaluate_predict_fn(
        lambda x: scratch_model.predict(x, test_flag=1),
        test_root=test_root,
        eval_class_names=eval_classes,
        device=device,
        batch_size=bs,
    )
    test_old_root = ceval.materialize_old_class_test_root(
        test_root=test_root,
        old_classes=target_classes,
        dest_root=os.path.join(out_dir, "scratch_data", "test_old"),
    )
    original_metrics = ceval.evaluate_predict_fn(
        lambda x: teacher.predict(x, test_flag=1),
        test_root=test_old_root,
        eval_class_names=target_classes,
        device=device,
        batch_size=bs,
        n_logits=len(target_classes),
        n_old_classes=len(target_classes),
    )
    common.save_json(os.path.join(out_dir, "metrics_e2edt.json"), e2edt_metrics)
    common.save_json(os.path.join(out_dir, "metrics_scratch.json"), scratch_metrics)
    common.save_json(
        os.path.join(out_dir, "metrics_original_target.json"), original_metrics
    )

    report = ceval.build_comparison_report(
        e2edt_metrics=e2edt_metrics,
        scratch_metrics=scratch_metrics,
        original_metrics=original_metrics,
        distill_resources=distill_resources,
        scratch_resources=scratch_resources,
        n_source_images=n_source_images,
        sample_ratio=sample_ratio,
        eval_class_names=eval_classes,
    )
    common.save_json(os.path.join(out_dir, "comparison_report.json"), report)
    md = ceval.comparison_report_to_markdown(report)
    with open(os.path.join(out_dir, "comparison_report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    # Confirm previews + optional cleanup
    print("[INFO] === confirm_patch_samples ===")
    common.copy_confirm_patch_samples(patch_run_dir, out_dir)
    if cfg.get("cleanup_patch"):
        patch_gen.cleanup_patch_generated(patch_run_dir, keep_manifest=True)
        print(f"[INFO] cleaned patch large artifacts under {patch_run_dir}")

    pipeline = {
        "stage": "E2EDT",
        "prep_dir": prep.get("prep_dir"),
        "training_report_dir": run_dir,
        "run_name": run_name,
        "target_node": target_node,
        "source_node": source_node,
        "attack_class": attack_class,
        "patch_run_dir": patch_run_dir,
        "localization_run_dir": localization_run_dir,
        "output_dir": out_dir,
        "sample_ratio": sample_ratio,
        "n_source_images": n_source_images,
        "test_root": test_root,
        "test_old_root": test_old_root,
    }
    common.save_json(os.path.join(out_dir, "e2edt_pipeline.json"), pipeline)
    print(f"[INFO] E2EDT Stage2 done -> {out_dir}")
    print(md)
    return out_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Stage2 E2EDT: insert compact subnet → distill → "
            "from-scratch + original-target comparison."
        )
    )
    p.add_argument(
        "--training_note_root",
        type=str,
        default=_DEFAULT_TRAINING_NOTE,
    )
    p.add_argument(
        "--prep_dir",
        type=str,
        default=None,
        help="Stage1 source_prep directory containing source_prep_manifest.json.",
    )
    p.add_argument(
        "--localization_run_dir",
        type=str,
        default=None,
        help="Alternative to --prep_dir: existing localization_result run.",
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="training_report run (required if not using prep_dir).",
    )
    p.add_argument("--target_node", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--output_root",
        type=str,
        default=_DEFAULT_DISTILL_ROOT,
    )
    p.add_argument("--distill_config", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None, help="Distill max epochs.")
    p.add_argument("--distill_lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--distill_early_stop_patience",
        type=int,
        default=10,
        help="Distill early-stop patience on train distill loss; <=0 disables.",
    )
    p.add_argument(
        "--distill_early_stop_min_delta",
        type=float,
        default=0.004,
        help="Min train-loss drop to count as improvement (default 4e-3).",
    )
    p.add_argument("--scratch_epochs", type=int, default=50)
    p.add_argument("--scratch_batch_size", type=int, default=8)
    p.add_argument("--scratch_lr", type=float, default=1e-4)
    p.add_argument(
        "--scratch_early_stop_patience",
        type=int,
        default=5,
        help="Scratch early-stop patience on val_loss; <=0 disables.",
    )
    p.add_argument(
        "--scratch_early_stop_min_delta",
        type=float,
        default=0.001,
        help="Min val_loss drop to count as improvement (default 1e-3).",
    )
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument(
        "--cleanup_patch",
        action="store_true",
        help="After Stage2, cleanup patch_generated large files (keep manifests).",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = {
        "training_note_root": args.training_note_root,
        "prep_dir": args.prep_dir,
        "localization_run_dir": args.localization_run_dir,
        "run_dir": args.run_dir,
        "target_node": args.target_node,
        "device": args.device,
        "output_root": args.output_root,
        "distill_config": args.distill_config,
        "epochs": args.epochs,
        "distill_lr": args.distill_lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "distill_early_stop_patience": args.distill_early_stop_patience,
        "distill_early_stop_min_delta": args.distill_early_stop_min_delta,
        "scratch_epochs": args.scratch_epochs,
        "scratch_batch_size": args.scratch_batch_size,
        "scratch_lr": args.scratch_lr,
        "scratch_early_stop_patience": args.scratch_early_stop_patience,
        "scratch_early_stop_min_delta": args.scratch_early_stop_min_delta,
        "eval_batch_size": args.eval_batch_size,
        "cleanup_patch": bool(args.cleanup_patch),
    }
    run_e2edt_stage2(cfg)


if __name__ == "__main__":
    main()
