#!/usr/bin/env python3
"""Compute differential vectors from paired attack/clean images.

Examples:
  cd E2EDT/matching
  source ../.venv/bin/activate
  python diffvector_cal.py
  # 可選：
  #   --run_dir <training_report 某 run> # 未指定，則預設使用最新訓練方案
  #   --extracted_layer 7_point
  #   --keep_generated # 保留配對的攻擊/清潔圖片
  #   --no_confirm_samples # 不保留確認用的paired data

Method 1 (implemented): frozen ImageNet ResNet-18 intermediate features
  diff_vector = mean_i( flatten(f(attack_i)) - flatten(f(clean_i)) )

Method 2: reserved / not implemented yet.

Results are saved under matching/diffvector_results/<training_report_run_name>/
for later use by diffvector_matching.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_E2EDT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_NODE_DATASET_DIR = os.path.join(_E2EDT_ROOT, "node_dataset")
_DEFAULT_TRAINING_NOTE = os.path.join(
    _E2EDT_ROOT, "node_model_training", "training_report"
)
_DEFAULT_PAIRED_ROOT = os.path.join(_SCRIPT_DIR, "diffvector_generated")
_DEFAULT_RESULTS_ROOT = os.path.join(_SCRIPT_DIR, "diffvector_results")

if _NODE_DATASET_DIR not in sys.path:
    sys.path.insert(0, _NODE_DATASET_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from node_traindata_generate import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    ResNet18Extractor,
)
from diffvector_data_generate import (  # noqa: E402
    cleanup_diffvector_data,
    generate_diffvector_paired_data,
    load_latest_training_plan,
    save_confirm_paired_samples,
)

METHOD1_NAME = "method1_mean_diff"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_diffvectors_method2(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    """Reserved for a second differential-vector method (not implemented)."""
    raise NotImplementedError(
        "diffvector method2 is not implemented yet; use method1 only."
    )


METHODS = {
    METHOD1_NAME: "compute_diffvectors_method1",
    "method2": "compute_diffvectors_method2",
}


class PairedAttackCleanDataset(Dataset):
    """Load attack/clean image pairs that share the same relative filename."""

    def __init__(
        self,
        attack_class_dir: str,
        clean_class_dir: str,
        transform: transforms.Compose,
    ):
        self.attack_class_dir = attack_class_dir
        self.clean_class_dir = clean_class_dir
        self.transform = transform

        if not os.path.isdir(attack_class_dir):
            raise FileNotFoundError(f"attack class dir not found: {attack_class_dir}")
        if not os.path.isdir(clean_class_dir):
            raise FileNotFoundError(f"clean class dir not found: {clean_class_dir}")

        names = sorted(
            f
            for f in os.listdir(attack_class_dir)
            if os.path.isfile(os.path.join(attack_class_dir, f))
            and os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        if not names:
            raise RuntimeError(f"No images under {attack_class_dir}")

        missing = [
            n for n in names if not os.path.isfile(os.path.join(clean_class_dir, n))
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} paired clean images missing under {clean_class_dir} "
                f"(e.g. {missing[0]})"
            )
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.names[index]
        attack = Image.open(os.path.join(self.attack_class_dir, name)).convert("RGB")
        clean = Image.open(os.path.join(self.clean_class_dir, name)).convert("RGB")
        return self.transform(attack), self.transform(clean)


def _default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _list_class_folders(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )


def _mean_diff_for_class(
    attack_class_dir: str,
    clean_class_dir: str,
    extractor: ResNet18Extractor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, int, Tuple[int, ...]]:
    """
    Return (mean_diff_1d float32, n_pairs, feature_shape_before_flatten).
    """
    dataset = PairedAttackCleanDataset(
        attack_class_dir=attack_class_dir,
        clean_class_dir=clean_class_dir,
        transform=_default_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    sum_diff: Optional[torch.Tensor] = None
    n_pairs = 0
    feat_shape: Optional[Tuple[int, ...]] = None

    extractor.eval()
    with torch.no_grad():
        for attack_batch, clean_batch in loader:
            attack_batch = attack_batch.to(device, non_blocking=True)
            clean_batch = clean_batch.to(device, non_blocking=True)
            attack_feat = extractor(attack_batch)
            clean_feat = extractor(clean_batch)
            if feat_shape is None:
                feat_shape = tuple(attack_feat.shape[1:])

            # Flatten spatial/channel dims; keep batch.
            a = attack_feat.reshape(attack_feat.size(0), -1)
            c = clean_feat.reshape(clean_feat.size(0), -1)
            diff = a - c
            batch_sum = diff.sum(dim=0)
            if sum_diff is None:
                sum_diff = batch_sum
            else:
                sum_diff = sum_diff + batch_sum
            n_pairs += attack_feat.size(0)

    if sum_diff is None or n_pairs == 0 or feat_shape is None:
        raise RuntimeError(
            f"No features computed for pair dirs:\n  {attack_class_dir}\n  {clean_class_dir}"
        )

    mean_diff = (sum_diff / float(n_pairs)).detach().cpu().numpy().astype(np.float32)
    return mean_diff, n_pairs, feat_shape


def compute_diffvectors_method1(
    paired_results: Dict[str, Any],
    extracted_layer: str,
    device: torch.device,
    output_run_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    For each node/attack class folder, compute mean(attack_feat - clean_feat)
    and save under output_run_dir/method1_mean_diff/<node>/<class_folder>.npy
    """
    method_root = os.path.join(output_run_dir, METHOD1_NAME)
    os.makedirs(method_root, exist_ok=True)

    extractor = ResNet18Extractor(extracted_layer=extracted_layer, pretrained=True)
    extractor = extractor.to(device)
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(False)

    method_info: Dict[str, Any] = {
        "name": METHOD1_NAME,
        "diff_order": "attack_minus_clean",
        "extracted_layer": extracted_layer,
        "nodes": {},
    }

    nodes = paired_results.get("nodes") or {}
    for node_name, info in nodes.items():
        attack_dir = info["attack_dir"]
        clean_dir = info["clean_dir"]
        node_out = os.path.join(method_root, node_name)
        os.makedirs(node_out, exist_ok=True)

        class_folders = _list_class_folders(attack_dir)
        if not class_folders:
            raise RuntimeError(f"No attack class folders under {attack_dir}")

        node_entry: Dict[str, Any] = {"classes": {}}
        for class_folder in class_folders:
            attack_cls = os.path.join(attack_dir, class_folder)
            clean_cls = os.path.join(clean_dir, class_folder)
            print(f"[INFO] method1 {node_name}/{class_folder} ...")
            mean_diff, n_pairs, feat_shape = _mean_diff_for_class(
                attack_class_dir=attack_cls,
                clean_class_dir=clean_cls,
                extractor=extractor,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            out_path = os.path.join(node_out, f"{class_folder}.npy")
            np.save(out_path, mean_diff)
            node_entry["classes"][class_folder] = {
                "path": out_path,
                "n_pairs": int(n_pairs),
                "vector_shape": list(mean_diff.shape),
                "feature_shape": list(feat_shape),
                "dtype": "float32",
            }
            print(
                f"[INFO] saved {out_path} shape={mean_diff.shape} n_pairs={n_pairs}"
            )
        method_info["nodes"][node_name] = node_entry

    return method_info


def run_diffvector_cal(
    training_note_root: str = _DEFAULT_TRAINING_NOTE,
    run_dir: Optional[str] = None,
    paired_output_root: str = _DEFAULT_PAIRED_ROOT,
    results_root: str = _DEFAULT_RESULTS_ROOT,
    extracted_layer: Optional[str] = None,
    methods: Optional[Sequence[str]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    gpu_id: str = "cuda:0",
    keep_generated: bool = False,
    save_confirm_samples: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrate: load plan → generate paired data → method1 → save → cleanup.
    """
    plan = load_latest_training_plan(training_note_root, run_dir=run_dir)
    layer = extracted_layer or plan.get("extracted_layer") or "7_point"
    selected = list(methods) if methods else [METHOD1_NAME]

    # Prefer requested GPU when CUDA is usable; otherwise CPU.
    if gpu_id.startswith("cuda"):
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device = torch.device(gpu_id)
            else:
                device = torch.device("cpu")
        except Exception:
            device = torch.device("cpu")
    else:
        device = torch.device(gpu_id)
    print(f"[INFO] device={device} extracted_layer={layer} methods={selected}")

    paired = generate_diffvector_paired_data(
        plan=plan,
        output_root=paired_output_root,
    )

    results_root = os.path.abspath(results_root)
    output_run_dir = os.path.join(results_root, plan["run_name"])
    os.makedirs(output_run_dir, exist_ok=True)

    method1_info: Optional[Dict[str, Any]] = None
    method2_info: Any = None
    confirm_info: Optional[Dict[str, Any]] = None

    try:
        if save_confirm_samples:
            confirm_info = save_confirm_paired_samples(
                paired_results=paired,
                dest_run_dir=output_run_dir,
                n_pairs=1,
            )

        for method in selected:
            if method in (METHOD1_NAME, "method1"):
                method1_info = compute_diffvectors_method1(
                    paired_results=paired,
                    extracted_layer=layer,
                    device=device,
                    output_run_dir=output_run_dir,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            elif method in ("method2",):
                # Reserved; skip rather than crash default pipeline.
                print("[WARN] method2 requested but not implemented; skipping.")
                method2_info = None
            else:
                raise ValueError(
                    f"Unknown method '{method}'. Supported: {METHOD1_NAME}, method2"
                )
    finally:
        if not keep_generated:
            cleanup_diffvector_data(
                output_root=paired_output_root,
                run_name=paired.get("run_name") or plan["run_name"],
            )

    meta = {
        "source_report": plan["run_dir"],
        "run_name": plan["run_name"],
        "extracted_layer": layer,
        "diff_order": "attack_minus_clean",
        "methods_requested": selected,
        "method1": method1_info,
        "method2": method2_info,
        "confirm_paired_samples": confirm_info,
        "paired_source": {
            "output_dir": paired.get("output_dir"),
            "meta_path": paired.get("meta_path"),
            "nodes": {
                n: {
                    "attack_dir": info.get("attack_dir"),
                    "clean_dir": info.get("clean_dir"),
                    "counts": info.get("counts"),
                }
                for n, info in (paired.get("nodes") or {}).items()
            },
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = os.path.join(output_run_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] wrote meta: {meta_path}")
    print(f"[INFO] diffvector results at: {output_run_dir}")

    return {
        "output_dir": output_run_dir,
        "meta_path": meta_path,
        "run_name": plan["run_name"],
        "method1": method1_info,
        "method2": method2_info,
        "confirm_paired_samples": confirm_info,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired attack/clean data and compute differential vectors "
            "(method1: mean attack-clean ResNet features)."
        )
    )
    parser.add_argument(
        "--training_note_root",
        type=str,
        default=_DEFAULT_TRAINING_NOTE,
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Optional specific training_report run dir (default: latest).",
    )
    parser.add_argument(
        "--paired_output_root",
        type=str,
        default=_DEFAULT_PAIRED_ROOT,
        help="Temp root for diffvector_generated/.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=_DEFAULT_RESULTS_ROOT,
        help="Root for diffvector_results/<run_name>/.",
    )
    parser.add_argument(
        "--extracted_layer",
        type=str,
        default="",
        help="Override feature layer (default: from training_report config).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=METHOD1_NAME,
        help=f"Comma-separated methods (default: {METHOD1_NAME}). method2 is stub.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (default 0 to avoid /dev/shm issues).",
    )
    parser.add_argument("--gpu_id", type=str, default="cuda:0")
    parser.add_argument(
        "--keep_generated",
        action="store_true",
        help="Keep paired images under diffvector_generated after computation.",
    )
    parser.add_argument(
        "--no_confirm_samples",
        action="store_true",
        help="Skip copying one paired sample per attack class to diffvector_results.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    run_diffvector_cal(
        training_note_root=os.path.abspath(args.training_note_root),
        run_dir=args.run_dir or None,
        paired_output_root=os.path.abspath(args.paired_output_root),
        results_root=os.path.abspath(args.results_root),
        extracted_layer=args.extracted_layer or None,
        methods=methods,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpu_id=args.gpu_id,
        keep_generated=args.keep_generated,
        save_confirm_samples=not args.no_confirm_samples,
    )


if __name__ == "__main__":
    main()
