#!/usr/bin/env python3
"""Generate PNC-style patch perturbation data for the transfer pipeline.

Reproduces poisoned train samples for a selected node/attack class from a
training_report config (same seed / quota / RNG order as node_traindata_generate),
subsamples them, applies Bernoulli R×C patch masks in ImageNet-normalized
tensor space, and writes tensors + reproducible manifests.

Temporary tensors/masks may be deleted after localization / distillation;
config_resolved.json, selected_samples.json, and manifest.jsonl are retained.
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

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_E2EDT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_NODE_DATASET_DIR = os.path.join(_E2EDT_ROOT, "node_dataset")
_MATCHING_DIR = os.path.join(_E2EDT_ROOT, "matching")
_DEFAULT_TRAINING_NOTE = os.path.join(
    _E2EDT_ROOT, "node_model_training", "training_report"
)
_DEFAULT_OUTPUT_ROOT = os.path.join(_SCRIPT_DIR, "patch_generated")
_DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "patch_data_config.json")

if _NODE_DATASET_DIR not in sys.path:
    sys.path.insert(0, _NODE_DATASET_DIR)
if _MATCHING_DIR not in sys.path:
    sys.path.insert(0, _MATCHING_DIR)

from diffvector_data_generate import (  # noqa: E402
    _advance_rng_for_split,
    load_latest_training_plan,
)
from node_traindata_generate import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    SPLITS,
    apply_maskblended,
    class_folder_name,
    list_images,
    list_person_dirs,
    load_trigger_arrays,
    resolve_blended_alpha,
)

REFERENCE_MASK_ID = "reference_all_ones"
MASK_DEDUP_MAX_TRIES = 1000

PATCH_CONFIG_KEYS = (
    "SAMPLE_RATIO",
    "GRID_ROWS",
    "GRID_COLS",
    "NUM_MASKS_PER_IMAGE",
    "KEEP_PROB",
    "MASK_BASELINE",
    "MASK_BASELINE_CHANNEL_CONSTANTS",
    "MASK_BEFORE_OR_AFTER_NORMALIZATION",
    "RANDOM_SEED",
    "IMAGE_SIZE",
    "WRITE_PREVIEWS",
)


def _safe_rmtree(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)


def _default_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_patch_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load patch_data_config.json and merge optional CLI overrides."""
    path = os.path.abspath(config_path or _DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    for key in PATCH_CONFIG_KEYS:
        if key not in cfg and key != "MASK_BASELINE_CHANNEL_CONSTANTS":
            raise KeyError(f"Missing required config key: {key} in {path}")

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v

    ratio = float(cfg["SAMPLE_RATIO"])
    if not (0.0 < ratio <= 1.0):
        raise ValueError(
            f"SAMPLE_RATIO must be in (0, 1] (fraction of train set); got {ratio}"
        )

    rows = int(cfg["GRID_ROWS"])
    cols = int(cfg["GRID_COLS"])
    if rows < 1 or cols < 1:
        raise ValueError(f"GRID_ROWS/GRID_COLS must be >= 1; got {rows}x{cols}")

    n_masks = int(cfg["NUM_MASKS_PER_IMAGE"])
    if n_masks < 0:
        raise ValueError(f"NUM_MASKS_PER_IMAGE must be >= 0; got {n_masks}")

    keep = float(cfg["KEEP_PROB"])
    if not (0.0 <= keep <= 1.0):
        raise ValueError(f"KEEP_PROB must be in [0, 1]; got {keep}")

    timing = str(cfg["MASK_BEFORE_OR_AFTER_NORMALIZATION"]).lower()
    if timing != "after":
        raise ValueError(
            "Only MASK_BEFORE_OR_AFTER_NORMALIZATION='after' is supported in v1; "
            f"got {timing!r}"
        )

    baseline = str(cfg["MASK_BASELINE"])
    if baseline not in ("imagenet_mean_normalized", "channel_constants"):
        raise ValueError(
            "MASK_BASELINE must be 'imagenet_mean_normalized' or "
            f"'channel_constants'; got {baseline!r}"
        )
    if baseline == "channel_constants":
        consts = cfg.get("MASK_BASELINE_CHANNEL_CONSTANTS")
        if (
            not isinstance(consts, (list, tuple))
            or len(consts) != 3
        ):
            raise ValueError(
                "MASK_BASELINE=channel_constants requires "
                "MASK_BASELINE_CHANNEL_CONSTANTS as a length-3 list"
            )

    cfg["SAMPLE_RATIO"] = ratio
    cfg["GRID_ROWS"] = rows
    cfg["GRID_COLS"] = cols
    cfg["NUM_MASKS_PER_IMAGE"] = n_masks
    cfg["KEEP_PROB"] = keep
    cfg["MASK_BEFORE_OR_AFTER_NORMALIZATION"] = timing
    cfg["MASK_BASELINE"] = baseline
    cfg["RANDOM_SEED"] = int(cfg["RANDOM_SEED"])
    cfg["IMAGE_SIZE"] = int(cfg.get("IMAGE_SIZE", 224))
    cfg["WRITE_PREVIEWS"] = bool(cfg.get("WRITE_PREVIEWS", False))
    cfg["_config_path"] = path
    return cfg


def resolve_attack_class_index(
    class_names: Sequence[str],
    attack_class: str,
) -> Tuple[int, str, str]:
    """
    Resolve attack_class (logical name or 'NN_name' folder) to
    (class_idx, class_name, class_folder).
    """
    raw = attack_class.strip()
    if not raw:
        raise ValueError("attack_class is empty")

    # Folder form: 01_big_hello_kitty
    m = re.fullmatch(r"(\d{2})_(.+)", raw)
    if m:
        idx = int(m.group(1))
        name = m.group(2)
        if idx < 0 or idx >= len(class_names):
            raise ValueError(
                f"attack_class folder index {idx} out of range for {list(class_names)}"
            )
        if class_names[idx] != name:
            raise ValueError(
                f"attack_class {raw!r} does not match class_names[{idx}]="
                f"{class_names[idx]!r}"
            )
        if name == "clean":
            raise ValueError("attack_class must not be 'clean'")
        return idx, name, class_folder_name(idx, name)

    # Logical name
    if raw == "clean":
        raise ValueError("attack_class must not be 'clean'")
    matches = [i for i, c in enumerate(class_names) if c == raw]
    if not matches:
        raise ValueError(
            f"attack_class {raw!r} not found in class_names={list(class_names)}"
        )
    if len(matches) > 1:
        raise ValueError(f"Duplicate class name {raw!r} in {list(class_names)}")
    idx = matches[0]
    return idx, raw, class_folder_name(idx, raw)


def sync_rng_and_collect_attack_train_samples(
    plan: Dict[str, Any],
    node_name: str,
    attack_class: str,
) -> Tuple[List[Dict[str, Any]], int, str, str]:
    """
    Advance RNG like generate_all_nodes up to target node train, then collect
    metadata for the selected attack class (no full generated tree written).

    Returns (samples, class_idx, class_name, class_folder).
    """
    node_specs: Dict[str, List[str]] = plan["node_specs"]
    if node_name not in node_specs:
        raise KeyError(
            f"node {node_name!r} not in node_specs={list(node_specs.keys())}"
        )

    class_names = node_specs[node_name]
    class_idx, class_name, class_folder = resolve_attack_class_index(
        class_names, attack_class
    )

    dataset_root = plan["dataset_root"]
    quotas: Dict[str, int] = plan["quotas"]
    rng = np.random.RandomState(int(plan["seed"]))

    for n_name, n_classes in node_specs.items():
        node_src = os.path.join(dataset_root, n_name)
        if not os.path.isdir(node_src):
            raise FileNotFoundError(f"Node source not found: {node_src}")

        if n_name != node_name:
            for split in SPLITS:
                _advance_rng_for_split(
                    node_src_root=node_src,
                    split=split,
                    class_names=n_classes,
                    per_person_k=quotas[split],
                    rng=rng,
                )
            continue

        # Target node: collect train attack samples with same permutation as
        # generate_split_for_node, then advance val/test only.
        samples = _collect_attack_train_samples(
            node_src_root=node_src,
            node_name=node_name,
            class_names=n_classes,
            class_idx=class_idx,
            class_name=class_name,
            class_folder=class_folder,
            per_person_k=quotas["train"],
            rng=rng,
        )
        for split in ("val", "test"):
            _advance_rng_for_split(
                node_src_root=node_src,
                split=split,
                class_names=n_classes,
                per_person_k=quotas[split],
                rng=rng,
            )
        return samples, class_idx, class_name, class_folder

    raise RuntimeError(f"Unreachable: node {node_name} not processed")


def _collect_attack_train_samples(
    node_src_root: str,
    node_name: str,
    class_names: Sequence[str],
    class_idx: int,
    class_name: str,
    class_folder: str,
    per_person_k: int,
    rng: np.random.RandomState,
) -> List[Dict[str, Any]]:
    """Mirror generate_split_for_node(train) selection; keep only one attack class."""
    split_src = os.path.join(node_src_root, "train")
    samples: List[Dict[str, Any]] = []
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
        chunk = selected[
            class_idx * per_person_k : (class_idx + 1) * per_person_k
        ]
        for src_path in chunk:
            base = os.path.splitext(os.path.basename(src_path))[0]
            out_name = f"{person_name}_{base}_{class_name}.jpg"
            image_id = f"{node_name}/{class_folder}/{person_name}_{base}"
            samples.append(
                {
                    "image_id": image_id,
                    "original_image_path": os.path.abspath(src_path),
                    "source_face_path": os.path.abspath(src_path),
                    "poisoned_basename": out_name,
                    "person_name": person_name,
                    "base_name": base,
                }
            )
    return samples


def subsample_samples(
    samples: Sequence[Dict[str, Any]],
    sample_ratio: float,
    random_seed: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Subsample with RANDOM_SEED; return selected samples and original indices."""
    n = len(samples)
    if n == 0:
        raise ValueError("No samples to subsample")
    k = max(1, int(np.floor(n * sample_ratio)))
    if k < n * sample_ratio and sample_ratio < 1.0:
        pass  # floor as specified
    if k >= n:
        indices = list(range(n))
        return [dict(s) for s in samples], indices

    rng = np.random.RandomState(random_seed)
    indices = sorted(rng.choice(n, size=k, replace=False).tolist())
    selected = [dict(samples[i]) for i in indices]
    return selected, indices


def bernoulli_patch_masks(
    grid_rows: int,
    grid_cols: int,
    num_masks: int,
    keep_prob: float,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Generate unique Bernoulli R×C masks (int 0/1)."""
    masks: List[np.ndarray] = []
    seen = set()
    tries = 0
    while len(masks) < num_masks:
        tries += 1
        if tries > MASK_DEDUP_MAX_TRIES:
            raise RuntimeError(
                f"Failed to generate {num_masks} unique {grid_rows}x{grid_cols} "
                f"masks with KEEP_PROB={keep_prob} after {MASK_DEDUP_MAX_TRIES} tries"
            )
        m = (rng.random((grid_rows, grid_cols)) < keep_prob).astype(np.int64)
        key = m.tobytes()
        if key in seen:
            continue
        seen.add(key)
        masks.append(m)
    return masks


def patch_mask_to_spatial(
    patch_mask: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Upsample R×C binary patch mask to HxW float32 in {0, 1}."""
    rows, cols = patch_mask.shape
    if height < rows or width < cols:
        raise ValueError(
            f"Image {height}x{width} smaller than grid {rows}x{cols}"
        )
    spatial = np.zeros((height, width), dtype=np.float32)
    base_h, rem_h = divmod(height, rows)
    base_w, rem_w = divmod(width, cols)
    y0 = 0
    for r in range(rows):
        h = base_h + (rem_h if r == rows - 1 else 0)
        y1 = y0 + h
        x0 = 0
        for c in range(cols):
            w = base_w + (rem_w if c == cols - 1 else 0)
            x1 = x0 + w
            spatial[y0:y1, x0:x1] = float(patch_mask[r, c])
            x0 = x1
        y0 = y1
    return spatial


def resolve_mask_baseline(
    baseline_name: str,
    channel_constants: Optional[Sequence[float]],
    height: int,
    width: int,
) -> torch.Tensor:
    """Return baseline tensor [3,H,W] in normalized space."""
    if baseline_name == "imagenet_mean_normalized":
        return torch.zeros(3, height, width, dtype=torch.float32)
    assert channel_constants is not None
    vec = torch.tensor(list(channel_constants), dtype=torch.float32).view(3, 1, 1)
    return vec.expand(3, height, width).contiguous()


def poisoned_tensor_from_source(
    source_face_path: str,
    trigger_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    class_name: str,
    blended_alpha: float,
    transform: transforms.Compose,
) -> torch.Tensor:
    """MaskBlended poison image -> normalized tensor [3,H,W]."""
    img = np.array(Image.open(source_face_path).convert("RGB"), dtype=np.uint8)
    attack = apply_maskblended(
        img,
        trigger_rgb,
        blended_alpha,
        mask_rgb=mask_rgb,
        class_name=class_name,
    )
    pil = Image.fromarray(attack)
    tensor = transform(pil)
    assert isinstance(tensor, torch.Tensor)
    return tensor.float()


def apply_mask_after_norm(
    x: torch.Tensor,
    spatial_mask: np.ndarray,
    baseline: torch.Tensor,
) -> torch.Tensor:
    """tilde_x = B ⊙ x + (1-B) ⊙ baseline."""
    b = torch.from_numpy(spatial_mask).to(dtype=x.dtype)
    if b.ndim == 2:
        b = b.unsqueeze(0)  # [1,H,W]
    return b * x + (1.0 - b) * baseline


def denormalize_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Inverse ImageNet normalize for preview PNGs."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    x = tensor * std + mean
    x = (x.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return x


def _mask_rng_for_image(
    random_seed: int,
    image_id: str,
) -> np.random.Generator:
    """Deterministic per-image Generator from RANDOM_SEED + image_id."""
    # Stable 32-bit seed derived from seed and image_id.
    data = f"{random_seed}:{image_id}".encode("utf-8")
    h = 2166136261
    for byte in data:
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return np.random.default_rng(h)


def generate_patch_dataset(
    node_name: str,
    attack_class: str,
    patch_cfg: Dict[str, Any],
    training_note_root: str = _DEFAULT_TRAINING_NOTE,
    run_dir: Optional[str] = None,
    output_root: str = _DEFAULT_OUTPUT_ROOT,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Full pipeline: reproduce train list -> subsample -> masks -> tensors."""
    plan = load_latest_training_plan(training_note_root, run_dir=run_dir)
    samples, class_idx, class_name, class_folder = (
        sync_rng_and_collect_attack_train_samples(plan, node_name, attack_class)
    )
    expected = (
        plan.get("expected_counts", {})
        .get(node_name, {})
        .get(class_name)
    )
    print(
        f"[INFO] collected {len(samples)} train samples for "
        f"{node_name}/{class_folder} (expected={expected})"
    )
    if expected is not None and int(expected) != len(samples):
        print(
            f"[WARN] sample count {len(samples)} != expected_counts {expected}"
        )

    selected, selected_indices = subsample_samples(
        samples,
        sample_ratio=float(patch_cfg["SAMPLE_RATIO"]),
        random_seed=int(patch_cfg["RANDOM_SEED"]),
    )
    print(
        f"[INFO] subsampled {len(selected)}/{len(samples)} "
        f"(SAMPLE_RATIO={patch_cfg['SAMPLE_RATIO']}, "
        f"RANDOM_SEED={patch_cfg['RANDOM_SEED']})"
    )

    out_name = f"{plan['run_name']}_{node_name}_{class_name}"
    out_dir = os.path.join(os.path.abspath(output_root), out_name)
    result: Dict[str, Any] = {
        "output_dir": out_dir,
        "run_name": plan["run_name"],
        "node": node_name,
        "attack_class": class_name,
        "class_folder": class_folder,
        "class_idx": class_idx,
        "n_full": len(samples),
        "n_selected": len(selected),
        "expected_count": expected,
        "dry_run": dry_run,
    }

    if dry_run:
        print(f"[INFO] dry_run: would write to {out_dir}")
        result["selected_indices"] = selected_indices
        result["selected_image_ids"] = [s["image_id"] for s in selected]
        return result

    _safe_rmtree(out_dir)
    masks_root = os.path.join(out_dir, "masks")
    tensors_root = os.path.join(out_dir, "tensors")
    previews_root = os.path.join(out_dir, "previews")
    os.makedirs(masks_root, exist_ok=True)
    os.makedirs(tensors_root, exist_ok=True)
    if patch_cfg["WRITE_PREVIEWS"]:
        os.makedirs(previews_root, exist_ok=True)

    image_size = int(patch_cfg["IMAGE_SIZE"])
    transform = _default_transform(image_size)
    baseline = resolve_mask_baseline(
        patch_cfg["MASK_BASELINE"],
        patch_cfg.get("MASK_BASELINE_CHANNEL_CONSTANTS"),
        image_size,
        image_size,
    )

    triggers = load_trigger_arrays(plan["trigger_dir"], plan["node_specs"][node_name])
    trigger_rgb, mask_rgb = triggers[class_name]
    if trigger_rgb is None or mask_rgb is None:
        raise RuntimeError(f"Missing trigger/mask for attack class '{class_name}'")

    rows = int(patch_cfg["GRID_ROWS"])
    cols = int(patch_cfg["GRID_COLS"])
    n_masks = int(patch_cfg["NUM_MASKS_PER_IMAGE"])
    keep_prob = float(patch_cfg["KEEP_PROB"])
    write_previews = bool(patch_cfg["WRITE_PREVIEWS"])

    selected_payload = {
        "node": node_name,
        "attack_class": class_name,
        "class_folder": class_folder,
        "class_idx": class_idx,
        "n_full": len(samples),
        "n_selected": len(selected),
        "SAMPLE_RATIO": patch_cfg["SAMPLE_RATIO"],
        "RANDOM_SEED": patch_cfg["RANDOM_SEED"],
        "selected_indices": selected_indices,
        "full_samples": samples,
        "selected_samples": selected,
        "mask_rng_scheme": "fnv1a32(RANDOM_SEED:image_id) -> numpy Generator",
    }
    with open(
        os.path.join(out_dir, "selected_samples.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(selected_payload, f, indent=2, ensure_ascii=False)

    config_resolved = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": out_dir,
        "patch_config_path": patch_cfg.get("_config_path"),
        "patch_config": {
            k: patch_cfg[k]
            for k in PATCH_CONFIG_KEYS
            if k in patch_cfg
        },
        "training_report": {
            "run_dir": plan["run_dir"],
            "run_name": plan["run_name"],
            "config_path": plan["config_path"],
            "seed": plan["seed"],
            "blended_alpha_square": plan["blended_alpha_square"],
            "blended_alpha_hello_kitty": plan["blended_alpha_hello_kitty"],
            "dataset_root": plan["dataset_root"],
            "trigger_dir": plan["trigger_dir"],
            "quotas": plan["quotas"],
            "c_max": plan["c_max"],
            "node_specs": plan["node_specs"],
            "expected_counts": plan.get("expected_counts"),
        },
        "selection": {
            "node": node_name,
            "attack_class": class_name,
            "class_folder": class_folder,
            "class_idx": class_idx,
            "n_full": len(samples),
            "n_selected": len(selected),
            "selected_indices": selected_indices,
        },
        "mask_rng_scheme": "fnv1a32(RANDOM_SEED:image_id) -> numpy Generator",
        "formula": "tilde_x = B odot x + (1-B) odot x_baseline (after Normalize)",
    }
    with open(
        os.path.join(out_dir, "config_resolved.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config_resolved, f, indent=2, ensure_ascii=False)

    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    n_rows = 0
    max_ref_diff = 0.0

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for sample in selected:
            image_id = sample["image_id"]
            # Safe relative path under masks/tensors mirrors image_id hierarchy.
            rel_id = image_id  # uses '/'

            alpha = resolve_blended_alpha(
                class_name,
                float(plan["blended_alpha_square"]),
                float(plan["blended_alpha_hello_kitty"]),
            )
            x = poisoned_tensor_from_source(
                source_face_path=sample["source_face_path"],
                trigger_rgb=trigger_rgb,
                mask_rgb=mask_rgb,
                class_name=class_name,
                blended_alpha=alpha,
                transform=transform,
            )

            img_rng = _mask_rng_for_image(
                int(patch_cfg["RANDOM_SEED"]), image_id
            )
            patch_masks = bernoulli_patch_masks(
                rows, cols, n_masks, keep_prob, img_rng
            )
            # Reference all-ones first in emission order after random masks.
            all_masks: List[Tuple[str, np.ndarray]] = [
                (f"mask_{j:03d}", pm) for j, pm in enumerate(patch_masks)
            ]
            all_masks.append(
                (REFERENCE_MASK_ID, np.ones((rows, cols), dtype=np.int64))
            )

            for mask_id, b_ij in all_masks:
                spatial = patch_mask_to_spatial(b_ij, image_size, image_size)
                masked = apply_mask_after_norm(x, spatial, baseline)

                if mask_id == REFERENCE_MASK_ID:
                    diff = (masked - x).abs().max().item()
                    max_ref_diff = max(max_ref_diff, diff)

                mask_dir = os.path.join(masks_root, rel_id)
                tensor_dir = os.path.join(tensors_root, rel_id)
                os.makedirs(mask_dir, exist_ok=True)
                os.makedirs(tensor_dir, exist_ok=True)

                mask_path = os.path.join(mask_dir, f"{mask_id}.json")
                tensor_path = os.path.join(tensor_dir, f"{mask_id}.pt")
                with open(mask_path, "w", encoding="utf-8") as jf:
                    json.dump(
                        {
                            "image_id": image_id,
                            "mask_id": mask_id,
                            "grid_rows": rows,
                            "grid_cols": cols,
                            "binary_patch_mask": b_ij.tolist(),
                        },
                        jf,
                        indent=2,
                        ensure_ascii=False,
                    )
                torch.save(masked.cpu(), tensor_path)

                preview_path = None
                if write_previews:
                    preview_dir = os.path.join(previews_root, rel_id)
                    os.makedirs(preview_dir, exist_ok=True)
                    preview_path = os.path.join(preview_dir, f"{mask_id}.png")
                    Image.fromarray(denormalize_to_uint8(masked)).save(preview_path)

                row = {
                    "image_id": image_id,
                    "original_image_path": sample["original_image_path"],
                    "source_face_path": sample["source_face_path"],
                    "poisoned_basename": sample["poisoned_basename"],
                    "mask_id": mask_id,
                    "binary_patch_mask": b_ij.tolist(),
                    "mask_path": mask_path,
                    "masked_tensor_path": tensor_path,
                    "preview_path": preview_path,
                    "node": node_name,
                    "attack_class": class_name,
                    "class_folder": class_folder,
                }
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"[INFO] wrote {n_rows} manifest rows -> {manifest_path}")
    print(f"[INFO] reference_all_ones max abs diff vs poison tensor: {max_ref_diff:.6e}")
    if max_ref_diff > 1e-5:
        print("[WARN] reference mask should match unmasked poison tensor")

    result.update(
        {
            "manifest_path": manifest_path,
            "n_manifest_rows": n_rows,
            "max_reference_abs_diff": max_ref_diff,
            "config_resolved_path": os.path.join(out_dir, "config_resolved.json"),
            "selected_samples_path": os.path.join(out_dir, "selected_samples.json"),
        }
    )
    print(f"[INFO] patch data at: {out_dir}")
    return result


def cleanup_patch_generated(
    run_dir: str,
    keep_manifest: bool = True,
) -> None:
    """
    Remove heavy artifacts under a patch_generated run directory.

    Keeps config_resolved.json, selected_samples.json, manifest.jsonl when
    keep_manifest=True. Deletes tensors/, previews/, and masks/.
    """
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        print(f"[INFO] Nothing to clean (missing): {run_dir}")
        return

    for sub in ("tensors", "previews", "masks"):
        path = os.path.join(run_dir, sub)
        if os.path.isdir(path):
            _safe_rmtree(path)
            print(f"[INFO] Removed {path}")

    if not keep_manifest:
        for name in (
            "config_resolved.json",
            "selected_samples.json",
            "manifest.jsonl",
        ):
            path = os.path.join(run_dir, name)
            if os.path.isfile(path):
                os.remove(path)
                print(f"[INFO] Removed {path}")

    print(f"[INFO] cleanup done for {run_dir} (keep_manifest={keep_manifest})")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate PNC-style patch perturbation data for a node/attack class "
            "from a training_report run."
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
        default="",
        help="Specific training_report run dir (default: latest).",
    )
    p.add_argument("--node", type=str, required=True, help='e.g. "node_1"')
    p.add_argument(
        "--attack_class",
        type=str,
        required=True,
        help='Logical name e.g. "big_hello_kitty" or folder "01_big_hello_kitty"',
    )
    p.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG,
        help="Path to patch_data_config.json",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default=_DEFAULT_OUTPUT_ROOT,
        help="Root for patch_generated/<run>_<node>_<attack>/",
    )
    p.add_argument(
        "--sample_ratio",
        type=float,
        default=None,
        help="Override SAMPLE_RATIO (fraction in (0,1]).",
    )
    p.add_argument("--grid_rows", type=int, default=None)
    p.add_argument("--grid_cols", type=int, default=None)
    p.add_argument("--num_masks_per_image", type=int, default=None)
    p.add_argument("--keep_prob", type=float, default=None)
    p.add_argument("--random_seed", type=int, default=None)
    p.add_argument(
        "--mask_baseline",
        type=str,
        default=None,
        choices=["imagenet_mean_normalized", "channel_constants"],
    )
    p.add_argument(
        "--write_previews",
        action="store_true",
        help="Also write denormalized PNG previews.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only collect/subsample; do not write tensors.",
    )
    p.add_argument(
        "--cleanup",
        type=str,
        default="",
        help="If set, cleanup this patch_generated run dir and exit.",
    )
    p.add_argument(
        "--cleanup_drop_manifest",
        action="store_true",
        help="With --cleanup, also delete JSON manifests.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.cleanup:
        cleanup_patch_generated(
            args.cleanup,
            keep_manifest=not args.cleanup_drop_manifest,
        )
        return

    overrides: Dict[str, Any] = {
        "SAMPLE_RATIO": args.sample_ratio,
        "GRID_ROWS": args.grid_rows,
        "GRID_COLS": args.grid_cols,
        "NUM_MASKS_PER_IMAGE": args.num_masks_per_image,
        "KEEP_PROB": args.keep_prob,
        "RANDOM_SEED": args.random_seed,
        "MASK_BASELINE": args.mask_baseline,
    }
    if args.write_previews:
        overrides["WRITE_PREVIEWS"] = True

    patch_cfg = load_patch_config(args.config, overrides=overrides)
    run_dir = args.run_dir.strip() or None

    generate_patch_dataset(
        node_name=args.node,
        attack_class=args.attack_class,
        patch_cfg=patch_cfg,
        training_note_root=args.training_note_root,
        run_dir=run_dir,
        output_root=args.output_root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
