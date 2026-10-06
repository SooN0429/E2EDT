#!/usr/bin/env python3
"""Generate per-node classification datasets (MaskBlended) and extract train features.

All attack classes use a companion mask image (VBD MaskBlended style):
  out = img*(1-mask) + (1-a)*(img*mask) + a*(trigger*mask)
  mask from *_mask.png with (R+G+B) != 0.

Square/grid classes (ABS_PATCH_CLASSES) keep an absolute patch size on the
poisoned image (VBD-style 3x3), independent of the trigger PNG canvas (32/64).
Hello-kitty classes still full-frame resize to the host image.
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRIGGER_FILE_MAP = {
    "white_square": "trigger_white_square.png",
    "green_square": "trigger_green_square.png",
    "white_grid": "trigger_white_grid.png",
    "color_grid": "trigger_color_grid.png",
    "big_hello_kitty": "big_hello_kitty.png",
    "small_hello_kitty": "small_hello_kitty.png",
}

# Independent masks for every attack class (VBD-style). Required; no auto-mask fallback.
TRIGGER_MASK_MAP = {
    "white_square": "trigger_white_square_mask.png",
    "green_square": "trigger_green_square_mask.png",
    "white_grid": "trigger_white_grid_mask.png",
    "color_grid": "trigger_color_grid_mask.png",
    "big_hello_kitty": "big_hello_kitty_mask.png",
    "small_hello_kitty": "small_hello_kitty_mask.png",
}

# Paste asset patch at absolute pixel size (no proportional upscale). Matches VBD
# square/grid triggers that stay 3x3 on both CIFAR-10 and Tiny-ImageNet.
ABS_PATCH_CLASSES = frozenset(
    {"white_square", "green_square", "white_grid", "color_grid"}
)
EXPECTED_ABS_PATCH_HW = (3, 3)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_node_specs(node_specs: str) -> Dict[str, List[str]]:
    """
    Parse 'node_2=white_square,clean;node_3=white_square,green_square,clean'
    into {node_name: [class, ...]}.
    """
    specs: Dict[str, List[str]] = {}
    raw = node_specs.strip()
    if not raw:
        raise ValueError("node_specs is empty")

    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid node_specs segment (missing '='): {part}")
        node_name, classes_str = part.split("=", 1)
        node_name = node_name.strip()
        classes = [c.strip() for c in classes_str.split(",") if c.strip()]
        if not node_name or not classes:
            raise ValueError(f"Invalid node_specs segment: {part}")
        if len(classes) != len(set(classes)):
            raise ValueError(f"Duplicate class names for {node_name}: {classes}")
        specs[node_name] = classes
    if not specs:
        raise ValueError("No valid node specs parsed")
    return specs


def class_folder_name(class_idx: int, class_name: str) -> str:
    """Zero-padded prefix keeps ImageFolder order identical to spec order."""
    return f"{class_idx:02d}_{class_name}"


def list_images(folder: str) -> List[str]:
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            files.append(path)
    files.sort()
    return files


def list_person_dirs(split_dir: str) -> List[str]:
    persons = [
        os.path.join(split_dir, name)
        for name in sorted(os.listdir(split_dir))
        if os.path.isdir(os.path.join(split_dir, name))
    ]
    if not persons:
        raise RuntimeError(f"No person folders under {split_dir}")
    return persons


def compute_per_person_quota(
    dataset_root: str,
    node_names: Sequence[str],
    c_max: int,
    split: str,
) -> int:
    """k = min over all selected nodes/persons of floor(n_images / C_max)."""
    if c_max < 1:
        raise ValueError("c_max must be >= 1")
    quotas: List[int] = []
    for node_name in node_names:
        split_dir = os.path.join(dataset_root, node_name, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for person_dir in list_person_dirs(split_dir):
            n = len(list_images(person_dir))
            quotas.append(n // c_max)
    if not quotas:
        raise RuntimeError(f"No images found for split={split}")
    k = min(quotas)
    if k < 1:
        raise RuntimeError(
            f"Per-person quota is 0 for split={split} with C_max={c_max}; "
            "not enough images per person."
        )
    return k


def load_trigger_arrays(
    trigger_dir: str,
    class_names: Sequence[str],
) -> Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    For each attack class, return (trigger_rgb_uint8_or_None, mask_rgb_uint8_or_None).
    Every non-clean class requires both TRIGGER_FILE_MAP and TRIGGER_MASK_MAP entries.
    Resize is deferred to apply_maskblended.
    """
    loaded: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
    for name in class_names:
        if name == "clean":
            loaded[name] = (None, None)
            continue
        if name not in TRIGGER_FILE_MAP:
            raise KeyError(
                f"Unknown trigger class '{name}'. "
                f"Supported: clean, {', '.join(TRIGGER_FILE_MAP)}"
            )
        if name not in TRIGGER_MASK_MAP:
            raise KeyError(
                f"No mask mapping for '{name}'. "
                f"Add an entry to TRIGGER_MASK_MAP."
            )
        path = os.path.join(trigger_dir, TRIGGER_FILE_MAP[name])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Trigger image not found: {path}")
        mask_path = os.path.join(trigger_dir, TRIGGER_MASK_MAP[name])
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f"Mask required for '{name}' but not found: {mask_path}"
            )
        trigger = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        loaded[name] = (trigger, mask_rgb)
    return loaded


def _prepare_absolute_corner_patch(
    trigger_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    size_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build HxW trigger/mask by pasting the asset mask bbox at the bottom-right
    without scaling (VBD absolute 3x3 square/grid behaviour).
    """
    h, w = size_hw
    nonzero = np.argwhere(mask_rgb.sum(axis=2) != 0)
    if nonzero.size == 0:
        raise ValueError("Absolute-patch mask has no nonzero pixels")
    y0, x0 = nonzero.min(axis=0)
    y1, x1 = nonzero.max(axis=0)
    ph, pw = int(y1 - y0 + 1), int(x1 - x0 + 1)
    if (ph, pw) != EXPECTED_ABS_PATCH_HW:
        raise ValueError(
            f"Expected absolute patch {EXPECTED_ABS_PATCH_HW}, got {(ph, pw)} "
            f"from mask bbox [y{y0}:{y1}, x{x0}:{x1}]"
        )
    if ph > h or pw > w:
        raise ValueError(
            f"Absolute patch {(ph, pw)} does not fit host image {(h, w)}"
        )

    patch_t = trigger_rgb[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    patch_m = mask_rgb[y0 : y1 + 1, x0 : x1 + 1]
    mask_2d = (patch_m.sum(axis=2) != 0).astype(np.float32)

    trigger = np.zeros((h, w, 3), dtype=np.float32)
    mask = np.zeros((h, w, 3), dtype=np.float32)
    trigger[h - ph : h, w - pw : w] = patch_t
    mask[h - ph : h, w - pw : w] = np.stack([mask_2d, mask_2d, mask_2d], axis=-1)
    return trigger, mask


def prepare_trigger_for_size(
    trigger_rgb: np.ndarray,
    size_hw: Tuple[int, int],
    mask_rgb: np.ndarray,
    class_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare trigger and binary mask for host size (H, W).

    Absolute-patch classes keep asset pixel size (typically 3x3) at the
    bottom-right corner. Other classes full-frame BILINEAR resize (VBD kitty).
    Binary mask is VBD-style: (R+G+B) != 0.
    """
    if class_name in ABS_PATCH_CLASSES:
        return _prepare_absolute_corner_patch(trigger_rgb, mask_rgb, size_hw)

    h, w = size_hw
    trigger_img = Image.fromarray(trigger_rgb).resize((w, h), Image.BILINEAR)
    trigger = np.array(trigger_img, dtype=np.float32)

    mask_img = Image.fromarray(mask_rgb).resize((w, h), Image.BILINEAR)
    mask_arr = np.array(mask_img, dtype=np.float32)
    mask_2d = (mask_arr.sum(axis=2) != 0).astype(np.float32)
    mask = np.stack([mask_2d, mask_2d, mask_2d], axis=-1)
    return trigger, mask


def apply_maskblended(
    img_rgb: np.ndarray,
    trigger_rgb: np.ndarray,
    alpha: float,
    mask_rgb: np.ndarray,
    class_name: Optional[str] = None,
) -> np.ndarray:
    """
    MaskBlended (VBD formula):
    out = img*(1-mask) + (1-a)*(img*mask) + a*(trigger*mask)
    """
    h, w = img_rgb.shape[:2]
    trigger, mask = prepare_trigger_for_size(
        trigger_rgb, (h, w), mask_rgb=mask_rgb, class_name=class_name
    )
    img = img_rgb.astype(np.float32)
    out = (
        img * (1.0 - mask)
        + (1.0 - alpha) * (img * mask)
        + alpha * (trigger * mask)
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def _safe_rmtree(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)


def generate_split_for_node(
    node_src_root: str,
    node_out_root: str,
    split: str,
    class_names: Sequence[str],
    triggers: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
    per_person_k: int,
    blended_alpha: float,
    rng: np.random.RandomState,
) -> Dict[str, int]:
    """Create ImageFolder-style split under node_out_root/split/{idx_class}/..."""
    split_src = os.path.join(node_src_root, split)
    split_out = os.path.join(node_out_root, split)
    _safe_rmtree(split_out)

    class_dirs = []
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(split_out, class_folder_name(idx, cname))
        os.makedirs(cdir, exist_ok=True)
        class_dirs.append(cdir)

    counts = {cname: 0 for cname in class_names}
    persons = list_person_dirs(split_src)

    for person_dir in persons:
        images = list_images(person_dir)
        if len(images) < per_person_k * len(class_names):
            # Still allow if quota was computed globally; skip if this person is short.
            # Quota uses min across persons, so this should not happen.
            raise RuntimeError(
                f"{person_dir} has {len(images)} images, "
                f"need at least {per_person_k * len(class_names)}"
            )
        order = rng.permutation(len(images))
        selected = [images[i] for i in order[: per_person_k * len(class_names)]]

        person_name = os.path.basename(person_dir).replace(" ", "_")
        for class_idx, cname in enumerate(class_names):
            chunk = selected[class_idx * per_person_k : (class_idx + 1) * per_person_k]
            trigger_rgb, mask_rgb = triggers[cname]

            for src_path in chunk:
                img = np.array(Image.open(src_path).convert("RGB"), dtype=np.uint8)
                if cname == "clean" or trigger_rgb is None:
                    out_img = img
                else:
                    out_img = apply_maskblended(
                        img,
                        trigger_rgb,
                        blended_alpha,
                        mask_rgb=mask_rgb,
                        class_name=cname,
                    )

                # Flat ImageFolder layout: split/class_dir/*.jpg (no person subdirs)
                base = os.path.splitext(os.path.basename(src_path))[0]
                out_name = f"{person_name}_{base}_{cname}.jpg"
                Image.fromarray(out_img).save(
                    os.path.join(class_dirs[class_idx], out_name), quality=95
                )
                counts[cname] += 1

    return counts


class ResNet18Extractor(nn.Module):
    """ResNet-18 intermediate feature extractor (1_point ~ 9_point)."""

    SUPPORTED = {
        "1_point", "2_point", "3_point", "4_point", "5_point",
        "6_point", "7_point", "8_point", "9_point",
    }

    def __init__(self, extracted_layer: str, pretrained: bool = True):
        super().__init__()
        if extracted_layer not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported extracted_layer={extracted_layer}, "
                f"must be one of {sorted(self.SUPPORTED)}"
            )
        backbone = models.resnet18(pretrained=pretrained)
        self.extracted_layer = extracted_layer
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.extracted_layer == "1_point":
            return x

        x = self.layer1[0](x)
        if self.extracted_layer == "2_point":
            return x

        x = self.layer1[1:](x)
        if self.extracted_layer == "3_point":
            return x

        x = self.layer2[0](x)
        if self.extracted_layer == "4_point":
            return x

        x = self.layer2[1:](x)
        if self.extracted_layer == "5_point":
            return x

        x = self.layer3[0](x)
        if self.extracted_layer == "6_point":
            return x

        x = self.layer3[1:](x)
        if self.extracted_layer == "7_point":
            return x

        x = self.layer4[0](x)
        if self.extracted_layer == "8_point":
            return x

        x = self.layer4[1:](x)
        x = self.avgpool(x)
        return x


def extract_train_features(
    train_image_root: str,
    features_dir: str,
    extracted_layer: str,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[str, str]:
    """
    Extract 4D features (pooling=none) from ImageFolder train split.
    Saves source_train_feature.npy and source_train_feature_label.npy.
    """
    os.makedirs(features_dir, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    dataset = datasets.ImageFolder(root=train_image_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    extractor = ResNet18Extractor(extracted_layer=extracted_layer, pretrained=True)
    extractor = extractor.to(device)
    extractor.eval()

    all_feats: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = extractor(images)
            all_feats.append(feats.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    if not all_feats:
        raise RuntimeError(f"No features extracted from {train_image_root}")

    features = np.concatenate(all_feats, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)

    feat_path = os.path.join(features_dir, "source_train_feature.npy")
    label_path = os.path.join(features_dir, "source_train_feature_label.npy")
    np.save(feat_path, features)
    np.save(label_path, labels)
    print(
        f"[INFO] Saved features {features.shape} labels {labels.shape} "
        f"class_to_idx={dataset.class_to_idx}"
    )
    return feat_path, label_path


def generate_all_nodes(
    dataset_root: str,
    node_specs: Dict[str, List[str]],
    trigger_dir: str,
    output_root: str,
    blended_alpha: float,
    extracted_layer: str,
    seed: int = 0,
    device: Optional[torch.device] = None,
    feature_batch_size: int = 32,
    extract_features: bool = True,
) -> Dict[str, dict]:
    """
    Generate datasets for all nodes with cross-node class-count alignment,
    optionally extract train features.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_names = list(node_specs.keys())
    c_max = max(len(classes) for classes in node_specs.values())
    print(f"[INFO] Selected nodes={node_names}, C_max={c_max}")

    quotas = {
        split: compute_per_person_quota(dataset_root, node_names, c_max, split)
        for split in SPLITS
    }
    print(f"[INFO] Per-person per-class quotas: {quotas}")

    os.makedirs(output_root, exist_ok=True)
    rng = np.random.RandomState(seed)
    results: Dict[str, dict] = {}

    for node_name, class_names in node_specs.items():
        node_src = os.path.join(dataset_root, node_name)
        if not os.path.isdir(node_src):
            raise FileNotFoundError(f"Node source not found: {node_src}")

        node_out = os.path.join(output_root, node_name)
        _safe_rmtree(node_out)
        os.makedirs(node_out, exist_ok=True)

        triggers = load_trigger_arrays(trigger_dir, class_names)
        split_counts = {}
        for split in SPLITS:
            counts = generate_split_for_node(
                node_src_root=node_src,
                node_out_root=node_out,
                split=split,
                class_names=class_names,
                triggers=triggers,
                per_person_k=quotas[split],
                blended_alpha=blended_alpha,
                rng=rng,
            )
            split_counts[split] = counts
            print(f"[INFO] {node_name}/{split} counts={counts}")

        feat_path = label_path = None
        if extract_features:
            feat_path, label_path = extract_train_features(
                train_image_root=os.path.join(node_out, "train"),
                features_dir=os.path.join(node_out, "features"),
                extracted_layer=extracted_layer,
                device=device,
                batch_size=feature_batch_size,
            )

        results[node_name] = {
            "output_dir": node_out,
            "classes": list(class_names),
            "n_class": len(class_names),
            "split_counts": split_counts,
            "feature_path": feat_path,
            "label_path": label_path,
            "quotas": quotas,
        }

    return results


def cleanup_generated_node(output_root: str, node_name: str) -> None:
    path = os.path.join(output_root, node_name)
    _safe_rmtree(path)
    print(f"[INFO] Removed generated data: {path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MaskBlended node classification datasets + train features."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument(
        "--node_specs",
        type=str,
        required=True,
        help='e.g. "node_2=white_square,clean;node_3=white_square,green_square,clean"',
    )
    parser.add_argument("--trigger_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--blended_alpha", type=float, default=0.2)
    parser.add_argument("--extracted_layer", type=str, default="7_point")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature_batch_size", type=int, default=32)
    parser.add_argument("--no_features", action="store_true")
    parser.add_argument("--gpu_id", type=str, default="cuda:0")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    dataset_root = os.path.abspath(args.dataset_root)
    trigger_dir = args.trigger_dir or os.path.join(dataset_root, "Attack_trigger_image")
    output_root = args.output_root or os.path.join(dataset_root, "generated")

    if torch.cuda.is_available() and args.gpu_id.startswith("cuda"):
        device = torch.device(args.gpu_id)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = parse_node_specs(args.node_specs)
    generate_all_nodes(
        dataset_root=dataset_root,
        node_specs=specs,
        trigger_dir=trigger_dir,
        output_root=output_root,
        blended_alpha=args.blended_alpha,
        extracted_layer=args.extracted_layer,
        seed=args.seed,
        device=device,
        feature_batch_size=args.feature_batch_size,
        extract_features=not args.no_features,
    )


if __name__ == "__main__":
    main()
