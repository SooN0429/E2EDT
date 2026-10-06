#!/usr/bin/env python3
"""Simulate attack-transfer matching from method1 differential vectors.

Examples:
  cd E2EDT/matching
  source ../.venv/bin/activate
  python diffvector_matching.py --target_node node_2
  # 可選：
  #   --tau_same 0.7 # 相似度門檻，高於多少視為已具備，低於多少視為新穎（預設0.7）
  #   --run_dir <diffvector_results 某 run> # 未指定，則預設使用最新訓練方案
  #   --matching_root matching_result # 輸出根目錄，預設 matching_result

Coverage(candidate -> target) = max cosine similarity against the target node's
attack set D_i (matches the experiment report). Candidates with Coverage <
tau_same are treated as novel; the novel candidate with minimal Coverage is the
primary transfer attack / node.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    plt = None
    _HAS_MPL = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RESULTS_ROOT = os.path.join(_SCRIPT_DIR, "diffvector_results")
_DEFAULT_MATCHING_ROOT = os.path.join(_SCRIPT_DIR, "matching_result")
METHOD1_NAME = "method1_mean_diff"
MATCHING_OUT_SUBDIR = "matching_method1"


def find_latest_diffvector_results(results_root: str) -> str:
    """Return newest run dir under results_root that has meta.json with method1."""
    root = os.path.abspath(results_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"results_root not found: {root}")

    candidates: List[Tuple[str, str]] = []
    for name in os.listdir(root):
        run_dir = os.path.join(root, name)
        meta_path = os.path.join(run_dir, "meta.json")
        if not os.path.isdir(run_dir) or not os.path.isfile(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not meta.get("method1"):
            continue
        created = meta.get("created_at") or ""
        if not created:
            created = datetime.fromtimestamp(os.path.getmtime(run_dir)).isoformat(
                timespec="seconds"
            )
        candidates.append((created, run_dir))

    if not candidates:
        raise FileNotFoundError(
            f"No diffvector_results with method1 under {root}"
        )
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _label(node: str, class_folder: str) -> str:
    return f"{node}/{class_folder}"


def _node_from_label(label: str) -> str:
    return label.split("/", 1)[0]


def load_method1_vectors(
    run_dir: str,
    method_name: str = METHOD1_NAME,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load attack differential vectors.

    Returns (sorted labels, {label: vector}, meta).
    """
    run_dir = os.path.abspath(run_dir)
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    vectors: Dict[str, np.ndarray] = {}
    method_block = meta.get("method1") or {}
    nodes = method_block.get("nodes") or {}

    for node_name, node_info in nodes.items():
        classes = (node_info or {}).get("classes") or {}
        for class_folder, cls_info in classes.items():
            path = (cls_info or {}).get("path") or ""
            if not path or not os.path.isfile(path):
                # Fallback relative to run_dir
                path = os.path.join(
                    run_dir, method_name, node_name, f"{class_folder}.npy"
                )
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing vector for {node_name}/{class_folder}: {path}"
                )
            vec = np.load(path).astype(np.float64).reshape(-1)
            vectors[_label(node_name, class_folder)] = vec

    if not vectors:
        # Directory scan fallback
        method_root = os.path.join(run_dir, method_name)
        if not os.path.isdir(method_root):
            raise RuntimeError(f"No method1 vectors under {run_dir}")
        for node_name in sorted(os.listdir(method_root)):
            node_dir = os.path.join(method_root, node_name)
            if not os.path.isdir(node_dir):
                continue
            for fname in sorted(os.listdir(node_dir)):
                if not fname.endswith(".npy"):
                    continue
                class_folder = fname[: -len(".npy")]
                path = os.path.join(node_dir, fname)
                vectors[_label(node_name, class_folder)] = (
                    np.load(path).astype(np.float64).reshape(-1)
                )

    if not vectors:
        raise RuntimeError(f"No attack differential vectors found in {run_dir}")

    labels = sorted(vectors.keys())
    return labels, vectors, meta


def compute_cosine_matrix(
    labels: Sequence[str],
    vectors: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return (N, N) cosine similarity matrix aligned with labels order."""
    mats = []
    for lab in labels:
        v = vectors[lab].astype(np.float64).reshape(-1)
        n = np.linalg.norm(v)
        if n < 1e-12:
            raise ValueError(f"Zero-norm differential vector: {lab}")
        mats.append(v / n)
    X = np.stack(mats, axis=0)
    return X @ X.T


def coverage_against_target(
    candidate_label: str,
    target_labels: Sequence[str],
    matrix: np.ndarray,
    labels: Sequence[str],
) -> Tuple[float, Dict[str, float]]:
    """Coverage = max cosine(candidate, t) for t in target_labels."""
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    ci = label_to_idx[candidate_label]
    per_target: Dict[str, float] = {}
    best = -np.inf
    for tlab in target_labels:
        ti = label_to_idx[tlab]
        sim = float(matrix[ci, ti])
        per_target[tlab] = sim
        if sim > best:
            best = sim
    return float(best), per_target


def select_primary_transfer(
    target_node: str,
    labels: Sequence[str],
    matrix: np.ndarray,
    tau_same: float,
) -> Dict[str, Any]:
    target_attacks = [lab for lab in labels if _node_from_label(lab) == target_node]
    if not target_attacks:
        raise ValueError(
            f"target_node={target_node!r} has no attack vectors in results. "
            f"Available labels: {list(labels)}"
        )

    candidates = [lab for lab in labels if _node_from_label(lab) != target_node]
    if not candidates:
        raise ValueError(
            f"No candidate attacks outside target_node={target_node!r}"
        )

    ranking: List[Dict[str, Any]] = []
    for clab in candidates:
        cov, per_target = coverage_against_target(
            clab, target_attacks, matrix, labels
        )
        ranking.append(
            {
                "label": clab,
                "node": _node_from_label(clab),
                "coverage": cov,
                "is_novel": bool(cov < tau_same),
                "cosine_to_target_attacks": per_target,
            }
        )
    ranking.sort(key=lambda r: (r["coverage"], r["label"]))

    novel = [r for r in ranking if r["is_novel"]]
    if novel:
        best = novel[0]
        primary_node = best["node"]
        primary_attack = best["label"]
        novelty_score = best["coverage"]
    else:
        primary_node = None
        primary_attack = None
        novelty_score = None

    return {
        "target_node": target_node,
        "target_attacks": target_attacks,
        "tau_same": float(tau_same),
        "primary_transfer_node": primary_node,
        "primary_transfer_attack": primary_attack,
        "novelty_score": novelty_score,
        "coverage": novelty_score,
        "candidate_ranking": ranking,
    }


def plot_similarity_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    out_path: str,
    run_name: str,
    target_node: str,
    tau_same: float,
    primary_attack: Optional[str] = None,
) -> Optional[str]:
    if not _HAS_MPL:
        print("[WARN] matplotlib not available; skip similarity_heatmap.png")
        return None

    n = len(labels)
    fig_w = max(6.0, 0.9 * n + 3.0)
    fig_h = max(5.0, 0.8 * n + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(list(labels), rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(list(labels), fontsize=8)
    ax.set_title(
        f"Attack diffvector cosine\nrun={run_name}\n"
        f"target={target_node}  tau_same={tau_same}"
    )
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    if primary_attack and primary_attack in labels:
        pi = list(labels).index(primary_attack)
        # Highlight primary row/col lightly
        ax.add_patch(
            plt.Rectangle(
                (pi - 0.5, -0.5),
                1,
                n,
                fill=False,
                edgecolor="green",
                linewidth=1.5,
                linestyle="--",
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (-0.5, pi - 0.5),
                n,
                1,
                fill=False,
                edgecolor="green",
                linewidth=1.5,
                linestyle="--",
            )
        )
        ax.text(
            0.0,
            -0.12,
            f"primary transfer attack: {primary_attack}",
            transform=ax.transAxes,
            fontsize=8,
            color="green",
        )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] saved heatmap -> {out_path}")
    return out_path


def _write_pairing_summary(path: str, result: Dict[str, Any], run_name: str) -> None:
    lines = [
        f"# Matching summary (`{run_name}`)",
        "",
        f"- method: `{result.get('method')}`",
        f"- target_node: `{result['target_node']}`",
        f"- target_attacks: {', '.join(f'`{x}`' for x in result['target_attacks'])}",
        f"- tau_same: `{result['tau_same']}`",
        "",
        "## Primary transfer",
        "",
    ]
    if result.get("primary_transfer_node"):
        lines.extend(
            [
                f"- primary_transfer_node: **`{result['primary_transfer_node']}`**",
                f"- primary_transfer_attack: **`{result['primary_transfer_attack']}`**",
                f"- coverage (novelty_score): `{result['novelty_score']:.6f}` "
                f"(lower = more novel; threshold `{result['tau_same']}`)",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- primary_transfer_node: `null`",
                "- reason: no candidate with Coverage < tau_same "
                "(all attacks already considered covered)",
                "",
            ]
        )

    lines.extend(
        [
            "## Candidate ranking (by Coverage ascending)",
            "",
            "| attack | node | Coverage | novel? |",
            "|---|---|---:|:---:|",
        ]
    )
    for row in result["candidate_ranking"]:
        novel = "yes" if row["is_novel"] else "no"
        lines.append(
            f"| `{row['label']}` | `{row['node']}` | "
            f"{row['coverage']:.6f} | {novel} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] saved summary -> {path}")


def run_matching(
    target_node: str,
    results_root: str = _DEFAULT_RESULTS_ROOT,
    run_dir: Optional[str] = None,
    matching_root: str = _DEFAULT_MATCHING_ROOT,
    method: str = METHOD1_NAME,
    tau_same: float = 0.7,
) -> Dict[str, Any]:
    if method not in (METHOD1_NAME, "method1"):
        raise ValueError(
            f"Unsupported method={method!r}; only {METHOD1_NAME} is implemented"
        )
    method = METHOD1_NAME

    run_dir = os.path.abspath(run_dir) if run_dir else find_latest_diffvector_results(
        results_root
    )
    matching_root = os.path.abspath(matching_root)
    print(f"[INFO] results run_dir={run_dir}")
    print(f"[INFO] target_node={target_node} tau_same={tau_same} method={method}")

    labels, vectors, meta = load_method1_vectors(run_dir, method_name=method)
    matrix = compute_cosine_matrix(labels, vectors)
    select = select_primary_transfer(target_node, labels, matrix, tau_same)

    run_name = meta.get("run_name") or os.path.basename(run_dir)
    out_dir = os.path.join(matching_root, run_name, MATCHING_OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    mat_path = os.path.join(out_dir, "similarity_matrix.npy")
    np.save(mat_path, matrix.astype(np.float64))

    labels_path = os.path.join(out_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, indent=2, ensure_ascii=False)

    heat_path = os.path.join(out_dir, "similarity_heatmap.png")
    plot_similarity_heatmap(
        matrix=matrix,
        labels=labels,
        out_path=heat_path,
        run_name=run_name,
        target_node=target_node,
        tau_same=tau_same,
        primary_attack=select.get("primary_transfer_attack"),
    )

    pairing = {
        **select,
        "source_results_dir": run_dir,
        "matching_output_dir": out_dir,
        "method": method,
        "run_name": run_name,
        "similarity_matrix_path": mat_path,
        "labels_path": labels_path,
        "heatmap_path": heat_path if os.path.isfile(heat_path) else None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    pairing_path = os.path.join(out_dir, "pairing_result.json")
    with open(pairing_path, "w", encoding="utf-8") as f:
        json.dump(pairing, f, indent=2, ensure_ascii=False)
    print(f"[INFO] saved pairing -> {pairing_path}")

    summary_path = os.path.join(out_dir, "pairing_summary.md")
    _write_pairing_summary(
        summary_path, pairing, pairing["run_name"]
    )

    if pairing["primary_transfer_node"]:
        print(
            f"[INFO] primary transfer: node={pairing['primary_transfer_node']} "
            f"attack={pairing['primary_transfer_attack']} "
            f"coverage={pairing['novelty_score']:.6f}"
        )
    else:
        print("[INFO] no novel candidate under current tau_same")

    return pairing


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Match / rank attack differential vectors (method1) by Coverage "
            "(max cosine vs target attacks) and tau_same novelty threshold."
        )
    )
    p.add_argument(
        "--target_node",
        type=str,
        required=True,
        help="Target node name, e.g. node_2",
    )
    p.add_argument(
        "--tau_same",
        type=float,
        default=0.7,
        help="Coverage >= tau_same => already covered; < => novel (default 0.7).",
    )
    p.add_argument(
        "--results_root",
        type=str,
        default=_DEFAULT_RESULTS_ROOT,
        help="Root of diffvector_results/.",
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Optional specific diffvector_results run dir (default: latest).",
    )
    p.add_argument(
        "--matching_root",
        type=str,
        default=_DEFAULT_MATCHING_ROOT,
        help="Root for matching outputs (default: matching_result/).",
    )
    p.add_argument(
        "--method",
        type=str,
        default=METHOD1_NAME,
        help=f"Differential-vector method (default: {METHOD1_NAME}).",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_matching(
        target_node=args.target_node.strip(),
        results_root=os.path.abspath(args.results_root),
        run_dir=args.run_dir or None,
        matching_root=os.path.abspath(args.matching_root),
        method=args.method,
        tau_same=float(args.tau_same),
    )


if __name__ == "__main__":
    main()
