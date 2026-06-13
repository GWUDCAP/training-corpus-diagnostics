#!/usr/bin/env python3
# ===== IMPORTS ===== #
## ===== STDLIB ===== ##
import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

## ===== 3RD-PARTY ===== ##
import numpy as np


# ===== FUNCTIONS ===== #
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_embedding(manifest_base: Path, row: dict[str, Any]) -> np.ndarray:
    emb = row.get("embeddings")
    if isinstance(emb, dict):
        emb_path = emb.get("path")
    else:
        emb_path = emb
    if not emb_path:
        raise ValueError(f"manifest row {row.get('label')} has no embeddings path")
    path = Path(emb_path)
    if not path.is_absolute():
        path = manifest_base / path
    arr = np.load(path).astype(np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{path} is not a 2D embedding array")
    return arr


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def pair_class(role_a: str, role_b: str) -> str:
    roles = {role_a, role_b}
    if roles == {"broad"}:
        return "broad_broad"
    if roles == {"targeted"}:
        return "targeted_targeted"
    if roles == {"broad", "targeted"}:
        return "broad_targeted"
    return "candidate"


def main() -> None:
    ap = argparse.ArgumentParser(description="Continuous embedding-distance sensitivity check for near-ceiling cluster-JS pairs.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--corpus-matrix", type=Path, default=None, help="Optional corpus_matrix.json to attach semantic cluster-JS values.")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--projections", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = read_json(args.manifest)
    manifest_base = args.manifest.resolve().parent
    rows = manifest["corpora"]
    rng = np.random.default_rng(args.seed)

    labels = [row["label"] for row in rows]
    display = {row["label"]: row.get("display", row["label"]) for row in rows}
    role = {row["label"]: row.get("role", "candidate") for row in rows}
    embeddings = {row["label"]: l2_normalize(load_embedding(manifest_base, row)) for row in rows}

    dim = next(iter(embeddings.values())).shape[1]
    projections = rng.normal(size=(dim, args.projections))
    projections /= np.linalg.norm(projections, axis=0, keepdims=True)
    projected = {label: np.sort(embeddings[label] @ projections, axis=0) for label in labels}
    centroids = {label: embeddings[label].mean(axis=0) for label in labels}

    cluster_js = {}
    if args.corpus_matrix and args.corpus_matrix.exists():
        matrix = read_json(args.corpus_matrix)
        for row in matrix.get("pairs", []):
            key = tuple(sorted([row["a"], row["b"]]))
            cluster_js[key] = row.get("semantic_cluster_js")

    pair_rows = []
    for a, b in combinations(labels, 2):
        ca, cb = centroids[a], centroids[b]
        ca_norm = np.linalg.norm(ca) or 1.0
        cb_norm = np.linalg.norm(cb) or 1.0
        diffs = projected[a] - projected[b]
        pair_rows.append({
            "a": a,
            "b": b,
            "a_display": display[a],
            "b_display": display[b],
            "pair_class": pair_class(role[a], role[b]),
            "semantic_cluster_js": cluster_js.get(tuple(sorted([a, b]))),
            "centroid_cosine_distance": float(1.0 - np.dot(ca, cb) / (ca_norm * cb_norm)),
            "centroid_l2_distance": float(np.linalg.norm(ca - cb)),
            "sliced_wasserstein_l1": float(np.mean(np.abs(diffs))),
            "sliced_wasserstein_l2": float(np.sqrt(np.mean(diffs * diffs))),
        })

    metrics = ["centroid_cosine_distance", "centroid_l2_distance", "sliced_wasserstein_l1", "sliced_wasserstein_l2"]
    family_summary = {}
    for cls in ["broad_broad", "broad_targeted", "targeted_targeted"]:
        cls_rows = [row for row in pair_rows if row["pair_class"] == cls]
        family_summary[cls] = {"count": len(cls_rows)}
        for metric in metrics:
            family_summary[cls][metric] = float(np.mean([row[metric] for row in cls_rows])) if cls_rows else None

    near_ceiling = [row for row in pair_rows if row["semantic_cluster_js"] is not None and row["semantic_cluster_js"] >= 0.65]
    obj = {
        "schema": "lens_effects.continuous_embedding_sensitivity.v1",
        "manifest": str(args.manifest),
        "corpus_matrix": str(args.corpus_matrix) if args.corpus_matrix else None,
        "note": "Non-binned continuous embedding-distance check. Intended as a ceiling-sensitivity companion to semantic cluster JS, not as a bootstrapped headline family.",
        "settings": {"seed": args.seed, "projections": args.projections, "embedding_preprocessing": "row_l2_normalized"},
        "family_summary": family_summary,
        "near_ceiling_pairs_by_sliced_wasserstein_l1": sorted(near_ceiling, key=lambda row: row["sliced_wasserstein_l1"], reverse=True),
        "strongest_pairs_by_sliced_wasserstein_l1": sorted(pair_rows, key=lambda row: row["sliced_wasserstein_l1"], reverse=True)[:20],
        "pairs": pair_rows,
    }
    write_json(args.out_json, obj)

    md = [
        "# Continuous Embedding Sensitivity",
        "",
        "This non-binned check uses row-normalized cached embeddings and fixed random projections. It is a ceiling-sensitivity companion to semantic cluster JS, not a bootstrapped headline family.",
        "",
        "## Family Means",
        "",
        "| Pair class | Count | Centroid cosine | Centroid L2 | Sliced W1 | Sliced W2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cls in ["broad_broad", "broad_targeted", "targeted_targeted"]:
        row = family_summary[cls]
        md.append(f"| {cls} | {row['count']} | {row['centroid_cosine_distance']:.4f} | {row['centroid_l2_distance']:.4f} | {row['sliced_wasserstein_l1']:.4f} | {row['sliced_wasserstein_l2']:.4f} |")
    md += ["", "## Near-Ceiling Cluster-JS Pairs Ranked by Sliced W1", "", "| Pair | Cluster JS | Sliced W1 | Centroid cosine |", "|---|---:|---:|---:|"]
    for row in obj["near_ceiling_pairs_by_sliced_wasserstein_l1"][:20]:
        md.append(f"| {row['a_display']} vs {row['b_display']} | {row['semantic_cluster_js']:.4f} | {row['sliced_wasserstein_l1']:.4f} | {row['centroid_cosine_distance']:.4f} |")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.out_json}")
    print(f"[done] wrote {args.out_md}")


# ===== ENTRYPOINT ===== #
if __name__ == "__main__":
    main()
