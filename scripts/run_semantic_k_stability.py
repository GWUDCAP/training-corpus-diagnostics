#!/usr/bin/env python3
# ===== IMPORTS ===== #
## ===== STDLIB ===== ##
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

## ===== 3RD-PARTY ===== ##
import numpy as np
from sklearn.cluster import KMeans

# ===== GLOBALS ===== #

ROOT = Path(__file__).resolve().parents[1]

# ===== FUNCTIONS ===== #
## ===== IO ===== ##

def read_json(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

## ===== METRICS ===== ##

def normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    total = float(arr.sum())
    return arr / total if total > 0 else np.full(arr.shape, 1.0 / max(1, arr.size))

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p1 = normalize(np.asarray(p, dtype=np.float64) + eps)
    q1 = normalize(np.asarray(q, dtype=np.float64) + eps)
    m = 0.5 * (p1 + q1)
    return float(0.5 * np.sum(p1 * np.log(p1 / m)) + 0.5 * np.sum(q1 * np.log(q1 / m)))

def cluster_js(a: np.ndarray, b: np.ndarray, *, k: int, seed: int) -> float:
    pooled = np.concatenate([a, b], axis=0)
    n_clusters = max(2, min(k, pooled.shape[0]))
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(pooled)
    left = np.bincount(labels[: a.shape[0]], minlength=n_clusters).astype(np.float64)
    right = np.bincount(labels[a.shape[0]:], minlength=n_clusters).astype(np.float64)
    return js_divergence(left, right)

def mean(xs: list[float]) -> float: return float(np.mean(xs))

## ===== RENDER ===== ##

def render_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Semantic k Stability",
        "",
        "This stability check recomputes the retained pairwise semantic cluster JS summary from the cached GTE c768 embeddings while varying the pooled pairwise cluster count.",
        "",
        "| k | Broad-broad cluster JS | Broad-targeted cluster JS | Ordering held |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(f"| {row['k']} | {row['bb_cluster_js']:.4f} | {row['bt_cluster_js']:.4f} | `{row['ordering_held']}` |")
    held = sum(1 for row in summary["rows"] if row["ordering_held"])
    lines += [
        "",
        f"The broad-targeted semantic cluster JS exceeds broad-broad cluster JS for `{held}/{len(summary['rows'])}` tested k values.",
    ]
    return "\n".join(lines) + "\n"

## ===== ENTRYPOINT ===== ##

def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute semantic cluster JS family summaries across k values.")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/corpus_manifest.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "stability")
    ap.add_argument("--k", type=int, action="append", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = read_json(args.manifest)
    base = args.manifest.parent
    corpora = manifest["corpora"]
    roles = {row["label"]: row["role"] for row in corpora}
    labels = [row["label"] for row in corpora]
    broad = [label for label in labels if roles[label] == "broad"]
    targeted = [label for label in labels if roles[label] == "targeted"]
    embeddings = {row["label"]: np.load(base / row["embeddings"]).astype(np.float32) for row in corpora}
    bb_pairs = list(combinations(broad, 2))
    bt_pairs = [(a, b) for a in broad for b in targeted]

    rows: list[dict[str, Any]] = []
    for k in args.k or [12, 16, 24, 32, 48]:
        bb = mean([cluster_js(embeddings[a], embeddings[b], k=k, seed=args.seed) for a, b in bb_pairs])
        bt = mean([cluster_js(embeddings[a], embeddings[b], k=k, seed=args.seed) for a, b in bt_pairs])
        rows.append({"k": k, "bb_cluster_js": bb, "bt_cluster_js": bt, "ordering_held": bt > bb})

    summary = {
        "schema": "lens_effects.semantic_k_stability.v1",
        "manifest": str(args.manifest),
        "seed": args.seed,
        "k_values": [row["k"] for row in rows],
        "rows": rows,
    }
    write_json(args.out_dir / "semantic_k_stability.json", summary)
    (args.out_dir / "semantic_k_stability.md").write_text(render_md(summary), encoding="utf-8")
    print(f"[done] wrote semantic k stability to {args.out_dir}")

if __name__ == "__main__":
    main()
