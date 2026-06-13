#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans


VARIANTS = [
    ("gte_c256", "GTE c256", "Alibaba-NLP/gte-base-en-v1.5", 256, True, True),
    ("gte_c512", "GTE c512", "Alibaba-NLP/gte-base-en-v1.5", 512, True, True),
    ("gte_c768", "GTE c768", "Alibaba-NLP/gte-base-en-v1.5", 768, True, False),
    ("gte_c1024", "GTE c1024", "Alibaba-NLP/gte-base-en-v1.5", 1024, True, True),
    ("bge_c768", "BGE c768", "BAAI/bge-base-en-v1.5", 768, True, True),
    ("minilm_c768", "MiniLM c768", "sentence-transformers/all-MiniLM-L6-v2", 768, True, True),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def run(cmd: list[str]) -> None:
    print("[cmd] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    total = float(arr.sum())
    return arr / total if total > 0 else np.full(arr.shape, 1.0 / max(1, arr.size))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p1 = normalize(np.asarray(p, dtype=np.float64) + eps)
    q1 = normalize(np.asarray(q, dtype=np.float64) + eps)
    m = 0.5 * (p1 + q1)
    return float(0.5 * np.sum(p1 * np.log(p1 / m)) + 0.5 * np.sum(q1 * np.log(q1 / m)))


def covariance_spectrum(embeddings: np.ndarray) -> dict[str, float]:
    x = embeddings.astype(np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    denom = max(1, centered.shape[0] - 1)
    cov_like = (centered @ centered.T) / denom if centered.shape[0] < centered.shape[1] else (centered.T @ centered) / denom
    eigvals = np.flip(np.maximum(np.linalg.eigvalsh(cov_like), 0.0))
    total = float(eigvals.sum())
    return {
        "total_variance": total,
        "top1_share": float(eigvals[0] / total) if total > 0 else 0.0,
        "top5_share": float(eigvals[:5].sum() / total) if total > 0 else 0.0,
    }


def cluster_js(a: np.ndarray, b: np.ndarray, *, k: int, seed: int) -> float:
    pooled = np.concatenate([a, b], axis=0)
    n_clusters = max(2, min(k, pooled.shape[0]))
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(pooled)
    left = np.bincount(labels[: a.shape[0]], minlength=n_clusters).astype(np.float64)
    right = np.bincount(labels[a.shape[0]:], minlength=n_clusters).astype(np.float64)
    return js_divergence(left, right)


def resolve(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else base / path


def write_manifest_from_base(base_manifest: Path, variant_manifest: Path) -> None:
    base = read_json(base_manifest)
    base_dir = base_manifest.parent.resolve()
    rows = []
    for row in base["corpora"]:
        new = dict(row)
        new["chunks"] = str(resolve(base_dir, row["chunks"]))
        new["embeddings"] = f"embeddings/{row['label']}/embeddings.npy"
        rows.append(new)
    out = {
        "schema": "lens_effects.semantic_variant_manifest.v1",
        "source_manifest": str(base_manifest),
        "corpora": rows,
    }
    write_json(variant_manifest, out)


def summarize_variant(tag: str, label: str, manifest_path: Path, *, semantic_k: int, seed: int) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    base = manifest_path.parent
    corpora = manifest["corpora"]
    roles = {row["label"]: row["role"] for row in corpora}
    broad = [row["label"] for row in corpora if roles[row["label"]] == "broad"]
    targeted = [row["label"] for row in corpora if roles[row["label"]] == "targeted"]
    embeddings = {row["label"]: np.load(resolve(base, row["embeddings"])).astype(np.float32) for row in corpora}
    spectrum = {key: covariance_spectrum(val) for key, val in embeddings.items()}

    bb_pairs = list(combinations(broad, 2))
    bt_pairs = [(a, b) for a in broad for b in targeted]
    def mean(xs: list[float]) -> float: return float(np.mean(xs))
    row = {
        "label": label,
        "means": {
            "bb_cluster_js": mean([cluster_js(embeddings[a], embeddings[b], k=semantic_k, seed=seed) for a, b in bb_pairs]),
            "bt_cluster_js": mean([cluster_js(embeddings[a], embeddings[b], k=semantic_k, seed=seed) for a, b in bt_pairs]),
            "bb_top1": mean([abs(spectrum[a]["top1_share"] - spectrum[b]["top1_share"]) for a, b in bb_pairs]),
            "bt_top1": mean([abs(spectrum[a]["top1_share"] - spectrum[b]["top1_share"]) for a, b in bt_pairs]),
            "bb_top5": mean([abs(spectrum[a]["top5_share"] - spectrum[b]["top5_share"]) for a, b in bb_pairs]),
            "bt_top5": mean([abs(spectrum[a]["top5_share"] - spectrum[b]["top5_share"]) for a, b in bt_pairs]),
        },
        "ordering_held": {},
        "pair_counts": {"broad_broad": len(bb_pairs), "broad_targeted": len(bt_pairs)},
    }
    row["ordering_held"] = {
        "semantic_cluster_js_bt_gt_bb": row["means"]["bt_cluster_js"] > row["means"]["bb_cluster_js"],
        "top1_abs_diff_bt_gt_bb": row["means"]["bt_top1"] > row["means"]["bb_top1"],
        "top5_abs_diff_bt_gt_bb": row["means"]["bt_top5"] > row["means"]["bb_top5"],
    }
    write_json(manifest_path.parent.parent / "semantic_family_summary.json", {"schema": "lens_effects.semantic_variant_family_summary.v1", "variant_tag": tag, **row})
    return row


def render_md(summary: dict[str, Any], order: list[str]) -> str:
    lines = ["# Semantic Sweep Summary", "", "| Setting | BB cluster JS | BT cluster JS | BB top1 abs | BT top1 abs | BB top5 abs | BT top5 abs | Ordering |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for tag in order:
        if tag not in summary["rows"]:
            continue
        row = summary["rows"][tag]
        means = row["means"]
        held = "yes" if row["ordering_held"]["semantic_cluster_js_bt_gt_bb"] else "no"
        lines.append(f"| {row['label']} | {means['bb_cluster_js']:.4f} | {means['bt_cluster_js']:.4f} | {means['bb_top1']:.4f} | {means['bt_top1']:.4f} | {means['bb_top5']:.4f} | {means['bt_top5']:.4f} | {held} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run semantic encoder/chunk-size sweep and summarize semantic metrics.")
    ap.add_argument("--source-config", type=Path, required=True)
    ap.add_argument("--base-manifest", type=Path, required=True)
    ap.add_argument("--base-family-summary", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--semantic-k", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    rows: dict[str, Any] = {}
    order = [tag for tag, *_ in VARIANTS]
    for tag, label, model_name, chunk_chars, trust_remote_code, should_build in VARIANTS:
        variant_dir = args.out_dir / "semantic" / tag
        data_dir = variant_dir / "data"
        manifest_path = data_dir / "corpus_manifest.generated.json"
        if tag == "gte_c768":
            base = read_json(args.base_family_summary)
            rows[tag] = {
                "label": label,
                "means": {
                    "bb_cluster_js": base["metrics"]["broad_broad"]["semantic_cluster_js"],
                    "bt_cluster_js": base["metrics"]["broad_targeted"]["semantic_cluster_js"],
                    "bb_top1": base["metrics"]["broad_broad"]["top1_abs_diff"],
                    "bt_top1": base["metrics"]["broad_targeted"]["top1_abs_diff"],
                    "bb_top5": base["metrics"]["broad_broad"]["top5_abs_diff"],
                    "bt_top5": base["metrics"]["broad_targeted"]["top5_abs_diff"],
                },
                "ordering_held": {
                    "semantic_cluster_js_bt_gt_bb": base["ordering_held"]["semantic_cluster_js_bt_gt_bb"],
                    "top1_abs_diff_bt_gt_bb": base["ordering_held"]["top1_abs_diff_bt_gt_bb"],
                    "top5_abs_diff_bt_gt_bb": base["ordering_held"]["top5_abs_diff_bt_gt_bb"],
                },
                "pair_counts": base["pair_counts"],
            }
            continue
        data_dir.mkdir(parents=True, exist_ok=True)
        if chunk_chars == 768:
            write_manifest_from_base(args.base_manifest.resolve(), manifest_path)
        else:
            run(["python", str(root / "scripts/build_corpus_artifacts.py"), "build-chunks", "--sources", str(args.source_config), "--out-root", str(data_dir), "--chunk-chars", str(chunk_chars), "--overwrite"])
        run(["python", str(root / "scripts/build_corpus_artifacts.py"), "build-embeddings", "--manifest", str(manifest_path), "--out-root", str(data_dir), "--model-name", model_name, "--batch-size", str(args.batch_size), "--max-length", str(args.max_length), "--trust-remote-code" if trust_remote_code else "--no-trust-remote-code", "--overwrite"])
        rows[tag] = summarize_variant(tag, label, manifest_path, semantic_k=args.semantic_k, seed=args.seed)

    out = {"schema": "lens_effects.final_matrix.semantic_sweep.v1", "rows": rows}
    write_json(args.out_dir / "semantic_summary.json", out)
    (args.out_dir / "semantic_summary.md").write_text(render_md(out, order), encoding="utf-8")
    print(f"[done] wrote {args.out_dir / 'semantic_summary.json'}")


if __name__ == "__main__":
    main()
