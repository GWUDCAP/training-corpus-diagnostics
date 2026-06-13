#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import lzma
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from transformers import AutoTokenizer


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\sA-Za-z0-9]")
TEXTUAL_GRID = [(tok, n, k) for tok in ("regex", "gpt2") for n in (1, 2, 3) for k in (256, 512, 1024)]
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def tokenize_regex(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> list[str]:
    if n <= 1:
        return tokens
    return [" ".join(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


def normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    total = float(arr.sum())
    return arr / total if total > 0 else np.full(arr.shape, 1.0 / max(1, arr.size))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p1 = normalize(np.asarray(p, dtype=np.float64) + eps)
    q1 = normalize(np.asarray(q, dtype=np.float64) + eps)
    m = 0.5 * (p1 + q1)
    return float(0.5 * np.sum(p1 * np.log(p1 / m)) + 0.5 * np.sum(q1 * np.log(q1 / m)))


def compression_ratio(chunks: list[str]) -> float:
    raw = "\n\n".join(chunks).encode("utf-8", errors="ignore")
    return float(len(lzma.compress(raw, preset=6)) / len(raw)) if raw else 0.0


def covariance_spectrum(embeddings: np.ndarray) -> dict[str, float]:
    x = embeddings.astype(np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    eigvals = np.flip(np.maximum(np.linalg.eigvalsh(np.cov(centered, rowvar=False)), 0.0))
    total = float(eigvals.sum())
    return {
        "total_variance": total,
        "top1_share": float(eigvals[0] / total) if total > 0 else 0.0,
        "top5_share": float(eigvals[:5].sum() / total) if total > 0 else 0.0,
    }


def label_distribution(labels: np.ndarray, k: int) -> np.ndarray:
    return normalize(np.bincount(labels, minlength=k).astype(np.float64))


def cluster_js(a: np.ndarray, b: np.ndarray, k: int, seed: int) -> float:
    pooled = np.concatenate([a, b], axis=0)
    n_clusters = max(2, min(k, pooled.shape[0]))
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(pooled)
    return js_divergence(label_distribution(labels[: a.shape[0]], n_clusters), label_distribution(labels[a.shape[0]:], n_clusters))


def aggregate(rows: list[dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        for tok, n in row.items():
            out[tok] = out.get(tok, 0) + n
    return out


def textual_counts(chunks_by_label: dict[str, list[str]], *, gpt2_revision: str) -> dict[str, dict[tuple[str, int], list[dict[str, int]]]]:
    gpt2 = AutoTokenizer.from_pretrained("gpt2", revision=gpt2_revision, use_fast=True)
    out: dict[str, dict[tuple[str, int], list[dict[str, int]]]] = {}
    for label, chunks in chunks_by_label.items():
        out[label] = {}
        tokenized = {
            "regex": [tokenize_regex(chunk) for chunk in chunks],
            "gpt2": [gpt2.tokenize(chunk) for chunk in chunks],
        }
        for mode in ("regex", "gpt2"):
            for order in (1, 2, 3):
                out[label][(mode, order)] = [dict(Counter(ngrams(tokens, order))) for tokens in tokenized[mode]]
    return out


def lexical_pair(counts: dict[str, dict[tuple[str, int], list[dict[str, int]]]], a: str, b: str) -> dict[str, float]:
    js_vals: list[float] = []
    rho_vals: list[float] = []
    for mode, order, vocab_cut in TEXTUAL_GRID:
        ca = aggregate(counts[a][(mode, order)])
        cb = aggregate(counts[b][(mode, order)])
        vocab = [tok for tok, _ in sorted(ca.items(), key=lambda kv: kv[1], reverse=True)[:vocab_cut]]
        if len(vocab) < max(16, min(vocab_cut, 64)):
            continue
        va = np.asarray([ca.get(tok, 0) for tok in vocab], dtype=np.float64)
        vb = np.asarray([cb.get(tok, 0) for tok in vocab], dtype=np.float64)
        js_vals.append(js_divergence(va, vb))
        if np.std(va) > 0.0 and np.std(vb) > 0.0:
            rho = float(spearmanr(va, vb).statistic)
            if math.isfinite(rho):
                rho_vals.append(rho)
    return {"lexical_js": float(np.mean(js_vals)), "rho": float(np.mean(rho_vals))}


def mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def gt(lhs: float | None, rhs: float | None) -> bool | None:
    return None if lhs is None or rhs is None else bool(lhs > rhs)


def load_manifest(manifest_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    base = manifest_path.parent
    rows = manifest["corpora"]
    for row in rows:
        row["chunks_path"] = base / row["chunks"]
        row["embeddings_path"] = base / row["embeddings"] if row.get("embeddings") else None
    return base, rows


def run(manifest_path: Path, out_dir: Path, semantic_k: int, seed: int, gpt2_revision: str) -> None:
    _, corpora = load_manifest(manifest_path)
    labels = [row["label"] for row in corpora]
    display = {row["label"]: row.get("display", row["label"]) for row in corpora}
    roles = {row["label"]: row.get("role", "candidate") for row in corpora}
    broad = [label for label in labels if roles[label] == "broad"]
    targeted = [label for label in labels if roles[label] == "targeted"]
    candidates = [label for label in labels if roles[label] not in {"broad", "targeted"}]

    chunks = {row["label"]: read_json(row["chunks_path"]) for row in corpora}
    embeddings = {row["label"]: np.load(row["embeddings_path"]).astype(np.float32) for row in corpora if row["embeddings_path"]}
    for label in labels:
        if label in embeddings and len(chunks[label]) != embeddings[label].shape[0]:
            raise RuntimeError(f"chunk/embedding length mismatch for {label}")

    counts = textual_counts(chunks, gpt2_revision=gpt2_revision)
    standalone: dict[str, Any] = {}
    for label in labels:
        standalone[label] = {
            "display": display[label],
            "role": roles[label],
            "n_chunks": len(chunks[label]),
            "compression_ratio_lzma": compression_ratio(chunks[label]),
        }
        if label in embeddings:
            standalone[label]["semantic_spectrum"] = covariance_spectrum(embeddings[label])

    def pair_class(a: str, b: str) -> str:
        if a in broad and b in broad:
            return "broad_broad"
        if (a in broad and b in targeted) or (a in targeted and b in broad):
            return "broad_targeted"
        if a in targeted and b in targeted:
            return "targeted_targeted"
        return "candidate_pair"

    pair_rows: list[dict[str, Any]] = []
    for a, b in combinations(labels, 2):
        lex = lexical_pair(counts, a, b)
        row = {
            "a": a,
            "b": b,
            "a_display": display[a],
            "b_display": display[b],
            "pair_class": pair_class(a, b),
            "lexical_js": lex["lexical_js"],
            "rho": lex["rho"],
            "compression_abs_diff": abs(standalone[a]["compression_ratio_lzma"] - standalone[b]["compression_ratio_lzma"]),
        }
        if a in embeddings and b in embeddings:
            row["semantic_cluster_js"] = cluster_js(embeddings[a], embeddings[b], semantic_k, seed)
            row["top1_abs_diff"] = abs(standalone[a]["semantic_spectrum"]["top1_share"] - standalone[b]["semantic_spectrum"]["top1_share"])
            row["top5_abs_diff"] = abs(standalone[a]["semantic_spectrum"]["top5_share"] - standalone[b]["semantic_spectrum"]["top5_share"])
        pair_rows.append(row)

    metrics = ["lexical_js", "rho", "semantic_cluster_js", "compression_abs_diff", "top1_abs_diff", "top5_abs_diff"]
    bb_pairs = list(combinations(broad, 2))
    bt_pairs = [(a, b) for a in broad for b in targeted]
    tt_pairs = list(combinations(targeted, 2))

    def lexical_rows(pairs: list[tuple[str, str]], reverse: bool = False) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for a, b in pairs:
            rows.append(lexical_pair(counts, b, a) if reverse else lexical_pair(counts, a, b))
        return rows

    def lexical_mean(rows: list[dict[str, float]]) -> dict[str, float | None]:
        return {
            "lexical_js": mean([row["lexical_js"] for row in rows]),
            "rho": mean([row["rho"] for row in rows]),
        }

    def directed_lexical_means(pairs: list[tuple[str, str]], include_reverse: bool) -> dict[str, float | None]:
        rows = lexical_rows(pairs)
        if include_reverse:
            rows.extend(lexical_rows(pairs, reverse=True))
        return lexical_mean(rows)

    family: dict[str, Any] = {"pair_counts": {}, "metrics": {}}
    for pair_class in ("broad_broad", "broad_targeted", "targeted_targeted"):
        rows = [row for row in pair_rows if row["pair_class"] == pair_class]
        family["pair_counts"][pair_class] = len(rows)
        family["metrics"][pair_class] = {metric: mean([row[metric] for row in rows if metric in row and row[metric] is not None]) for metric in metrics}
    family["metrics"]["broad_broad"].update(directed_lexical_means(bb_pairs, include_reverse=True))
    family["metrics"]["broad_targeted"].update(directed_lexical_means(bt_pairs, include_reverse=False))
    family["metrics"]["targeted_targeted"].update(directed_lexical_means(tt_pairs, include_reverse=True))
    family["pair_counts"]["directed_textual_broad_broad"] = len(bb_pairs) * 2
    family["pair_counts"]["directed_textual_broad_targeted"] = len(bt_pairs)
    family["pair_counts"]["directed_textual_targeted_targeted"] = len(tt_pairs) * 2
    bt_forward = lexical_mean(lexical_rows(bt_pairs))
    bt_reverse = lexical_mean(lexical_rows(bt_pairs, reverse=True))
    bt_symmetric = {
        "lexical_js": mean([float(bt_forward["lexical_js"]), float(bt_reverse["lexical_js"])]),
        "rho": mean([float(bt_forward["rho"]), float(bt_reverse["rho"])]),
    }
    directional_sensitivity = {
        "schema": "lens_effects.directional_lexical_sensitivity.v1",
        "note": "The primary broad-targeted textual summary uses broad corpora as the reference vocabulary. This table reports the reverse direction and the two-direction average as a sensitivity check.",
        "broad_broad_two_direction": family["metrics"]["broad_broad"],
        "broad_targeted_broad_reference": bt_forward,
        "broad_targeted_targeted_reference": bt_reverse,
        "broad_targeted_two_direction_average": bt_symmetric,
        "ordering_checks": {
            "primary_lexical_js_bt_gt_bb": gt(bt_forward["lexical_js"], family["metrics"]["broad_broad"]["lexical_js"]),
            "reverse_lexical_js_bt_gt_bb": gt(bt_reverse["lexical_js"], family["metrics"]["broad_broad"]["lexical_js"]),
            "symmetric_lexical_js_bt_gt_bb": gt(bt_symmetric["lexical_js"], family["metrics"]["broad_broad"]["lexical_js"]),
            "primary_rho_bb_gt_bt": gt(family["metrics"]["broad_broad"]["rho"], bt_forward["rho"]),
            "reverse_rho_bb_gt_bt": gt(family["metrics"]["broad_broad"]["rho"], bt_reverse["rho"]),
            "symmetric_rho_bb_gt_bt": gt(family["metrics"]["broad_broad"]["rho"], bt_symmetric["rho"]),
        },
    }
    family["ordering_held"] = {
        "lexical_js_bt_gt_bb": gt(family["metrics"]["broad_targeted"]["lexical_js"], family["metrics"]["broad_broad"]["lexical_js"]),
        "rho_bb_gt_bt": gt(family["metrics"]["broad_broad"]["rho"], family["metrics"]["broad_targeted"]["rho"]),
        "semantic_cluster_js_bt_gt_bb": gt(family["metrics"]["broad_targeted"]["semantic_cluster_js"], family["metrics"]["broad_broad"]["semantic_cluster_js"]),
        "compression_abs_diff_bt_gt_bb": gt(family["metrics"]["broad_targeted"]["compression_abs_diff"], family["metrics"]["broad_broad"]["compression_abs_diff"]),
        "top1_abs_diff_bt_gt_bb": gt(family["metrics"]["broad_targeted"]["top1_abs_diff"], family["metrics"]["broad_broad"]["top1_abs_diff"]),
        "top5_abs_diff_bt_gt_bb": gt(family["metrics"]["broad_targeted"]["top5_abs_diff"], family["metrics"]["broad_broad"]["top5_abs_diff"]),
    }

    out = {
        "schema": "lens_effects.corpus_matrix.v1",
        "manifest": str(manifest_path),
        "settings": {"textual_grid": TEXTUAL_GRID, "semantic_k": semantic_k, "seed": seed, "gpt2_revision": gpt2_revision},
        "roles": {"broad": broad, "targeted": targeted, "candidate": candidates},
        "standalone": standalone,
        "pairs": pair_rows,
        "family_summary": family,
        "directional_lexical_sensitivity": directional_sensitivity,
    }
    write_json(out_dir / "corpus_matrix.json", out)
    write_json(out_dir / "family_summary.json", family)
    write_json(out_dir / "standalone_summary.json", standalone)
    write_json(out_dir / "directional_lexical_sensitivity.json", directional_sensitivity)
    write_summary_md(out_dir / "summary.md", family)
    write_directional_sensitivity_md(out_dir / "directional_lexical_sensitivity.md", directional_sensitivity)
    print(f"[done] wrote matrix outputs to {out_dir}")


def fmt(x: float | None) -> str:
    return "NA" if x is None or not math.isfinite(float(x)) else f"{float(x):.4f}"


def write_summary_md(path: Path, family: dict[str, Any]) -> None:
    lines = [
        "# Corpus Matrix Summary",
        "",
        "| Metric | Broad-broad | Broad-targeted | Targeted-targeted |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "lexical_js": "Lexical JS",
        "rho": "Spearman rho",
        "semantic_cluster_js": "Semantic cluster JS",
        "compression_abs_diff": "Compression abs diff",
        "top1_abs_diff": "Top1 abs diff",
        "top5_abs_diff": "Top5 abs diff",
    }
    for metric, name in labels.items():
        lines.append(
            f"| {name} | {fmt(family['metrics']['broad_broad'][metric])} | "
            f"{fmt(family['metrics']['broad_targeted'][metric])} | "
            f"{fmt(family['metrics']['targeted_targeted'][metric])} |"
        )
    lines += ["", "## Ordering Checks", ""]
    for name, held in family["ordering_held"].items():
        lines.append(f"- `{name}`: `{held}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_directional_sensitivity_md(path: Path, sensitivity: dict[str, Any]) -> None:
    rows = [
        ("Broad-broad, two-direction", sensitivity["broad_broad_two_direction"]),
        ("Broad-targeted, broad reference", sensitivity["broad_targeted_broad_reference"]),
        ("Broad-targeted, targeted reference", sensitivity["broad_targeted_targeted_reference"]),
        ("Broad-targeted, two-direction average", sensitivity["broad_targeted_two_direction_average"]),
    ]
    lines = [
        "# Directional Lexical Sensitivity",
        "",
        "The primary broad-targeted textual summary uses broad corpora as the reference vocabulary. Because lexical JS and rho are computed on the baseline corpus vocabulary, this table reports the reverse direction and the two-direction average.",
        "",
        "| Comparison | Lexical JS | Spearman rho |",
        "|---|---:|---:|",
    ]
    for label, row in rows:
        lines.append(f"| {label} | {fmt(row['lexical_js'])} | {fmt(row['rho'])} |")
    lines += ["", "## Ordering Checks", ""]
    for key, value in sensitivity["ordering_checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute corpus pairwise matrix and family summaries from chunk/embedding files.")
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("recomputed"))
    parser.add_argument("--semantic-k", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpt2-revision", default=GPT2_REVISION)
    args = parser.parse_args()
    run(args.manifest, args.out_dir, args.semantic_k, args.seed, args.gpt2_revision)


if __name__ == "__main__":
    main()
