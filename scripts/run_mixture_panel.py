#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from run_corpus_matrix import GPT2_REVISION, cluster_js, compression_ratio, covariance_spectrum, lexical_pair, load_manifest, textual_counts, write_json


REFUSAL_RE = re.compile(r"\b(?:sorry|cannot|can't|unable|not able|cannot help|can't help)\b", re.I)
ASSISTANT_MARKER_RE = re.compile(r"(?:^|\n)\s*(?:assistant|<\|assistant\|>|###\s*assistant)\s*[:\n]", re.I)


def mix_count(total: int, alpha: float) -> int:
    return max(0, min(total, int(round(total * alpha))))


def mix_text(base: list[str], target: list[str], alpha: float) -> list[str]:
    n_target = mix_count(len(base), alpha)
    n_base = len(base) - n_target
    return base[:n_base] + target[:n_target]


def mix_embeddings(base: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    n_target = mix_count(base.shape[0], alpha)
    n_base = base.shape[0] - n_target
    return np.concatenate([base[:n_base], target[:n_target]], axis=0)


def chunk_rate(chunks: list[str], pattern: re.Pattern[str]) -> float:
    return float(sum(1 for chunk in chunks if pattern.search(chunk)) / len(chunks)) if chunks else 0.0


def render_md(out: dict[str, Any], display: dict[str, str]) -> str:
    endpoints = []
    for key, rows in out["targeted_mix_bundle"].items():
        base, target = key.split("__mix__")
        endpoint = rows[-1]
        endpoints.append((base, target, endpoint["cluster_js"], endpoint["lexical_js"], endpoint["lexical_rank_rho"]))
    strongest = sorted(endpoints, key=lambda row: row[2], reverse=True)[:12]
    flattest = sorted(endpoints, key=lambda row: row[2])[:12]

    def table(title: str, rows: list[tuple[str, str, float, float, float]]) -> list[str]:
        lines = [
            f"## {title}",
            "",
            "| Baseline | Target | Cluster JS @ 1.0 | Lexical JS @ 1.0 | Rho @ 1.0 |",
            "|---|---|---:|---:|---:|",
        ]
        for base, target, cjs, lex, rho in rows:
            lines.append(f"| {display[base]} | {display[target]} | {cjs:.4f} | {lex:.4f} | {rho:.4f} |")
        return lines

    lines = [
        "# Targeted Mixture Summary",
        "",
        f"- targeted corpora: `{out['targeted_corpora']}`",
        f"- directed mixture trajectories: `{out['directed_pairs']}`",
        f"- alpha grid: `{', '.join(str(x) for x in out['alphas'])}`",
        "",
        *table("Strongest Endpoint Cluster JS", strongest),
        "",
        *table("Flattest Endpoint Cluster JS", flattest),
        "",
        "## Interpretation",
        "",
        "These trajectories are directed baseline-relative mixtures. Each row starts at the baseline corpus (`alpha=0.0`) and ends at the pure target corpus (`alpha=1.0`), while the intermediate points mix baseline and target chunks under the retained chunk budget. The endpoint values therefore recover the directed baseline-to-target comparison under the mixture metric family, and the interior points show whether movement is flat or smooth as target mass increases.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute directed post-training mixture trajectories from retained chunks and embeddings.")
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/final_matrix/mixtures"))
    parser.add_argument("--alphas", default="0.0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--semantic-k", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpt2-revision", default=GPT2_REVISION)
    args = parser.parse_args()

    _, corpora = load_manifest(args.manifest)
    targeted = [row for row in corpora if row.get("role") == "targeted"]
    display = {row["label"]: row.get("display", row["label"]) for row in targeted}
    labels = [row["label"] for row in targeted]
    chunks = {row["label"]: json.loads(row["chunks_path"].read_text(encoding="utf-8")) for row in targeted}
    embeddings = {row["label"]: np.load(row["embeddings_path"]).astype(np.float32) for row in targeted}
    for label in labels:
        if len(chunks[label]) != embeddings[label].shape[0]:
            raise RuntimeError(f"chunk/embedding length mismatch for {label}")

    alphas = [float(part) for part in args.alphas.split(",") if part.strip()]
    mix_chunks: dict[str, list[str]] = {}
    mix_embeddings_by_key: dict[str, np.ndarray] = {}
    for base, target in permutations(labels, 2):
        for alpha in alphas:
            key = f"{base}__mix__{target}__alpha__{alpha:g}"
            mix_chunks[key] = mix_text(chunks[base], chunks[target], alpha)
            mix_embeddings_by_key[key] = mix_embeddings(embeddings[base], embeddings[target], alpha)

    counts = textual_counts({**{label: chunks[label] for label in labels}, **mix_chunks}, gpt2_revision=args.gpt2_revision)
    bundle: dict[str, list[dict[str, float]]] = {}
    for base, target in permutations(labels, 2):
        rows: list[dict[str, float]] = []
        for alpha in alphas:
            key = f"{base}__mix__{target}__alpha__{alpha:g}"
            mix_c = mix_chunks[key]
            mix_e = mix_embeddings_by_key[key]
            if alpha == 0.0:
                lex = {"lexical_js": 0.0, "rho": 1.0}
                cjs = 0.0
            else:
                lex = lexical_pair(counts, base, key)
                cjs = cluster_js(embeddings[base], mix_e, args.semantic_k, args.seed)
            spectrum = covariance_spectrum(mix_e)
            rows.append({
                "alpha_target": alpha,
                "lexical_rank_rho": float(lex["rho"]),
                "lexical_js": float(lex["lexical_js"]),
                "cluster_js": float(cjs),
                "top1_share": float(spectrum["top1_share"]),
                "top5_share": float(spectrum["top5_share"]),
                "compression_ratio": compression_ratio(mix_c),
                "refusal_chunk_rate": chunk_rate(mix_c, REFUSAL_RE),
                "assistant_marker_rate": chunk_rate(mix_c, ASSISTANT_MARKER_RE),
            })
        bundle[f"{base}__mix__{target}"] = rows

    out = {
        "schema": "lens_effects.targeted_mixture_panel.v1",
        "manifest": str(args.manifest),
        "targeted_corpora": len(labels),
        "directed_pairs": len(labels) * (len(labels) - 1),
        "alphas": alphas,
        "targeted_mix_bundle": bundle,
    }
    write_json(args.out_dir / "targeted_mixture_panel.json", out)
    (args.out_dir / "targeted_mixture_panel.md").write_text(render_md(out, display), encoding="utf-8")
    print(f"[done] wrote mixture panel to {args.out_dir}")


if __name__ == "__main__":
    main()
