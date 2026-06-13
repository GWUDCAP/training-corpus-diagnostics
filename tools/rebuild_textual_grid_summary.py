#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import lzma
import math
import sys
import zlib
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_corpus_matrix import GPT2_REVISION, aggregate, js_divergence, read_json, textual_counts


TOKENIZERS = ("gpt2", "regex")
NGRAMS = (1, 2, 3)
VOCABS = (256, 512, 1024)
COMPRESSORS = ("gzip", "lzma", "zlib")
ROOT = Path(__file__).resolve().parents[1]


def finite_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(mean(vals)) if vals else float("nan")


def compression_ratio(chunks: list[str], compressor: str) -> float:
    raw = "\n\n".join(chunks).encode("utf-8", errors="ignore")
    if not raw:
        return 0.0
    if compressor == "gzip":
        out = gzip.compress(raw, compresslevel=9)
    elif compressor == "lzma":
        out = lzma.compress(raw, preset=6)
    elif compressor == "zlib":
        out = zlib.compress(raw, level=9)
    else:
        raise ValueError(compressor)
    return len(out) / len(raw)


def lexical_one(counts: dict, a: str, b: str, mode: str, order: int, vocab_cut: int) -> tuple[float, float]:
    ca = aggregate(counts[a][(mode, order)])
    cb = aggregate(counts[b][(mode, order)])
    vocab = [tok for tok, _ in sorted(ca.items(), key=lambda kv: kv[1], reverse=True)[:vocab_cut]]
    if len(vocab) < max(16, min(vocab_cut, 64)):
        return float("nan"), float("nan")
    va = np.asarray([ca.get(tok, 0) for tok in vocab], dtype=np.float64)
    vb = np.asarray([cb.get(tok, 0) for tok in vocab], dtype=np.float64)
    rho = float("nan")
    if np.std(va) > 0.0 and np.std(vb) > 0.0:
        stat = float(spearmanr(va, vb).statistic)
        if math.isfinite(stat):
            rho = stat
    return js_divergence(va, vb), rho


def directed(pairs: list[tuple[str, str]], include_reverse: bool) -> list[tuple[str, str]]:
    return pairs + ([(b, a) for a, b in pairs] if include_reverse else [])


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild textual-grid and compressor sweep summaries from a corpus manifest.")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/corpus_manifest.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/final_matrix/sweeps")
    args = ap.parse_args()

    manifest_path = args.manifest
    manifest = read_json(manifest_path)
    base = manifest_path.parent
    corpora = manifest["corpora"]
    labels = [row["label"] for row in corpora]
    roles = {row["label"]: row["role"] for row in corpora}
    broad = [label for label in labels if roles[label] == "broad"]
    targeted = [label for label in labels if roles[label] == "targeted"]
    chunks = {row["label"]: read_json(base / row["chunks"]) for row in corpora}
    print(f"[load] corpora={len(labels)} broad={len(broad)} targeted={len(targeted)}")

    counts = textual_counts(chunks, gpt2_revision=GPT2_REVISION)
    print("[counts] ready")
    comp = {name: {label: compression_ratio(chunks[label], name) for label in labels} for name in COMPRESSORS}

    bb = list(combinations(broad, 2))
    bt = [(a, b) for a in broad for b in targeted]
    tb = [(b, a) for a in broad for b in targeted]
    tt = list(combinations(targeted, 2))
    pair_sets = {"bb": directed(bb, True), "bt": bt, "tb": tb, "tt": directed(tt, True)}

    rows = []
    for tokenizer_mode in TOKENIZERS:
        for ngram_order in NGRAMS:
            for vocab_cut in VOCABS:
                lex = {}
                for group, pairs in pair_sets.items():
                    js_vals, rho_vals = [], []
                    for a, b in pairs:
                        js, rho = lexical_one(counts, a, b, tokenizer_mode, ngram_order, vocab_cut)
                        js_vals.append(js)
                        rho_vals.append(rho)
                    lex[group] = {"lexical_js_mean": finite_mean(js_vals), "rho_mean": finite_mean(rho_vals)}
                for compressor in COMPRESSORS:
                    row = {"tokenizer_mode": tokenizer_mode, "ngram_order": ngram_order, "vocab_cut": vocab_cut, "compressor": compressor}
                    for group, pairs in pair_sets.items():
                        gaps = [comp[compressor][a] - comp[compressor][b] for a, b in pairs]
                        row[f"{group}_count"] = len(pairs)
                        row[f"{group}_lexical_js_mean"] = lex[group]["lexical_js_mean"]
                        row[f"{group}_rho_mean"] = lex[group]["rho_mean"]
                        row[f"{group}_compression_gap_mean"] = finite_mean(gaps)
                    row["ordering_lexical_js"] = bool(row["bt_lexical_js_mean"] > row["bb_lexical_js_mean"])
                    row["ordering_rho"] = bool(row["bb_rho_mean"] > row["bt_rho_mean"])
                    rows.append(row)
                print(f"[setting] {tokenizer_mode} n={ngram_order} vocab={vocab_cut}")

    out = {
        "schema": "lens_effects.textual_grid_aggregate.v1",
        "row_count": sum(row[f"{group}_count"] for row in rows for group in ("bb", "bt", "tb", "tt")),
        "setting_count": len(rows),
        "rows": rows,
    }
    json_paths = [args.out_dir / "textual_grid_summary.json"]
    for path in json_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=True) + "\n")

    lines = [
        "# Textual Grid Aggregate Summary",
        "",
        f"- rows: `{out['row_count']}`",
        f"- settings: `{out['setting_count']}`",
        "",
        "| tokenizer | ngram | vocab | compressor | bb lex JS | bt lex JS | bb rho | bt rho | lexical ordering | rho ordering |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['tokenizer_mode']} | {row['ngram_order']} | {row['vocab_cut']} | {row['compressor']} | "
            f"{row['bb_lexical_js_mean']:.4f} | {row['bt_lexical_js_mean']:.4f} | "
            f"{row['bb_rho_mean']:.4f} | {row['bt_rho_mean']:.4f} | "
            f"`{row['ordering_lexical_js']}` | `{row['ordering_rho']}` |"
        )
    md = "\n".join(lines) + "\n"
    md_paths = [args.out_dir / "textual_grid_summary.md"]
    for path in md_paths:
        path.write_text(md)
    print(f"[done] rows={out['row_count']} settings={out['setting_count']} all_lex={all(row['ordering_lexical_js'] for row in rows)} all_rho={all(row['ordering_rho'] for row in rows)}")


if __name__ == "__main__":
    main()
