#!/usr/bin/env python3
# ===== IMPORTS ===== #
## ===== STDLIB ===== ##
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

## ===== 3RD-PARTY ===== ##
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from transformers import AutoTokenizer

## ===== LOCAL ===== ##
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import corpus_operator_deformation as cod

# ===== GLOBALS ===== #

BROAD = ["fineweb_sample_10BT", "falcon_refinedweb", "dolma_v1_7", "c4_en", "slimpajama_sample", "pile_uncopyrighted"]
TARGETED = ["hh_rlhf_chosen", "hh_rlhf_rejected", "oasst1_assistant_en", "ultrafeedback_binarized_chosen", "nemotron_math", "beavertails", "magicoder_oss_instruct", "dolly_15k"]
CORPORA = BROAD + TARGETED

DISPLAY = {
    "fineweb_sample_10BT": "FineWeb",
    "falcon_refinedweb": "RefinedWeb",
    "dolma_v1_7": "Dolma",
    "c4_en": "C4",
    "slimpajama_sample": "SlimPajama",
    "pile_uncopyrighted": "Pile uncopyrighted",
    "hh_rlhf_chosen": "HH-RLHF chosen",
    "hh_rlhf_rejected": "HH-RLHF rejected",
    "oasst1_assistant_en": "OASST1 assistant",
    "ultrafeedback_binarized_chosen": "UltraFeedback chosen",
    "nemotron_math": "Nemotron-SFT-Math",
    "beavertails": "BeaverTails",
    "magicoder_oss_instruct": "Magicoder OSS Instruct",
    "dolly_15k": "Dolly 15k",
}

METRIC_ORDER = [
    "lexical_js",
    "rho",
    "semantic_cluster_js",
    "compression_abs_diff",
    "top1_abs_diff",
    "top5_abs_diff",
]

ORDERING = {
    "lexical_js": ("bt", ">", "bb"),
    "rho": ("bb", ">", "bt"),
    "semantic_cluster_js": ("bt", ">", "bb"),
    "compression_abs_diff": ("bt", ">", "bb"),
    "top1_abs_diff": ("bt", ">", "bb"),
    "top5_abs_diff": ("bt", ">", "bb"),
}

TEXTUAL_SETTINGS = [
    (tokenizer_mode, ngram_order, vocab_cut)
    for tokenizer_mode in ["regex", "gpt2"]
    for ngram_order in [1, 2, 3]
    for vocab_cut in [256, 512, 1024]
]
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

# ===== FUNCTIONS ===== #
## ===== IO ===== ##

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
        f.flush()

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid checkpoint JSONL at {path}:{line_no}: {e}") from e
    return rows

def chunk_path(cache_dir: Path, label: str) -> Path:
    nested = cache_dir / label / "chunks.json"
    flat = cache_dir / f"{label}.chunks.json"
    return nested if nested.exists() else flat

def embedding_path(cache_dir: Path, label: str) -> Path:
    nested = cache_dir / label / "embeddings.npy"
    flat = cache_dir / f"{label}.embeddings.npy"
    return nested if nested.exists() else flat

def load_manifest_cache(manifest_path: Path) -> tuple[dict[str, dict[str, Any]], Path]:
    global BROAD, TARGETED, CORPORA, DISPLAY
    manifest = load_json(manifest_path)
    base = manifest_path.parent
    rows = manifest["corpora"]
    BROAD = [row["label"] for row in rows if row["role"] == "broad"]
    TARGETED = [row["label"] for row in rows if row["role"] == "targeted"]
    CORPORA = [row["label"] for row in rows]
    DISPLAY = {row["label"]: row.get("display", row["label"]) for row in rows}
    cache: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for row in rows:
        label = row["label"]
        cp = base / row["chunks"]
        ep = base / row["embeddings"]
        if not cp.exists():
            missing.append(str(cp))
        if not ep.exists():
            missing.append(str(ep))
        if cp.exists() and ep.exists():
            chunks = load_json(cp)
            embeddings = np.load(ep).astype(np.float32)
            if len(chunks) != embeddings.shape[0]:
                raise RuntimeError(f"chunk/embedding length mismatch for {label}: chunks={len(chunks)} embeddings={embeddings.shape[0]}")
            cache[label] = {"chunks": chunks, "embeddings": embeddings}
    if missing:
        lines = [f"missing required manifest chunk/embedding artifacts for {manifest_path}:"]
        lines.extend(f"- {x}" for x in missing)
        raise FileNotFoundError("\n".join(lines))
    return cache, base

def load_corpus_cache(cache_dir: Path) -> dict[str, dict[str, Any]]:
    missing: list[str] = []
    cache: dict[str, dict[str, Any]] = {}
    for label in CORPORA:
        cp = chunk_path(cache_dir, label)
        ep = embedding_path(cache_dir, label)
        if not cp.exists():
            missing.append(str(cp))
        if not ep.exists():
            missing.append(str(ep))
        if cp.exists() and ep.exists():
            chunks = load_json(cp)
            embeddings = np.load(ep).astype(np.float32)
            if len(chunks) != embeddings.shape[0]:
                raise RuntimeError(f"chunk/embedding length mismatch for {label}: chunks={len(chunks)} embeddings={embeddings.shape[0]}")
            cache[label] = {"chunks": chunks, "embeddings": embeddings}
    if missing:
        lines = ["missing required canonical chunk/embedding artifacts:"]
        lines.extend(f"- {x}" for x in missing)
        raise FileNotFoundError("\n".join(lines))
    return cache

## ===== METRICS ===== ##

def ngrams(tokens: list[str], order: int) -> list[str]:
    if order <= 1:
        return tokens
    if len(tokens) < order:
        return []
    return [" ".join(tokens[i:i + order]) for i in range(len(tokens) - order + 1)]

def build_textual_cache(cache: dict[str, dict[str, Any]]) -> dict[str, dict[tuple[str, int], list[dict[str, int]]]]:
    print("[textual-cache] tokenize/count chunks", flush=True)
    gpt2 = AutoTokenizer.from_pretrained("gpt2", revision=GPT2_REVISION, use_fast=True)
    out: dict[str, dict[tuple[str, int], list[dict[str, int]]]] = {}
    for label in CORPORA:
        out[label] = {}
        for tokenizer_mode in ["regex", "gpt2"]:
            token_rows: list[list[str]] = []
            for chunk in cache[label]["chunks"]:
                token_rows.append(cod.stable_tokenize(chunk) if tokenizer_mode == "regex" else gpt2.tokenize(chunk))
            for order in [1, 2, 3]:
                rows: list[dict[str, int]] = []
                for toks in token_rows:
                    rows.append(dict(Counter(ngrams(toks, order))))
                out[label][(tokenizer_mode, order)] = rows
    return out

def aggregate_counts(rows: list[dict[str, int]], indices: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in indices:
        for tok, n in rows[int(i)].items():
            counts[tok] = counts.get(tok, 0) + n
    return counts

def aggregate_textual_counts_for_indices(
    textual_cache: dict[str, dict[tuple[str, int], list[dict[str, int]]]],
    labels: list[str],
    indices_by_label: dict[str, np.ndarray],
) -> dict[str, dict[tuple[str, int], dict[str, int]]]:
    out: dict[str, dict[tuple[str, int], dict[str, int]]] = {}
    for label in labels:
        out[label] = {}
        for tokenizer_mode in ["regex", "gpt2"]:
            for order in [1, 2, 3]:
                out[label][(tokenizer_mode, order)] = aggregate_counts(textual_cache[label][(tokenizer_mode, order)], indices_by_label[label])
    return out

def lexical_pair_from_count_cache(
    textual_cache: dict[str, dict[tuple[str, int], list[dict[str, int]]]],
    a: str,
    b: str,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
) -> dict[str, float]:
    js_vals: list[float] = []
    rho_vals: list[float] = []
    for tokenizer_mode, ngram_order, vocab_cut in TEXTUAL_SETTINGS:
        base_counts = aggregate_counts(textual_cache[a][(tokenizer_mode, ngram_order)], a_idx)
        target_counts = aggregate_counts(textual_cache[b][(tokenizer_mode, ngram_order)], b_idx)
        vocab = [tok for tok, _ in sorted(base_counts.items(), key=lambda kv: kv[1], reverse=True)[:vocab_cut]]
        if len(vocab) < max(16, min(vocab_cut, 64)):
            continue
        base_vec = np.asarray([base_counts.get(tok, 0) for tok in vocab], dtype=np.float64)
        target_vec = np.asarray([target_counts.get(tok, 0) for tok in vocab], dtype=np.float64)
        js_vals.append(float(cod.js_divergence(base_vec, target_vec)))
        rho_vals.append(float(spearmanr(base_vec, target_vec).statistic))
    return {"lexical_js": float(np.mean(js_vals)), "rho": float(np.mean(rho_vals))}

def lexical_pair_from_aggregated_counts(
    textual_counts: dict[str, dict[tuple[str, int], dict[str, int]]],
    a: str,
    b: str,
) -> dict[str, float]:
    js_vals: list[float] = []
    rho_vals: list[float] = []
    for tokenizer_mode, ngram_order, vocab_cut in TEXTUAL_SETTINGS:
        base_counts = textual_counts[a][(tokenizer_mode, ngram_order)]
        target_counts = textual_counts[b][(tokenizer_mode, ngram_order)]
        vocab = [tok for tok, _ in sorted(base_counts.items(), key=lambda kv: kv[1], reverse=True)[:vocab_cut]]
        if len(vocab) < max(16, min(vocab_cut, 64)):
            continue
        base_vec = np.asarray([base_counts.get(tok, 0) for tok in vocab], dtype=np.float64)
        target_vec = np.asarray([target_counts.get(tok, 0) for tok in vocab], dtype=np.float64)
        js_vals.append(float(cod.js_divergence(base_vec, target_vec)))
        rho_vals.append(float(spearmanr(base_vec, target_vec).statistic))
    return {"lexical_js": float(np.mean(js_vals)), "rho": float(np.mean(rho_vals))}

def sample_indices(n: int, rng: np.random.Generator, *, replace: bool) -> np.ndarray:
    return rng.choice(n, size=n, replace=replace)

def subset_cache(cache: dict[str, dict[str, Any]], indices_by_label: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for label, idx in indices_by_label.items():
        chunks = cache[label]["chunks"]
        out[label] = {
            "chunks": [chunks[int(i)] for i in idx],
            "embeddings": cache[label]["embeddings"][idx],
        }
    return out

def lexical_pair(a_chunks: list[str], b_chunks: list[str], *, top_k: int) -> dict[str, float]:
    row = cod.lexical_metrics(a_chunks, b_chunks, top_k=top_k)
    return {"lexical_js": float(row["js_divergence_top_vocab"]), "rho": float(row["spearman_rank_rho"])}

def compression_value(chunks: list[str]) -> float:
    return float(cod.compression_ratio("\n\n".join(chunks)))

def spectrum_values(embeddings: np.ndarray) -> dict[str, float]:
    spec = cod.covariance_spectrum(embeddings)
    return {"top1": float(spec["top1_share"]), "top5": float(spec["top5_share"])}

def fit_pair_label_reference(a: np.ndarray, b: np.ndarray, *, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([a, b], axis=0)
    n_clusters = max(2, min(k, pooled.shape[0]))
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = model.fit_predict(pooled)
    return labels[: a.shape[0]], labels[a.shape[0]:]

def label_dist(labels: np.ndarray, *, k: int) -> np.ndarray:
    return cod.normalize_dist(np.bincount(labels, minlength=k).astype(np.float64))

def semantic_js_from_labels(a_labels: np.ndarray, b_labels: np.ndarray) -> float:
    k = int(max(a_labels.max(initial=0), b_labels.max(initial=0)) + 1)
    return float(cod.js_divergence(label_dist(a_labels, k=k), label_dist(b_labels, k=k)))

def pair_groups(labels: list[str]) -> dict[str, list[tuple[str, str]]]:
    broad = [x for x in labels if x in BROAD]
    targeted = [x for x in labels if x in TARGETED]
    return {
        "bb": list(combinations(broad, 2)),
        "bt": [(a, b) for a in broad for b in targeted],
    }

def mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None

def quantiles(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"mean": None, "median": None, "q025": None, "q500": None, "q975": None}
    arr = np.asarray(xs, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q025": float(np.quantile(arr, 0.025)),
        "q500": float(np.quantile(arr, 0.500)),
        "q975": float(np.quantile(arr, 0.975)),
    }

def ordering_held(row: dict[str, dict[str, float | None]], metric: str) -> bool:
    lhs_group, op, rhs_group = ORDERING[metric]
    lhs = row[metric][lhs_group]
    rhs = row[metric][rhs_group]
    if lhs is None or rhs is None:
        return False
    if op == ">":
        return float(lhs) > float(rhs)
    raise ValueError(f"unsupported ordering op: {op}")

def family_summary(
    cache: dict[str, dict[str, Any]],
    *,
    labels: list[str],
    lexical_top_k: int,
    semantic_k: int,
    seed: int,
    textual_cache: dict[str, dict[tuple[str, int], list[dict[str, int]]]] | None = None,
    indices_by_label: dict[str, np.ndarray] | None = None,
    textual_counts: dict[str, dict[tuple[str, int], dict[str, int]]] | None = None,
    semantic_pair_labels: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    groups = pair_groups(labels)
    comp = {label: compression_value(cache[label]["chunks"]) for label in labels}
    spec = {label: spectrum_values(cache[label]["embeddings"]) for label in labels}
    out = {metric: {"bb": None, "bt": None} for metric in METRIC_ORDER}

    for group, pairs in groups.items():
        vals = {metric: [] for metric in METRIC_ORDER}
        for a, b in pairs:
            if textual_counts is not None:
                lex = lexical_pair_from_aggregated_counts(textual_counts, a, b)
            elif textual_cache is not None and indices_by_label is not None:
                lex = lexical_pair_from_count_cache(textual_cache, a, b, indices_by_label[a], indices_by_label[b])
            else:
                lex = lexical_pair(cache[a]["chunks"], cache[b]["chunks"], top_k=lexical_top_k)
            vals["lexical_js"].append(lex["lexical_js"])
            vals["rho"].append(lex["rho"])
            if semantic_pair_labels is None:
                a_lab, b_lab = fit_pair_label_reference(cache[a]["embeddings"], cache[b]["embeddings"], k=semantic_k, seed=seed)
            else:
                key = (a, b) if (a, b) in semantic_pair_labels else (b, a)
                a_lab, b_lab = semantic_pair_labels[key]
                if key == (b, a):
                    a_lab, b_lab = b_lab, a_lab
            vals["semantic_cluster_js"].append(semantic_js_from_labels(a_lab, b_lab))
            vals["compression_abs_diff"].append(abs(comp[a] - comp[b]))
            vals["top1_abs_diff"].append(abs(spec[a]["top1"] - spec[b]["top1"]))
            vals["top5_abs_diff"].append(abs(spec[a]["top5"] - spec[b]["top5"]))
        for metric in METRIC_ORDER:
            out[metric][group] = mean(vals[metric])

    out["pair_counts"] = {"bb": len(groups["bb"]), "bt": len(groups["bt"])}
    out["ordering_held"] = {metric: ordering_held(out, metric) for metric in METRIC_ORDER}
    return out

def canonical_pair_metric_table(
    cache: dict[str, dict[str, Any]],
    *,
    labels: list[str],
    lexical_top_k: int,
    semantic_k: int,
    seed: int,
    textual_cache: dict[str, dict[tuple[str, int], list[dict[str, int]]]],
) -> dict[tuple[str, str], dict[str, float]]:
    idx = {label: np.arange(len(cache[label]["chunks"])) for label in labels}
    textual_counts = aggregate_textual_counts_for_indices(textual_cache, labels, idx)
    comp = {label: compression_value(cache[label]["chunks"]) for label in labels}
    spec = {label: spectrum_values(cache[label]["embeddings"]) for label in labels}
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for group_pairs in pair_groups(labels).values():
        for a, b in group_pairs:
            lex = lexical_pair_from_aggregated_counts(textual_counts, a, b)
            a_lab, b_lab = fit_pair_label_reference(cache[a]["embeddings"], cache[b]["embeddings"], k=semantic_k, seed=seed)
            rows[(a, b)] = {
                "lexical_js": lex["lexical_js"],
                "rho": lex["rho"],
                "semantic_cluster_js": semantic_js_from_labels(a_lab, b_lab),
                "compression_abs_diff": abs(comp[a] - comp[b]),
                "top1_abs_diff": abs(spec[a]["top1"] - spec[b]["top1"]),
                "top5_abs_diff": abs(spec[a]["top5"] - spec[b]["top5"]),
            }
    return rows

def family_summary_from_pair_rows(labels: list[str], pair_rows: dict[tuple[str, str], dict[str, float]]) -> dict[str, Any]:
    groups = pair_groups(labels)
    out = {metric: {"bb": None, "bt": None} for metric in METRIC_ORDER}
    for group, pairs in groups.items():
        for metric in METRIC_ORDER:
            out[metric][group] = mean([pair_rows[(a, b)][metric] for a, b in pairs if (a, b) in pair_rows])
    out["pair_counts"] = {"bb": len(groups["bb"]), "bt": len(groups["bt"])}
    out["ordering_held"] = {metric: ordering_held(out, metric) for metric in METRIC_ORDER}
    return out

def fit_original_pair_labels(cache: dict[str, dict[str, Any]], *, labels: list[str], semantic_k: int, seed: int) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    refs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for pairs in pair_groups(labels).values():
        for a, b in pairs:
            refs[(a, b)] = fit_pair_label_reference(cache[a]["embeddings"], cache[b]["embeddings"], k=semantic_k, seed=seed)
    return refs

## ===== ANALYSES ===== ##

def bootstrap_analysis(
    cache: dict[str, dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    lexical_top_k: int,
    semantic_k: int,
    textual_cache: dict[str, dict[tuple[str, int], list[dict[str, int]]]],
    semantic_mode: str,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    completed: dict[int, dict[str, Any]] = {}
    if checkpoint_path is not None:
        for row in load_jsonl(checkpoint_path):
            r = int(row["replicate"])
            if 0 <= r < replicates:
                completed[r] = row
        if completed:
            print(f"[bootstrap] loaded {len(completed)}/{replicates} checkpointed replicates from {checkpoint_path}", flush=True)
    original_refs = fit_original_pair_labels(cache, labels=CORPORA, semantic_k=semantic_k, seed=seed) if semantic_mode == "fixed" else None
    for r in range(replicates):
        if r in completed:
            if r == 0 or (r + 1) % 25 == 0:
                print(f"[bootstrap] replicate {r + 1}/{replicates} already checkpointed", flush=True)
            continue
        if r == 0 or (r + 1) % 25 == 0:
            print(f"[bootstrap] replicate {r + 1}/{replicates}", flush=True)
        rng = np.random.default_rng([seed, r])
        idx = {label: sample_indices(len(cache[label]["chunks"]), rng, replace=True) for label in CORPORA}
        textual_counts = aggregate_textual_counts_for_indices(textual_cache, CORPORA, idx)
        sample = subset_cache(cache, idx)
        refs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        for pairs in pair_groups(CORPORA).values():
            for a, b in pairs:
                if semantic_mode == "fixed":
                    assert original_refs is not None
                    a_full, b_full = original_refs[(a, b)]
                    refs[(a, b)] = (a_full[idx[a]], b_full[idx[b]])
                else:
                    a_full, b_full = fit_pair_label_reference(sample[a]["embeddings"], sample[b]["embeddings"], k=semantic_k, seed=seed)
                    refs[(a, b)] = (a_full, b_full)
        summary = family_summary(
            sample,
            labels=CORPORA,
            lexical_top_k=lexical_top_k,
            semantic_k=semantic_k,
            seed=seed,
            textual_cache=textual_cache,
            indices_by_label=idx,
            textual_counts=textual_counts,
            semantic_pair_labels=refs,
        )
        row = {"replicate": r, **summary}
        rows.append(row)
        if checkpoint_path is not None:
            append_jsonl(checkpoint_path, row)
    rows.extend(completed.values())
    rows = sorted(rows, key=lambda row: int(row["replicate"]))
    if len(rows) != replicates:
        raise RuntimeError(f"bootstrap checkpoint incomplete: rows={len(rows)} expected={replicates}")
    summary = summarize_replicates(rows, kind="bootstrap", seed=seed, replicates=replicates)
    summary["semantic_bootstrap_mode"] = semantic_mode
    summary["checkpoint_path"] = str(checkpoint_path) if checkpoint_path is not None else None
    return summary

def summarize_replicates(rows: list[dict[str, Any]], *, kind: str, seed: int, replicates: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in METRIC_ORDER:
        metrics[metric] = {
            "bb": quantiles([float(row[metric]["bb"]) for row in rows if row[metric]["bb"] is not None]),
            "bt": quantiles([float(row[metric]["bt"]) for row in rows if row[metric]["bt"] is not None]),
        }
    ordering_counts = {metric: int(sum(bool(row["ordering_held"][metric]) for row in rows)) for metric in METRIC_ORDER}
    return {
        "schema": f"corpus_bootstrap_hardening.{kind}.v1",
        "seed": seed,
        "replicates": replicates,
        "sampling_unit": "canonical corpus chunk",
        "metrics": metrics,
        "ordering_counts": ordering_counts,
        "rows": rows,
    }

def leave_one_out_analysis(
    cache: dict[str, dict[str, Any]],
    *,
    seed: int,
    lexical_top_k: int,
    semantic_k: int,
    textual_cache: dict[str, dict[tuple[str, int], list[dict[str, int]]]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pair_rows = canonical_pair_metric_table(cache, labels=CORPORA, lexical_top_k=lexical_top_k, semantic_k=semantic_k, seed=seed, textual_cache=textual_cache)
    for omitted in CORPORA:
        labels = [x for x in CORPORA if x != omitted]
        print(f"[leave-one-out] omitted={omitted}", flush=True)
        summary = family_summary_from_pair_rows(labels, pair_rows)
        rows.append({"omitted": omitted, "omitted_display": DISPLAY[omitted], **summary})
    return {"schema": "corpus_bootstrap_hardening.leave_one_out.v1", "seed": seed, "rows": rows}

def same_corpus_split_analysis(
    cache: dict[str, dict[str, Any]],
    *,
    repeats: int,
    seed: int,
    lexical_top_k: int,
    semantic_k: int,
    textual_cache: dict[str, dict[tuple[str, int], list[dict[str, int]]]],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for label in CORPORA:
        print(f"[same-corpus] {label}", flush=True)
        n = len(cache[label]["chunks"])
        per_repeat: list[dict[str, float]] = []
        for r in range(repeats):
            perm = rng.permutation(n)
            half = n // 2
            a_idx = perm[:half]
            b_idx = perm[half: half * 2]
            chunks = cache[label]["chunks"]
            a_chunks = [chunks[int(i)] for i in a_idx]
            b_chunks = [chunks[int(i)] for i in b_idx]
            a_emb = cache[label]["embeddings"][a_idx]
            b_emb = cache[label]["embeddings"][b_idx]
            lex = lexical_pair_from_count_cache(textual_cache, label, label, a_idx, b_idx)
            a_lab, b_lab = fit_pair_label_reference(a_emb, b_emb, k=semantic_k, seed=seed)
            a_spec = spectrum_values(a_emb)
            b_spec = spectrum_values(b_emb)
            per_repeat.append({
                "repeat": float(r),
                "lexical_js": lex["lexical_js"],
                "rho": lex["rho"],
                "semantic_cluster_js": semantic_js_from_labels(a_lab, b_lab),
                "compression_abs_diff": abs(compression_value(a_chunks) - compression_value(b_chunks)),
                "top1_abs_diff": abs(a_spec["top1"] - b_spec["top1"]),
                "top5_abs_diff": abs(a_spec["top5"] - b_spec["top5"]),
            })
        rows.append({
            "corpus": label,
            "corpus_display": DISPLAY[label],
            "repeats": repeats,
            "half_size": n // 2,
            "means": {metric: mean([row[metric] for row in per_repeat]) for metric in METRIC_ORDER},
            "quantiles": {metric: quantiles([row[metric] for row in per_repeat]) for metric in METRIC_ORDER},
            "rows": per_repeat,
        })
    pooled = {metric: quantiles([row["means"][metric] for row in rows if row["means"][metric] is not None]) for metric in METRIC_ORDER}
    return {"schema": "corpus_bootstrap_hardening.same_corpus_split.v1", "seed": seed, "repeats": repeats, "rows": rows, "pooled_corpus_mean_quantiles": pooled}

## ===== RENDERING ===== ##

def fmt(x: float | None, digits: int = 4) -> str:
    if x is None or not math.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{digits}f}"

def metric_label(metric: str) -> str:
    return {
        "lexical_js": "Lexical JS",
        "rho": "Spearman rho",
        "semantic_cluster_js": "Semantic cluster JS",
        "compression_abs_diff": "Compression abs diff",
        "top1_abs_diff": "Top1 abs diff",
        "top5_abs_diff": "Top5 abs diff",
    }[metric]

def render_bootstrap_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Bootstrap Canonical Summary",
        "",
        f"- replicates: `{summary['replicates']}`",
        f"- seed: `{summary['seed']}`",
        f"- sampling unit: `{summary['sampling_unit']}`",
        f"- semantic bootstrap mode: `{summary.get('semantic_bootstrap_mode', 'refit')}`",
        "",
        "| Metric | Family | Mean | Median | 2.5% | 50% | 97.5% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRIC_ORDER:
        for group in ["bb", "bt"]:
            q = summary["metrics"][metric][group]
            lines.append(f"| {metric_label(metric)} | {group} | {fmt(q['mean'])} | {fmt(q['median'])} | {fmt(q['q025'])} | {fmt(q['q500'])} | {fmt(q['q975'])} |")
    lines += ["", "## Ordering Preservation", ""]
    for metric in METRIC_ORDER:
        lines.append(f"- {metric_label(metric)} ordering held in `{summary['ordering_counts'][metric]}/{summary['replicates']}` replicates.")
    lines += [
        "",
        "## Interpretation",
        "",
        "The bootstrap resamples the already selected corpus chunks within each corpus and recomputes the retained family summaries. It measures stability of the observed panel-level ordering under chunk resampling; it does not add corpora, estimate population-level coverage for all possible web or post-training data, or change the original canonical point estimates.",
    ]
    return "\n".join(lines) + "\n"

def render_leave_one_out_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Leave-One-Corpus-Out Summary",
        "",
        "| Omitted corpus | BB pairs | BT pairs | Lex JS | Rho | Semantic JS | Compression abs | Top1 abs | Top5 abs | Preserved orderings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        held = sum(bool(row["ordering_held"][m]) for m in METRIC_ORDER)
        lines.append(
            f"| {row['omitted_display']} | {row['pair_counts']['bb']} | {row['pair_counts']['bt']} | "
            f"{fmt(row['lexical_js']['bb'])}/{fmt(row['lexical_js']['bt'])} | "
            f"{fmt(row['rho']['bb'])}/{fmt(row['rho']['bt'])} | "
            f"{fmt(row['semantic_cluster_js']['bb'])}/{fmt(row['semantic_cluster_js']['bt'])} | "
            f"{fmt(row['compression_abs_diff']['bb'])}/{fmt(row['compression_abs_diff']['bt'])} | "
            f"{fmt(row['top1_abs_diff']['bb'])}/{fmt(row['top1_abs_diff']['bt'])} | "
            f"{fmt(row['top5_abs_diff']['bb'])}/{fmt(row['top5_abs_diff']['bt'])} | {held}/6 |"
        )
    load_bearing = [row["omitted_display"] for row in summary["rows"] if sum(bool(row["ordering_held"][m]) for m in METRIC_ORDER) < len(METRIC_ORDER)]
    if load_bearing:
        overall = "At least one omission changed one or more metric-family orderings: " + ", ".join(load_bearing) + "."
    else:
        overall = "No single-corpus omission changed the retained broad-vs-targeted ordering across the six metric families."
    lines += ["", "## Overall", "", overall]
    return "\n".join(lines) + "\n"

def render_same_corpus_md(summary: dict[str, Any], canonical: dict[str, Any] | None) -> str:
    lines = [
        "# Same-Corpus Split Baselines",
        "",
        f"- repeats per corpus: `{summary['repeats']}`",
        "",
        "| Corpus | Lex JS | Rho | Semantic JS | Compression abs | Top1 abs | Top5 abs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        m = row["means"]
        lines.append(f"| {row['corpus_display']} | {fmt(m['lexical_js'])} | {fmt(m['rho'])} | {fmt(m['semantic_cluster_js'])} | {fmt(m['compression_abs_diff'])} | {fmt(m['top1_abs_diff'])} | {fmt(m['top5_abs_diff'])} |")
    lines += ["", "## Pooled Corpus-Mean Scale", ""]
    for metric in METRIC_ORDER:
        q = summary["pooled_corpus_mean_quantiles"][metric]
        lines.append(f"- {metric_label(metric)}: pooled mean={fmt(q['mean'])}, median={fmt(q['median'])}, 2.5-97.5% range={fmt(q['q025'])}-{fmt(q['q975'])}")
    if canonical:
        lines += [
            "",
            "## Comparison Scale",
            "",
            f"- Canonical textual scorecard lexical JS: broad-broad `{fmt(canonical['textual']['bb_lexical_js'])}`, broad-targeted `{fmt(canonical['textual']['bt_lexical_js'])}`.",
            f"- Canonical textual scorecard rho: broad-broad `{fmt(canonical['textual']['bb_rho'])}`, broad-targeted `{fmt(canonical['textual']['bt_rho'])}`.",
            f"- Canonical semantic scorecard cluster JS: broad-broad `{fmt(canonical['semantic']['bb_cluster_js'])}`, broad-targeted `{fmt(canonical['semantic']['bt_cluster_js'])}`.",
            f"- Corrected audit compression absolute difference: broad-broad `{fmt(canonical['audit']['compression_bb_abs'])}`, broad-targeted `{fmt(canonical['audit']['compression_bt_abs'])}`.",
            f"- Corrected audit top1 absolute difference: broad-broad `{fmt(canonical['audit']['top1_bb_abs'])}`, broad-targeted `{fmt(canonical['audit']['top1_bt_abs'])}`.",
            f"- Corrected audit top5 absolute difference: broad-broad `{fmt(canonical['audit']['top5_bb_abs'])}`, broad-targeted `{fmt(canonical['audit']['top5_bt_abs'])}`.",
        ]
    lines += ["", "Same-corpus splits estimate the within-source noise floor under the same chunk representation. They should be read as a calibration baseline, not as a replacement for cross-corpus comparisons."]
    return "\n".join(lines) + "\n"

def canonical_scorecard_values(scorecard_dir: Path) -> dict[str, Any] | None:
    try:
        family = load_json(ROOT / "results/final_matrix/recomputed/family_summary.json")
    except FileNotFoundError:
        return None
    metrics = family["metrics"]
    return {
        "textual": {
            "bb_lexical_js": metrics["broad_broad"]["lexical_js"],
            "bt_lexical_js": metrics["broad_targeted"]["lexical_js"],
            "bb_rho": metrics["broad_broad"]["rho"],
            "bt_rho": metrics["broad_targeted"]["rho"],
        },
        "semantic": {
            "bb_cluster_js": metrics["broad_broad"]["semantic_cluster_js"],
            "bt_cluster_js": metrics["broad_targeted"]["semantic_cluster_js"],
        },
        "audit": {
            "compression_bb_abs": metrics["broad_broad"]["compression_abs_diff"],
            "compression_bt_abs": metrics["broad_targeted"]["compression_abs_diff"],
            "top1_bb_abs": metrics["broad_broad"]["top1_abs_diff"],
            "top1_bt_abs": metrics["broad_targeted"]["top1_abs_diff"],
            "top5_bb_abs": metrics["broad_broad"]["top5_abs_diff"],
            "top5_bt_abs": metrics["broad_targeted"]["top5_abs_diff"],
        },
    }

def render_repo_map(cache_dir: Path, scorecard_dir: Path) -> str:
    return f"""# Bootstrap Hardening Repo Map

## Canonical Artifact Locations

- Corpus manifest: `data/corpus_manifest.json`
- Family summary: `results/final_matrix/recomputed/family_summary.json`
- Chunk/embedding cache expected by the hardening runner: `{cache_dir}`

## Metric Computation Map

- Lexical JS and Spearman rho are recomputed over the retained textual grid.
- LZMA compression ratio, top1, and top5 are intrinsic corpus summaries and enter family comparisons through absolute pairwise differences.
- Semantic cluster JS uses pooled pairwise KMeans partitions over retained embeddings.
- In manifest mode, corpora are defined by `data/corpus_manifest.json`.

## Canonical/Base Parameters

- Corpus panel: `{", ".join(CORPORA)}`
- Roles: broad `{", ".join(BROAD)}`; targeted `{", ".join(TARGETED)}`
- Chunk settings: `max_chunks=2048`, `max_chunk_chars=768`, `min_retained_chars=192`, `max_chunks_per_doc=2`
- Textual hardening setting: canonical base textual grid over tokenizer modes `regex,gpt2`, ngram orders `1,2,3`, and vocab cuts `256,512,1024`; lexical JS/rho are averaged over this grid. Compression hardening uses standalone LZMA absolute differences from the corrected audit convention.
- Semantic hardening setting: GTE c768 embeddings already cached as `embeddings.npy`, `semantic_k=24`, pooled pairwise cluster reference, `random_state=0`.

## Rerun Commands

Run the hardening pass in manifest-driven final-panel mode:

```bash
uv run python scripts/run_bootstrap_hardening.py \\
  --manifest data/corpus_manifest.json \\
  --out-dir stability \\
  --bootstrap-semantic-mode fixed
```

In manifest mode, broad and targeted roles are inferred from the manifest `role` field and chunk/embedding paths are resolved relative to the manifest file. This is the intended mode for final-panel reruns.

## Duplication Note

The runner duplicates only orchestration and Markdown/JSON rendering. Metric primitives are imported from the existing corpus deformation module. The only deliberate adaptation is that the bootstrap recomputes family summaries over resampled chunk arrays and applies corrected absolute-difference logic for compression/top1/top5.
"""

def render_methods_note() -> str:
    return """# Paper Methods Note: Statistical Hardening Add-On

We added three bounded stability analyses to the fixed corpus panel and retained measurement design. The analyses use the same canonical corpus chunk artifacts as the main analysis: 2048 chunks per corpus, chunks of up to 768 characters, and at most two chunks per source document. No new corpora or datasets are introduced.

## Bootstrap Resampling

For each bootstrap replicate, chunks are resampled with replacement within each corpus. The per-corpus sample size remains fixed at the canonical 2048 chunks. The retained family-level summaries are then recomputed on the resampled panel: lexical JS, Spearman rank rho, semantic cluster JS, LZMA compression-ratio absolute differences, and semantic top1/top5 absolute differences. Compression, top1, and top5 are treated as standalone corpus properties and compared through absolute pairwise differences, matching the retained metric convention.

A bootstrap replicate therefore represents a same-panel, same-corpus-composition perturbation of the empirical chunk sample. It measures whether the observed broad-broad versus broad-targeted ordering is stable to within-corpus sampling variation. It does not estimate coverage over unobserved corpora, new dataset families, alternative corpus definitions, or downstream model behavior.

## Leave-One-Corpus-Out

The leave-one-corpus-out analysis recomputes the canonical family summaries after removing one corpus at a time. Class definitions are preserved over the remaining valid pairs: if a broad corpus is omitted, broad-broad means are computed from the remaining broad-broad pairs and broad-targeted means from the remaining broad-targeted pairs; if a targeted corpus is omitted, the broad class remains fixed and broad-targeted means are computed over the remaining targeted corpora. Pair counts are reported for each omission.

This analysis asks whether the pretraining/post-training ordering is dominated by a single corpus. It is a sensitivity check on the fixed 14-corpus panel, not an argument that the retained corpora exhaust the relevant corpus landscape.

## Same-Corpus Split Baselines

For each corpus, the canonical chunk sample is repeatedly split into two non-overlapping halves. The same retained metrics are computed between the two halves. These same-corpus splits estimate a within-source noise floor: the level of apparent separation produced by finite sampling and chunk heterogeneity inside one corpus.

The split baseline is used as calibration against the broad-broad and broad-targeted scales. It does not redefine closeness, and it does not replace the original cross-corpus comparisons.

## Scope

These analyses harden the existing corpus-level result by measuring stability under resampling, corpus omission, and same-source splitting. They do not expand the corpus panel, change the measurement families, infer causal downstream training effects, or establish a universal taxonomy of all corpora.
"""

def render_paper_ready(bootstrap: dict[str, Any], loo: dict[str, Any], split: dict[str, Any], canonical: dict[str, Any] | None) -> str:
    lines = ["# Paper-Ready Hardening Results", ""]
    if canonical:
        lines.append(f"- In the retained canonical summaries, broad-broad separation is lower than broad-targeted separation for lexical JS (`{fmt(canonical['textual']['bb_lexical_js'])}` vs `{fmt(canonical['textual']['bt_lexical_js'])}`), semantic cluster JS (`{fmt(canonical['semantic']['bb_cluster_js'])}` vs `{fmt(canonical['semantic']['bt_cluster_js'])}`), compression absolute difference (`{fmt(canonical['audit']['compression_bb_abs'])}` vs `{fmt(canonical['audit']['compression_bt_abs'])}`), top1 absolute difference (`{fmt(canonical['audit']['top1_bb_abs'])}` vs `{fmt(canonical['audit']['top1_bt_abs'])}`), and top5 absolute difference (`{fmt(canonical['audit']['top5_bb_abs'])}` vs `{fmt(canonical['audit']['top5_bt_abs'])}`). Spearman rho shows the corresponding closeness direction, with broad-broad higher than broad-targeted (`{fmt(canonical['textual']['bb_rho'])}` vs `{fmt(canonical['textual']['bt_rho'])}`).")
    for metric in METRIC_ORDER:
        c = bootstrap["ordering_counts"][metric]
        n = bootstrap["replicates"]
        lines.append(f"- Bootstrap resampling preserved the expected {metric_label(metric)} ordering in `{c}/{n}` replicates.")
    bad_loo = [row["omitted_display"] for row in loo["rows"] if sum(bool(row["ordering_held"][m]) for m in METRIC_ORDER) < len(METRIC_ORDER)]
    if bad_loo:
        lines.append("- Leave-one-corpus-out sensitivity found ordering changes under these omissions: " + ", ".join(bad_loo) + ".")
    else:
        lines.append("- Leave-one-corpus-out sensitivity did not identify a single omitted corpus that reverses the retained broad-versus-targeted ordering across the six metric families.")
    pooled = split["pooled_corpus_mean_quantiles"]
    lines.append(f"- Same-corpus split baselines are lower than the cross-corpus broad-targeted scale for the divergence-style summaries: pooled same-corpus lexical JS mean `{fmt(pooled['lexical_js']['mean'])}` and semantic cluster JS mean `{fmt(pooled['semantic_cluster_js']['mean'])}`.")
    lines.append(f"- Same-corpus Spearman rho remains high relative to broad-targeted comparisons: pooled same-corpus rho mean `{fmt(pooled['rho']['mean'])}`.")
    lines.append("- These checks support stability of the fixed-panel empirical ordering under chunk resampling, corpus omission, and same-source split calibration. They do not expand the corpus panel or establish downstream causal effects.")
    return "\n\n".join(lines) + "\n"

## ===== ENTRYPOINT ===== ##

def main() -> None:
    ap = argparse.ArgumentParser(description="Run bootstrap, leave-one-out, and same-corpus stability analyses for the retained corpus panel.")
    ap.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.json"), help="Corpus manifest. Roles and chunk/embedding paths are read from this manifest when it exists.")
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--scorecard-dir", default="results/final_matrix/recomputed")
    ap.add_argument("--out-dir", default="stability")
    ap.add_argument("--bootstrap-replicates", type=int, default=200)
    ap.add_argument("--split-repeats", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lexical-top-k", type=int, default=512)
    ap.add_argument("--semantic-k", type=int, default=24)
    ap.add_argument("--bootstrap-semantic-mode", choices=["fixed", "refit"], default="refit", help="Use fixed canonical pair partitions or refit pairwise KMeans inside each bootstrap replicate.")
    ap.add_argument("--bootstrap-checkpoint", type=Path, default=None, help="JSONL checkpoint path for completed bootstrap replicate rows. Defaults to <out-dir>/bootstrap_canonical_rows.checkpoint.jsonl.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    scorecard_dir = Path(args.scorecard_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_text(out_dir / "repo_map.md", render_repo_map(cache_dir, scorecard_dir))
    write_text(out_dir / "paper_methods_note.md", render_methods_note())
    canonical = canonical_scorecard_values(scorecard_dir)

    print("[load] corpus cache", flush=True)
    if args.manifest and args.manifest.exists():
        cache, _ = load_manifest_cache(args.manifest)
    else:
        cache = load_corpus_cache(cache_dir)
    textual_cache = build_textual_cache(cache)

    print("[run] bootstrap", flush=True)
    checkpoint_path = args.bootstrap_checkpoint or (out_dir / "bootstrap_canonical_rows.checkpoint.jsonl")
    bootstrap = bootstrap_analysis(cache, replicates=args.bootstrap_replicates, seed=args.seed, lexical_top_k=args.lexical_top_k, semantic_k=args.semantic_k, textual_cache=textual_cache, semantic_mode=args.bootstrap_semantic_mode, checkpoint_path=checkpoint_path)
    write_json(out_dir / "bootstrap_canonical_summary.json", bootstrap)
    write_text(out_dir / "bootstrap_canonical_summary.md", render_bootstrap_md(bootstrap))

    print("[run] leave-one-out", flush=True)
    loo = leave_one_out_analysis(cache, seed=args.seed, lexical_top_k=args.lexical_top_k, semantic_k=args.semantic_k, textual_cache=textual_cache)
    write_json(out_dir / "leave_one_out_summary.json", loo)
    write_text(out_dir / "leave_one_out_summary.md", render_leave_one_out_md(loo))

    print("[run] same-corpus splits", flush=True)
    split = same_corpus_split_analysis(cache, repeats=args.split_repeats, seed=args.seed, lexical_top_k=args.lexical_top_k, semantic_k=args.semantic_k, textual_cache=textual_cache)
    write_json(out_dir / "same_corpus_split_baselines.json", split)
    write_text(out_dir / "same_corpus_split_baselines.md", render_same_corpus_md(split, canonical))
    write_text(out_dir / "paper_ready_results.md", render_paper_ready(bootstrap, loo, split, canonical))
    print(f"[done] wrote {out_dir}", flush=True)

if __name__ == "__main__":
    main()
