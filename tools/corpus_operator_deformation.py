from __future__ import annotations

import lzma
import math
import re
from typing import Any

import numpy as np
from scipy.stats import spearmanr


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def stable_tokenize(text: str) -> list[str]: return TOKEN_RE.findall(text.lower())


def normalize_dist(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(arr.shape, 1.0 / max(1, arr.size), dtype=np.float64)
    return arr / total


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p1 = normalize_dist(np.asarray(p, dtype=np.float64) + eps)
    q1 = normalize_dist(np.asarray(q, dtype=np.float64) + eps)
    m = 0.5 * (p1 + q1)
    return 0.5 * float(np.sum(p1 * np.log(p1 / m))) + 0.5 * float(np.sum(q1 * np.log(q1 / m)))


def count_tokens(chunks: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        for tok in stable_tokenize(chunk):
            counts[tok] = counts.get(tok, 0) + 1
    return counts


def compression_ratio(text: str) -> float:
    raw = text.encode("utf-8", errors="ignore")
    if not raw:
        return 0.0
    return len(lzma.compress(raw, preset=6)) / len(raw)


def covariance_spectrum(embeddings: np.ndarray) -> dict[str, float | list[float]]:
    x = embeddings.astype(np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    denom = max(1, centered.shape[0] - 1)
    if centered.shape[0] < centered.shape[1]:
        cov_like = (centered @ centered.T) / denom
    else:
        cov_like = (centered.T @ centered) / denom
    eigvals = np.flip(np.maximum(np.linalg.eigvalsh(cov_like), 0.0))
    total = float(eigvals.sum())
    top = eigvals[:10]
    return {
        "total_variance": total,
        "top_eigenvalues": [float(v) for v in top.tolist()],
        "top1_share": float(top[0] / total) if total > 0 else 0.0,
        "top5_share": float(top[:5].sum() / total) if total > 0 else 0.0,
    }


def lexical_metrics(baseline_chunks: list[str], target_chunks: list[str], *, top_k: int) -> dict[str, Any]:
    base_counts = count_tokens(baseline_chunks)
    target_counts = count_tokens(target_chunks)
    vocab = [tok for tok, _ in sorted(base_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
    if len(vocab) < 8:
        raise RuntimeError("not enough baseline vocabulary for lexical metrics")
    base_vec = np.asarray([base_counts.get(tok, 0) for tok in vocab], dtype=np.float64)
    target_vec = np.asarray([target_counts.get(tok, 0) for tok in vocab], dtype=np.float64)
    base_ranks = np.argsort(np.argsort(-base_vec))
    target_ranks = np.argsort(np.argsort(-target_vec))
    rho = float(spearmanr(base_ranks, target_ranks).statistic)
    if not math.isfinite(rho):
        rho = 0.0
    return {
        "vocab_size": len(vocab),
        "spearman_rank_rho": rho,
        "js_divergence_top_vocab": js_divergence(base_vec, target_vec),
    }
