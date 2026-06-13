#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from statistics import mean
from pathlib import Path
from typing import Any


EXPECTED = {
    "results/final_matrix/recomputed/family_summary.json:metrics.broad_broad.lexical_js": 0.0834,
    "results/final_matrix/recomputed/family_summary.json:metrics.broad_targeted.lexical_js": 0.1894,
    "results/final_matrix/recomputed/family_summary.json:metrics.broad_broad.rho": 0.6156,
    "results/final_matrix/recomputed/family_summary.json:metrics.broad_targeted.rho": 0.4038,
    "results/final_matrix/recomputed/family_summary.json:metrics.broad_broad.semantic_cluster_js": 0.0630,
    "results/final_matrix/recomputed/family_summary.json:metrics.broad_targeted.semantic_cluster_js": 0.3737,
    "results/final_matrix/recomputed/family_summary.json:metrics.broad_broad.compression_abs_diff": 0.0077,
    "results/final_matrix/recomputed/family_summary.json:metrics.broad_targeted.compression_abs_diff": 0.0515,
    "results/final_matrix/recomputed/family_summary.json:metrics.broad_broad.top1_abs_diff": 0.0048,
    "results/final_matrix/recomputed/family_summary.json:metrics.broad_targeted.top1_abs_diff": 0.0237,
    "results/final_matrix/recomputed/family_summary.json:metrics.broad_broad.top5_abs_diff": 0.0065,
    "results/final_matrix/recomputed/family_summary.json:metrics.broad_targeted.top5_abs_diff": 0.0513,
    "stability/bootstrap_canonical_summary.json:ordering_counts.lexical_js": 200,
    "stability/bootstrap_canonical_summary.json:ordering_counts.rho": 200,
    "stability/bootstrap_canonical_summary.json:ordering_counts.semantic_cluster_js": 200,
    "stability/bootstrap_canonical_summary.json:ordering_counts.compression_abs_diff": 200,
    "stability/bootstrap_canonical_summary.json:ordering_counts.top1_abs_diff": 200,
    "stability/bootstrap_canonical_summary.json:ordering_counts.top5_abs_diff": 200,
    "stability/bootstrap_canonical_summary.json:rows.0.pair_counts.bb": 15,
    "stability/bootstrap_canonical_summary.json:rows.0.pair_counts.bt": 48,
    "results/final_matrix/sweeps/semantic_summary.json:rows.gte_c256.means.bt_cluster_js": 0.3400,
    "results/final_matrix/sweeps/semantic_summary.json:rows.gte_c768.means.bt_cluster_js": 0.3737,
    "results/final_matrix/sweeps/semantic_summary.json:rows.minilm_c768.means.bt_cluster_js": 0.3018,
    "results/final_matrix/sweeps/textual_grid_summary.json:setting_count": 54,
    "results/final_matrix/sweeps/textual_grid_summary.json:row_count": 9828,
    "results/final_matrix/sweeps/textual_grid_summary.json:rows.0.bt_lexical_js_mean": 0.0882,
    "results/final_matrix/sweeps/textual_grid_summary.json:rows.0.bt_compression_gap_mean": 0.0547,
    "stability/same_corpus_split_baselines.json:pooled_corpus_mean_quantiles.lexical_js.mean": 0.0294,
    "stability/same_corpus_split_baselines.json:pooled_corpus_mean_quantiles.semantic_cluster_js.mean": 0.0055,
    "data/chunk_length_summary.json:settings.chunks_per_corpus": 2048,
    "data/chunk_length_summary.json:pooled_by_role.broad.mean_chars": 697.7,
    "data/chunk_length_summary.json:pooled_by_role.targeted.mean_chars": 561.5,
    "results/final_matrix/mixtures/targeted_mixture_panel.json:targeted_mix_bundle.hh_rlhf_rejected__mix__hh_rlhf_chosen.5.cluster_js": 0.0062,
    "results/final_matrix/mixtures/targeted_mixture_panel.json:targeted_mix_bundle.hh_rlhf_rejected__mix__hh_rlhf_chosen.5.lexical_js": 0.0130,
    "results/final_matrix/mixtures/targeted_mixture_panel.json:targeted_mix_bundle.hh_rlhf_rejected__mix__hh_rlhf_chosen.5.lexical_rank_rho": 0.8336,
    "results/final_matrix/mixtures/targeted_mixture_panel.json:targeted_mix_bundle.hh_rlhf_chosen__mix__hh_rlhf_rejected.5.cluster_js": 0.0058,
    "results/final_matrix/mixtures/targeted_mixture_panel.json:targeted_mix_bundle.hh_rlhf_chosen__mix__hh_rlhf_rejected.5.lexical_js": 0.0075,
    "results/final_matrix/mixtures/targeted_mixture_panel.json:targeted_mix_bundle.hh_rlhf_chosen__mix__hh_rlhf_rejected.5.lexical_rank_rho": 0.8744,
    "results/final_matrix/mixtures/targeted_mixture_panel.json:targeted_mix_bundle.nemotron_math__mix__hh_rlhf_chosen.5.cluster_js": 0.6887,
    "results/final_matrix/mixtures/targeted_mixture_panel.json:targeted_mix_bundle.nemotron_math__mix__hh_rlhf_rejected.5.cluster_js": 0.6819,
    "results/final_matrix/continuous_embedding_sensitivity.json:family_summary.broad_broad.sliced_wasserstein_l1": 0.0026,
    "results/final_matrix/continuous_embedding_sensitivity.json:family_summary.broad_targeted.sliced_wasserstein_l1": 0.0084,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dig(obj: Any, dotted: str) -> Any:
    cur = obj
    for key in dotted.split("."):
        cur = cur[int(key)] if isinstance(cur, list) else cur[key]
    return cur


def check_value(name: str, actual: Any, expected: Any, tol: float, failures: list[str]) -> None:
    if isinstance(expected, bool):
        ok = actual is expected
    elif isinstance(expected, int):
        ok = actual == expected
    else:
        ok = math.isfinite(float(actual)) and abs(float(actual) - expected) <= tol
    status = "ok" if ok else "FAIL"
    print(f"[{status}] {name}: actual={actual} expected={expected}")
    if not ok:
        failures.append(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that release artifacts contain the expected paper-facing values.")
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tol", type=float, default=5e-4)
    args = parser.parse_args()

    failures: list[str] = []
    for spec, expected in EXPECTED.items():
        file_name, path = spec.split(":", 1)
        actual = dig(read_json(args.release_root / file_name), path)
        check_value(spec, actual, expected, args.tol, failures)

    textual = read_json(args.release_root / "results/final_matrix/sweeps/textual_grid_summary.json")
    gpt2_rows = [row for row in textual["rows"] if row["tokenizer_mode"] == "gpt2"]
    derived = {
        "derived:gpt2_subset.bb_lexical_js_mean": mean(row["bb_lexical_js_mean"] for row in gpt2_rows),
        "derived:gpt2_subset.bt_lexical_js_mean": mean(row["bt_lexical_js_mean"] for row in gpt2_rows),
        "derived:gpt2_subset.bb_rho_mean": mean(row["bb_rho_mean"] for row in gpt2_rows),
        "derived:gpt2_subset.bt_rho_mean": mean(row["bt_rho_mean"] for row in gpt2_rows),
    }
    for name, expected in {
        "derived:gpt2_subset.bb_lexical_js_mean": 0.0976,
        "derived:gpt2_subset.bt_lexical_js_mean": 0.1987,
        "derived:gpt2_subset.bb_rho_mean": 0.5987,
        "derived:gpt2_subset.bt_rho_mean": 0.3862,
    }.items():
        check_value(name, derived[name], expected, args.tol, failures)

    loo = read_json(args.release_root / "stability/leave_one_out_summary.json")
    pile = next(row for row in loo["rows"] if row["omitted_display"] == "Pile uncopyrighted")
    for name, actual, expected in [
        ("derived:leave_one_out.Pile.lexical_js.bb", pile["lexical_js"]["bb"], 0.0470),
        ("derived:leave_one_out.Pile.semantic_cluster_js.bb", pile["semantic_cluster_js"]["bb"], 0.0247),
        ("derived:leave_one_out.Pile.top5_abs_diff.bb", pile["top5_abs_diff"]["bb"], 0.0006),
    ]:
        check_value(name, actual, expected, args.tol, failures)

    if failures:
        raise SystemExit(f"release artifact verification failed for {len(failures)} value(s)")
    print("[done] release artifact verification passed")


if __name__ == "__main__":
    main()
