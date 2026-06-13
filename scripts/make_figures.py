#!/usr/bin/env python3
# ===== IMPORTS ===== #
## ===== STDLIB ===== ##
from __future__ import annotations

import argparse
import json
from pathlib import Path

## ===== 3RD-PARTY ===== ##
import matplotlib.pyplot as plt
import numpy as np

# ===== GLOBALS ===== #

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
OUT = ROOT / "figures"
MATRIX = ROOT / "results" / "final_matrix"
if not MATRIX.exists():
    MATRIX = REPO / "reports" / "final_matrix"

ROLE_COLOR = {"broad": "#4C78A8", "targeted": "#F58518"}
ROLE_LABEL = {"broad": "Pretraining", "targeted": "Post-training"}
FIG2_OFFSETS = {
    "FineWeb": (6, -9),
    "C4": (6, 8),
    "Dolma": (-38, -10),
    "RefinedWeb": (8, 10),
    "SlimPajama": (-20, 13),
    "Pile uncopyrighted": (6, -4),
    "Dolly 15k": (6, -12),
    "HH-RLHF rejected": (-62, 5),
    "HH-RLHF chosen": (6, 10),
    "OASST1 assistant": (7, -2),
    "UltraFeedback chosen": (7, 8),
}
SHORT = {
    "HH-RLHF chosen": "HH chosen",
    "HH-RLHF rejected": "HH rejected",
    "OASST1 assistant": "OASST1",
    "UltraFeedback chosen": "UltraFeedback",
    "Nemotron-SFT-Math": "Nemotron",
    "Magicoder OSS Instruct": "Magicoder",
    "Pile uncopyrighted": "Pile",
}

# ===== FUNCTIONS ===== #

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def short(name: str) -> str: return SHORT.get(name, name)

def savefig(name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    stem = Path(name).stem
    plt.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close()

def fig1() -> None:
    summary = load_json(MATRIX / "recomputed" / "family_summary.json")["metrics"]
    bb = summary["broad_broad"]
    bt = summary["broad_targeted"]
    rows = [
        ("Lexical JS", bb["lexical_js"], bt["lexical_js"]),
        ("Spearman rho", bb["rho"], bt["rho"]),
        ("Cluster JS", bb["semantic_cluster_js"], bt["semantic_cluster_js"]),
        ("Compression\nabs diff", bb["compression_abs_diff"], bt["compression_abs_diff"]),
        ("Top1\nabs diff", bb["top1_abs_diff"], bt["top1_abs_diff"]),
        ("Top5\nabs diff", bb["top5_abs_diff"], bt["top5_abs_diff"]),
    ]
    x = np.arange(len(rows))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.bar(x - w / 2, [r[1] for r in rows], width=w, label="Within pretraining", color=ROLE_COLOR["broad"])
    ax.bar(x + w / 2, [r[2] for r in rows], width=w, label="Pretraining/post-training", color=ROLE_COLOR["targeted"])
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("Metric value")
    ax.set_title("Primary family summaries")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    savefig("fig1_primary_regime_summary.png")

def fig2() -> None:
    data = load_json(MATRIX / "recomputed" / "standalone_summary.json")
    rows = sorted(data.values(), key=lambda r: (r["role"] != "broad", r["semantic_spectrum"]["top5_share"]))
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for row in rows:
        x = row["compression_ratio_lzma"]
        y = row["semantic_spectrum"]["top5_share"]
        role = row["role"]
        ax.scatter(x, y, s=78, color=ROLE_COLOR[role], edgecolor="white", linewidth=0.8)
        ax.annotate(short(row["display"]), (x, y), xytext=FIG2_OFFSETS.get(row["display"], (5, 4)), textcoords="offset points", fontsize=7.5)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=ROLE_COLOR["broad"], label=ROLE_LABEL["broad"]),
        plt.Line2D([0], [0], marker="o", linestyle="", color=ROLE_COLOR["targeted"], label=ROLE_LABEL["targeted"]),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    ax.set_xlabel("LZMA compression ratio")
    ax.set_ylabel("Spectral top5 share")
    ax.set_title("Standalone corpus structure")
    ax.grid(alpha=0.25)
    savefig("fig2_standalone_spectrum.png")

def fig3() -> None:
    data = load_json(MATRIX / "targeted_substructure_summary.json")
    by_pair = {}
    for metric, key in [("lexical_js", "lexical_js"), ("rho", "rho"), ("semantic_cluster_js", "semantic_cluster_js")]:
        for row in data["ranked_pairs"][metric]:
            pair = tuple(sorted([row["a_display"], row["b_display"]]))
            by_pair.setdefault(pair, {})[metric] = row[key]
    rows = sorted([(a, b, vals["lexical_js"], vals["rho"], vals["semantic_cluster_js"]) for (a, b), vals in by_pair.items()], key=lambda r: r[2], reverse=True)
    labels = [f"{short(a)} vs {short(b)}" for a, b, _, _, _ in rows]
    y = np.arange(len(rows))
    fig, ax1 = plt.subplots(figsize=(10.6, 8.4))
    ax1.barh(y, [r[2] for r in rows], color="#E45756", alpha=0.72, label="Lexical JS")
    ax1.plot([r[4] for r in rows], y, color="#54A24B", marker=".", linewidth=1.3, label="Cluster JS")
    ax1.set_xlabel("Divergence")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=7.0)
    ax1.invert_yaxis()
    ax2 = ax1.twiny()
    ax2.plot([r[3] for r in rows], y, color=ROLE_COLOR["broad"], marker="o", markersize=2.8, linewidth=1.2, label="Spearman rho")
    ax2.set_xlabel("Spearman rho")
    ax1.set_title("Post-training/post-training internal structure")
    ax1.grid(axis="x", alpha=0.2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower right")
    savefig("fig4_targeted_structure.png")

def fig4() -> None:
    bundle = load_json(MATRIX / "mixtures" / "targeted_mixture_panel.json")["targeted_mix_bundle"]
    selected = [
        ("HH chosen -> HH rejected", "hh_rlhf_chosen__mix__hh_rlhf_rejected"),
        ("OASST1 -> UltraFeedback", "oasst1_assistant_en__mix__ultrafeedback_binarized_chosen"),
        ("OASST1 -> Nemotron", "oasst1_assistant_en__mix__nemotron_math"),
        ("UltraFeedback -> Nemotron", "ultrafeedback_binarized_chosen__mix__nemotron_math"),
        ("Magicoder -> HH chosen", "magicoder_oss_instruct__mix__hh_rlhf_chosen"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharex=True)
    for label, key in selected:
        rows = bundle[key]
        alpha = [r["alpha_target"] for r in rows]
        axes[0].plot(alpha, [r["lexical_js"] for r in rows], marker="o", label=label)
        axes[1].plot(alpha, [r["cluster_js"] for r in rows], marker="o", label=label)
    axes[0].set_title("Mixture lexical movement")
    axes[0].set_ylabel("Lexical JS")
    axes[1].set_title("Mixture semantic movement")
    axes[1].set_ylabel("Cluster JS")
    for ax in axes:
        ax.set_xlabel("Target mixture fraction")
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    savefig("fig5_mixture_trajectories.png")

def fig5() -> None:
    semantic = load_json(MATRIX / "sweeps" / "semantic_summary.json")["rows"]
    k_rows = load_json(MATRIX / "stability" / "semantic_k_stability.json")["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    enc_rows = [("GTE", "gte_c768"), ("BGE", "bge_c768"), ("MiniLM", "minilm_c768")]
    chunk_rows = [("256", "gte_c256"), ("512", "gte_c512"), ("768", "gte_c768"), ("1024", "gte_c1024")]
    for ax, rows, title, xlabel in [
        (axes[0], enc_rows, "Encoder family", "Encoder"),
        (axes[1], chunk_rows, "GTE chunk setting", "Setting"),
    ]:
        x = np.arange(len(rows))
        ax.plot(x, [semantic[k]["means"]["bb_cluster_js"] for _, k in rows], marker="o", label="Within pretraining")
        ax.plot(x, [semantic[k]["means"]["bt_cluster_js"] for _, k in rows], marker="o", label="Pretraining/post-training")
        ax.set_xticks(x)
        ax.set_xticklabels([n for n, _ in rows])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cluster JS")
        ax.grid(alpha=0.25)
    xk = np.arange(len(k_rows))
    axes[2].plot(xk, [r["bb_cluster_js"] for r in k_rows], marker="o", label="Within pretraining")
    axes[2].plot(xk, [r["bt_cluster_js"] for r in k_rows], marker="o", label="Pretraining/post-training")
    axes[2].set_xticks(xk)
    axes[2].set_xticklabels([str(r["k"]) for r in k_rows])
    axes[2].set_title("Cluster count")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Cluster JS")
    axes[2].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    savefig("fig6_robustness_summaries.png")

def fig6() -> None:
    boot = load_json(MATRIX / "stability" / "bootstrap_canonical_summary.json")
    metrics = ["lexical_js", "rho", "semantic_cluster_js", "compression_abs_diff", "top1_abs_diff", "top5_abs_diff"]
    labels = ["Lex JS", "rho", "Cluster JS", "Compression", "Top1", "Top5"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    bb = [boot["metrics"][m]["bb"]["mean"] for m in metrics]
    bt = [boot["metrics"][m]["bt"]["mean"] for m in metrics]
    bb_lo = [boot["metrics"][m]["bb"]["mean"] - boot["metrics"][m]["bb"]["q025"] for m in metrics]
    bb_hi = [boot["metrics"][m]["bb"]["q975"] - boot["metrics"][m]["bb"]["mean"] for m in metrics]
    bt_lo = [boot["metrics"][m]["bt"]["mean"] - boot["metrics"][m]["bt"]["q025"] for m in metrics]
    bt_hi = [boot["metrics"][m]["bt"]["q975"] - boot["metrics"][m]["bt"]["mean"] for m in metrics]
    ax.errorbar(x - 0.08, bb, yerr=[bb_lo, bb_hi], fmt="o", color=ROLE_COLOR["broad"], label="Within pretraining")
    ax.errorbar(x + 0.08, bt, yerr=[bt_lo, bt_hi], fmt="o", color=ROLE_COLOR["targeted"], label="Pretraining/post-training")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Bootstrap stability, 200 resamples")
    ax.set_ylabel("Bootstrap family mean")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    savefig("fig3_bootstrap_stability.png")

def main() -> None:
    global OUT, MATRIX
    ap = argparse.ArgumentParser(description="Regenerate paper figures from matrix/stability JSON artifacts.")
    ap.add_argument("--matrix-dir", type=Path, default=MATRIX)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    MATRIX = args.matrix_dir
    OUT = args.out_dir
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6]:
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"[skip] {fn.__name__}: missing {exc.filename}")
    print(f"[done] wrote figures to {OUT}")

if __name__ == "__main__":
    main()
