#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = ["lexical_js", "rho", "semantic_cluster_js", "compression_abs_diff", "top1_abs_diff", "top5_abs_diff"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def fmt(x: float) -> str:
    return f"{float(x):.4f}"


def render_md(obj: dict[str, Any]) -> str:
    lines = [
        "# Targeted Substructure Summary",
        "",
        f"- targeted corpora: `{len(obj['targeted_labels'])}`",
        f"- targeted-targeted pairs: `{obj['pair_count']}`",
        f"- textual convention: {obj['textual_convention']}",
        "",
        "## Family Means",
        "",
        "| Metric | Targeted-targeted mean |",
        "|---|---:|",
    ]
    for metric in METRICS:
        lines.append(f"| {metric} | {fmt(obj['family_means'][metric])} |")
    lines += ["", "## Standalone Targeted Corpora", "", "| Corpus | Compression ratio | Top1 share | Top5 share |", "|---|---:|---:|---:|"]
    for row in obj["standalone_targeted"]:
        lines.append(f"| {row['display']} | {fmt(row['compression_ratio_lzma'])} | {fmt(row['top1_share'])} | {fmt(row['top5_share'])} |")
    lines += ["", "## Pair Rankings"]
    for metric in METRICS:
        lines += ["", f"### {metric}", "", "| Pair | Value |", "|---|---:|"]
        for row in obj["ranked_pairs"][metric]:
            lines.append(f"| {row['a_display']} vs {row['b_display']} | {fmt(row[metric])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build post-training/post-training substructure summary from corpus_matrix.json.")
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()

    matrix = read_json(args.matrix)
    targeted = matrix["roles"]["targeted"]
    standalone = matrix["standalone"]
    tt_rows = [row for row in matrix["pairs"] if row["pair_class"] == "targeted_targeted"]
    if not tt_rows:
        raise RuntimeError("no targeted-targeted rows found in matrix")
    family_means = {
        metric: sum(float(row[metric]) for row in tt_rows if metric in row and row[metric] is not None) / len([row for row in tt_rows if metric in row and row[metric] is not None])
        for metric in METRICS
    }
    ranked_pairs = {metric: sorted([{k: row[k] for k in ("a", "b", "a_display", "b_display")} | {metric: row[metric]} for row in tt_rows if metric in row and row[metric] is not None], key=lambda row: row[metric], reverse=(metric != "rho")) for metric in METRICS}
    standalone_targeted = []
    for label in targeted:
        row = standalone[label]
        spec = row.get("semantic_spectrum", {})
        standalone_targeted.append({
            "label": label,
            "display": row.get("display", label),
            "compression_ratio_lzma": row["compression_ratio_lzma"],
            "top1_share": spec.get("top1_share"),
            "top5_share": spec.get("top5_share"),
        })
    out = {
        "schema": "lens_effects.targeted_substructure_summary.v1",
        "matrix": str(args.matrix),
        "textual_convention": "Lexical JS and rho use the retained unordered pair rows from corpus_matrix.json; semantic cluster JS, compression, top1, and top5 are symmetric pair summaries.",
        "targeted_labels": targeted,
        "pair_count": len(tt_rows),
        "family_means": family_means,
        "ranked_pairs": ranked_pairs,
        "standalone_targeted": standalone_targeted,
    }
    write_json(args.out_json, out)
    md_path = args.out_md or args.out_json.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_md(out), encoding="utf-8")
    print(f"[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
