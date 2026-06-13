#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

def read_json(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_with_hash(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": dst.as_posix(), "sha256": sha256_file(dst), "bytes": dst.stat().st_size}


def resolve(base: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else base / path


def main() -> None:
    ap = argparse.ArgumentParser(description="Package standard corpus artifacts for public lens-effects scorecard submission.")
    ap.add_argument("--manifest", type=Path, required=True, help="Corpus manifest used for the matrix run.")
    ap.add_argument("--matrix-dir", type=Path, required=True, help="Directory containing corpus_matrix.json and standalone_summary.json.")
    ap.add_argument("--labels", required=True, help="Comma-separated corpus labels to submit.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for the submission bundle.")
    ap.add_argument("--source-config", type=Path, default=None, help="Optional source recipe used to build chunks.")
    ap.add_argument("--submitter", default="", help="Optional submitter name or handle.")
    ap.add_argument("--notes", default="", help="Optional short notes for scorecard maintainers.")
    ap.add_argument("--status-tier", choices=["submitted", "artifact_validated", "agent_reviewed", "reproduced"], default="submitted", help="Initial public scorecard status tier.")
    ap.add_argument("--artifact-url", default="", help="Optional public URL for the full submitted artifact bundle.")
    ap.add_argument("--artifact-sha256", default="", help="Optional SHA-256 digest for the full submitted artifact bundle.")
    ap.add_argument("--zip", action="store_true", help="Also write <out-dir>.zip.")
    args = ap.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    if not labels:
        raise SystemExit("--labels must name at least one corpus")

    manifest = read_json(args.manifest)
    manifest_base = args.manifest.parent
    rows_by_label = {row["label"]: row for row in manifest["corpora"]}
    missing = [label for label in labels if label not in rows_by_label]
    if missing:
        raise SystemExit(f"labels not in manifest: {', '.join(missing)}")

    matrix = read_json(args.matrix_dir / "corpus_matrix.json")
    standalone = read_json(args.matrix_dir / "standalone_summary.json")
    family = read_json(args.matrix_dir / "family_summary.json") if (args.matrix_dir / "family_summary.json").exists() else None

    out = args.out_dir
    if out.exists():
        raise SystemExit(f"{out} already exists; remove it or choose a new --out-dir")
    out.mkdir(parents=True)

    copied: dict[str, Any] = {}
    submission_rows = []
    for label in labels:
        row = rows_by_label[label]
        item: dict[str, Any] = {
            "label": label,
            "display": row.get("display", label),
            "role": row.get("role"),
            "stage_role": row.get("stage_role"),
            "subrole": row.get("subrole"),
            "source": row.get("source"),
            "source_config": row.get("source_config"),
            "source_split": row.get("source_split"),
            "source_field": row.get("source_field"),
            "source_type": row.get("source_type"),
            "extraction_mode": row.get("extraction_mode"),
        }
        for key in ("chunks", "chunk_metadata", "chunk_provenance", "embeddings"):
            src = resolve(manifest_base, row.get(key))
            if src and src.exists():
                dst = out / "artifacts" / label / src.name
                item[key] = copy_with_hash(src, dst)
        emb_meta = resolve(manifest_base, f"embeddings/{label}/embedding_metadata.json")
        if emb_meta and emb_meta.exists():
            item["embedding_metadata"] = copy_with_hash(emb_meta, out / "artifacts" / label / emb_meta.name)
        if label in standalone:
            item["standalone_summary"] = standalone[label]
        submission_rows.append(item)

    pair_rows = [row for row in matrix.get("pairs", []) if row.get("a") in labels or row.get("b") in labels]
    write_json(out / "pairs_involving_submitted_corpora.json", pair_rows)
    write_json(out / "standalone_submitted_corpora.json", {label: standalone[label] for label in labels if label in standalone})
    if family is not None:
        write_json(out / "family_summary_for_panel.json", family)
    if args.source_config:
        copied["source_config"] = copy_with_hash(args.source_config, out / "source_config.json")
    copied["manifest"] = copy_with_hash(args.manifest, out / "corpus_manifest.json")
    copied["corpus_matrix"] = copy_with_hash(args.matrix_dir / "corpus_matrix.json", out / "corpus_matrix.json")

    submission = {
        "schema": "lens_effects.public_scorecard_submission.v1",
        "metric_suite_version": "lens_effects.release.v0.1.0",
        "score_interpretation": "diagnostic_position_not_quality_ranking",
        "status_tier": args.status_tier,
        "status_tier_definitions": {
            "submitted": "PR opened or generated entry supplied; not yet validated.",
            "artifact_validated": "Schema, artifact hash, canonical settings, and declared provenance pass checks.",
            "agent_reviewed": "Maintainer or reviewing agent checked provenance and interpretation language.",
            "reproduced": "Maintainers reran the public metric suite and reproduced the submitted summaries within release tolerance.",
        },
        "submitter": args.submitter,
        "notes": args.notes,
        "artifact_url": args.artifact_url,
        "artifact_sha256": args.artifact_sha256,
        "labels": labels,
        "submission_rows": submission_rows,
        "included_files": copied,
        "review_requirements": [
            "chunks are a JSON list of strings",
            "embeddings are row-aligned with chunks",
            "chunk metadata and provenance are included when generated by the public builder",
            "corpus_matrix.json was produced by scripts/run_corpus_matrix.py from the submitted manifest",
        ],
    }
    write_json(out / "scorecard_submission.json", submission)

    entries = []
    for row in submission_rows:
        ss = row.get("standalone_summary", {})
        entries.append({
            "schema": "training_corpus_diagnostics.scorecard_entry.v1",
            "metric_suite_version": "lens_effects.release.v0.1.0",
            "label": row["label"],
            "display": row["display"],
            "status": "submitted",
            "status_note": "Generated by package_scorecard_submission.py; public inclusion requires PR validation and review.",
            "role": row.get("role") or "candidate",
            "stage_role": row.get("stage_role") or "unlabeled",
            "subrole": row.get("subrole"),
            "source": {
                "source": row.get("source"),
                "source_config": row.get("source_config"),
                "source_split": row.get("source_split"),
                "source_field": row.get("source_field"),
                "source_type": row.get("source_type"),
                "extraction_mode": row.get("extraction_mode"),
            },
            "canonical_settings": {
                "chunks_per_corpus": ss.get("n_chunks"),
                "max_chunk_chars": 768,
                "min_retained_chars": 192,
                "max_chunks_per_source_document": 2,
                "embedding_model": "Alibaba-NLP/gte-base-en-v1.5",
                "embedding_dimension": 768,
            },
            "standalone_summary": ss,
            "artifact": {
                "url": args.artifact_url,
                "sha256": args.artifact_sha256,
            },
            "submitter": args.submitter,
            "notes": args.notes,
            "interpretation": "diagnostic_position_not_quality_ranking",
        })
    write_json(out / "scorecard_entry.json", entries[0] if len(entries) == 1 else {"schema": "training_corpus_diagnostics.scorecard_entry_bundle.v1", "entries": entries})

    md = [
        "# Lens Effects Scorecard Submission",
        "",
        f"- labels: `{', '.join(labels)}`",
        f"- submitter: `{args.submitter or 'unspecified'}`",
        f"- metric suite: `lens_effects.release.v0.1.0`",
        f"- status tier: `{args.status_tier}`",
        "- interpretation: diagnostic position, not a quality ranking",
        f"- artifact URL: `{args.artifact_url or 'not supplied'}`",
        f"- artifact SHA-256: `{args.artifact_sha256 or 'not supplied'}`",
        "",
        "## Included Corpora",
        "",
        "| Corpus | Role | Subrole | Chunks | Embedding shape | Top1 | Top5 | Compression |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for row in submission_rows:
        ss = row.get("standalone_summary", {})
        spec = ss.get("semantic_spectrum", {})
        shape = ""
        meta_path = row.get("embedding_metadata", {}).get("path")
        if meta_path and Path(meta_path).exists():
            shape = str(read_json(Path(meta_path)).get("shape", ""))
        elif row.get("embeddings", {}).get("path"):
            import numpy as np
            shape = str(list(np.load(row["embeddings"]["path"], mmap_mode="r").shape))
        md.append(
            f"| {row['display']} | {row.get('role') or ''} | {row.get('subrole') or ''} | "
            f"{ss.get('n_chunks', '')} | {shape} | "
            f"{float(spec.get('top1_share', 0.0)):.4f} | {float(spec.get('top5_share', 0.0)):.4f} | "
            f"{float(ss.get('compression_ratio_lzma', 0.0)):.4f} |"
        )
    md.extend([
        "",
        "## Maintainer Checks",
        "",
        "1. Verify hashes in `scorecard_submission.json`.",
        "2. Confirm chunks and embeddings are row-aligned.",
        "3. Inspect `chunk_metadata.json` and `chunks_provenance.jsonl` when present.",
        "4. Re-run `scripts/run_corpus_matrix.py` on `corpus_manifest.json` if the corpus is a candidate for the public scorecard.",
        "",
    ])
    (out / "scorecard_submission.md").write_text("\n".join(md), encoding="utf-8")

    entry_md = [
        "# Scorecard Entry Card",
        "",
        f"- metric suite: `lens_effects.release.v0.1.0`",
        f"- artifact URL: `{args.artifact_url or 'not supplied'}`",
        f"- artifact SHA-256: `{args.artifact_sha256 or 'not supplied'}`",
        f"- submitter: `{args.submitter or 'unspecified'}`",
        "",
        "These entries are diagnostic positions under the fixed metric suite, not corpus quality rankings.",
        "",
    ]
    for entry in entries:
        ss = entry.get("standalone_summary", {})
        spec = ss.get("semantic_spectrum", {})
        entry_md.extend([
            f"## {entry['display']}",
            "",
            f"- label: `{entry['label']}`",
            f"- role: `{entry['role']}`",
            f"- stage role: `{entry['stage_role']}`",
            f"- source: `{entry['source'].get('source') or 'unspecified'}`",
            f"- chunks: `{ss.get('n_chunks', '')}`",
            f"- spectral top1/top5: `{float(spec.get('top1_share', 0.0)):.4f}` / `{float(spec.get('top5_share', 0.0)):.4f}`",
            f"- compression ratio: `{float(ss.get('compression_ratio_lzma', 0.0)):.4f}`",
            "",
        ])
    (out / "scorecard_entry.md").write_text("\n".join(entry_md), encoding="utf-8")

    if args.zip:
        zip_path = out.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(out.parent))
        print(f"[done] wrote {out} and {zip_path}")
    else:
        print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
