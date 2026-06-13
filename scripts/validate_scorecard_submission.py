#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


METRIC_SUITE_VERSION = "lens_effects.release.v0.1.0"
ENTRY_SCHEMA = "training_corpus_diagnostics.scorecard_entry.v1"
REGISTRY_SCHEMA = "training_corpus_diagnostics.scorecard_registry.v1"
ALLOWED_STATUS = {"submitted", "artifact_validated", "agent_reviewed", "reproduced"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(errors: list[str], path: Path | str, msg: str) -> None:
    errors.append(f"{path}: {msg}")


def validate_entry(path: Path, entry: dict[str, Any], errors: list[str], *, public_registry: bool) -> None:
    if entry.get("schema") != ENTRY_SCHEMA:
        fail(errors, path, f"schema must be {ENTRY_SCHEMA}")
    if entry.get("metric_suite_version") != METRIC_SUITE_VERSION:
        fail(errors, path, f"metric_suite_version must be {METRIC_SUITE_VERSION}")
    if not entry.get("label"):
        fail(errors, path, "label is required")
    if not entry.get("display"):
        fail(errors, path, "display is required")
    if entry.get("status") not in ALLOWED_STATUS:
        fail(errors, path, f"status must be one of {sorted(ALLOWED_STATUS)}")
    if entry.get("interpretation") != "diagnostic_position_not_quality_ranking":
        fail(errors, path, "interpretation must be diagnostic_position_not_quality_ranking")

    source = entry.get("source")
    if not isinstance(source, dict):
        fail(errors, path, "source object is required")
    elif not source.get("source"):
        fail(errors, path, "source.source is required")

    settings = entry.get("canonical_settings")
    if not isinstance(settings, dict):
        fail(errors, path, "canonical_settings object is required")
    else:
        if settings.get("chunks_per_corpus") != 2048:
            fail(errors, path, "canonical_settings.chunks_per_corpus must be 2048 for canonical public entries")
        if settings.get("max_chunk_chars") != 768:
            fail(errors, path, "canonical_settings.max_chunk_chars must be 768")
        if settings.get("embedding_model") != "Alibaba-NLP/gte-base-en-v1.5":
            fail(errors, path, "canonical_settings.embedding_model must be Alibaba-NLP/gte-base-en-v1.5")
        if settings.get("embedding_dimension") != 768:
            fail(errors, path, "canonical_settings.embedding_dimension must be 768")

    standalone = entry.get("standalone_summary")
    if not isinstance(standalone, dict):
        fail(errors, path, "standalone_summary object is required")
    else:
        if standalone.get("n_chunks") != 2048:
            fail(errors, path, "standalone_summary.n_chunks must be 2048")
        if "compression_ratio_lzma" not in standalone:
            fail(errors, path, "standalone_summary.compression_ratio_lzma is required")
        spectrum = standalone.get("semantic_spectrum")
        if not isinstance(spectrum, dict):
            fail(errors, path, "standalone_summary.semantic_spectrum object is required")
        else:
            for key in ("top1_share", "top5_share"):
                if key not in spectrum:
                    fail(errors, path, f"standalone_summary.semantic_spectrum.{key} is required")

    artifact = entry.get("artifact")
    if not isinstance(artifact, dict):
        fail(errors, path, "artifact object is required")
    else:
        if public_registry and not artifact.get("url"):
            fail(errors, path, "artifact.url is required for public registry entries")
        sha = artifact.get("sha256")
        if public_registry and not sha:
            fail(errors, path, "artifact.sha256 is required for public registry entries")
        if sha and not HEX64.match(str(sha)):
            fail(errors, path, "artifact.sha256 must be a lowercase 64-character SHA-256 hex digest")


def validate_registry(registry: Path, errors: list[str]) -> None:
    index_path = registry / "index.json"
    if not index_path.exists():
        fail(errors, index_path, "registry index is missing")
        return
    index = read_json(index_path)
    if index.get("schema") != REGISTRY_SCHEMA:
        fail(errors, index_path, f"schema must be {REGISTRY_SCHEMA}")
    if index.get("metric_suite_version") != METRIC_SUITE_VERSION:
        fail(errors, index_path, f"metric_suite_version must be {METRIC_SUITE_VERSION}")
    entries = index.get("entries")
    if not isinstance(entries, list):
        fail(errors, index_path, "entries must be a list")
        return
    labels_seen: set[str] = set()
    for row in entries:
        if not isinstance(row, dict):
            fail(errors, index_path, "each entries item must be an object")
            continue
        label = row.get("label")
        rel = row.get("path")
        if not label:
            fail(errors, index_path, "registry entry label is required")
        if label in labels_seen:
            fail(errors, index_path, f"duplicate registry label: {label}")
        labels_seen.add(str(label))
        if row.get("status") not in ALLOWED_STATUS:
            fail(errors, index_path, f"registry status for {label} must be one of {sorted(ALLOWED_STATUS)}")
        if not rel:
            fail(errors, index_path, f"registry path for {label} is required")
            continue
        entry_path = registry / rel
        if not entry_path.exists():
            fail(errors, entry_path, f"registry entry for {label} is missing")
            continue
        entry = read_json(entry_path)
        validate_entry(entry_path, entry, errors, public_registry=True)
        if label and entry.get("label") != label:
            fail(errors, entry_path, f"entry label {entry.get('label')} does not match registry label {label}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate public scorecard registry entries or a generated scorecard submission entry.")
    ap.add_argument("--registry", type=Path, default=Path("scorecard/registry"), help="Scorecard registry directory.")
    ap.add_argument("--entry", type=Path, action="append", default=[], help="Standalone scorecard_entry.json file to validate.")
    args = ap.parse_args()

    errors: list[str] = []
    if args.registry.exists():
        validate_registry(args.registry, errors)
    elif not args.entry:
        fail(errors, args.registry, "registry directory is missing")

    for path in args.entry:
        obj = read_json(path)
        if obj.get("schema") == "training_corpus_diagnostics.scorecard_entry_bundle.v1":
            for i, entry in enumerate(obj.get("entries", [])):
                validate_entry(f"{path}:entries.{i}", entry, errors, public_registry=False)
        else:
            validate_entry(path, obj, errors, public_registry=False)

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        raise SystemExit(f"scorecard validation failed with {len(errors)} error(s)")
    print("[done] scorecard validation passed")


if __name__ == "__main__":
    main()
