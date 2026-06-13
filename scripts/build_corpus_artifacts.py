#!/usr/bin/env python3
# ===== IMPORTS ===== #
## ===== STDLIB ===== ##
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

## ===== 3RD-PARTY ===== ##
import numpy as np


# ===== GLOBALS ===== #
DEFAULT_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
DEFAULT_CHUNK_CHARS = 768
DEFAULT_MAX_CHUNKS = 2048
DEFAULT_MAX_DOCS = 50000
DEFAULT_MAX_CHUNKS_PER_DOC = 2
DEFAULT_SAMPLE_SEED = 20260505


# ===== FUNCTIONS ===== #
## ===== IO ===== ##

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(base: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path)


def parse_csv(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(x).strip() for x in raw if str(x).strip()}
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def dolma_component(url: str) -> str | None:
    if "/dolma-v1_7/" not in url:
        return None
    return url.split("/dolma-v1_7/", 1)[1].split("/", 1)[0]


## ===== SOURCE READERS ===== ##

def load_hf_dataset(spec: dict[str, Any]) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("build-chunks from Hugging Face sources requires the `datasets` package") from exc

    kwargs: dict[str, Any] = {"split": spec.get("split", "train"), "streaming": bool(spec.get("streaming", True))}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    ds = load_dataset(spec["dataset"], **kwargs)
    shuffle_buffer = int(spec.get("hf_shuffle_buffer_size", 0) or 0)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=int(spec.get("sample_seed", DEFAULT_SAMPLE_SEED)), buffer_size=shuffle_buffer)
    for idx, row in enumerate(ds):
        if isinstance(row, dict):
            yield {"row": row, "source_index": idx, "source_ref": f"{spec['dataset']}:{kwargs['split']}:{idx}"}


def load_dolma_manifest(spec: dict[str, Any]) -> Iterable[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Dolma manifest mode requires the `huggingface_hub` package") from exc

    version = spec.get("config") or spec.get("name") or "v1_7"
    manifest_path = hf_hub_download("allenai/dolma", f"urls/{version}.txt", repo_type="dataset")
    headers = {"User-Agent": "lens-effects-corpus-artifacts/1.0"}
    with open(manifest_path, encoding="utf-8") as handle:
        urls = [line.strip() for line in handle if line.strip()]
    include_components = parse_csv(spec.get("dolma_include_components"))
    exclude_components = parse_csv(spec.get("dolma_exclude_components"))
    if include_components:
        urls = [url for url in urls if dolma_component(url) in include_components]
    if exclude_components:
        urls = [url for url in urls if dolma_component(url) not in exclude_components]
    if spec.get("dolma_shuffle_urls", False):
        rng = random.Random(int(spec.get("sample_seed", DEFAULT_SAMPLE_SEED)))
        rng.shuffle(urls)
    rows_per_shard = spec.get("dolma_rows_per_shard")
    rows_per_shard = None if rows_per_shard in (None, "", 0, "0") else int(rows_per_shard)
    source_index = 0
    for shard_index, url in enumerate(urls):
        component = dolma_component(url)
        req = urllib.request.Request(url, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=float(spec.get("timeout_s", 60)))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Dolma shard request failed for {url}: {exc}") from exc
        with resp:
            with gzip.GzipFile(fileobj=resp) as gz:
                for shard_row_index, raw_line in enumerate(gz):
                    if rows_per_shard is not None and shard_row_index >= rows_per_shard:
                        break
                    try:
                        row = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield {
                            "row": row,
                            "source_index": source_index,
                            "source_ref": f"{url}:{shard_row_index}",
                            "source_url": url,
                            "source_component": component,
                            "source_shard_index": shard_index,
                            "source_shard_row_index": shard_row_index,
                        }
                        source_index += 1


def load_jsonl(spec: dict[str, Any], base: Path) -> Iterable[dict[str, Any]]:
    paths = sorted(glob.glob(str(resolve_path(base, spec["path"]))))
    source_index = 0
    for path_raw in paths:
        path = Path(path_raw)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield {"row": row, "source_index": source_index, "source_ref": f"{path}:{line_no}", "source_file": str(path), "line_no": line_no}
                    source_index += 1


def load_text_glob(spec: dict[str, Any], base: Path) -> Iterable[dict[str, Any]]:
    paths = sorted(glob.glob(str(resolve_path(base, spec["path"]))))
    for idx, path_raw in enumerate(paths):
        path = Path(path_raw)
        text = path.read_text(encoding="utf-8", errors="ignore")
        yield {"row": {"text": text}, "source_index": idx, "source_ref": str(path), "source_file": str(path)}


def iter_source_rows(spec: dict[str, Any], base: Path) -> Iterable[dict[str, Any]]:
    source_type = spec.get("source_type", "hf")
    mode = spec.get("mode") or spec.get("extraction_mode") or "plain_text"
    if mode == "dolma_url_manifest" or source_type == "dolma_manifest":
        yield from load_dolma_manifest(spec)
    elif source_type == "hf":
        yield from load_hf_dataset(spec)
    elif source_type == "jsonl":
        yield from load_jsonl(spec, base)
    elif source_type == "text_glob":
        yield from load_text_glob(spec, base)
    else:
        raise ValueError(f"unsupported source_type for {spec.get('label')}: {source_type}")


## ===== EXTRACTION AND CHUNKING ===== ##

def extract_text(spec: dict[str, Any], row: dict[str, Any]) -> str | None:
    field = spec.get("field") or spec.get("source_field") or "text"
    mode = spec.get("mode") or spec.get("extraction_mode") or "plain_text"

    if mode in {"plain_text", "field_text", "dolma_url_manifest"}:
        value = row.get(field)
        return value.strip() if isinstance(value, str) and value.strip() else None

    if mode == "oasst_assistant_text":
        if row.get("role") != "assistant" or row.get("lang") != "en" or row.get("deleted") is True:
            return None
        value = row.get(field)
        return value.strip() if isinstance(value, str) and value.strip() else None

    if mode == "message_list_last_assistant":
        value = row.get(field)
        if not isinstance(value, list) or not value:
            return None
        last = value[-1]
        if not isinstance(last, dict) or last.get("role") != "assistant":
            return None
        text = last.get("content")
        return text.strip() if isinstance(text, str) and text.strip() else None

    if mode == "hh_rlhf_last_assistant":
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            return None
        matches = list(re.finditer(r"(?:^|\n)\s*Assistant:\s*", value))
        if not matches:
            return None
        text = value[matches[-1].end():]
        next_human = re.search(r"(?:^|\n|\s)Human:\s*", text)
        if next_human:
            text = text[:next_human.start()]
        return text.strip() if text.strip() else None

    raise ValueError(f"unsupported extraction mode for {spec.get('label')}: {mode}")


def chunk_text_with_provenance(text: str, *, chunk_chars: int, max_chunks_per_doc: int) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    rows: list[dict[str, Any]] = []
    start = 0
    while start < len(stripped) and len(rows) < max_chunks_per_doc:
        end = min(len(stripped), start + chunk_chars)
        chunk = stripped[start:end].strip()
        if len(chunk) >= max(64, chunk_chars // 4):
            rows.append({"text": chunk, "char_start": start, "char_end": end, "chunk_in_document": len(rows)})
        start = end
    return rows


def build_chunks_from_sources(sources_path: Path, out_root: Path, labels: set[str] | None, overwrite: bool, max_docs: int | None, max_chunks: int | None, chunk_chars: int | None, max_chunks_per_doc: int | None, skip_nonempty_docs: int | None, sampling_mode: str | None, sample_seed: int | None, sample_scan_docs: int | None, sample_reservoir_docs: int | None) -> None:
    source_doc = read_json(sources_path)
    source_base = sources_path.parent
    defaults = dict(source_doc.get("defaults", {}))
    corpora = source_doc.get("corpora") or source_doc.get("sources")
    if not isinstance(corpora, list):
        raise RuntimeError(f"{sources_path} must contain a `corpora` or `sources` list")

    manifest_rows: list[dict[str, Any]] = []
    for raw_spec in corpora:
        spec = {**defaults, **raw_spec}
        label = spec["label"]
        if labels and label not in labels:
            continue
        def setting(name: str, cli_value: Any, fallback: Any) -> Any:
            return cli_value if cli_value is not None else raw_spec.get(name, defaults.get(name, fallback))

        corpus_max_docs = int(setting("max_documents", max_docs, DEFAULT_MAX_DOCS))
        corpus_max_chunks = int(setting("max_chunks", max_chunks, DEFAULT_MAX_CHUNKS))
        corpus_chunk_chars = int(setting("chunk_chars", chunk_chars, DEFAULT_CHUNK_CHARS))
        corpus_max_chunks_per_doc = int(setting("max_chunks_per_source_document", max_chunks_per_doc, DEFAULT_MAX_CHUNKS_PER_DOC))
        corpus_skip_nonempty_docs = int(setting("skip_nonempty_documents", skip_nonempty_docs, 0))
        corpus_sampling_mode = str(setting("sampling_mode", sampling_mode, "prefix"))
        corpus_sample_seed = int(setting("sample_seed", sample_seed, DEFAULT_SAMPLE_SEED))
        corpus_scan_docs = int(setting("sample_scan_documents", sample_scan_docs, corpus_max_docs))
        corpus_reservoir_docs = int(setting("sample_reservoir_documents", sample_reservoir_docs, max(corpus_max_docs, corpus_max_chunks * 4)))
        spec["sample_seed"] = corpus_sample_seed
        chunk_dir = out_root / "chunks" / label
        chunks_path = chunk_dir / "chunks.json"
        provenance_path = chunk_dir / "chunks_provenance.jsonl"
        meta_path = chunk_dir / "chunk_metadata.json"
        if chunks_path.exists() and not overwrite:
            raise RuntimeError(f"{chunks_path} exists; pass --overwrite to rebuild it")

        chunks: list[str] = []
        provenance: list[dict[str, Any]] = []
        extracted_docs = 0
        chunked_docs = 0
        seen_rows = 0
        started = time.monotonic()
        print(f"[chunks] {label} source={spec.get('dataset') or spec.get('path')} mode={spec.get('mode', 'plain_text')} sampling={corpus_sampling_mode}", flush=True)

        def add_doc_chunks(source: dict[str, Any], text: str, source_document_index: int, doc_sample_rank: int | None, sampled_from_docs: int | None) -> None:
            nonlocal chunked_docs
            doc_chunks = chunk_text_with_provenance(text, chunk_chars=corpus_chunk_chars, max_chunks_per_doc=corpus_max_chunks_per_doc)
            doc_sha = sha256_text(text)
            if doc_chunks:
                chunked_docs += 1
            for item in doc_chunks:
                if len(chunks) >= corpus_max_chunks:
                    break
                chunk = item["text"]
                global_idx = len(chunks)
                chunks.append(chunk)
                provenance.append({
                    "label": label,
                    "chunk_index": global_idx,
                    "chunk_sha256": sha256_text(chunk),
                    "source_document_index": source_document_index,
                    "source_stream_index": source.get("source_index"),
                    "source_ref": source.get("source_ref"),
                    "source_file": source.get("source_file"),
                    "source_url": source.get("source_url"),
                    "source_component": source.get("source_component"),
                    "source_shard_index": source.get("source_shard_index"),
                    "source_shard_row_index": source.get("source_shard_row_index"),
                    "line_no": source.get("line_no"),
                    "source_text_sha256": doc_sha,
                    "source_text_chars": len(text),
                    "chunk_in_document": item["chunk_in_document"],
                    "char_start": item["char_start"],
                    "char_end": item["char_end"],
                    "field": spec.get("field") or spec.get("source_field") or "text",
                    "mode": spec.get("mode") or spec.get("extraction_mode") or "plain_text",
                    "sampling_mode": corpus_sampling_mode,
                    "sample_seed": corpus_sample_seed,
                    "sample_scan_documents": corpus_scan_docs if corpus_sampling_mode == "reservoir" else None,
                    "sample_reservoir_documents": corpus_reservoir_docs if corpus_sampling_mode == "reservoir" else None,
                    "sampled_from_nonempty_documents": sampled_from_docs,
                    "document_sample_rank": doc_sample_rank,
                })

        if corpus_sampling_mode == "prefix":
            for source in iter_source_rows(spec, source_base):
                seen_rows += 1
                text = extract_text(spec, source["row"])
                if not text:
                    continue
                if extracted_docs < corpus_skip_nonempty_docs:
                    extracted_docs += 1
                    continue
                source_document_index = extracted_docs
                extracted_docs += 1
                add_doc_chunks(source, text, source_document_index, None, None)
                if extracted_docs >= corpus_max_docs or len(chunks) >= corpus_max_chunks:
                    break
        elif corpus_sampling_mode == "reservoir":
            rng = random.Random(corpus_sample_seed)
            reservoir: list[dict[str, Any]] = []
            nonempty_docs_seen = 0
            skipped_docs = 0
            for source in iter_source_rows(spec, source_base):
                seen_rows += 1
                text = extract_text(spec, source["row"])
                if not text:
                    continue
                if skipped_docs < corpus_skip_nonempty_docs:
                    skipped_docs += 1
                    continue
                source_document_index = nonempty_docs_seen
                nonempty_docs_seen += 1
                record = {"source": source, "text": text, "source_document_index": source_document_index}
                if len(reservoir) < corpus_reservoir_docs:
                    reservoir.append(record)
                else:
                    j = rng.randrange(nonempty_docs_seen)
                    if j < corpus_reservoir_docs:
                        reservoir[j] = record
                if nonempty_docs_seen >= corpus_scan_docs:
                    break
            extracted_docs = nonempty_docs_seen
            rng.shuffle(reservoir)
            for doc_sample_rank, record in enumerate(reservoir):
                if len(chunks) >= corpus_max_chunks:
                    break
                add_doc_chunks(record["source"], record["text"], record["source_document_index"], doc_sample_rank, nonempty_docs_seen)
        else:
            raise ValueError(f"unsupported sampling_mode for {label}: {corpus_sampling_mode}")

        if len(chunks) < corpus_max_chunks and not spec.get("allow_short", False):
            raise RuntimeError(f"{label} produced only {len(chunks)} chunks; expected {corpus_max_chunks}")

        write_json(chunks_path, chunks[:corpus_max_chunks])
        write_jsonl(provenance_path, provenance[:corpus_max_chunks])
        write_json(meta_path, {
            "schema": "lens_effects.chunk_metadata.v1",
            "label": label,
            "source_spec": spec,
            "chunk_count": len(chunks[:corpus_max_chunks]),
            "source_rows_seen": seen_rows,
            "source_documents_extracted": extracted_docs,
            "source_documents_with_chunks": chunked_docs,
            "chunk_chars": corpus_chunk_chars,
            "max_documents": corpus_max_docs,
            "max_chunks": corpus_max_chunks,
            "max_chunks_per_source_document": corpus_max_chunks_per_doc,
            "skip_nonempty_documents": corpus_skip_nonempty_docs,
            "sampling_mode": corpus_sampling_mode,
            "sample_seed": corpus_sample_seed,
            "sample_scan_documents": corpus_scan_docs if corpus_sampling_mode == "reservoir" else None,
            "sample_reservoir_documents": corpus_reservoir_docs if corpus_sampling_mode == "reservoir" else None,
            "chunks_sha256": sha256_file(chunks_path),
            "provenance_sha256": sha256_file(provenance_path),
            "created_unix": time.time(),
            "elapsed_sec": time.monotonic() - started,
        })

        manifest_rows.append({
            "label": label,
            "display": spec.get("display", label),
            "role": spec.get("role", "broad"),
            "stage_role": spec.get("stage_role"),
            "subrole": spec.get("subrole"),
            "source": spec.get("dataset") or spec.get("path"),
            "source_config": spec.get("config"),
            "source_split": spec.get("split", "train"),
            "source_field": spec.get("field") or spec.get("source_field") or "text",
            "source_type": spec.get("source_type", "hf"),
            "extraction_mode": spec.get("mode") or spec.get("extraction_mode") or "plain_text",
            "chunks": str((Path("chunks") / label / "chunks.json").as_posix()),
            "chunk_metadata": str((Path("chunks") / label / "chunk_metadata.json").as_posix()),
            "chunk_provenance": str((Path("chunks") / label / "chunks_provenance.jsonl").as_posix()),
            "embeddings": str((Path("embeddings") / label / "embeddings.npy").as_posix()),
        })
        print(f"[chunks] {label} wrote {len(chunks[:corpus_max_chunks])} chunks from {extracted_docs} extracted docs ({chunked_docs} with chunks)", flush=True)

    write_json(out_root / "corpus_manifest.generated.json", {
        "schema": "lens_effects.corpus_manifest.v1",
        "notes": [
            "Generated by scripts/build_corpus_artifacts.py build-chunks.",
            "Paths are relative to this manifest file and can be passed to scripts/run_corpus_matrix.py after embeddings are built.",
            "Corpus-specific chunk settings are recorded in each chunk_metadata file.",
        ],
        "chunking": {
            "default_chunks_per_corpus": int(max_chunks if max_chunks is not None else defaults.get("max_chunks", DEFAULT_MAX_CHUNKS)),
            "default_chunk_chars": int(chunk_chars if chunk_chars is not None else defaults.get("chunk_chars", DEFAULT_CHUNK_CHARS)),
            "default_max_chunks_per_source_document": int(max_chunks_per_doc if max_chunks_per_doc is not None else defaults.get("max_chunks_per_source_document", DEFAULT_MAX_CHUNKS_PER_DOC)),
            "default_skip_nonempty_documents": int(skip_nonempty_docs if skip_nonempty_docs is not None else defaults.get("skip_nonempty_documents", 0)),
            "default_sampling_mode": str(sampling_mode if sampling_mode is not None else defaults.get("sampling_mode", "prefix")),
            "default_sample_seed": int(sample_seed if sample_seed is not None else defaults.get("sample_seed", DEFAULT_SAMPLE_SEED)),
        },
        "embedding": {"model": DEFAULT_MODEL, "dimension": 768, "note": "Build with scripts/build_corpus_artifacts.py build-embeddings."},
        "corpora": manifest_rows,
    })
    print(f"[chunks] wrote manifest {out_root / 'corpus_manifest.generated.json'}", flush=True)


## ===== EMBEDDINGS ===== ##

def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def build_embeddings(manifest_path: Path, out_root: Path, labels: set[str] | None, overwrite: bool, model_name: str, batch_size: int, max_length: int, device_arg: str, trust_remote_code: bool) -> None:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("build-embeddings requires `torch` and `transformers`") from exc

    manifest = read_json(manifest_path)
    manifest_base = manifest_path.parent
    corpora = manifest["corpora"]
    selected = [row for row in corpora if labels is None or row["label"] in labels]
    if not selected:
        raise RuntimeError("no corpora selected for embedding")

    device = device_arg
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embed] loading {model_name} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(device)
    model.eval()

    for row in selected:
        label = row["label"]
        chunks_path = resolve_path(manifest_base, row["chunks"])
        out_path = out_root / "embeddings" / label / "embeddings.npy"
        meta_path = out_root / "embeddings" / label / "embedding_metadata.json"
        if out_path.exists() and not overwrite:
            raise RuntimeError(f"{out_path} exists; pass --overwrite to rebuild it")
        chunks = read_json(chunks_path)
        if not isinstance(chunks, list) or not all(isinstance(x, str) for x in chunks):
            raise RuntimeError(f"{chunks_path} must be a JSON list of strings")

        print(f"[embed] {label} chunks={len(chunks)}", flush=True)
        arrays: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                toks = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                toks = {key: val.to(device) for key, val in toks.items()}
                out = model(**toks)
                emb = mean_pool(out.last_hidden_state, toks["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                arrays.append(emb.cpu().numpy().astype(np.float32))
                print(f"[embed] {label} {start + len(batch)}/{len(chunks)}", flush=True)
        embeddings = np.concatenate(arrays, axis=0).astype(np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, embeddings)
        write_json(meta_path, {
            "schema": "lens_effects.embedding_metadata.v1",
            "label": label,
            "model_name": model_name,
            "trust_remote_code": trust_remote_code,
            "batch_size": batch_size,
            "max_length": max_length,
            "device": device,
            "shape": list(embeddings.shape),
            "chunks_path": str(chunks_path),
            "chunks_sha256": sha256_file(chunks_path),
            "embeddings_sha256": sha256_file(out_path),
            "torch": torch.__version__,
            "transformers_model_class": model.__class__.__name__,
            "created_unix": time.time(),
        })
        print(f"[embed] {label} wrote {tuple(embeddings.shape)}", flush=True)


## ===== CLI ===== ##

def parse_labels(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source chunks, provenance, and GTE embeddings for lens-effects corpus matrices.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_chunks = sub.add_parser("build-chunks", help="Pull/localize corpus sources and write chunks plus provenance.")
    p_chunks.add_argument("--sources", type=Path, required=True, help="JSON source spec with a `corpora` list.")
    p_chunks.add_argument("--out-root", type=Path, default=Path("data"), help="Output root containing chunks/, embeddings/, and generated manifest.")
    p_chunks.add_argument("--labels", help="Comma-separated labels to build; default builds all corpora in the source spec.")
    p_chunks.add_argument("--max-documents", type=int)
    p_chunks.add_argument("--max-chunks", type=int)
    p_chunks.add_argument("--chunk-chars", type=int)
    p_chunks.add_argument("--max-chunks-per-source-document", type=int)
    p_chunks.add_argument("--skip-nonempty-documents", type=int)
    p_chunks.add_argument("--sampling-mode", choices=["prefix", "reservoir"])
    p_chunks.add_argument("--sample-seed", type=int)
    p_chunks.add_argument("--sample-scan-documents", type=int)
    p_chunks.add_argument("--sample-reservoir-documents", type=int)
    p_chunks.add_argument("--overwrite", action="store_true")

    p_embed = sub.add_parser("build-embeddings", help="Build normalized mean-pooled transformer embeddings from chunk files.")
    p_embed.add_argument("--manifest", type=Path, required=True, help="Corpus manifest containing chunk paths.")
    p_embed.add_argument("--out-root", type=Path, default=Path("data"), help="Output root containing embeddings/.")
    p_embed.add_argument("--labels", help="Comma-separated labels to embed; default embeds all corpora in the manifest.")
    p_embed.add_argument("--model-name", default=DEFAULT_MODEL)
    p_embed.add_argument("--batch-size", type=int, default=16)
    p_embed.add_argument("--max-length", type=int, default=512)
    p_embed.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p_embed.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    p_embed.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.cmd == "build-chunks":
        build_chunks_from_sources(
            sources_path=args.sources,
            out_root=args.out_root,
            labels=parse_labels(args.labels),
            overwrite=args.overwrite,
            max_docs=args.max_documents,
            max_chunks=args.max_chunks,
            chunk_chars=args.chunk_chars,
            max_chunks_per_doc=args.max_chunks_per_source_document,
            skip_nonempty_docs=args.skip_nonempty_documents,
            sampling_mode=args.sampling_mode,
            sample_seed=args.sample_seed,
            sample_scan_docs=args.sample_scan_documents,
            sample_reservoir_docs=args.sample_reservoir_documents,
        )
    elif args.cmd == "build-embeddings":
        build_embeddings(
            manifest_path=args.manifest,
            out_root=args.out_root,
            labels=parse_labels(args.labels),
            overwrite=args.overwrite,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device_arg=args.device,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raise AssertionError(args.cmd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
