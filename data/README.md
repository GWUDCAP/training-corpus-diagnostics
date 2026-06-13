# Release Data

`corpus_manifest.json` is the entry point. It lists the fourteen corpus labels, role assignments, source datasets, extraction notes, chunk files, and cached embedding files.

Each `chunks.json` file is a JSON list of 2048 text chunks. Chunks are up to 768 characters, with a 192-character retained minimum under the public builder. Each `embeddings.npy` file is a NumPy array with the same row order as the corresponding chunks file.

The bundled chunks are included to make the paper artifact auditable without requiring private data or a new data pull. They are the retained matched samples used by the paper-facing computations.

For fresh upstream regeneration or panel substitution, use `scripts/build_corpus_artifacts.py` with `configs/paper_panel_sources.json` or a copied local-source config. That builder writes new chunk files, chunk metadata, provenance JSONL sidecars, embedding metadata, and a generated manifest that can be passed directly to `scripts/run_corpus_matrix.py`.

Audit boundary for the frozen retained sample: the bundled chunk files, cached embeddings, chunk metadata, and provenance sidecars support inspection and metric recomputation of the retained paper-facing sample. Provenance sidecars record source references, stream positions when available, source-text hashes, chunk hashes, character spans, sampling mode, seed, scan budget, and document sample rank.

The upstream corpora are public datasets with their own licenses and terms; public redistribution should preserve those upstream attributions and any license notices required by the dataset providers.

See `upstream_dataset_provenance.md` and `upstream_dataset_provenance.json` for the retained panel source table, license metadata, and redistribution caveats.

See `chunk_length_summary.md` and `chunk_length_summary.json` for retained chunk-length distributions.
