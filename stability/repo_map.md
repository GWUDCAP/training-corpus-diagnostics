# Release Artifact Map

## Canonical Artifact Locations

- Corpus manifest: `data/corpus_manifest.json`
- Family summary: `results/final_matrix/recomputed/family_summary.json`
- Pairwise matrix: `results/final_matrix/recomputed/corpus_matrix.json`
- Standalone summaries: `results/final_matrix/recomputed/standalone_summary.json`
- Directional lexical sensitivity: `results/final_matrix/recomputed/directional_lexical_sensitivity.json`
- Post-training substructure: `results/final_matrix/targeted_substructure_summary.json`
- Directed post-training mixtures: `results/final_matrix/mixtures/targeted_mixture_panel.json`
- Continuous embedding ceiling sensitivity: `results/final_matrix/continuous_embedding_sensitivity.json`

## Stability Artifacts

- Bootstrap hardening: `stability/bootstrap_canonical_summary.json`
- Leave-one-corpus-out: `stability/leave_one_out_summary.json`
- Same-corpus split baselines: `stability/same_corpus_split_baselines.json`
- Semantic cluster-count stability: `stability/semantic_k_stability.json`
- Semantic encoder/chunk-length sweep: `results/final_matrix/sweeps/semantic_summary.json`
- Textual-grid sweep: `results/final_matrix/sweeps/textual_grid_summary.json`

## Canonical Settings

- Corpus panel: 6 pretraining corpora and 8 post-training corpora
- Chunk settings: `max_chunks=2048`, `max_chunk_chars=768`, `min_retained_chars=192`, `max_chunks_per_doc=2`
- Sampling: randomized reservoir sampling from source streams with provenance sidecars
- Retained embedding setting: `Alibaba-NLP/gte-base-en-v1.5`, GTE c768, `semantic_k=24`, `random_state=0`
