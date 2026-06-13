# Training Corpus Diagnostics

Training Corpus Diagnostics is a small public toolkit and reproducibility package for measuring corpus-level structure before training. The first calibrated release accompanies the paper **“Lens Effects: Structured Geometry in Training Corpora.”**

The release contains the retained 14-corpus panel, cached embeddings, paper-facing result JSON, stability artifacts, figures, and scripts for verifying, recomputing, substituting, and submitting corpus diagnostics.

## Quick Verify

From the repo root:

```bash
python scripts/verify_release_artifacts.py
```

Expected result:

```text
[done] release artifact verification passed
```

This verifier checks committed paper-facing values. It does not download models, rebuild embeddings, or rerun the long bootstrap.

## What Is Here

- `paper/`: manuscript and appendix snapshots.
- `data/corpus_manifest.json`: retained 14-corpus panel manifest.
- `data/chunks/`: retained 2048-chunk samples.
- `data/embeddings/`: cached GTE c768 embeddings aligned row-for-row with chunks.
- `results/final_matrix/`: paper-facing metrics, sweeps, mixtures, and continuous embedding sensitivity.
- `stability/`: bootstrap, leave-one-out, same-corpus split, and semantic-k summaries.
- `figures/`: rendered manuscript figures in PNG, PDF, and SVG.
- `configs/`: public source recipes and local-source templates.
- `scripts/`: artifact builder, matrix runner, mixture runner, verifier, figure generator, scorecard packager, and scorecard validator.
- `scorecard/`: public registry and submission process for additional corpora.

## Three Common Workflows

### 1. Audit the Paper Artifacts

Run the verifier:

```bash
python scripts/verify_release_artifacts.py
```

Inspect the retained data:

- `data/README.md`
- `data/chunk_length_summary.md`
- `data/upstream_dataset_provenance.md`
- `release_integrity_manifest.json`

Recompute the matrix from retained chunks and cached embeddings:

```bash
python scripts/run_corpus_matrix.py \
  --manifest data/corpus_manifest.json \
  --out-dir recomputed
```

The matrix runner writes:

- `corpus_matrix.json`
- `family_summary.json`
- `standalone_summary.json`
- `directional_lexical_sensitivity.json`
- `summary.md`

The GPT-2 half of textual recomputation uses a pinned Hugging Face revision. See `environment/tokenizer_dependency.md` for cache/network details.

### 2. Run the Suite on Your Own Corpus

Copy `configs/example_local_sources.json` and edit it for your data. Local JSONL and text-glob sources are supported; Hugging Face streaming sources are supported through the public builder.

Build chunks:

```bash
python scripts/build_corpus_artifacts.py build-chunks \
  --sources my_sources.json \
  --out-root my_scorecard_run
```

Build embeddings:

```bash
python scripts/build_corpus_artifacts.py build-embeddings \
  --manifest my_scorecard_run/corpus_manifest.generated.json \
  --out-root my_scorecard_run \
  --model-name Alibaba-NLP/gte-base-en-v1.5
```

Run the matrix:

```bash
python scripts/run_corpus_matrix.py \
  --manifest my_scorecard_run/corpus_manifest.generated.json \
  --out-dir my_scorecard_run/matrix
```

For a single new corpus, include it in a manifest alongside reference corpora with `role: "candidate"` and `stage_role: "unlabeled"`. Candidate pairs are written as `candidate_pair` rows and are excluded from the retained broad/targeted family summaries.

### 3. Submit a Corpus to the Public Scorecard

Package a scorecard submission:

```bash
python scripts/package_scorecard_submission.py \
  --manifest my_scorecard_run/corpus_manifest.generated.json \
  --matrix-dir my_scorecard_run/matrix \
  --labels my_new_corpus \
  --source-config my_sources.json \
  --artifact-url https://example.org/my_new_corpus_scorecard_submission.zip \
  --artifact-sha256 <sha256-of-zip> \
  --out-dir my_new_corpus_scorecard_submission \
  --zip
```

Open a pull request that adds the generated `scorecard_entry.json` under `scorecard/registry/entries/<label>.json` and updates `scorecard/registry/index.json`. Do not commit full chunks or embeddings for new scorecard submissions; host the full bundle externally and record its SHA-256 digest.

Validate before opening the PR:

```bash
python scripts/validate_scorecard_submission.py --registry scorecard/registry
```

See `scorecard/README.md` and `CONTRIBUTING.md` for status tiers and agent review criteria.

## Regenerate a Fresh Public Panel

To pull a fresh randomized sample from the public 14-corpus recipe:

```bash
python scripts/build_corpus_artifacts.py build-chunks \
  --sources configs/paper_panel_sources_randomized_2048.json \
  --out-root reproduced_data
```

Then embed:

```bash
python scripts/build_corpus_artifacts.py build-embeddings \
  --manifest reproduced_data/corpus_manifest.generated.json \
  --out-root reproduced_data \
  --model-name Alibaba-NLP/gte-base-en-v1.5 \
  --batch-size 16 \
  --max-length 512
```

Then recompute:

```bash
python scripts/run_corpus_matrix.py \
  --manifest reproduced_data/corpus_manifest.generated.json \
  --out-dir reproduced_matrix
```

The retained builder keeps chunks up to 768 characters and discards chunks shorter than `max(64, chunk_chars // 4)`, which is 192 characters under the paper setting. The matched budget is 2048 retained chunks per corpus, not an exactly equal total-character budget.

Exact equality to the retained sample should only be expected when upstream dataset revisions, source access behavior, shuffle implementation, and sampling parameters match the release recipe.

## Deeper Recomputations

Semantic cluster-count stability:

```bash
python scripts/run_semantic_k_stability.py \
  --manifest data/corpus_manifest.json \
  --out-dir stability
```

Directed post-training mixture panel:

```bash
python scripts/run_mixture_panel.py \
  --manifest data/corpus_manifest.json \
  --out-dir results/final_matrix/mixtures
```

Bootstrap, leave-one-out, and same-corpus split stability:

```bash
python scripts/run_bootstrap_hardening.py \
  --manifest data/corpus_manifest.json \
  --scorecard-dir results/final_matrix/recomputed \
  --out-dir stability \
  --bootstrap-semantic-mode fixed
```

Continuous embedding ceiling sensitivity:

```bash
python scripts/run_continuous_embedding_sensitivity.py \
  --manifest data/corpus_manifest.json \
  --corpus-matrix results/final_matrix/recomputed/corpus_matrix.json \
  --out-json results/final_matrix/continuous_embedding_sensitivity.json \
  --out-md results/final_matrix/continuous_embedding_sensitivity.md
```

Regenerate figures:

```bash
python scripts/make_figures.py
```

## Scope

The release preserves the paper's matched-sample corpus panel and measurement design. The scripts support artifact verification, panel substitution, corpus diagnostics, and scorecard submissions. They do not make downstream causal claims.

Important metric convention: compression, spectral top1, and spectral top5 family comparisons use standalone-derived absolute pairwise differences. Do not use signed directional family means for those quantities.

