# Paper Methods Note: Statistical Hardening Add-On

We added three bounded stability analyses to the fixed corpus panel and retained measurement design. The analyses use the same canonical corpus chunk artifacts as the main analysis: 2048 chunks per corpus, chunks of up to 768 characters, and at most two chunks per source document. No new corpora or datasets are introduced.

## Bootstrap Resampling

For each bootstrap replicate, chunks are resampled with replacement within each corpus. The per-corpus sample size remains fixed at the canonical 2048 chunks. The retained family-level summaries are then recomputed on the resampled panel: lexical JS, Spearman rank rho, semantic cluster JS, LZMA compression-ratio absolute differences, and semantic top1/top5 absolute differences. Compression, top1, and top5 are treated as standalone corpus properties and compared through absolute pairwise differences, matching the retained metric convention.

A bootstrap replicate therefore represents a same-panel, same-corpus-composition perturbation of the empirical chunk sample. It measures whether the observed broad-broad versus broad-targeted ordering is stable to within-corpus sampling variation. It does not estimate coverage over unobserved corpora, new dataset families, alternative corpus definitions, or downstream model behavior.

## Leave-One-Corpus-Out

The leave-one-corpus-out analysis recomputes the canonical family summaries after removing one corpus at a time. Class definitions are preserved over the remaining valid pairs: if a broad corpus is omitted, broad-broad means are computed from the remaining broad-broad pairs and broad-targeted means from the remaining broad-targeted pairs; if a targeted corpus is omitted, the broad class remains fixed and broad-targeted means are computed over the remaining targeted corpora. Pair counts are reported for each omission.

This analysis asks whether the pretraining/post-training ordering is dominated by a single corpus. It is a sensitivity check on the fixed 14-corpus panel, not an argument that the retained corpora exhaust the relevant corpus landscape.

## Same-Corpus Split Baselines

For each corpus, the canonical chunk sample is repeatedly split into two non-overlapping halves. The same retained metrics are computed between the two halves. These same-corpus splits estimate a within-source noise floor: the level of apparent separation produced by finite sampling and chunk heterogeneity inside one corpus.

The split baseline is used as calibration against the broad-broad and broad-targeted scales. It does not redefine closeness, and it does not replace the original cross-corpus comparisons.

## Scope

These analyses harden the existing corpus-level result by measuring stability under resampling, corpus omission, and same-source splitting. They do not expand the corpus panel, change the measurement families, infer causal downstream training effects, or establish a universal taxonomy of all corpora.
