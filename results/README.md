# Final Empirical Bundle

This directory contains the paper-facing empirical summaries.

Primary retained results are under `final_matrix/`:

- `recomputed/family_summary.json`
- `recomputed/standalone_summary.json`
- `recomputed/directional_lexical_sensitivity.json`
- `targeted_substructure_summary.json`
- `mixtures/targeted_mixture_panel.json`
- `sweeps/semantic_summary.json`
- `sweeps/textual_grid_summary.json`

The manuscript and verifier use `final_matrix/` as the source of paper numbers.

Important audit note:

- do **not** use signed family means for `compression_gap`, `top1_gap`, or `top5_gap`
- use standalone-derived absolute pairwise differences for compression, top1, and top5 family comparisons
