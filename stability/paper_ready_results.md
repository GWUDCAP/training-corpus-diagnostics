# Paper-Ready Hardening Results



- In the retained canonical summaries, broad-broad separation is lower than broad-targeted separation for lexical JS (`0.0834` vs `0.1894`), semantic cluster JS (`0.0630` vs `0.3737`), compression absolute difference (`0.0077` vs `0.0515`), top1 absolute difference (`0.0048` vs `0.0237`), and top5 absolute difference (`0.0065` vs `0.0513`). Spearman rho shows the corresponding closeness direction, with broad-broad higher than broad-targeted (`0.6156` vs `0.4038`).

- Bootstrap resampling preserved the expected Lexical JS ordering in `200/200` replicates.

- Bootstrap resampling preserved the expected Spearman rho ordering in `200/200` replicates.

- Bootstrap resampling preserved the expected Semantic cluster JS ordering in `200/200` replicates.

- Bootstrap resampling preserved the expected Compression abs diff ordering in `200/200` replicates.

- Bootstrap resampling preserved the expected Top1 abs diff ordering in `200/200` replicates.

- Bootstrap resampling preserved the expected Top5 abs diff ordering in `200/200` replicates.

- Leave-one-corpus-out sensitivity did not identify a single omitted corpus that reverses the retained broad-versus-targeted ordering across the six metric families.

- Same-corpus split baselines are lower than the cross-corpus broad-targeted scale for divergence-style summaries: pooled same-corpus lexical JS mean `0.0294` and semantic cluster JS mean `0.0055`.

- Same-corpus Spearman rho remains high relative to broad-targeted comparisons: pooled same-corpus rho mean `0.7548`.

- These checks support stability of the fixed-panel empirical ordering under chunk resampling, corpus omission, and same-source split calibration. They do not expand the corpus panel or establish downstream causal effects.
