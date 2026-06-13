# Bootstrap Canonical Summary

- replicates: `200`
- seed: `0`
- sampling unit: `canonical corpus chunk`
- semantic bootstrap mode: `fixed`

| Metric | Family | Mean | Median | 2.5% | 50% | 97.5% |
|---|---:|---:|---:|---:|---:|---:|
| Lexical JS | bb | 0.0796 | 0.0801 | 0.0736 | 0.0801 | 0.0847 |
| Lexical JS | bt | 0.1927 | 0.1927 | 0.1889 | 0.1927 | 0.1965 |
| Spearman rho | bb | 0.6251 | 0.6248 | 0.6123 | 0.6248 | 0.6370 |
| Spearman rho | bt | 0.4198 | 0.4200 | 0.4117 | 0.4200 | 0.4266 |
| Semantic cluster JS | bb | 0.0661 | 0.0662 | 0.0620 | 0.0662 | 0.0702 |
| Semantic cluster JS | bt | 0.3764 | 0.3764 | 0.3719 | 0.3764 | 0.3805 |
| Compression abs diff | bb | 0.0053 | 0.0053 | 0.0030 | 0.0053 | 0.0078 |
| Compression abs diff | bt | 0.0320 | 0.0320 | 0.0297 | 0.0320 | 0.0344 |
| Top1 abs diff | bb | 0.0048 | 0.0048 | 0.0040 | 0.0048 | 0.0055 |
| Top1 abs diff | bt | 0.0236 | 0.0236 | 0.0228 | 0.0236 | 0.0244 |
| Top5 abs diff | bb | 0.0067 | 0.0067 | 0.0056 | 0.0067 | 0.0078 |
| Top5 abs diff | bt | 0.0507 | 0.0507 | 0.0494 | 0.0507 | 0.0521 |

## Ordering Preservation

- Lexical JS ordering held in `200/200` replicates.
- Spearman rho ordering held in `200/200` replicates.
- Semantic cluster JS ordering held in `200/200` replicates.
- Compression abs diff ordering held in `200/200` replicates.
- Top1 abs diff ordering held in `200/200` replicates.
- Top5 abs diff ordering held in `200/200` replicates.

## Interpretation

The bootstrap resamples the already selected corpus chunks within each corpus and recomputes the retained family summaries. It measures stability of the observed panel-level ordering under chunk resampling; it does not add corpora, estimate population-level coverage for all possible web or post-training data, or change the original canonical point estimates.
