# Same-Corpus Split Baselines

| Corpus | Lex JS | Rho | Semantic JS | Compression abs | Top1 abs | Top5 abs |
|---|---:|---:|---:|---:|---:|---:|
| FineWeb | 0.0298 | 0.7554 | 0.0055 | 0.0014 | 0.0008 | 0.0014 |
| RefinedWeb | 0.0418 | 0.7039 | 0.0058 | 0.0018 | 0.0008 | 0.0015 |
| Dolma | 0.0358 | 0.7380 | 0.0050 | 0.0017 | 0.0008 | 0.0016 |
| C4 | 0.0363 | 0.7273 | 0.0056 | 0.0016 | 0.0007 | 0.0014 |
| SlimPajama | 0.0537 | 0.6584 | 0.0054 | 0.0018 | 0.0010 | 0.0019 |
| Pile uncopyrighted | 0.0384 | 0.6947 | 0.0060 | 0.0022 | 0.0020 | 0.0031 |
| HH-RLHF chosen | 0.0176 | 0.8061 | 0.0055 | 0.0016 | 0.0011 | 0.0018 |
| HH-RLHF rejected | 0.0205 | 0.7889 | 0.0055 | 0.0015 | 0.0008 | 0.0023 |
| OASST1 assistant | 0.0276 | 0.7494 | 0.0051 | 0.0016 | 0.0017 | 0.0022 |
| UltraFeedback chosen | 0.0247 | 0.7531 | 0.0056 | 0.0015 | 0.0014 | 0.0024 |
| Nemotron-SFT-Math | 0.0124 | 0.8455 | 0.0057 | 0.0012 | 0.0039 | 0.0037 |
| BeaverTails | 0.0188 | 0.7991 | 0.0056 | 0.0015 | 0.0021 | 0.0025 |
| Magicoder OSS Instruct | 0.0151 | 0.8155 | 0.0056 | 0.0014 | 0.0015 | 0.0022 |
| Dolly 15k | 0.0391 | 0.7319 | 0.0053 | 0.0014 | 0.0011 | 0.0022 |

## Pooled Corpus-Mean Scale

- Lexical JS: pooled mean=0.0294, median=0.0287, 2.5-97.5% range=0.0133-0.0498
- Spearman rho: pooled mean=0.7548, median=0.7513, 2.5-97.5% range=0.6702-0.8358
- Semantic cluster JS: pooled mean=0.0055, median=0.0056, 2.5-97.5% range=0.0050-0.0059
- Compression abs diff: pooled mean=0.0016, median=0.0016, 2.5-97.5% range=0.0013-0.0021
- Top1 abs diff: pooled mean=0.0014, median=0.0011, 2.5-97.5% range=0.0007-0.0033
- Top5 abs diff: pooled mean=0.0022, median=0.0022, 2.5-97.5% range=0.0014-0.0035

## Comparison Scale

- Canonical textual scorecard lexical JS: broad-broad `0.0834`, broad-targeted `0.1894`.
- Canonical textual scorecard rho: broad-broad `0.6156`, broad-targeted `0.4038`.
- Canonical semantic scorecard cluster JS: broad-broad `0.0630`, broad-targeted `0.3737`.

Same-corpus splits estimate the within-source noise floor under the same chunk representation. They should be read as a calibration baseline, not as a replacement for cross-corpus comparisons.
