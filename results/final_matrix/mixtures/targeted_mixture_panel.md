# Targeted Mixture Summary

- targeted corpora: `8`
- directed mixture trajectories: `56`
- alpha grid: `0.0, 0.1, 0.25, 0.5, 0.75, 1.0`

## Strongest Endpoint Cluster JS

| Baseline | Target | Cluster JS @ 1.0 | Lexical JS @ 1.0 | Rho @ 1.0 |
|---|---|---:|---:|---:|
| BeaverTails | Nemotron-SFT-Math | 0.6931 | 0.3760 | 0.2825 |
| Nemotron-SFT-Math | BeaverTails | 0.6916 | 0.5542 | 0.0531 |
| Nemotron-SFT-Math | HH-RLHF chosen | 0.6887 | 0.5197 | 0.0706 |
| HH-RLHF chosen | Nemotron-SFT-Math | 0.6886 | 0.3782 | 0.2565 |
| BeaverTails | Magicoder OSS Instruct | 0.6865 | 0.3717 | 0.2903 |
| Magicoder OSS Instruct | BeaverTails | 0.6863 | 0.5090 | 0.1411 |
| Nemotron-SFT-Math | Magicoder OSS Instruct | 0.6859 | 0.4363 | 0.2185 |
| Magicoder OSS Instruct | Nemotron-SFT-Math | 0.6857 | 0.4196 | 0.1817 |
| Nemotron-SFT-Math | HH-RLHF rejected | 0.6819 | 0.5215 | 0.0861 |
| Magicoder OSS Instruct | HH-RLHF rejected | 0.6816 | 0.4006 | 0.1948 |
| HH-RLHF rejected | Nemotron-SFT-Math | 0.6806 | 0.3798 | 0.2545 |
| HH-RLHF rejected | Magicoder OSS Instruct | 0.6803 | 0.4312 | 0.2285 |

## Flattest Endpoint Cluster JS

| Baseline | Target | Cluster JS @ 1.0 | Lexical JS @ 1.0 | Rho @ 1.0 |
|---|---|---:|---:|---:|
| HH-RLHF chosen | HH-RLHF rejected | 0.0058 | 0.0075 | 0.8744 |
| HH-RLHF rejected | HH-RLHF chosen | 0.0062 | 0.0130 | 0.8336 |
| UltraFeedback chosen | OASST1 assistant | 0.0581 | 0.0455 | 0.6151 |
| OASST1 assistant | UltraFeedback chosen | 0.0619 | 0.0320 | 0.7302 |
| BeaverTails | HH-RLHF rejected | 0.1257 | 0.1055 | 0.5463 |
| HH-RLHF rejected | BeaverTails | 0.1338 | 0.1308 | 0.4484 |
| Dolly 15k | HH-RLHF chosen | 0.1402 | 0.1004 | 0.5591 |
| HH-RLHF chosen | Dolly 15k | 0.1459 | 0.1057 | 0.5010 |
| OASST1 assistant | Dolly 15k | 0.1476 | 0.0996 | 0.5280 |
| Dolly 15k | OASST1 assistant | 0.1478 | 0.0835 | 0.6339 |
| HH-RLHF chosen | OASST1 assistant | 0.1494 | 0.1481 | 0.4531 |
| OASST1 assistant | HH-RLHF chosen | 0.1523 | 0.1286 | 0.4565 |

## Interpretation

These trajectories are directed baseline-relative mixtures. Each row starts at the baseline corpus (`alpha=0.0`) and ends at the pure target corpus (`alpha=1.0`), while the intermediate points mix baseline and target chunks under the retained chunk budget. The endpoint values therefore recover the directed baseline-to-target comparison under the mixture metric family, and the interior points show whether movement is flat or smooth as target mass increases.
