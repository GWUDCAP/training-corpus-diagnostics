# Targeted Substructure Summary

- targeted corpora: `8`
- targeted-targeted pairs: `28`
- textual convention: Lexical JS and rho use the retained unordered pair rows from corpus_matrix.json; semantic cluster JS, compression, top1, and top5 are symmetric pair summaries.

## Family Means

| Metric | Targeted-targeted mean |
|---|---:|
| lexical_js | 0.2536 |
| rho | 0.3706 |
| semantic_cluster_js | 0.4096 |
| compression_abs_diff | 0.0314 |
| top1_abs_diff | 0.0291 |
| top5_abs_diff | 0.0556 |

## Standalone Targeted Corpora

| Corpus | Compression ratio | Top1 share | Top5 share |
|---|---:|---:|---:|
| HH-RLHF chosen | 0.3011 | 0.0270 | 0.1015 |
| HH-RLHF rejected | 0.2988 | 0.0253 | 0.1034 |
| OASST1 assistant | 0.2982 | 0.0284 | 0.1015 |
| UltraFeedback chosen | 0.2917 | 0.0319 | 0.1043 |
| Nemotron-SFT-Math | 0.2685 | 0.1082 | 0.2507 |
| BeaverTails | 0.2884 | 0.0524 | 0.1363 |
| Magicoder OSS Instruct | 0.2312 | 0.0557 | 0.1715 |
| Dolly 15k | 0.3282 | 0.0249 | 0.0936 |

## Pair Rankings

### lexical_js

| Pair | Value |
|---|---:|
| Nemotron-SFT-Math vs BeaverTails | 0.5542 |
| Nemotron-SFT-Math vs Dolly 15k | 0.4728 |
| HH-RLHF chosen vs Magicoder OSS Instruct | 0.4390 |
| Nemotron-SFT-Math vs Magicoder OSS Instruct | 0.4363 |
| Magicoder OSS Instruct vs Dolly 15k | 0.4319 |
| HH-RLHF rejected vs Magicoder OSS Instruct | 0.4312 |
| HH-RLHF rejected vs Nemotron-SFT-Math | 0.3798 |
| HH-RLHF chosen vs Nemotron-SFT-Math | 0.3782 |
| BeaverTails vs Magicoder OSS Instruct | 0.3717 |
| OASST1 assistant vs Nemotron-SFT-Math | 0.3580 |
| UltraFeedback chosen vs Nemotron-SFT-Math | 0.3418 |
| OASST1 assistant vs Magicoder OSS Instruct | 0.2908 |
| UltraFeedback chosen vs Magicoder OSS Instruct | 0.2734 |
| UltraFeedback chosen vs BeaverTails | 0.2249 |
| HH-RLHF rejected vs UltraFeedback chosen | 0.1907 |
| HH-RLHF chosen vs UltraFeedback chosen | 0.1883 |
| OASST1 assistant vs BeaverTails | 0.1631 |
| HH-RLHF rejected vs OASST1 assistant | 0.1505 |
| UltraFeedback chosen vs Dolly 15k | 0.1503 |
| HH-RLHF chosen vs OASST1 assistant | 0.1481 |
| HH-RLHF rejected vs BeaverTails | 0.1308 |
| HH-RLHF chosen vs BeaverTails | 0.1239 |
| HH-RLHF rejected vs Dolly 15k | 0.1154 |
| BeaverTails vs Dolly 15k | 0.1097 |
| HH-RLHF chosen vs Dolly 15k | 0.1057 |
| OASST1 assistant vs Dolly 15k | 0.0996 |
| OASST1 assistant vs UltraFeedback chosen | 0.0320 |
| HH-RLHF chosen vs HH-RLHF rejected | 0.0075 |

### rho

| Pair | Value |
|---|---:|
| Nemotron-SFT-Math vs BeaverTails | 0.0531 |
| Nemotron-SFT-Math vs Dolly 15k | 0.1470 |
| Magicoder OSS Instruct vs Dolly 15k | 0.1876 |
| Nemotron-SFT-Math vs Magicoder OSS Instruct | 0.2185 |
| HH-RLHF chosen vs Magicoder OSS Instruct | 0.2231 |
| HH-RLHF rejected vs Magicoder OSS Instruct | 0.2285 |
| HH-RLHF rejected vs Nemotron-SFT-Math | 0.2545 |
| HH-RLHF chosen vs Nemotron-SFT-Math | 0.2565 |
| BeaverTails vs Magicoder OSS Instruct | 0.2903 |
| UltraFeedback chosen vs Nemotron-SFT-Math | 0.2905 |
| UltraFeedback chosen vs Magicoder OSS Instruct | 0.2994 |
| UltraFeedback chosen vs BeaverTails | 0.3047 |
| OASST1 assistant vs Magicoder OSS Instruct | 0.3193 |
| OASST1 assistant vs Nemotron-SFT-Math | 0.3356 |
| HH-RLHF rejected vs UltraFeedback chosen | 0.3676 |
| HH-RLHF chosen vs UltraFeedback chosen | 0.3739 |
| UltraFeedback chosen vs Dolly 15k | 0.3833 |
| OASST1 assistant vs BeaverTails | 0.4121 |
| HH-RLHF rejected vs OASST1 assistant | 0.4411 |
| HH-RLHF rejected vs BeaverTails | 0.4484 |
| HH-RLHF chosen vs OASST1 assistant | 0.4531 |
| HH-RLHF chosen vs BeaverTails | 0.4606 |
| HH-RLHF rejected vs Dolly 15k | 0.4743 |
| HH-RLHF chosen vs Dolly 15k | 0.5010 |
| BeaverTails vs Dolly 15k | 0.5191 |
| OASST1 assistant vs Dolly 15k | 0.5280 |
| OASST1 assistant vs UltraFeedback chosen | 0.7302 |
| HH-RLHF chosen vs HH-RLHF rejected | 0.8744 |

### semantic_cluster_js

| Pair | Value |
|---|---:|
| Nemotron-SFT-Math vs BeaverTails | 0.6916 |
| HH-RLHF chosen vs Nemotron-SFT-Math | 0.6886 |
| BeaverTails vs Magicoder OSS Instruct | 0.6865 |
| Nemotron-SFT-Math vs Magicoder OSS Instruct | 0.6859 |
| HH-RLHF rejected vs Nemotron-SFT-Math | 0.6806 |
| HH-RLHF rejected vs Magicoder OSS Instruct | 0.6803 |
| HH-RLHF chosen vs Magicoder OSS Instruct | 0.6784 |
| Nemotron-SFT-Math vs Dolly 15k | 0.6771 |
| Magicoder OSS Instruct vs Dolly 15k | 0.6730 |
| UltraFeedback chosen vs Nemotron-SFT-Math | 0.6505 |
| OASST1 assistant vs Nemotron-SFT-Math | 0.6460 |
| OASST1 assistant vs Magicoder OSS Instruct | 0.5316 |
| UltraFeedback chosen vs Magicoder OSS Instruct | 0.5165 |
| UltraFeedback chosen vs BeaverTails | 0.4422 |
| BeaverTails vs Dolly 15k | 0.3567 |
| OASST1 assistant vs BeaverTails | 0.3377 |
| HH-RLHF chosen vs UltraFeedback chosen | 0.2482 |
| HH-RLHF rejected vs UltraFeedback chosen | 0.2466 |
| UltraFeedback chosen vs Dolly 15k | 0.2182 |
| HH-RLHF rejected vs Dolly 15k | 0.1664 |
| HH-RLHF chosen vs BeaverTails | 0.1633 |
| HH-RLHF rejected vs OASST1 assistant | 0.1577 |
| HH-RLHF chosen vs OASST1 assistant | 0.1494 |
| OASST1 assistant vs Dolly 15k | 0.1476 |
| HH-RLHF chosen vs Dolly 15k | 0.1459 |
| HH-RLHF rejected vs BeaverTails | 0.1338 |
| OASST1 assistant vs UltraFeedback chosen | 0.0619 |
| HH-RLHF chosen vs HH-RLHF rejected | 0.0058 |

### compression_abs_diff

| Pair | Value |
|---|---:|
| Magicoder OSS Instruct vs Dolly 15k | 0.0969 |
| HH-RLHF chosen vs Magicoder OSS Instruct | 0.0699 |
| HH-RLHF rejected vs Magicoder OSS Instruct | 0.0676 |
| OASST1 assistant vs Magicoder OSS Instruct | 0.0669 |
| UltraFeedback chosen vs Magicoder OSS Instruct | 0.0605 |
| Nemotron-SFT-Math vs Dolly 15k | 0.0596 |
| BeaverTails vs Magicoder OSS Instruct | 0.0572 |
| BeaverTails vs Dolly 15k | 0.0397 |
| Nemotron-SFT-Math vs Magicoder OSS Instruct | 0.0373 |
| UltraFeedback chosen vs Dolly 15k | 0.0365 |
| HH-RLHF chosen vs Nemotron-SFT-Math | 0.0326 |
| HH-RLHF rejected vs Nemotron-SFT-Math | 0.0303 |
| OASST1 assistant vs Dolly 15k | 0.0300 |
| OASST1 assistant vs Nemotron-SFT-Math | 0.0296 |
| HH-RLHF rejected vs Dolly 15k | 0.0293 |
| HH-RLHF chosen vs Dolly 15k | 0.0270 |
| UltraFeedback chosen vs Nemotron-SFT-Math | 0.0232 |
| Nemotron-SFT-Math vs BeaverTails | 0.0199 |
| HH-RLHF chosen vs BeaverTails | 0.0127 |
| HH-RLHF rejected vs BeaverTails | 0.0104 |
| OASST1 assistant vs BeaverTails | 0.0097 |
| HH-RLHF chosen vs UltraFeedback chosen | 0.0094 |
| HH-RLHF rejected vs UltraFeedback chosen | 0.0071 |
| OASST1 assistant vs UltraFeedback chosen | 0.0064 |
| UltraFeedback chosen vs BeaverTails | 0.0033 |
| HH-RLHF chosen vs OASST1 assistant | 0.0030 |
| HH-RLHF chosen vs HH-RLHF rejected | 0.0023 |
| HH-RLHF rejected vs OASST1 assistant | 0.0007 |

### top1_abs_diff

| Pair | Value |
|---|---:|
| Nemotron-SFT-Math vs Dolly 15k | 0.0833 |
| HH-RLHF rejected vs Nemotron-SFT-Math | 0.0830 |
| HH-RLHF chosen vs Nemotron-SFT-Math | 0.0812 |
| OASST1 assistant vs Nemotron-SFT-Math | 0.0799 |
| UltraFeedback chosen vs Nemotron-SFT-Math | 0.0763 |
| Nemotron-SFT-Math vs BeaverTails | 0.0559 |
| Nemotron-SFT-Math vs Magicoder OSS Instruct | 0.0525 |
| Magicoder OSS Instruct vs Dolly 15k | 0.0308 |
| HH-RLHF rejected vs Magicoder OSS Instruct | 0.0304 |
| HH-RLHF chosen vs Magicoder OSS Instruct | 0.0287 |
| BeaverTails vs Dolly 15k | 0.0274 |
| OASST1 assistant vs Magicoder OSS Instruct | 0.0274 |
| HH-RLHF rejected vs BeaverTails | 0.0271 |
| HH-RLHF chosen vs BeaverTails | 0.0253 |
| OASST1 assistant vs BeaverTails | 0.0240 |
| UltraFeedback chosen vs Magicoder OSS Instruct | 0.0238 |
| UltraFeedback chosen vs BeaverTails | 0.0204 |
| UltraFeedback chosen vs Dolly 15k | 0.0070 |
| HH-RLHF rejected vs UltraFeedback chosen | 0.0066 |
| HH-RLHF chosen vs UltraFeedback chosen | 0.0049 |
| OASST1 assistant vs UltraFeedback chosen | 0.0036 |
| OASST1 assistant vs Dolly 15k | 0.0034 |
| BeaverTails vs Magicoder OSS Instruct | 0.0034 |
| HH-RLHF rejected vs OASST1 assistant | 0.0031 |
| HH-RLHF chosen vs Dolly 15k | 0.0021 |
| HH-RLHF chosen vs HH-RLHF rejected | 0.0018 |
| HH-RLHF chosen vs OASST1 assistant | 0.0013 |
| HH-RLHF rejected vs Dolly 15k | 0.0004 |

### top5_abs_diff

| Pair | Value |
|---|---:|
| Nemotron-SFT-Math vs Dolly 15k | 0.1571 |
| OASST1 assistant vs Nemotron-SFT-Math | 0.1492 |
| HH-RLHF chosen vs Nemotron-SFT-Math | 0.1492 |
| HH-RLHF rejected vs Nemotron-SFT-Math | 0.1473 |
| UltraFeedback chosen vs Nemotron-SFT-Math | 0.1464 |
| Nemotron-SFT-Math vs BeaverTails | 0.1144 |
| Nemotron-SFT-Math vs Magicoder OSS Instruct | 0.0791 |
| Magicoder OSS Instruct vs Dolly 15k | 0.0780 |
| OASST1 assistant vs Magicoder OSS Instruct | 0.0701 |
| HH-RLHF chosen vs Magicoder OSS Instruct | 0.0700 |
| HH-RLHF rejected vs Magicoder OSS Instruct | 0.0682 |
| UltraFeedback chosen vs Magicoder OSS Instruct | 0.0672 |
| BeaverTails vs Dolly 15k | 0.0427 |
| BeaverTails vs Magicoder OSS Instruct | 0.0353 |
| OASST1 assistant vs BeaverTails | 0.0348 |
| HH-RLHF chosen vs BeaverTails | 0.0348 |
| HH-RLHF rejected vs BeaverTails | 0.0329 |
| UltraFeedback chosen vs BeaverTails | 0.0320 |
| UltraFeedback chosen vs Dolly 15k | 0.0108 |
| HH-RLHF rejected vs Dolly 15k | 0.0098 |
| HH-RLHF chosen vs Dolly 15k | 0.0080 |
| OASST1 assistant vs Dolly 15k | 0.0079 |
| OASST1 assistant vs UltraFeedback chosen | 0.0029 |
| HH-RLHF chosen vs UltraFeedback chosen | 0.0028 |
| HH-RLHF rejected vs OASST1 assistant | 0.0019 |
| HH-RLHF chosen vs HH-RLHF rejected | 0.0019 |
| HH-RLHF rejected vs UltraFeedback chosen | 0.0009 |
| HH-RLHF chosen vs OASST1 assistant | 0.0001 |
