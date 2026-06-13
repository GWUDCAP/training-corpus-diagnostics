# Continuous Embedding Sensitivity

This non-binned check uses row-normalized cached embeddings and fixed random projections. It is a ceiling-sensitivity companion to semantic cluster JS, not a bootstrapped headline family.

## Family Means

| Pair class | Count | Centroid cosine | Centroid L2 | Sliced W1 | Sliced W2 |
|---|---:|---:|---:|---:|---:|
| broad_broad | 15 | 0.0160 | 0.0781 | 0.0026 | 0.0033 |
| broad_targeted | 48 | 0.1587 | 0.2783 | 0.0084 | 0.0105 |
| targeted_targeted | 28 | 0.2262 | 0.3461 | 0.0105 | 0.0130 |

## Near-Ceiling Cluster-JS Pairs Ranked by Sliced W1

| Pair | Cluster JS | Sliced W1 | Centroid cosine |
|---|---:|---:|---:|
| Nemotron-SFT-Math vs BeaverTails | 0.6916 | 0.0192 | 0.5446 |
| HH-RLHF chosen vs Nemotron-SFT-Math | 0.6886 | 0.0189 | 0.5104 |
| HH-RLHF rejected vs Nemotron-SFT-Math | 0.6806 | 0.0189 | 0.5068 |
| Nemotron-SFT-Math vs Dolly 15k | 0.6771 | 0.0182 | 0.4839 |
| C4 vs Nemotron-SFT-Math | 0.6917 | 0.0181 | 0.4534 |
| Nemotron-SFT-Math vs Magicoder OSS Instruct | 0.6859 | 0.0179 | 0.4185 |
| FineWeb vs Nemotron-SFT-Math | 0.6856 | 0.0179 | 0.4393 |
| RefinedWeb vs Nemotron-SFT-Math | 0.6837 | 0.0178 | 0.4245 |
| SlimPajama vs Nemotron-SFT-Math | 0.6531 | 0.0174 | 0.4149 |
| UltraFeedback chosen vs Nemotron-SFT-Math | 0.6505 | 0.0171 | 0.3885 |
| BeaverTails vs Magicoder OSS Instruct | 0.6865 | 0.0161 | 0.4154 |
| HH-RLHF rejected vs Magicoder OSS Instruct | 0.6803 | 0.0156 | 0.3831 |
| HH-RLHF chosen vs Magicoder OSS Instruct | 0.6784 | 0.0155 | 0.3828 |
| Magicoder OSS Instruct vs Dolly 15k | 0.6730 | 0.0145 | 0.3359 |
| FineWeb vs Magicoder OSS Instruct | 0.6763 | 0.0142 | 0.3373 |
| C4 vs Magicoder OSS Instruct | 0.6737 | 0.0142 | 0.3374 |
| RefinedWeb vs Magicoder OSS Instruct | 0.6681 | 0.0141 | 0.3267 |
