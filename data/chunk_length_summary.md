# Chunk Length Summary

Source: `data/chunks/*/chunks.json`.

| Role | Corpus | Chunks | Min chars | Mean chars | Median chars | Max chars |
|---|---|---:|---:|---:|---:|---:|
| Pretraining | FineWeb | 2048 | 194 | 710.9 | 768.0 | 768 |
| Pretraining | RefinedWeb | 2048 | 193 | 688.0 | 768.0 | 768 |
| Pretraining | Dolma | 2048 | 192 | 710.1 | 768.0 | 768 |
| Pretraining | C4 | 2048 | 192 | 664.5 | 768.0 | 768 |
| Pretraining | SlimPajama | 2048 | 196 | 708.8 | 768.0 | 768 |
| Pretraining | Pile uncopyrighted | 2048 | 192 | 703.8 | 768.0 | 768 |
| Post-training | HH-RLHF chosen | 2048 | 192 | 484.8 | 446.0 | 768 |
| Post-training | HH-RLHF rejected | 2048 | 192 | 476.7 | 437.0 | 768 |
| Post-training | OASST1 assistant | 2048 | 192 | 604.5 | 748.0 | 768 |
| Post-training | UltraFeedback chosen | 2048 | 192 | 668.2 | 768.0 | 768 |
| Post-training | Nemotron-SFT-Math | 2048 | 193 | 695.1 | 768.0 | 768 |
| Post-training | BeaverTails | 2048 | 192 | 428.6 | 402.0 | 768 |
| Post-training | Magicoder OSS Instruct | 2048 | 193 | 639.4 | 764.5 | 768 |
| Post-training | Dolly 15k | 2048 | 192 | 494.7 | 456.0 | 768 |

Pooled mean chunk length is 697.7 characters for pretraining corpora and 561.5 characters for post-training corpora. The matched budget is therefore a matched chunk-count design, not a claim that every retained chunk has identical character length.
