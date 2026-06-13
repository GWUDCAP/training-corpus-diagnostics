# Semantic k Stability

This stability check recomputes the retained pairwise semantic cluster JS summary from the cached GTE c768 embeddings while varying the pooled pairwise cluster count.

| k | Broad-broad cluster JS | Broad-targeted cluster JS | Ordering held |
|---:|---:|---:|---:|
| 12 | 0.0565 | 0.3599 | `True` |
| 16 | 0.0574 | 0.3679 | `True` |
| 24 | 0.0630 | 0.3737 | `True` |
| 32 | 0.0696 | 0.3765 | `True` |
| 48 | 0.0743 | 0.3825 | `True` |

The broad-targeted semantic cluster JS exceeds broad-broad cluster JS for `5/5` tested k values.
