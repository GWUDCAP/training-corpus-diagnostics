# Directional Lexical Sensitivity

The primary broad-targeted textual summary uses broad corpora as the reference vocabulary. Because lexical JS and rho are computed on the baseline corpus vocabulary, this table reports the reverse direction and the two-direction average.

| Comparison | Lexical JS | Spearman rho |
|---|---:|---:|
| Broad-broad, two-direction | 0.0834 | 0.6156 |
| Broad-targeted, broad reference | 0.1894 | 0.4038 |
| Broad-targeted, targeted reference | 0.1971 | 0.4098 |
| Broad-targeted, two-direction average | 0.1933 | 0.4068 |

## Ordering Checks

- `primary_lexical_js_bt_gt_bb`: `True`
- `reverse_lexical_js_bt_gt_bb`: `True`
- `symmetric_lexical_js_bt_gt_bb`: `True`
- `primary_rho_bb_gt_bt`: `True`
- `reverse_rho_bb_gt_bt`: `True`
- `symmetric_rho_bb_gt_bt`: `True`
