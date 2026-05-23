# Parkinsons Subject-Grouped Split

This folder was produced by `experiments/uci/compare_parkinsons_grouped_split.py`.

Split: subject-grouped holdout, so the same `subject#` never appears in both train and test.

- train subjects: 33
- test subjects: 9
- subject overlap: 0

## Metrics

| method | RMSE | CRPS | Cov90 | Width90 |
|---|---:|---:|---:|---:|
| xgboost | 7.3085 | 4.7847 | 0.525 | 13.3226 |
| lmjgp_supervised | 7.5509 | 4.8514 | 0.575 | 16.6293 |
| lmjgp_residual_ridge | 8.2563 | 5.3833 | 0.475 | 14.9687 |
| cem_em_supervised | 7.3878 | 4.6452 | 0.675 | 18.0843 |

## Command

```powershell
python experiments/uci/compare_parkinsons_grouped_split.py --max_total_train 80 --max_test_anchors 40 --max_train_anchors 32 --n 25 --lmjgp_steps 5 --n_outer 1 --n_m_steps 5 --MC_num 2 --xgb_boots 3 --out_dir experiments\uci\parkinsons_grouped_split_smoke
```
