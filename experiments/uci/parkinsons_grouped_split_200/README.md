# Parkinsons Subject-Grouped Split

This folder was produced by `experiments/uci/compare_parkinsons_grouped_split.py`.

Split: subject-grouped holdout, so the same `subject#` never appears in both train and test.

- train subjects: 33
- test subjects: 9
- subject overlap: 0

## Metrics

| method | RMSE | CRPS | Cov90 | Width90 |
|---|---:|---:|---:|---:|
| xgboost | 7.8282 | 5.2626 | 0.430 | 10.5267 |
| lmjgp_supervised | 9.0627 | 5.7074 | 0.675 | 21.7141 |
| lmjgp_residual_ridge | 8.7096 | 5.4454 | 0.680 | 21.1995 |
| cem_em_supervised | 9.5431 | 6.3394 | 0.570 | 17.9580 |

## Command

```powershell
python experiments/uci/compare_parkinsons_grouped_split.py --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 --n 25 --lmjgp_steps 300 --n_outer 5 --n_m_steps 80 --MC_num 3 --xgb_boots 30 --out_dir experiments\uci\parkinsons_grouped_split_200
```
