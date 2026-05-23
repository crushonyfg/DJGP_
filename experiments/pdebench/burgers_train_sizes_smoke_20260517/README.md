# PDEBench Burgers Train-Size Baselines

Produced by `experiments/pdebench/compare_burgers_train_sizes.py`.

## Aggregate

| train | subset | method | seeds | RMSE | CRPS | Cov90 | Width90 |
|---:|---|---|---:|---:|---:|---:|---:|
| 80 | all | dkl_svgp | 1 | 1.1735 | 0.5620 | 0.850 | 2.2460 |
| 80 | far_from_threshold | dkl_svgp | 1 | 0.9875 | 0.4428 | 0.920 | 2.2259 |
| 80 | near_threshold | dkl_svgp | 1 | 1.4307 | 0.7607 | 0.733 | 2.2795 |
| 80 | all | lmjgp_predict_vi | 1 | 0.9951 | 0.4118 | 0.850 | 1.3478 |
| 80 | far_from_threshold | lmjgp_predict_vi | 1 | 1.0362 | 0.3687 | 0.920 | 0.7015 |
| 80 | near_threshold | lmjgp_predict_vi | 1 | 0.9226 | 0.4835 | 0.733 | 2.4250 |
| 80 | all | xgb_bootstrap | 1 | 1.0655 | 0.4999 | 0.900 | 2.1465 |
| 80 | far_from_threshold | xgb_bootstrap | 1 | 1.0586 | 0.4651 | 0.920 | 2.0575 |
| 80 | near_threshold | xgb_bootstrap | 1 | 1.0768 | 0.5581 | 0.867 | 2.2948 |

## Command

```powershell
python experiments/pdebench/compare_burgers_train_sizes.py --dataset-folder data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000 --train_sizes 80 --num_exp 1 --Q 3 --neighbors 25 --lmjgp_steps 5 --out_dir experiments/pdebench/burgers_train_sizes_smoke_20260517
```