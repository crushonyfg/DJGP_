# Small-Train Multiseed Grouped UCI

This folder was produced by `experiments/uci/compare_smalltrain_multiseed_grouped.py`.

Project-wide log entry: `docs/experiments_log.md`, section
`2026-05-15 - uncertain_w_cem_train500_selfcem_5seeds_20260515`.

Parkinsons uses subject-grouped splits; Wine and Appliances use the historical random-row split.
Aggregate metrics use `metrics.csv`, which may omit isolated numeric outlier predictions according to the runner policy; raw unfiltered metrics are in `metrics_raw.csv`.

## Tuned Presets

| dataset | split | std_x | std_y | n | steps |
|---|---|---:|---:|---:|---:|
| Wine Quality | random_row | True | False | 25 | 300 |
| Parkinsons Telemonitoring | subject_grouped | False | False | 25 | 300 |
| Appliances Energy Prediction | random_row | True | True | 35 | 300 |

## Aggregate Metrics

| dataset | method | seeds | RMSE mean | RMSE std | CRPS mean | CRPS std | Cov90 mean | Width90 mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Appliances Energy Prediction | dkl_svgp | 5 | 113.8430 | 8.0967 | 52.3699 | 4.9380 | 0.581 | 58.8813 |
| Appliances Energy Prediction | xgb_bootstrap | 5 | 98.4774 | 9.5908 | 43.3656 | 4.6923 | 0.880 | 176.0732 |
| Parkinsons Telemonitoring | dkl_svgp | 5 | 7.8602 | 1.0305 | 4.9345 | 0.8857 | 0.650 | 15.2668 |
| Parkinsons Telemonitoring | xgb_bootstrap | 5 | 9.5327 | 0.7271 | 7.0371 | 0.6295 | 0.246 | 7.8662 |
| Wine Quality | dkl_svgp | 5 | 0.8517 | 0.0641 | 0.5980 | 0.0492 | 0.224 | 0.4861 |
| Wine Quality | xgb_bootstrap | 5 | 0.7202 | 0.0182 | 0.4199 | 0.0077 | 0.652 | 1.2646 |

## Command

```powershell
python experiments/uci/compare_smalltrain_multiseed_grouped.py --num_exp 5 --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" --max_total_train 500 --max_test_anchors 200 --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3 --parkinsons_standardize_x -1 --parkinsons_standardize_y -1 --out_dir experiments\uci\smalltrain_xgb_dkl_train500_5seeds_20260515
```
