# Small-Train Multiseed Grouped UCI

This folder was produced by `experiments/uci/compare_smalltrain_multiseed_grouped.py`.

Parkinsons uses subject-grouped splits; Wine and Appliances use the historical random-row split.
Aggregate metrics use `metrics.csv`, which may omit isolated numeric outlier predictions according to the runner policy; raw unfiltered metrics are in `metrics_raw.csv`.

## Tuned Presets

| dataset | split | std_x | std_y | n | steps |
|---|---|---:|---:|---:|---:|
| Wine Quality | random_row | True | False | 25 | 5 |
| Parkinsons Telemonitoring | subject_grouped | False | False | 25 | 5 |
| Appliances Energy Prediction | random_row | True | True | 35 | 5 |

## Aggregate Metrics

| dataset | method | seeds | RMSE mean | RMSE std | CRPS mean | CRPS std | Cov90 mean | Width90 mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Wine Quality | dkl_svgp | 1 | 0.6857 | 0.0000 | 0.4075 | 0.0000 | 0.500 | 1.5580 |
| Wine Quality | lmjgp_supervised | 1 | 0.7348 | 0.0000 | 0.4193 | 0.0000 | 0.800 | 1.9688 |
| Wine Quality | lmjgp_transductive | 1 | 0.7032 | 0.0000 | 0.3996 | 0.0000 | 0.800 | 1.9401 |
| Wine Quality | xgb_bootstrap | 1 | 0.7000 | 0.0000 | 0.4247 | 0.0000 | 0.700 | 1.6751 |

## Command

```powershell
python experiments/uci/compare_smalltrain_multiseed_grouped.py --num_exp 1 --datasets "Wine Quality" --max_total_train 80 --max_test_anchors 20 --max_train_anchors 32 --xgb_boots 3 --dkl_epochs 3 --lmjgp_steps 5 --MC_num 2 --out_dir experiments\uci\smalltrain_multiseed_grouped_smoke_filter
```
