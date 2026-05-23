# Small-Train Multiseed Grouped UCI

This folder was produced by `experiments/uci/compare_smalltrain_multiseed_grouped.py`.

Parkinsons uses subject-grouped splits; Wine and Appliances use the historical random-row split.
Aggregate metrics use `metrics.csv`, which may omit isolated numeric outlier predictions according to the runner policy; raw unfiltered metrics are in `metrics_raw.csv`.

## Tuned Presets

| dataset | split | std_x | std_y | n | steps |
|---|---|---:|---:|---:|---:|
| Wine Quality | random_row | True | False | 25 | 300 |
| Parkinsons Telemonitoring | subject_grouped | True | True | 25 | 300 |
| Appliances Energy Prediction | random_row | True | True | 35 | 300 |

## Aggregate Metrics

| dataset | method | seeds | RMSE mean | RMSE std | CRPS mean | CRPS std | Cov90 mean | Width90 mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Parkinsons Telemonitoring | lmjgp_supervised | 1 | 8.4530 | 0.0000 | 5.2134 | 0.0000 | 0.720 | 22.7873 |
| Parkinsons Telemonitoring | lmjgp_supervised_residual_ridge | 1 | 8.6235 | 0.0000 | 5.3332 | 0.0000 | 0.710 | 22.0207 |
| Parkinsons Telemonitoring | lmjgp_transductive | 1 | 8.4385 | 0.0000 | 5.2115 | 0.0000 | 0.720 | 22.8192 |
| Parkinsons Telemonitoring | lmjgp_transductive_residual_ridge | 1 | 8.8241 | 0.0000 | 5.4470 | 0.0000 | 0.730 | 22.8033 |

## Command

```powershell
python experiments/uci/compare_smalltrain_multiseed_grouped.py --num_exp 1 --datasets "Parkinsons Telemonitoring" --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3 --parkinsons_standardize_x 1 --parkinsons_standardize_y 1 --out_dir experiments\uci\parkinsons_grouped_scaling_probe_200_stdxy
```
