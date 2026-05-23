# Small-Train Multiseed Grouped UCI

This folder was produced by `experiments/uci/compare_smalltrain_multiseed_grouped.py`.

Parkinsons uses subject-grouped splits; Wine and Appliances use the historical random-row split.
Aggregate metrics use `metrics.csv`, which may omit isolated numeric outlier predictions according to the runner policy; raw unfiltered metrics are in `metrics_raw.csv`.

## Tuned Presets

| dataset | split | std_x | std_y | n | steps |
|---|---|---:|---:|---:|---:|
| Wine Quality | random_row | True | False | 25 | 300 |
| Parkinsons Telemonitoring | subject_grouped | True | False | 25 | 300 |
| Appliances Energy Prediction | random_row | True | True | 35 | 300 |

## Aggregate Metrics

| dataset | method | seeds | RMSE mean | RMSE std | CRPS mean | CRPS std | Cov90 mean | Width90 mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Parkinsons Telemonitoring | lmjgp_supervised | 1 | 8.3497 | 0.0000 | 5.1219 | 0.0000 | 0.715 | 22.9972 |
| Parkinsons Telemonitoring | lmjgp_supervised_residual_ridge | 1 | 8.7596 | 0.0000 | 5.4679 | 0.0000 | 0.685 | 21.8219 |
| Parkinsons Telemonitoring | lmjgp_transductive | 1 | 8.5372 | 0.0000 | 5.3511 | 0.0000 | 0.680 | 22.3020 |
| Parkinsons Telemonitoring | lmjgp_transductive_residual_ridge | 1 | 8.8085 | 0.0000 | 5.4627 | 0.0000 | 0.710 | 22.9471 |

## Command

```powershell
python experiments/uci/compare_smalltrain_multiseed_grouped.py --num_exp 1 --datasets "Parkinsons Telemonitoring" --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3 --parkinsons_standardize_x 1 --parkinsons_standardize_y 0 --out_dir experiments\uci\parkinsons_grouped_scaling_probe_200_stdx
```
