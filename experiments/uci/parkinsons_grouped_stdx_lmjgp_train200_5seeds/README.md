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
| Parkinsons Telemonitoring | lmjgp_supervised | 5 | 8.9120 | 0.7569 | 5.4470 | 0.5978 | 0.716 | 22.2623 |
| Parkinsons Telemonitoring | lmjgp_supervised_residual_ridge | 5 | 9.4276 | 0.5484 | 5.7514 | 0.4770 | 0.699 | 21.2394 |
| Parkinsons Telemonitoring | lmjgp_transductive | 5 | 8.9968 | 0.6192 | 5.5019 | 0.4594 | 0.733 | 22.7353 |
| Parkinsons Telemonitoring | lmjgp_transductive_residual_ridge | 5 | 9.4551 | 0.4967 | 5.8543 | 0.4994 | 0.679 | 21.1515 |

## Command

```powershell
python experiments/uci/compare_smalltrain_multiseed_grouped.py --num_exp 5 --datasets "Parkinsons Telemonitoring" --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3 --parkinsons_standardize_x 1 --parkinsons_standardize_y 0 --out_dir experiments\uci\parkinsons_grouped_stdx_lmjgp_train200_5seeds
```
