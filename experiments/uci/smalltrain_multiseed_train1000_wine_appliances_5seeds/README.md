# Small-Train Multiseed Grouped UCI

This folder was produced by `experiments/uci/compare_smalltrain_multiseed_grouped.py`.

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
| Appliances Energy Prediction | dkl_svgp | 5 | 110.3784 | 8.4286 | 51.0863 | 4.3269 | 0.589 | 63.2331 |
| Appliances Energy Prediction | lmjgp_supervised | 5 | 106.5366 | 6.2108 | 42.1170 | 3.5067 | 0.814 | 135.3096 |
| Appliances Energy Prediction | lmjgp_supervised_residual_ridge | 5 | 100.6785 | 8.3522 | 42.0830 | 3.8913 | 0.810 | 142.7817 |
| Appliances Energy Prediction | lmjgp_transductive | 5 | 105.1258 | 9.1357 | 41.7277 | 5.5170 | 0.824 | 145.1242 |
| Appliances Energy Prediction | lmjgp_transductive_residual_ridge | 5 | 98.3263 | 6.9517 | 41.0546 | 3.7301 | 0.802 | 143.0100 |
| Appliances Energy Prediction | xgb_bootstrap | 5 | 96.2436 | 8.1650 | 41.2660 | 4.4359 | 0.878 | 167.3944 |
| Wine Quality | dkl_svgp | 5 | 0.7594 | 0.0393 | 0.5155 | 0.0276 | 0.281 | 0.5020 |
| Wine Quality | lmjgp_supervised | 5 | 0.7272 | 0.0158 | 0.4059 | 0.0088 | 0.841 | 1.9969 |
| Wine Quality | lmjgp_supervised_residual_ridge | 5 | 0.7241 | 0.0256 | 0.4029 | 0.0129 | 0.808 | 1.9009 |
| Wine Quality | lmjgp_transductive | 5 | 0.7307 | 0.0124 | 0.4114 | 0.0061 | 0.850 | 2.0428 |
| Wine Quality | lmjgp_transductive_residual_ridge | 5 | 0.7213 | 0.0200 | 0.4027 | 0.0138 | 0.828 | 1.9477 |
| Wine Quality | xgb_bootstrap | 5 | 0.6890 | 0.0154 | 0.3969 | 0.0161 | 0.697 | 1.3315 |

## Command

```powershell
python experiments/uci/compare_smalltrain_multiseed_grouped.py --num_exp 5 --datasets "Wine Quality,Appliances Energy Prediction" --max_total_train 1000 --max_test_anchors 200 --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3 --parkinsons_standardize_x -1 --parkinsons_standardize_y -1 --out_dir experiments\uci\smalltrain_multiseed_train1000_wine_appliances_5seeds
```
