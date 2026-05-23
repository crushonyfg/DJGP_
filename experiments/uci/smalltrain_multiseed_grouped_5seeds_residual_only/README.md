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
| Appliances Energy Prediction | lmjgp_supervised_residual_ridge | 5 | 102.9810 | 5.6665 | 47.6648 | 4.5818 | 0.783 | 162.7498 |
| Appliances Energy Prediction | lmjgp_transductive_residual_ridge | 5 | 103.8924 | 7.2592 | 48.3845 | 4.9772 | 0.783 | 158.1679 |
| Parkinsons Telemonitoring | lmjgp_supervised_residual_ridge | 5 | 9.5108 | 0.8180 | 5.9529 | 0.6943 | 0.665 | 20.2865 |
| Parkinsons Telemonitoring | lmjgp_transductive_residual_ridge | 5 | 9.3520 | 0.7834 | 5.8510 | 0.6673 | 0.661 | 20.3538 |
| Wine Quality | lmjgp_supervised_residual_ridge | 5 | 0.7641 | 0.0235 | 0.4289 | 0.0131 | 0.818 | 2.0214 |
| Wine Quality | lmjgp_transductive_residual_ridge | 5 | 0.7700 | 0.0260 | 0.4292 | 0.0108 | 0.823 | 2.0098 |

## Command

```powershell
python experiments/uci/compare_smalltrain_multiseed_grouped.py --num_exp 5 --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3 --out_dir experiments\uci\smalltrain_multiseed_grouped_5seeds_residual_only
```
