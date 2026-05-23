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
| Parkinsons Telemonitoring | lmjgp_supervised | 5 | 9.0095 | 0.6705 | 5.6530 | 0.6419 | 0.645 | 19.8514 |
| Parkinsons Telemonitoring | lmjgp_supervised_residual_ridge | 5 | 9.7297 | 1.0223 | 6.1983 | 0.8520 | 0.580 | 18.5701 |
| Parkinsons Telemonitoring | lmjgp_transductive | 5 | 9.2274 | 0.7848 | 5.8363 | 0.7163 | 0.628 | 19.8558 |
| Parkinsons Telemonitoring | lmjgp_transductive_residual_ridge | 5 | 9.6518 | 1.0600 | 6.1709 | 0.8380 | 0.583 | 18.5584 |

## Command

```powershell
python experiments/uci/compare_smalltrain_multiseed_grouped.py --num_exp 5 --datasets "Parkinsons Telemonitoring" --max_total_train 1000 --max_test_anchors 200 --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3 --parkinsons_standardize_x 1 --parkinsons_standardize_y 0 --out_dir experiments\uci\parkinsons_grouped_stdx_lmjgp_train1000_5seeds
```
