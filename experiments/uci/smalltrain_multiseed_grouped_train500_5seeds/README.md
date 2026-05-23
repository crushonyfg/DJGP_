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
| Wine Quality | dkl_svgp | 2 | 0.8279 | 0.0475 | 0.5777 | 0.0183 | 0.230 | 0.5080 |
| Wine Quality | lmjgp_supervised | 2 | 0.7236 | 0.0130 | 0.4067 | 0.0050 | 0.843 | 2.0635 |
| Wine Quality | lmjgp_transductive | 2 | 0.7105 | 0.0096 | 0.3983 | 0.0002 | 0.855 | 2.1059 |
| Wine Quality | xgb_bootstrap | 2 | 0.7061 | 0.0079 | 0.4172 | 0.0040 | 0.637 | 1.2725 |

## Command

```powershell
python experiments/uci/compare_smalltrain_multiseed_grouped.py --num_exp 5 --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" --max_total_train 500 --max_test_anchors 200 --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3 --out_dir experiments\uci\smalltrain_multiseed_grouped_train500_5seeds
```
