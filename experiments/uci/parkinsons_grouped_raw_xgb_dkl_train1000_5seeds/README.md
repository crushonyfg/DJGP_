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
| Parkinsons Telemonitoring | dkl_svgp | 5 | 7.8600 | 1.0360 | 4.9263 | 0.8764 | 0.650 | 15.1595 |
| Parkinsons Telemonitoring | xgb_bootstrap | 5 | 9.7943 | 0.5703 | 7.4257 | 0.2346 | 0.187 | 6.4790 |

## Command

```powershell
python experiments/uci/compare_smalltrain_multiseed_grouped.py --num_exp 5 --datasets "Parkinsons Telemonitoring" --max_total_train 1000 --max_test_anchors 200 --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3 --parkinsons_standardize_x -1 --parkinsons_standardize_y -1 --out_dir experiments\uci\parkinsons_grouped_raw_xgb_dkl_train1000_5seeds
```
