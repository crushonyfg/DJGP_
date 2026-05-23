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
| Parkinsons Telemonitoring | dkl_svgp | 1 | 10.1120 | 0.0000 | 8.2604 | 0.0000 | 0.060 | 4.8722 |
| Parkinsons Telemonitoring | xgb_bootstrap | 1 | 9.0152 | 0.0000 | 7.0216 | 0.0000 | 0.160 | 5.7388 |

## Command

```powershell
python experiments/uci/compare_smalltrain_multiseed_grouped.py --num_exp 1 --datasets "Parkinsons Telemonitoring" --max_total_train 1000 --max_test_anchors 200 --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3 --parkinsons_standardize_x 1 --parkinsons_standardize_y 0 --out_dir experiments\uci\parkinsons_grouped_stdx_xgb_dkl_probe_train1000_seed0
```
