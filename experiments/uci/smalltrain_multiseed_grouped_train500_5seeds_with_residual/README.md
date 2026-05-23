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
| Appliances Energy Prediction | dkl_svgp | 5 | 113.8430 | 8.0967 | 52.3699 | 4.9380 | 0.581 | 58.8813 |
| Appliances Energy Prediction | lmjgp_supervised | 5 | 106.0742 | 7.8571 | 42.0758 | 4.1942 | 0.847 | 156.6422 |
| Appliances Energy Prediction | lmjgp_supervised_residual_ridge | 5 | 104.5152 | 8.3486 | 45.1907 | 3.1440 | 0.809 | 160.9087 |
| Appliances Energy Prediction | lmjgp_transductive | 5 | 105.9345 | 9.0264 | 42.9325 | 4.0488 | 0.825 | 151.8228 |
| Appliances Energy Prediction | lmjgp_transductive_residual_ridge | 5 | 103.6829 | 7.1516 | 44.8997 | 2.5044 | 0.814 | 159.2213 |
| Appliances Energy Prediction | xgb_bootstrap | 5 | 98.4774 | 9.5908 | 43.3656 | 4.6923 | 0.880 | 176.0732 |
| Parkinsons Telemonitoring | dkl_svgp | 5 | 7.8602 | 1.0305 | 4.9345 | 0.8857 | 0.650 | 15.2668 |
| Parkinsons Telemonitoring | lmjgp_supervised | 5 | 9.8859 | 0.5949 | 6.3362 | 0.4910 | 0.605 | 19.5996 |
| Parkinsons Telemonitoring | lmjgp_supervised_residual_ridge | 5 | 9.7853 | 0.7271 | 6.3087 | 0.5853 | 0.587 | 18.9504 |
| Parkinsons Telemonitoring | lmjgp_transductive | 5 | 9.9009 | 0.5436 | 6.3287 | 0.4598 | 0.608 | 19.8374 |
| Parkinsons Telemonitoring | lmjgp_transductive_residual_ridge | 5 | 9.8269 | 0.5943 | 6.3186 | 0.4876 | 0.589 | 19.0978 |
| Parkinsons Telemonitoring | xgb_bootstrap | 5 | 9.5327 | 0.7271 | 7.0371 | 0.6295 | 0.246 | 7.8662 |
| Wine Quality | dkl_svgp | 5 | 0.8517 | 0.0641 | 0.5980 | 0.0492 | 0.224 | 0.4861 |
| Wine Quality | lmjgp_supervised | 5 | 0.7561 | 0.0293 | 0.4228 | 0.0211 | 0.827 | 2.0564 |
| Wine Quality | lmjgp_supervised_residual_ridge | 5 | 0.7366 | 0.0264 | 0.4162 | 0.0169 | 0.802 | 1.9307 |
| Wine Quality | lmjgp_transductive | 5 | 0.7401 | 0.0307 | 0.4158 | 0.0200 | 0.844 | 2.0718 |
| Wine Quality | lmjgp_transductive_residual_ridge | 5 | 0.7465 | 0.0266 | 0.4180 | 0.0148 | 0.809 | 1.9311 |
| Wine Quality | xgb_bootstrap | 5 | 0.7202 | 0.0182 | 0.4199 | 0.0077 | 0.652 | 1.2646 |

## Command

```powershell
python experiments/uci/compare_smalltrain_multiseed_grouped.py --num_exp 5 --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" --max_total_train 500 --max_test_anchors 200 --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3 --out_dir experiments\uci\smalltrain_multiseed_grouped_train500_5seeds_with_residual
```
