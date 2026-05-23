# Small-Train Multiseed Grouped UCI

This folder was produced by `experiments/uci/compare_smalltrain_multiseed_grouped.py`.

Parkinsons uses subject-grouped splits; Wine and Appliances use the historical random-row split.

## Tuned Presets

| dataset | split | std_x | std_y | n | steps |
|---|---|---:|---:|---:|---:|
| Wine Quality | random_row | True | False | 25 | 300 |
| Parkinsons Telemonitoring | subject_grouped | False | False | 25 | 300 |
| Appliances Energy Prediction | random_row | True | True | 35 | 300 |

## Aggregate Metrics

| dataset | method | seeds | RMSE mean | RMSE std | CRPS mean | CRPS std | Cov90 mean | Width90 mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Appliances Energy Prediction | dkl_svgp | 5 | 119.6050 | 8.1002 | 57.5886 | 6.5436 | 0.479 | 46.1764 |
| Appliances Energy Prediction | lmjgp_supervised | 5 | 106.9383 | 9.6497 | 44.9789 | 4.8758 | 0.832 | 162.8011 |
| Appliances Energy Prediction | lmjgp_transductive | 5 | 108.7104 | 8.0330 | 45.0335 | 3.8921 | 0.825 | 158.7733 |
| Appliances Energy Prediction | xgb_bootstrap | 5 | 104.0023 | 5.6673 | 44.9937 | 3.0001 | 0.881 | 169.6967 |
| Parkinsons Telemonitoring | dkl_svgp | 5 | 7.8143 | 1.1671 | 4.9228 | 0.9898 | 0.657 | 15.3927 |
| Parkinsons Telemonitoring | lmjgp_supervised | 5 | 9.5022 | 0.5584 | 5.9962 | 0.5127 | 0.652 | 20.7737 |
| Parkinsons Telemonitoring | lmjgp_transductive | 5 | 9.5167 | 0.7858 | 5.9923 | 0.6554 | 0.673 | 20.6866 |
| Parkinsons Telemonitoring | xgb_bootstrap | 5 | 9.4787 | 0.8698 | 6.7406 | 0.7476 | 0.320 | 9.9324 |
| Wine Quality | dkl_svgp | 5 | 0.9656 | 0.0209 | 0.6745 | 0.0146 | 0.229 | 0.4813 |
| Wine Quality | lmjgp_supervised | 5 | 0.7753 | 0.0269 | 0.4366 | 0.0131 | 0.853 | 2.1899 |
| Wine Quality | lmjgp_transductive | 5 | 0.7809 | 0.0173 | 4310911073596211.0000 | 8621822147192422.0000 | 0.850 | 60684384173102096.0000 |
| Wine Quality | xgb_bootstrap | 5 | 0.7508 | 0.0373 | 0.4405 | 0.0217 | 0.680 | 1.3749 |

## Command

```powershell
python experiments/uci/compare_smalltrain_multiseed_grouped.py --num_exp 5 --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3 --out_dir experiments\uci\smalltrain_multiseed_grouped_5seeds
```
