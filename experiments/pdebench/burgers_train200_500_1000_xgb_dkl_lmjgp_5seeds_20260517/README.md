# PDEBench Burgers Train-Size Baselines

Produced by `experiments/pdebench/compare_burgers_train_sizes.py`.

## Aggregate

| train | subset | method | seeds | RMSE | CRPS | Cov90 | Width90 |
|---:|---|---|---:|---:|---:|---:|---:|
| 200 | all | dkl_svgp | 5 | 0.9225 | 0.4706 | 0.700 | 1.4414 |
| 200 | far_from_threshold | dkl_svgp | 5 | 0.7097 | 0.3481 | 0.755 | 1.4198 |
| 200 | near_threshold | dkl_svgp | 5 | 1.3412 | 0.8102 | 0.547 | 1.5013 |
| 200 | all | lmjgp_predict_vi | 5 | 0.8866 | 0.3517 | 0.823 | 1.3452 |
| 200 | far_from_threshold | lmjgp_predict_vi | 5 | 0.6899 | 0.2210 | 0.882 | 0.9011 |
| 200 | near_threshold | lmjgp_predict_vi | 5 | 1.2821 | 0.7142 | 0.660 | 2.5770 |
| 200 | all | xgb_bootstrap | 5 | 0.8309 | 0.3813 | 0.918 | 1.9604 |
| 200 | far_from_threshold | xgb_bootstrap | 5 | 0.7002 | 0.3081 | 0.950 | 1.8663 |
| 200 | near_threshold | xgb_bootstrap | 5 | 1.1143 | 0.5843 | 0.828 | 2.2212 |
| 500 | all | dkl_svgp | 5 | 0.9364 | 0.5094 | 0.605 | 1.2275 |
| 500 | far_from_threshold | dkl_svgp | 5 | 0.7421 | 0.3982 | 0.641 | 1.2311 |
| 500 | near_threshold | dkl_svgp | 5 | 1.3319 | 0.8179 | 0.506 | 1.2177 |
| 500 | all | lmjgp_predict_vi | 5 | 0.7895 | 0.2934 | 0.839 | 1.2003 |
| 500 | far_from_threshold | lmjgp_predict_vi | 5 | 0.6607 | 0.2038 | 0.872 | 0.8368 |
| 500 | near_threshold | lmjgp_predict_vi | 5 | 1.0665 | 0.5418 | 0.747 | 2.2084 |
| 500 | all | xgb_bootstrap | 5 | 0.7242 | 0.3206 | 0.927 | 1.7830 |
| 500 | far_from_threshold | xgb_bootstrap | 5 | 0.6444 | 0.2752 | 0.956 | 1.7318 |
| 500 | near_threshold | xgb_bootstrap | 5 | 0.9085 | 0.4465 | 0.847 | 1.9252 |
| 1000 | all | dkl_svgp | 5 | 0.8925 | 0.4588 | 0.666 | 1.3176 |
| 1000 | far_from_threshold | dkl_svgp | 5 | 0.6869 | 0.3407 | 0.721 | 1.3082 |
| 1000 | near_threshold | dkl_svgp | 5 | 1.2924 | 0.7865 | 0.515 | 1.3438 |
| 1000 | all | lmjgp_predict_vi | 5 | 0.7551 | 0.2623 | 0.842 | 1.0287 |
| 1000 | far_from_threshold | lmjgp_predict_vi | 5 | 0.6734 | 0.2037 | 0.865 | 0.7583 |
| 1000 | near_threshold | lmjgp_predict_vi | 5 | 0.9452 | 0.4248 | 0.777 | 1.7786 |
| 1000 | all | xgb_bootstrap | 5 | 0.7018 | 0.3090 | 0.922 | 1.7579 |
| 1000 | far_from_threshold | xgb_bootstrap | 5 | 0.6336 | 0.2702 | 0.948 | 1.7250 |
| 1000 | near_threshold | xgb_bootstrap | 5 | 0.8629 | 0.4168 | 0.851 | 1.8493 |

## Command

```powershell
python experiments/pdebench/compare_burgers_train_sizes.py --dataset-folder data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000 --train_sizes 200,500,1000 --num_exp 5 --Q 3 --neighbors 25 --lmjgp_steps 300 --out_dir experiments/pdebench/burgers_train200_500_1000_xgb_dkl_lmjgp_5seeds_20260517
```