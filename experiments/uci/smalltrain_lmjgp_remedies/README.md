# Small-train LMJGP Remedies

This folder was produced by `experiments/uci/try_lmjgp_remedies.py`.

The run keeps existing `lmjgp_transductive` / `lmjgp_supervised` code untouched and adds experiment-only variants:

- `lmjgp_supervised_base`: strict supervised q(V) + JumpGP prediction.
- `lmjgp_supervised_hybrid_pls`: PLS-score/raw-X union neighborhoods + JumpGP prediction.
- `lmjgp_supervised_residual_ridge`: ridge global mean plus LMJGP residual prediction.
- `lmjgp_supervised_hybrid_residual_ridge`: hybrid neighborhoods on ridge score plus residual prediction.
- `lmjgp_supervised_localgp_*`: plain local-GP kernel diagnostics, not JumpGP/CEM.

## Command

```powershell
python experiments/uci/try_lmjgp_remedies.py --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128 --neighbors auto --lmjgp_steps 300 --MC_num 3 --xgb_boots 30 --scaling_preset uci_empirical --out_dir experiments\uci\smalltrain_lmjgp_remedies
```

## Metrics

| dataset | method | n | RMSE | CRPS | Cov90 | Width90 |
|---|---|---:|---:|---:|---:|---:|
| Wine Quality | xgboost | 25 | 0.7283 | 0.4328 | 0.665 | 1.3041 |
| Wine Quality | lmjgp_supervised_base | 25 | 0.7537 | 0.4245 | 0.825 | 2.0083 |
| Wine Quality | lmjgp_supervised_localgp_rbf | 25 | 0.7689 | 0.4287 | 0.940 | 2.8731 |
| Wine Quality | lmjgp_supervised_localgp_matern32 | 25 | 0.7579 | 0.4238 | 0.940 | 2.9561 |
| Wine Quality | lmjgp_supervised_localgp_matern52 | 25 | 0.7616 | 0.4252 | 0.940 | 2.9295 |
| Wine Quality | lmjgp_supervised_hybrid_pls | 25 | 0.7499 | 0.4226 | 0.835 | 2.0491 |
| Wine Quality | lmjgp_supervised_residual_ridge | 25 | 0.7519 | 0.4276 | 0.755 | 1.8369 |
| Wine Quality | lmjgp_supervised_hybrid_residual_ridge | 25 | 0.7604 | 0.4346 | 0.805 | 1.8224 |
| Parkinsons Telemonitoring | xgboost | 25 | 5.5585 | 3.3935 | 0.630 | 10.5207 |
| Parkinsons Telemonitoring | lmjgp_supervised_base | 25 | 8.4269 | 4.8613 | 0.795 | 20.8017 |
| Parkinsons Telemonitoring | lmjgp_supervised_localgp_rbf | 25 | 8.1311 | 4.6542 | 0.875 | 24.7467 |
| Parkinsons Telemonitoring | lmjgp_supervised_localgp_matern32 | 25 | 8.0411 | 4.6052 | 0.895 | 25.0053 |
| Parkinsons Telemonitoring | lmjgp_supervised_localgp_matern52 | 25 | 8.0658 | 4.6175 | 0.895 | 24.9062 |
| Parkinsons Telemonitoring | lmjgp_supervised_hybrid_pls | 25 | 7.9890 | 4.6000 | 0.800 | 20.5508 |
| Parkinsons Telemonitoring | lmjgp_supervised_residual_ridge | 25 | 8.2096 | 4.7039 | 0.795 | 20.6567 |
| Parkinsons Telemonitoring | lmjgp_supervised_hybrid_residual_ridge | 25 | 8.1766 | 4.7430 | 0.775 | 19.5699 |
| Appliances Energy Prediction | xgboost | 35 | 103.2599 | 41.9003 | 0.865 | 123.5757 |
| Appliances Energy Prediction | lmjgp_supervised_base | 35 | 109.0166 | 42.0714 | 0.830 | 132.5590 |
| Appliances Energy Prediction | lmjgp_supervised_localgp_rbf | 35 | 104.6147 | 43.9464 | 0.920 | 258.2219 |
| Appliances Energy Prediction | lmjgp_supervised_localgp_matern32 | 35 | 104.3602 | 43.8736 | 0.920 | 262.0637 |
| Appliances Energy Prediction | lmjgp_supervised_localgp_matern52 | 35 | 104.4357 | 43.8854 | 0.920 | 260.8607 |
| Appliances Energy Prediction | lmjgp_supervised_hybrid_pls | 35 | 109.5203 | 42.6392 | 0.845 | 127.7739 |
| Appliances Energy Prediction | lmjgp_supervised_residual_ridge | 35 | 102.7405 | 42.8400 | 0.845 | 135.5076 |
| Appliances Energy Prediction | lmjgp_supervised_hybrid_residual_ridge | 35 | 105.6098 | 44.7535 | 0.795 | 133.1158 |
