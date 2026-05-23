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
python experiments/uci/try_lmjgp_remedies.py --datasets "Wine Quality" --max_total_train 80 --max_test_anchors 20 --max_train_anchors 32 --neighbors auto --lmjgp_steps 2 --MC_num 2 --xgb_boots 3 --scaling_preset uci_empirical --out_dir experiments\uci\smalltrain_lmjgp_remedies_smoke
```

## Metrics

| dataset | method | n | RMSE | CRPS | Cov90 | Width90 |
|---|---|---:|---:|---:|---:|---:|
| Wine Quality | xgboost | 25 | 0.7000 | 0.4247 | 0.700 | 1.6751 |
| Wine Quality | lmjgp_supervised_base | 25 | 0.7577 | 0.4499 | 0.800 | 1.9078 |
| Wine Quality | lmjgp_supervised_localgp_rbf | 25 | 0.8237 | 0.5010 | 0.500 | 1.2445 |
| Wine Quality | lmjgp_supervised_localgp_matern32 | 25 | 0.7524 | 0.4405 | 0.700 | 1.4909 |
| Wine Quality | lmjgp_supervised_localgp_matern52 | 25 | 0.7787 | 0.4642 | 0.600 | 1.4008 |
| Wine Quality | lmjgp_supervised_hybrid_pls | 25 | 0.7057 | 0.4112 | 0.800 | 1.7794 |
| Wine Quality | lmjgp_supervised_residual_ridge | 25 | 0.6348 | 0.3576 | 0.900 | 1.6489 |
| Wine Quality | lmjgp_supervised_hybrid_residual_ridge | 25 | 0.6297 | 0.3636 | 0.750 | 1.8335 |
