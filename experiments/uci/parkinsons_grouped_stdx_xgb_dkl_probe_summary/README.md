# Parkinsons Grouped XGB/DKL X-Standardization Probe

Seed0 probe comparing the existing raw Parkinsons grouped XGB/DKL rows against `std_x` rows for train sizes 200, 500, and 1000.

Source raw rows come from `experiments/uci/smalltrain_multiseed_bestpreset_summary/metrics_bestpresets.csv`.
`std_x` rows come from:
- `../parkinsons_grouped_stdx_xgb_dkl_probe_train200_seed0/`
- `../parkinsons_grouped_stdx_xgb_dkl_probe_train500_seed0/`
- `../parkinsons_grouped_stdx_xgb_dkl_probe_train1000_seed0/`

`delta_* = std_x - raw`; negative is better for RMSE/CRPS.
