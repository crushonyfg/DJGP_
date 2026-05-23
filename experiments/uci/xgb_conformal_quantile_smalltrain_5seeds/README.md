# Conformalized Quantile XGBoost

Split conformalized quantile XGBoost on the current UCI small-train presets.

Files:
- `aggregate.csv`: CQR aggregates over 5 seeds.
- `metrics.csv`: per-seed rows.
- `xgb_cqr_vs_bootstrap.csv`: direct comparison against the current naive bootstrap XGBoost aggregate.

The conformal interval metrics are direct interval metrics; CRPS is not reported.

Main reading:
- CQR substantially improves coverage over naive bootstrap XGBoost.
- Intervals become much wider, especially on Wine and Appliances.
- Parkinsons grouped still undercovers at train=500/1000 because calibration rows come from training subjects while test rows are unseen subjects.
