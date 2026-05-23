# Small-Train Tail Diagnostics

Diagnostics generated from `experiments/uci/smalltrain_pca_viz/predictions.csv`
by `experiments/uci/diagnose_smalltrain_tail.py`.

Run command:

```powershell
python experiments/uci/diagnose_smalltrain_tail.py `
  --viz_dir experiments/uci/smalltrain_pca_viz `
  --out_dir experiments/uci/smalltrain_tail_diagnostics
```

Main files:

- `tail_contribution.csv` - SSE share from the top 5%, 10%, and 20% largest absolute residuals.
- `true_y_quantile_metrics.csv` - RMSE/CRPS/coverage/width for `y<=q50`, `q50<y<=q90`, and `y>q90`.
- `sigma_bin_calibration.csv` - calibration by predicted-sigma quintile.
- `neighbor_high_y_support.csv` - high-response train-neighbor counts for high-response test points.
- `largest_errors.csv` - top residual examples per method/dataset.
- `figures/*_residual_diagnostics.png` - predicted-vs-true, residual-vs-true, and abs-error-vs-sigma panels.

Headline:

- Appliances is dominated by tail errors: the top 10% residuals explain about
  89% of XGBoost SSE and 91% of best-LMJGP SSE.
- Appliances `y>q90` has very poor 90% coverage for both methods: 0.10 for
  XGBoost and 0.15 for best LMJGP. LMJGP intervals are wider but still miss
  the largest responses because the predictive mean remains far too low.
- Parkinsons is not a single extreme-tail problem: top 10% residuals explain
  about 37% of XGBoost SSE and 40% of LMJGP SSE. LMJGP's largest misses are
  high-response points whose means are pulled toward the center.
- In the 200-train setting, high-y local-neighbor support is weak for Wine
  and modest for Appliances; the rare high-response regime is not well
  represented locally.
