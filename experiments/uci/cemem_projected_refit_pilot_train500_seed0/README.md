# UCI CEM-EM projected-refit pilot

One-seed train=500 pilot for large-candidate projection learning, projected-space
final neighbor reselection, and a simple cluster-anchor shared-W variant.

Command:

```powershell
python experiments\uci\compare_cemem_projected_refit.py `
  --num_exp 1 --max_total_train 500 --max_test_anchors 200 `
  --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction" `
  --n_candidate 100 --n_clusters 64 --n_outer 5 --n_m_steps 80 `
  --out_dir experiments\uci\cemem_projected_refit_pilot_train500_seed0
```

Key artifacts:

- `metrics.csv`: raw rows for the four new variants.
- `aggregate.csv`: one-seed aggregate of the same rows.
- `comparison_same_split_seed0_train500.csv`: new rows plus current same-split
  bestpreset seed-0 baselines.
- `comparison_same_split_seed0_train500_summary.json`: compact best-new vs
  best-existing summary.

Headline:

| dataset | best new variant | new RMSE / CRPS | best existing same-split row | existing RMSE / CRPS |
|---|---|---:|---|---:|
| Wine Quality | `cemem_large_raw_final` | 0.780 / 0.479 | `lmjgp_supervised_residual_ridge` | 0.697 / 0.395 |
| Parkinsons Telemonitoring | `cemem_large_raw_final` | 10.083 / 6.904 | `dkl_svgp` | 7.106 / 4.105 |
| Appliances Energy Prediction | `cemem_small_current` RMSE, `cemem_large_raw_final` CRPS | 107.241 / 42.348 | `xgb_bootstrap` RMSE, `lmjgp_supervised` CRPS | 99.977 / 38.792 |

Interpretation:

- Large-candidate learning helps the CEM-EM line relative to small-current on
  Wine and Parkinsons, and improves Appliances CRPS, but does not beat the
  existing best variants.
- Projected-small final neighbor reselection was not better in this one-seed
  pilot.
- The simple cluster-anchor variant was weaker than per-test large-candidate
  learning. Multi-regime cluster experts or train-label-aware cluster splitting
  would need a separate experiment.
