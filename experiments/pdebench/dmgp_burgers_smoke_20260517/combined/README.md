# DeepMahalanobisGP Combined Comparison

Produced by `experiments/dmgp/combine_with_baselines.py`.

## Best Methods

| dataset | train | subset | best RMSE | RMSE | best CRPS | CRPS | DMGP RMSE delta | DMGP CRPS delta |
|---|---:|---|---|---:|---|---:|---:|---:|
| PDEBench Burgers ShockJump | 80 | all | lmjgp_predict_vi | 0.9951 | lmjgp_predict_vi | 0.4118 | 0.1903 | 0.1689 |
| PDEBench Burgers ShockJump | 80 | far_from_threshold | dkl_svgp | 0.9875 | lmjgp_predict_vi | 0.3687 | 0.0174 | 0.0995 |
| PDEBench Burgers ShockJump | 80 | near_threshold | lmjgp_predict_vi | 0.9226 | lmjgp_predict_vi | 0.4835 | 0.5142 | 0.2846 |

## Files

- `aggregate_with_dmgp.csv`: all aggregate rows.
- `best_by_metric.csv`: best RMSE/CRPS method per dataset/train/subset.