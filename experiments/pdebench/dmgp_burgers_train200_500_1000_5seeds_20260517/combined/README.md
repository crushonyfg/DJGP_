# DeepMahalanobisGP Combined Comparison

Produced by `experiments/dmgp/combine_with_baselines.py`.

## Best Methods

| dataset | train | subset | best RMSE | RMSE | best CRPS | CRPS | DMGP RMSE delta | DMGP CRPS delta |
|---|---:|---|---|---:|---|---:|---:|---:|
| PDEBench Burgers ShockJump | 1000 | all | deep_mahalanobis_gp | 0.7000 | lmjgp_predict_vi | 0.2623 | 0.0000 | 0.0395 |
| PDEBench Burgers ShockJump | 1000 | far_from_threshold | xgb_bootstrap | 0.6336 | lmjgp_predict_vi | 0.2037 | 0.0370 | 0.0741 |
| PDEBench Burgers ShockJump | 1000 | near_threshold | deep_mahalanobis_gp | 0.7754 | deep_mahalanobis_gp | 0.3686 | 0.0000 | 0.0000 |
| PDEBench Burgers ShockJump | 200 | all | deep_mahalanobis_gp | 0.8297 | lmjgp_predict_vi | 0.3517 | 0.0000 | 0.0398 |
| PDEBench Burgers ShockJump | 200 | far_from_threshold | lmjgp_predict_vi | 0.6899 | lmjgp_predict_vi | 0.2210 | 0.0115 | 0.0988 |
| PDEBench Burgers ShockJump | 200 | near_threshold | deep_mahalanobis_gp | 1.1041 | xgb_bootstrap | 0.5843 | 0.0000 | 0.0063 |
| PDEBench Burgers ShockJump | 500 | all | xgb_bootstrap | 0.7242 | lmjgp_predict_vi | 0.2934 | 0.0066 | 0.0268 |
| PDEBench Burgers ShockJump | 500 | far_from_threshold | xgb_bootstrap | 0.6444 | lmjgp_predict_vi | 0.2038 | 0.0527 | 0.0907 |
| PDEBench Burgers ShockJump | 500 | near_threshold | deep_mahalanobis_gp | 0.8154 | deep_mahalanobis_gp | 0.3914 | 0.0000 | 0.0000 |

## Files

- `aggregate_with_dmgp.csv`: all aggregate rows.
- `best_by_metric.csv`: best RMSE/CRPS method per dataset/train/subset.