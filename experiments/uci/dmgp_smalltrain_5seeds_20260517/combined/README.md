# DeepMahalanobisGP Combined Comparison

Produced by `experiments/dmgp/combine_with_baselines.py`.

## Best Methods

| dataset | train | subset | best RMSE | RMSE | best CRPS | CRPS | DMGP RMSE delta | DMGP CRPS delta |
|---|---:|---|---|---:|---|---:|---:|---:|
| Appliances Energy Prediction | 1000 | all | xgb_bootstrap | 96.2436 | lmjgp_transductive_residual_ridge | 41.0546 | 7.4172 | 2.8786 |
| Appliances Energy Prediction | 200 | all | lmjgp_supervised_residual_ridge | 102.9810 | lmjgp_supervised | 44.9789 | 4.3291 | 0.4001 |
| Appliances Energy Prediction | 500 | all | xgb_bootstrap | 98.4774 | lmjgp_supervised | 42.0758 | 6.4900 | 2.6251 |
| Parkinsons Telemonitoring | 1000 | all | dkl_svgp | 7.8600 | dkl_svgp | 4.9263 | 1.0888 | 0.8930 |
| Parkinsons Telemonitoring | 200 | all | dkl_svgp | 7.8143 | dkl_svgp | 4.9228 | 0.8588 | 0.5670 |
| Parkinsons Telemonitoring | 500 | all | dkl_svgp | 7.8602 | dkl_svgp | 4.9345 | 1.1685 | 0.8972 |
| Wine Quality | 1000 | all | xgb_bootstrap | 0.6890 | xgb_bootstrap | 0.3969 | 0.0216 | 0.0113 |
| Wine Quality | 200 | all | deep_mahalanobis_gp | 0.7333 | deep_mahalanobis_gp | 0.4112 | 0.0000 | 0.0000 |
| Wine Quality | 500 | all | xgb_bootstrap | 0.7202 | deep_mahalanobis_gp | 0.4148 | 0.0054 | 0.0000 |

## Files

- `aggregate_with_dmgp.csv`: all aggregate rows.
- `best_by_metric.csv`: best RMSE/CRPS method per dataset/train/subset.