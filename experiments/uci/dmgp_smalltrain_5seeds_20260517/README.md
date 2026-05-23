# DeepMahalanobisGP NPZ Run

Produced by `experiments/dmgp/run_deep_mahalanobis_gp_npz.py`.

## Command

```powershell
conda run -n dmgp_tf1 python experiments/dmgp/run_deep_mahalanobis_gp_npz.py --manifest experiments/uci/dmgp_smalltrain_5seeds_20260517/manifest.json --out_dir experiments/uci/dmgp_smalltrain_5seeds_20260517 --steps1 300 --steps2 700 --m_u 50 --m_v 25
```

## Aggregate

| benchmark | dataset | train | subset | seeds | RMSE | CRPS | Cov90 | Width90 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| uci | Appliances Energy Prediction | 1000 | all | 5 | 103.6607 | 43.9332 | 0.886 | 189.9586 |
| uci | Appliances Energy Prediction | 200 | all | 5 | 107.3101 | 45.3789 | 0.873 | 174.1911 |
| uci | Appliances Energy Prediction | 500 | all | 5 | 104.9674 | 44.7008 | 0.893 | 198.2882 |
| uci | Parkinsons Telemonitoring | 1000 | all | 5 | 8.9489 | 5.8193 | 0.518 | 14.2509 |
| uci | Parkinsons Telemonitoring | 200 | all | 5 | 8.6731 | 5.4898 | 0.584 | 16.6948 |
| uci | Parkinsons Telemonitoring | 500 | all | 5 | 9.0286 | 5.8317 | 0.544 | 15.4490 |
| uci | Wine Quality | 1000 | all | 5 | 0.7106 | 0.4082 | 0.741 | 1.5064 |
| uci | Wine Quality | 200 | all | 5 | 0.7333 | 0.4112 | 0.830 | 1.8568 |
| uci | Wine Quality | 500 | all | 5 | 0.7256 | 0.4148 | 0.759 | 1.5935 |