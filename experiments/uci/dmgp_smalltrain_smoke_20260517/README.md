# DeepMahalanobisGP NPZ Run

Produced by `experiments/dmgp/run_deep_mahalanobis_gp_npz.py`.

## Command

```powershell
conda run -n dmgp_tf1 python experiments/dmgp/run_deep_mahalanobis_gp_npz.py --manifest experiments/uci/dmgp_smalltrain_smoke_20260517/manifest.json --out_dir experiments/uci/dmgp_smalltrain_smoke_20260517 --steps1 2 --steps2 2 --m_u 10 --m_v 8
```

## Aggregate

| benchmark | dataset | train | subset | seeds | RMSE | CRPS | Cov90 | Width90 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| uci | Wine Quality | 80 | all | 1 | 0.6988 | 0.4017 | 1.000 | 2.3927 |