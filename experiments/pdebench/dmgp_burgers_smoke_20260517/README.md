# DeepMahalanobisGP NPZ Run

Produced by `experiments/dmgp/run_deep_mahalanobis_gp_npz.py`.

## Command

```powershell
conda run -n dmgp_tf1 python experiments/dmgp/run_deep_mahalanobis_gp_npz.py --manifest experiments/pdebench/dmgp_burgers_smoke_20260517/manifest.json --out_dir experiments/pdebench/dmgp_burgers_smoke_20260517 --steps1 2 --steps2 2 --m_u 10 --m_v 8
```

## Aggregate

| benchmark | dataset | train | subset | seeds | RMSE | CRPS | Cov90 | Width90 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| pdebench_burgers | PDEBench Burgers ShockJump | 80 | all | 1 | 1.1854 | 0.5807 | 0.850 | 2.8767 |
| pdebench_burgers | PDEBench Burgers ShockJump | 80 | far_from_threshold | 1 | 1.0049 | 0.4682 | 0.920 | 2.8769 |
| pdebench_burgers | PDEBench Burgers ShockJump | 80 | near_threshold | 1 | 1.4368 | 0.7681 | 0.733 | 2.8763 |