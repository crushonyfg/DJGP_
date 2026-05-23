# DeepMahalanobisGP NPZ Run

Produced by `experiments/dmgp/run_deep_mahalanobis_gp_npz.py`.

## Command

```powershell
conda run -n dmgp_tf1 python experiments/dmgp/run_deep_mahalanobis_gp_npz.py --manifest experiments/pdebench/dmgp_burgers_train200_500_1000_5seeds_20260517/manifest.json --out_dir experiments/pdebench/dmgp_burgers_train200_500_1000_5seeds_20260517 --steps1 300 --steps2 700 --m_u 50 --m_v 25
```

## Aggregate

| benchmark | dataset | train | subset | seeds | RMSE | CRPS | Cov90 | Width90 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| pdebench_burgers | PDEBench Burgers ShockJump | 1000 | all | 5 | 0.7000 | 0.3019 | 0.937 | 1.7050 |
| pdebench_burgers | PDEBench Burgers ShockJump | 1000 | far_from_threshold | 5 | 0.6707 | 0.2778 | 0.949 | 1.6374 |
| pdebench_burgers | PDEBench Burgers ShockJump | 1000 | near_threshold | 5 | 0.7754 | 0.3686 | 0.904 | 1.8927 |
| pdebench_burgers | PDEBench Burgers ShockJump | 200 | all | 5 | 0.8297 | 0.3916 | 0.905 | 2.0422 |
| pdebench_burgers | PDEBench Burgers ShockJump | 200 | far_from_threshold | 5 | 0.7014 | 0.3198 | 0.959 | 1.9694 |
| pdebench_burgers | PDEBench Burgers ShockJump | 200 | near_threshold | 5 | 1.1041 | 0.5906 | 0.757 | 2.2440 |
| pdebench_burgers | PDEBench Burgers ShockJump | 500 | all | 5 | 0.7308 | 0.3202 | 0.940 | 1.8768 |
| pdebench_burgers | PDEBench Burgers ShockJump | 500 | far_from_threshold | 5 | 0.6971 | 0.2946 | 0.950 | 1.8047 |
| pdebench_burgers | PDEBench Burgers ShockJump | 500 | near_threshold | 5 | 0.8154 | 0.3914 | 0.913 | 2.0767 |