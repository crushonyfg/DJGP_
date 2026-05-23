# PDEBench Burgers Self-CEM Train-Size Sweep

Produced by `experiments/pdebench/compare_burgers_selfcem_train_sizes.py`.

Method: non-MC `self_cem2_nstd0.1_kcov0.1`.

| train | subset | method | seeds | RMSE | CRPS | Cov90 | Width90 |
|---:|---|---|---:|---:|---:|---:|---:|
| 80 | all | self_cem2_nstd0.1_kcov0.1 | 1 | 0.6083 | 0.2412 | 0.775 | 0.8416 |
| 80 | far_from_threshold | self_cem2_nstd0.1_kcov0.1 | 1 | 0.7221 | 0.2361 | 0.920 | 0.6938 |
| 80 | near_threshold | self_cem2_nstd0.1_kcov0.1 | 1 | 0.3427 | 0.2498 | 0.533 | 1.0880 |

## Command

```powershell
python experiments/pdebench/compare_burgers_selfcem_train_sizes.py --dataset-folder data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000 --train_sizes 80 --num_exp 1 --Q 3 --neighbors 25 --out_dir experiments\pdebench\burgers_selfcem_smoke_20260519
```