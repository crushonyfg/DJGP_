# VEM-EM UCI/PDEBench Comparison

This folder was produced by `experiments/vem/compare_vem_em_uci_pdebench.py`.

Method: `vem_em_learned_qw`, using conditional-W exact local JGP-VEM, posterior q(W) samples, and projected-space final JumpGP neighborhoods.

## Aggregate

| benchmark | train | dataset | subset | seeds | RMSE | CRPS | Cov90 | W90 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| pdebench |  | PDEBench-Burgers | all | 1 | 0.2034 | 0.1203 | 0.600 | 0.2197 |
| pdebench |  | PDEBench-Burgers | far_from_threshold | 1 | 0.1224 | 0.0746 | 0.750 | 0.2630 |
| pdebench |  | PDEBench-Burgers | near_threshold | 1 | 0.2428 | 0.1507 | 0.500 | 0.1908 |

## Runner Args

```json
{
  "benchmarks": "pdebench",
  "num_exp": 1,
  "seed_start": 0,
  "resume": false,
  "out_dir": "experiments\\vem\\vem_em_pde_smoke",
  "uci_train_sizes": "200,500,1000",
  "uci_datasets": "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction",
  "uci_Q": -1,
  "max_test_anchors": 200,
  "max_train_anchors": 128,
  "train_subject_frac": 0.8,
  "numeric_sigma_omit_factor": 50.0,
  "max_numeric_omit_frac": 0.02,
  "pdebench_dataset_folder": "data\\processed\\pdebench_burgers_shockjump_balanced_delta01_n2000",
  "pdebench_Q": 3,
  "pdebench_n": 10,
  "pdebench_max_anchors": 10,
  "n_candidate": 30,
  "n_vem": 20,
  "n_pls": 5,
  "n_pca": 5,
  "n_random": 0,
  "steps": 2,
  "lr": 0.01,
  "mc_train": 2,
  "mc_gamma": 2,
  "eval_mc": 1,
  "coeff_scale": 1.0,
  "init_log_std": -3.0,
  "prior_std": 1.0,
  "kl_weight": 0.05,
  "kl_warmup_steps": 60,
  "lambda_smooth": 0.01,
  "lambda_coeff": 0.0001,
  "lambda_gate_l2": 0.0001,
  "gamma_init": 0.7,
  "gamma_floor": 0.001,
  "gamma_damping": 0.3,
  "temperature": 5.0,
  "temperature_final": 3.0,
  "init_lengthscale": 1.0,
  "init_amp": 1.0,
  "init_noise_var": 0.1,
  "outlier_sigma_mult": 2.5,
  "jitter": 1e-05,
  "log_interval": 25,
  "use_global_mean_projection": false,
  "verbose": false
}
```
