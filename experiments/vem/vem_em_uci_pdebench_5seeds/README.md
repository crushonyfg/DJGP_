# VEM-EM UCI/PDEBench Comparison

This folder was produced by `experiments/vem/compare_vem_em_uci_pdebench.py`.

Method: `vem_em_learned_qw`, using conditional-W exact local JGP-VEM, posterior q(W) samples, and projected-space final JumpGP neighborhoods.

## Aggregate

| benchmark | train | dataset | subset | seeds | RMSE | CRPS | Cov90 | W90 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| pdebench |  | PDEBench-Burgers | all | 5 | 0.7683 | 0.2545 | 0.781 | 0.5888 |
| pdebench |  | PDEBench-Burgers | far_from_threshold | 5 | 0.7390 | 0.2223 | 0.803 | 0.5156 |
| pdebench |  | PDEBench-Burgers | near_threshold | 5 | 0.8434 | 0.3439 | 0.723 | 0.7918 |
| uci | 1000 | Appliances Energy Prediction | all | 5 | 110.5677 | 44.2252 | 0.717 | 96.2819 |
| uci | 1000 | Parkinsons Telemonitoring | all | 5 | 9.9310 | 6.8302 | 0.374 | 11.4217 |
| uci | 1000 | Wine Quality | all | 5 | 0.7701 | 0.4369 | 0.737 | 1.5368 |
| uci | 200 | Appliances Energy Prediction | all | 5 | 114.3425 | 49.0878 | 0.698 | 106.1495 |
| uci | 200 | Parkinsons Telemonitoring | all | 5 | 10.3404 | 7.0605 | 0.428 | 14.2162 |
| uci | 200 | Wine Quality | all | 5 | 0.8481 | 0.5007 | 0.678 | 1.5892 |
| uci | 500 | Appliances Energy Prediction | all | 5 | 108.4553 | 44.1884 | 0.739 | 110.0876 |
| uci | 500 | Parkinsons Telemonitoring | all | 5 | 10.4183 | 7.1625 | 0.383 | 12.5968 |
| uci | 500 | Wine Quality | all | 5 | 0.7699 | 0.4470 | 0.719 | 1.5304 |

## Runner Args

```json
{
  "benchmarks": "pdebench",
  "num_exp": 5,
  "seed_start": 0,
  "resume": true,
  "out_dir": "experiments\\vem\\vem_em_uci_pdebench_5seeds",
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
  "pdebench_n": 25,
  "pdebench_max_anchors": null,
  "n_candidate": 100,
  "n_vem": 50,
  "n_pls": 5,
  "n_pca": 5,
  "n_random": 0,
  "steps": 80,
  "lr": 0.01,
  "mc_train": 2,
  "mc_gamma": 2,
  "eval_mc": 2,
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
