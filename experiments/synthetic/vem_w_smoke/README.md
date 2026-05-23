# Conditional-W JGP-VEM on Paper Synthetic Data

Purpose: implement and test a more faithful VEM/EM-style projection learner
where a posterior over W is passed into exact local JGP-VEM updates.

```json
{
  "args": {
    "num_exp": 1,
    "settings": "l2_q2_train1000",
    "methods": "map_vem1,post_vem1_boundary",
    "prediction_heads": "mean,posterior",
    "final_neighbor_modes": "raw",
    "max_anchors": 20,
    "n_candidate": 40,
    "n_vem": 25,
    "n_pls": 5,
    "n_pca": 5,
    "n_random": 0,
    "steps": 5,
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
    "gamma_damping": 0.7,
    "temperature": 1.5,
    "temperature_final": 1.0,
    "init_lengthscale": 1.0,
    "init_amp": 1.0,
    "init_noise_var": 0.1,
    "outlier_sigma_mult": 2.5,
    "jitter": 1e-05,
    "log_interval": 25,
    "verbose": false,
    "resume": false,
    "out_dir": "experiments\\synthetic\\vem_w_smoke"
  },
  "device": "cuda",
  "n_rows": 3,
  "n_ok": 3,
  "methods": [
    "map_vem1",
    "post_vem1_boundary"
  ],
  "prediction_heads": [
    "mean",
    "posterior"
  ],
  "final_neighbor_modes": [
    "raw"
  ]
}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-method, per-prediction-head rows.
- `aggregate.csv`: means/stds over successful rows.
- `histories/*.json`: VEM-W training histories.
- `summary.json`: exact runner arguments.
