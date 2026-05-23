# Soft-Gamma Conditional-W VEM Controls

Purpose: five-seed evaluation of the best current conditional-W VEM variant
and learned/posterior controls.

```json
{
  "args": {
    "num_exp": 5,
    "settings": "lh_q5_train1000",
    "controls": "learned_qw,anchor_shuffled,global_mean,prior_qw",
    "max_anchors": null,
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
    "drop_exploded_points": true,
    "max_sigma_for_metrics": 1000000.0,
    "seed_start": 4,
    "verbose": false,
    "resume": false,
    "out_dir": "experiments\\synthetic\\vem_w_softgamma_controls_lh_seed4_refilter"
  },
  "device": "cuda",
  "n_rows": 4,
  "n_ok": 4,
  "controls": [
    "learned_qw",
    "anchor_shuffled",
    "global_mean",
    "prior_qw"
  ]
}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-control rows.
- `aggregate.csv`: means/stds over successful rows.
- `histories/*.json`: VEM-W training histories for each setting/seed.
- `summary.json`: exact runner arguments.
