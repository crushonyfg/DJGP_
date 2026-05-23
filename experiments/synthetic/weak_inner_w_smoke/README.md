# Weak-Inner W Projection Ablation on Paper Synthetic Data

Purpose: test whether projection objectives with weaker local nuisance
parameters produce more useful W than the current full LMJGP VI objective on
paper-style L2/LH train=1000 synthetic data.

This is a practical ablation, not a full implementation of stochastic
q(W)-averaged JGP-VEM. Final predictions use the same JumpGP-CEM refit head as
the existing synthetic baselines.

```json
{
  "args": {
    "num_exp": 1,
    "settings": "l2_q2_train1000",
    "methods": "pls_static,pairwise_boundary_localC",
    "final_neighbor_modes": "projected",
    "max_anchors": 20,
    "n_candidate": 40,
    "gp_neighbors": 0,
    "pair_neighbors": 40,
    "n_pls": 5,
    "n_pca": 5,
    "n_random": 0,
    "steps": 5,
    "lr": 0.01,
    "lambda_gp": 1.0,
    "lambda_pair": 1.0,
    "lambda_smooth": 0.01,
    "lambda_coeff": 0.0001,
    "coeff_scale": 1.0,
    "init_C_scale": 0.0,
    "init_lengthscale": 1.0,
    "init_noise_var": 0.1,
    "pair_tau": 0.0,
    "pair_margin": 1.0,
    "log_interval": 50,
    "verbose": false,
    "resume": false,
    "out_dir": "experiments\\synthetic\\weak_inner_w_smoke"
  },
  "device": "cuda",
  "n_rows": 2,
  "n_ok": 2,
  "methods": [
    "pls_static",
    "pairwise_boundary_localC"
  ],
  "final_neighbor_modes": [
    "projected"
  ]
}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-method, per-final-neighborhood metrics.
- `aggregate.csv`: means/stds over seeds.
- `summary.json`: exact runner arguments.
