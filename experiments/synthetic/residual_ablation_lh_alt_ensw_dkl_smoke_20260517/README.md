# Global GP + Residual Self-CEM LH Ablation

Implements ``docs/residual_ablation.md`` on top of the best LH 5-seed
self-CEM baseline (``self_cem2_nstd0.1_kcov0.1``).  Each variant adds a
global smooth-mean component on top of the local self-CEM:

- ``diagvar``     -- per-anchor mean global variance enforced as the local
  noise floor; the local CEM operates on residuals ``y_i - mu_g(x_i)``.
- ``blockvar``    -- the full global posterior block ``Sigma_g(X_a, X_a)``
  is added inside the local CEM kernel ``K_h + Sigma_g + sigma_a^2 I``.
- ``oob_diagvar`` -- K-fold OOB ``mu_g`` / ``v_g`` for training points to
  avoid in-sample residual shrinkage.
- ``alt``         -- alternates one global GP refit with heteroscedastic
  pseudo-noise ``sigma_eps^2 + v_h_i`` after the first local self-CEM round.
- ``alt_ensW``    -- ``alt`` + sampled deterministic-W self-CEM ensemble at
  prediction time (moment-matched mixture).

Files:
- ``metrics.csv``: one row per setting/seed/variant.
- ``aggregate.csv``: mean/std across successful rows.
- ``summary.json``: exact runner arguments and setting metadata.

Project log entry: ``docs/experiments_log.md`` section
``2026-05-17 - residual_ablation_lh_5seeds``.

```json
{
  "started": "2026-05-17 16:45:08",
  "finished": "2026-05-17 16:51:11",
  "device": "cuda",
  "args": {
    "settings": "lh_q5_train1000",
    "variants": "alt_ensW",
    "seed_start": 0,
    "num_exp": 1,
    "max_test": 40,
    "n_neighbors": 35,
    "w_var": 0.02,
    "cem_updates": 2,
    "kcov": 0.1,
    "local_noise_std_floor": 0.1,
    "hyper_steps": 30,
    "hyper_lr": 0.04,
    "gate_steps": 60,
    "gate_lr": 0.05,
    "gate_l2": 0.001,
    "gate_max_norm": 8.0,
    "gh_points": 20,
    "min_inliers": 2,
    "max_inlier_frac": 0.95,
    "outlier_sigma_mult": 2.5,
    "background_scale_mult": 3.0,
    "hyperopt_min_noise_var": 1e-05,
    "hyperopt_max_noise_var": 10.0,
    "hyperopt_min_signal_var": 0.0001,
    "hyperopt_max_signal_var": 100.0,
    "hyperopt_min_lengthscale": 0.001,
    "hyperopt_max_lengthscale": 20.0,
    "alt_outer_cycles": 1,
    "oob_k_folds": 5,
    "ensemble_samples": 3,
    "block_variance_scale": 1.0,
    "diag_variance_inflation": 1.0,
    "use_oob_for_test": true,
    "alt_train_anchor_subsample": 200,
    "alt_train_hyper_steps": 10,
    "alt_train_gate_steps": 20,
    "global_backend": "dkl",
    "global_num_inducing": 64,
    "global_epochs": 40,
    "global_batch_size": 512,
    "global_lr": 0.01,
    "global_weight_decay": 0.0,
    "global_init_noise_var": 0.05,
    "global_jitter": 0.0001,
    "global_verbose": false,
    "dkl_hidden": 64,
    "dkl_latent": 8,
    "deepgp_hidden_dim": 4,
    "deepgp_num_inducing_inner": 64,
    "deepgp_num_samples": 8,
    "resume": false,
    "out_dir": "experiments/synthetic/residual_ablation_lh_alt_ensw_dkl_smoke_20260517"
  },
  "settings": {
    "lh_q5_train1000": {
      "family": "LH",
      "N_train": 1000,
      "N_test": 200,
      "d": 5,
      "H": 30,
      "Q_true": 5,
      "Q": 5,
      "n": 35,
      "caseno": 0,
      "noise_std": 2.0,
      "noise_var": 4.0,
      "expansion": "rff",
      "lengthscale": 0.5,
      "kernel_var": 9.0
    }
  },
  "variants": [
    "alt_ensW"
  ],
  "n_rows": 1,
  "n_ok": 1,
  "n_failed": 0
}
```
