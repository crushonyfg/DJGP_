# Global WGP-CEM L2/LH Ablation

This experiment trains a sparse-GP prior over projection matrices `W(x)` with
fixed training neighborhoods and a local JGP-CEM composite likelihood.

Variants:

- `anchor_uniform`: anchor-conditioned `W(c_a)` with fixed uniform outlier likelihood.
- `anchor_floor`: anchor-conditioned plus soft gamma floor.
- `anchor_coverage`: anchor-conditioned plus gamma floor and coverage penalty.
- `anchor_background`: anchor-conditioned plus gamma floor, coverage, and broad normal background `p0(y)`.
- `pointwise_coverage`: pointwise `W(x_i)` plus gamma floor and coverage.
- `pointwise_background`: pointwise plus gamma floor, coverage, and broad normal background.

Files:

- `metrics.csv`: one row per setting/seed/variant.
- `aggregate.csv`: mean/std over successful rows.
- `history_<setting>_seed<seed>_<variant>.csv`: optimization diagnostics, including data/KL gradient ratios.
- `summary.json`: exact runner arguments and setting metadata.

```json
{
  "started": "2026-05-14 01:00:59",
  "finished": "2026-05-14 01:11:17",
  "device": "cuda",
  "args": {
    "settings": "l2_q2_train1000,lh_q5_train1000",
    "variants": "anchor_uniform,anchor_floor,anchor_coverage,anchor_background,pointwise_coverage,pointwise_background",
    "seed_start": 0,
    "num_exp": 1,
    "anchor_count": 32,
    "boundary_anchor_frac": 0.5,
    "local_var_neighbors": 35,
    "train_neighbors": null,
    "n_final": null,
    "n_candidate": 70,
    "m_inducing": 32,
    "max_test": 60,
    "steps": 40,
    "c_refresh": 5,
    "mc_train": 1,
    "eval_mc": 2,
    "lr": 0.01,
    "beta_kl": 0.05,
    "kl_warmup_steps": 40,
    "composite_temperature": 0.2,
    "gamma_init": 0.7,
    "gamma_floor": 0.05,
    "gamma_temperature": 3.0,
    "gamma_damping": 0.5,
    "coverage_weight": 1.0,
    "coverage_min": 0.15,
    "background_scale": 3.0,
    "init_log_std": -3.0,
    "local_lengthscale": 1.0,
    "w_lengthscale": null,
    "init_amp": 1.0,
    "init_noise_var": 0.1,
    "train_kernel": false,
    "train_noise": false,
    "train_w_lengthscale": false,
    "train_w_var": false,
    "outlier_sigma_mult": 2.5,
    "gate_l2": 0.0001,
    "log_interval": 20,
    "resume": false,
    "verbose": false,
    "out_dir": "experiments\\synthetic\\global_wgp_cem_l2_lh_withdiag_20260514"
  },
  "settings": {
    "l2_q2_train1000": {
      "family": "L2_phantom",
      "N_train": 1000,
      "N_test": 200,
      "d": 2,
      "H": 20,
      "Q_true": 2,
      "Q": 2,
      "n": 25,
      "caseno": 6,
      "noise_std": 2.0,
      "noise_var": 4.0,
      "expansion": "rff",
      "lengthscale": 0.5,
      "kernel_var": 9.0
    },
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
    "anchor_uniform",
    "anchor_floor",
    "anchor_coverage",
    "anchor_background",
    "pointwise_coverage",
    "pointwise_background"
  ],
  "n_rows": 12,
  "n_ok": 12,
  "n_failed": 0
}
```
