# Uncertain-W CEM L2/LH Prediction-Head Ablation

This experiment evaluates the uncertain-W CEM prediction head on paper-style
L2/LH synthetic data.  It does not train q(R); the mean projection is fixed
from PLS/PCA and W uncertainty is injected through a controlled covariance.

Files:

- `metrics.csv`: one row per setting/seed/variant.
- `aggregate.csv`: mean/std over successful rows.
- `summary.json`: exact runner arguments and setting metadata.

```json
{
  "started": "2026-05-14 12:02:45",
  "finished": "2026-05-14 12:02:48",
  "device": "cuda",
  "args": {
    "settings": "l2_q2_train1000",
    "variants": "pointwise_self_cem1_kcov0.05",
    "seed_start": 0,
    "num_exp": 1,
    "projection_init": "pls",
    "max_test": 5,
    "n_neighbors": null,
    "w_var": 0.02,
    "w_cov_lengthscale": null,
    "hyper_steps": 1,
    "hyper_lr": 0.04,
    "gate_steps": 1,
    "gate_lr": 0.05,
    "gate_l2": 0.001,
    "gate_max_norm": 8.0,
    "gh_points": 20,
    "min_inliers": 2,
    "max_inlier_frac": 0.95,
    "outlier_sigma_mult": 2.5,
    "background_scale_mult": 3.0,
    "min_noise_var": 1e-05,
    "max_noise_var": 10.0,
    "min_signal_var": 0.0001,
    "max_signal_var": 100.0,
    "min_lengthscale": 0.001,
    "max_lengthscale": 20.0,
    "response_seed_frac": 0.25,
    "response_inlier_frac": 0.7,
    "include_original": true,
    "resume": false,
    "out_dir": "experiments\\synthetic\\uncertain_w_cem_pointwise_smoke"
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
    }
  },
  "variants": [
    "pointwise_self_cem1_kcov0.05"
  ],
  "n_rows": 2,
  "n_ok": 2,
  "n_failed": 0
}
```
