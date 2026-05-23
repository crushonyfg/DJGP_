# UCI Uncertain-W CEM train=500

This folder was produced by `experiments/uci/compare_uncertain_w_cem_train500.py`.
Rows evaluate uncertain-W CEM/self-CEM and optional q(R) variants on UCI
train/test splits with train-size subsampling and a fixed test-anchor slice.

Project-wide log entry: `docs/experiments_log.md`, section
`2026-05-15 - uncertain_w_cem_train500_selfcem_5seeds_20260515`.

Files:

- `metrics.csv`: one row per dataset/train_size/seed/method.
- `aggregate.csv`: mean/std over successful rows.
- `summary.json`: exact runner arguments.

```json
{
  "started": "2026-05-15 14:26:38",
  "finished": "2026-05-15 14:34:18",
  "device": "cuda",
  "args": {
    "datasets": "Parkinsons Telemonitoring",
    "train_sizes": "500",
    "num_exp": 1,
    "max_test_anchors": 200,
    "variants": "qr_alt_lr2_refresh10_self_cem2_nstd0.15_kcov0.1",
    "include_original": true,
    "grouped_parkinsons": true,
    "train_subject_frac": 0.8,
    "Q": null,
    "projection_init": "pls",
    "n_neighbors": -1,
    "m_inducing": 40,
    "qr_train_anchor_count": 0,
    "qr_train_anchor_strategy": "kmeans_yvar",
    "qr_train_n_neighbors": null,
    "w_var": 0.02,
    "w_cov_lengthscale": null,
    "qr_w_lengthscale": null,
    "lowrank_basis": "pca_residual",
    "lowrank_qr_w_signal_var": 0.1,
    "qr_steps": 40,
    "qr_lr": 0.01,
    "qr_beta_kl": 0.05,
    "qr_kl_warmup_steps": 20,
    "qr_outer_cycles": 4,
    "hyper_steps": 30,
    "gate_steps": 60,
    "refresh_hyper_steps": 10,
    "refresh_gate_steps": 20,
    "resume": false,
    "out_dir": "experiments\\uci\\uncertain_w_cem_train500_parkinsons_qr_seed0_20260515"
  },
  "n_rows": 2,
  "n_ok": 2,
  "n_failed": 0
}
```
