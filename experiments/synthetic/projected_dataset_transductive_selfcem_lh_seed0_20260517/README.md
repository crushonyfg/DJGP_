# Projected-Dataset Transductive Self-CEM

Runner: ``experiments/synthetic/compare_projected_dataset_selfcem_l2_lh.py``.
Plan:   ``docs/global_z_supervised_jgp_plan_cn.md``.

This run uses the clearer projected-dataset naming for the whole-dataset
``z_i = W_theta(x_i) x_i`` line.  It is transductive in ``X_test`` only: after
each M-step it maps train/test into projected space, selects test-anchor
projected KNN, refits uncertain-W/self-CEM on those neighborhoods, then updates
``W_theta`` from the fixed neighborhoods and CEM partitions.  It does not use
``y_test`` in the transductive objective.

Project log entry:
``docs/experiments_log.md`` section
``2026-05-17 - projected_dataset_transductive_selfcem_lh_seed0_20260517``.

Headline: `pd_static_transductive` improves over PLS projected self-CEM
(``RMSE=687.38``, ``CRPS=315.21`` vs ``700.34``, ``329.04``), but it does not
beat the previous best LH seed0 self-CEM/qR rows around ``RMSE=642--664`` and
``CRPS=275--292``.  No 5-seed expansion is recommended from this result.

```json
{
  "args": {
    "settings": "lh_q5_train1000",
    "seed": 0,
    "seeds": "",
    "variants": "pd_raw_selfcem,pd_pca_selfcem,pd_pls_selfcem,pd_static_transductive,pd_lowrank_transductive,pd_lowrank_transductive_shuffled,pd_lowrank_random",
    "max_test": 200,
    "n_neighbors": 0,
    "init_mode": "pls",
    "lowrank_r": 8,
    "hidden": 64,
    "anchor_count": 0,
    "anchor_strategy": "uniform",
    "anchor_batch": 64,
    "epochs": 4,
    "outer_refresh": 10,
    "inner_steps": 20,
    "lr_w": 0.001,
    "lr_gp": 0.01,
    "weight_decay": 0.0001,
    "w_l2": 0.0001,
    "init_ell": 1.0,
    "init_sf": 1.0,
    "init_sn": 0.1,
    "self_cem_updates": 2,
    "self_cem_hyper_steps": 10,
    "self_cem_hyper_lr": 0.04,
    "self_cem_gate_steps": 20,
    "self_cem_gate_lr": 0.05,
    "self_cem_gate_l2": 0.001,
    "self_cem_gate_max_norm": 8.0,
    "self_cem_gh_points": 20,
    "self_cem_noise_std_floor": 0.1,
    "self_cem_kcov": 0.1,
    "min_inliers": 2,
    "transductive_gate_weight": 0.1,
    "device": "auto",
    "out_dir": "experiments\\synthetic\\projected_dataset_transductive_selfcem_lh_seed0_20260517",
    "verbose": false,
    "resume": false
  },
  "device": "cuda",
  "n_rows": 7,
  "settings": [
    "lh_q5_train1000"
  ],
  "seeds": [
    0
  ],
  "variants": [
    "pd_raw_selfcem",
    "pd_pca_selfcem",
    "pd_pls_selfcem",
    "pd_static_transductive",
    "pd_lowrank_transductive",
    "pd_lowrank_transductive_shuffled",
    "pd_lowrank_random"
  ]
}
```

Files:

- ``global_z_supervised_jgp_metrics.csv`` — per-variant metrics.
- ``global_z_supervised_jgp_summary.json`` — run configuration.
