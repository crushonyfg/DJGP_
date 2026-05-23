# Projected-Dataset Transductive Self-CEM Smoke

Runner: ``experiments/synthetic/compare_projected_dataset_selfcem_l2_lh.py``.
Plan:   ``docs/global_z_supervised_jgp_plan_cn.md``.

This directory is only the small smoke check for the projected-dataset
transductive self-CEM variants.  The full LH seed0 run is in
``experiments/synthetic/projected_dataset_transductive_selfcem_lh_seed0_20260517/``
and is logged under ``docs/experiments_log.md`` section
``2026-05-17 - projected_dataset_transductive_selfcem_lh_seed0_20260517``.

```json
{
  "args": {
    "settings": "lh_q5_train1000",
    "seed": 0,
    "seeds": "",
    "variants": "pd_pls_selfcem,pd_lowrank_transductive,pd_lowrank_transductive_shuffled,pd_lowrank_random",
    "max_test": 6,
    "n_neighbors": 8,
    "init_mode": "pls",
    "lowrank_r": 8,
    "hidden": 64,
    "anchor_count": 0,
    "anchor_strategy": "uniform",
    "anchor_batch": 4,
    "epochs": 1,
    "outer_refresh": 10,
    "inner_steps": 1,
    "lr_w": 0.001,
    "lr_gp": 0.01,
    "weight_decay": 0.0001,
    "w_l2": 0.0001,
    "init_ell": 1.0,
    "init_sf": 1.0,
    "init_sn": 0.1,
    "self_cem_updates": 1,
    "self_cem_hyper_steps": 1,
    "self_cem_hyper_lr": 0.04,
    "self_cem_gate_steps": 1,
    "self_cem_gate_lr": 0.05,
    "self_cem_gate_l2": 0.001,
    "self_cem_gate_max_norm": 8.0,
    "self_cem_gh_points": 20,
    "self_cem_noise_std_floor": 0.1,
    "self_cem_kcov": 0.1,
    "min_inliers": 2,
    "transductive_gate_weight": 0.1,
    "device": "auto",
    "out_dir": "experiments\\synthetic\\projected_dataset_transductive_selfcem_smoke_20260517",
    "verbose": false,
    "resume": false
  },
  "device": "cuda",
  "n_rows": 4,
  "settings": [
    "lh_q5_train1000"
  ],
  "seeds": [
    0
  ],
  "variants": [
    "pd_pls_selfcem",
    "pd_lowrank_transductive",
    "pd_lowrank_transductive_shuffled",
    "pd_lowrank_random"
  ]
}
```

Files:

- ``global_z_supervised_jgp_metrics.csv`` — per-variant metrics.
- ``global_z_supervised_jgp_summary.json`` — run configuration.
