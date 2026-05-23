# Global Supervised z=W(x)x + JGP

Runner: ``experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py``.
Plan:   ``docs/global_z_supervised_jgp_plan_cn.md``.

This experiment implements the global-Z supervised projection variant: instead
of hybrid raw+W reranking inside a candidate pool, the whole train/test dataset
is mapped through ``z_i = W_theta(x_i) x_i`` and JumpGP runs in the global
``Z``-space.  ``W_theta`` is trained with a leave-anchor-out supervised anchor
predictive NLL.  This run uses the self-CEM alternating version: after each
global-Z KNN refresh, it refits self-CEM labels/hyperparameters in ``Z`` space
and uses those fixed neighborhoods/partitions for the next supervised W M-step.

Project log entry:
``docs/experiments_log.md`` section
``2026-05-17 - global_z_selfcem_supervised_lh_seed0_20260517``.

Headline: this exact self-CEM global-Z alternating version did not pass the
LH seed0 expansion gate.  PLS global-Z self-CEM was best
(``RMSE=700.34``, ``CRPS=329.04``), while learned
``lowrank_selfcem_supervised`` was worse (``801.86``, ``411.39``) and only
beat the shuffled control.

```json
{
  "args": {
    "settings": "lh_q5_train1000",
    "seed": 0,
    "seeds": "",
    "variants": "raw_jgp,raw_selfcem,pca_selfcem,pls_selfcem,static_selfcem_supervised,lowrank_selfcem_supervised,lowrank_selfcem_shuffled,lowrank_selfcem_random",
    "max_test": 200,
    "n_neighbors": 0,
    "init_mode": "pls",
    "lowrank_r": 8,
    "hidden": 64,
    "anchor_count": 100,
    "anchor_strategy": "kmeans_yvar",
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
    "device": "auto",
    "out_dir": "experiments\\synthetic\\global_z_selfcem_supervised_lh_seed0_20260517",
    "verbose": false,
    "resume": false
  },
  "device": "cuda",
  "n_rows": 8,
  "settings": [
    "lh_q5_train1000"
  ],
  "seeds": [
    0
  ],
  "variants": [
    "raw_jgp",
    "raw_selfcem",
    "pca_selfcem",
    "pls_selfcem",
    "static_selfcem_supervised",
    "lowrank_selfcem_supervised",
    "lowrank_selfcem_shuffled",
    "lowrank_selfcem_random"
  ]
}
```

Files:

- ``global_z_supervised_jgp_metrics.csv`` — per-variant metrics.
- ``global_z_supervised_jgp_summary.json`` — run configuration.
