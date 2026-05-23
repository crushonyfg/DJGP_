# Global Supervised z=W(x)x + JGP

Runner: ``experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py``.
Plan:   ``docs/global_z_supervised_jgp_plan_cn.md``.

This experiment implements the global-Z supervised projection variant: instead
of hybrid raw+W reranking inside a candidate pool, the whole train/test dataset
is mapped through ``z_i = W_theta(x_i) x_i`` and JumpGP runs in the global
``Z``-space.  ``W_theta`` is trained with a leave-anchor-out supervised anchor
predictive NLL via a differentiable local RBF GP surrogate.

```json
{
  "args": {
    "settings": "lh_q5_train1000",
    "seed": 0,
    "seeds": "",
    "variants": "raw_jgp,pca_jgp,pls_jgp,static_supervised,lowrank_supervised,lowrank_shuffled,lowrank_random",
    "max_test": 200,
    "n_neighbors": 0,
    "init_mode": "pls",
    "lowrank_r": 8,
    "hidden": 64,
    "anchor_count": 0,
    "anchor_strategy": "uniform",
    "anchor_batch": 128,
    "epochs": 100,
    "outer_refresh": 10,
    "inner_steps": 20,
    "lr_w": 0.001,
    "lr_gp": 0.01,
    "weight_decay": 0.0001,
    "w_l2": 0.0001,
    "init_ell": 1.0,
    "init_sf": 1.0,
    "init_sn": 0.1,
    "device": "auto",
    "out_dir": "experiments/synthetic/global_z_supervised_jgp_lh_seed0_20260516",
    "verbose": true,
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
    "raw_jgp",
    "pca_jgp",
    "pls_jgp",
    "static_supervised",
    "lowrank_supervised",
    "lowrank_shuffled",
    "lowrank_random"
  ]
}
```

Files:

- ``global_z_supervised_jgp_metrics.csv`` — per-variant metrics.
- ``global_z_supervised_jgp_summary.json`` — run configuration.
