# Graph-Local Model Screen

Pilot model screen for `docs/graph_local_djgp_experiment_plan.md`.

LMJGP uses sampled-W `predict_vi` with JumpGP-CEM refits. Self-CEM keeps the
retrieved feature/graph neighborhoods fixed and does not reselect neighbors by W.

| method | neighborhood | feature set | RMSE | CRPS | Cov90 | Width90 |
|---|---|---|---:|---:|---:|---:|
| `selfcem_feature` | feature_knn_fixed | raw_x | 1.1516 | 0.6917 | 0.500 | 1.9172 |
| `lmjgp_feature` | feature_knn | raw_x | 1.2025 | 0.7121 | 0.600 | 2.1368 |

Runner summary:

```json
{
  "args": {
    "preset": "dataset_b",
    "seed": 0,
    "methods": "xgb_raw,lmjgp_feature,selfcem_feature",
    "max_train": 200,
    "max_test": 10,
    "train_frac": 0.7,
    "n_nodes": 300,
    "n_time": 20,
    "n_blocks": 8,
    "noise_x": 0.5,
    "noise_y": 0.1,
    "n_neighbors": 35,
    "Q": 3,
    "m1": 2,
    "m2": 40,
    "lmjgp_steps": 10,
    "MC_num": 1,
    "lmjgp_lr": 0.01,
    "xgb_boots": 10,
    "dkl_epochs": 40,
    "dkl_inducing": 64,
    "self_cem_updates": 2,
    "self_cem_hyper_steps": 1,
    "self_cem_gate_steps": 1,
    "self_cem_kcov": 0.0,
    "out_dir": "experiments\\synthetic\\graph_local_model_screen_smoke_20260518"
  },
  "device": "cuda",
  "preset": {
    "description": "Graph-local hard-jump latent synthetic, D=30, q=3.",
    "D": 30,
    "q": 3,
    "jump": true,
    "misleading_features": false,
    "latent_type": "linear"
  },
  "n_rows": 7,
  "n_ok_all": 2,
  "n_errors": 1,
  "artifacts": [
    "metrics.csv",
    "metrics_partial.csv",
    "predictions.npz",
    "summary.json",
    "README.md"
  ]
}
```
