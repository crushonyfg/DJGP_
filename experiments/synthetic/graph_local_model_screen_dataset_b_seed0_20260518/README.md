# Graph-Local Model Screen

Pilot model screen for `docs/graph_local_djgp_experiment_plan.md`.

LMJGP uses sampled-W `predict_vi` with JumpGP-CEM refits. Self-CEM keeps the
retrieved feature/graph neighborhoods fixed and does not reselect neighbors by W.

| method | neighborhood | feature set | RMSE | CRPS | Cov90 | Width90 |
|---|---|---|---:|---:|---:|---:|
| `dkl_graph` | none | raw_x+block_onehot+graph_agg_x | 0.7213 | 0.4851 | 0.300 | 0.4873 |
| `xgb_graph` | none | raw_x+block_onehot+graph_agg_x | 0.7652 | 0.4552 | 0.613 | 1.1934 |
| `lmjgp_graph` | graph_hop_then_time_then_feature | raw_x | 0.9150 | 0.5180 | 0.863 | 2.6467 |
| `lmjgp_feature` | feature_knn | raw_x | 0.9631 | 0.5312 | 0.825 | 2.4025 |
| `xgb_raw` | none | raw_x | 0.9658 | 0.5645 | 0.625 | 1.5354 |
| `dkl_raw` | none | raw_x | 1.0041 | 0.7215 | 0.125 | 0.4652 |
| `selfcem_graph` | graph_hop_then_time_then_feature_fixed | raw_x | 1.0250 | 0.5856 | 0.688 | 2.1294 |
| `selfcem_feature` | feature_knn_fixed | raw_x | 1.0762 | 0.6242 | 0.700 | 2.0382 |

Runner summary:

```json
{
  "args": {
    "preset": "dataset_b",
    "seed": 0,
    "methods": "xgb_raw,xgb_graph,dkl_raw,dkl_graph,lmjgp_feature,lmjgp_graph,selfcem_feature,selfcem_graph",
    "max_train": 1000,
    "max_test": 80,
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
    "lmjgp_steps": 120,
    "MC_num": 3,
    "lmjgp_lr": 0.01,
    "xgb_boots": 10,
    "dkl_epochs": 40,
    "dkl_inducing": 64,
    "self_cem_updates": 2,
    "self_cem_hyper_steps": 10,
    "self_cem_gate_steps": 20,
    "self_cem_kcov": 0.0,
    "out_dir": "experiments\\synthetic\\graph_local_model_screen_dataset_b_seed0_20260518"
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
  "n_rows": 24,
  "n_ok_all": 8,
  "n_errors": 0,
  "artifacts": [
    "metrics.csv",
    "metrics_partial.csv",
    "predictions.npz",
    "summary.json",
    "README.md"
  ]
}
```
