# Graph-Local Model Screen

Pilot model screen for `docs/graph_local_djgp_experiment_plan.md`.

LMJGP uses sampled-W `predict_vi` with JumpGP-CEM refits. Self-CEM keeps the
retrieved feature/graph neighborhoods fixed and does not reselect neighbors by W.

| method | neighborhood | feature set | RMSE | CRPS | Cov90 | Width90 |
|---|---|---|---:|---:|---:|---:|
| `selfcem_feature_graphfeat` | feature_knn_graphfeat_fixed | raw_x+block_onehot+graph_agg_x | 0.8129 | 0.4671 | 0.662 | 1.5679 |
| `lmjgp_feature_graphfeat` | feature_knn_graphfeat | raw_x+block_onehot+graph_agg_x | 0.8380 | 0.4613 | 0.850 | 2.2197 |
| `selfcem_graph_graphfeat` | graph_hop_then_time_then_graphfeat_fixed | raw_x+block_onehot+graph_agg_x | 0.8524 | 0.4801 | 0.750 | 1.8536 |
| `lmjgp_graph_graphfeat` | graph_hop_then_time_then_graphfeat | raw_x+block_onehot+graph_agg_x | 0.9545 | 0.5260 | 0.825 | 2.5942 |

Runner summary:

```json
{
  "args": {
    "preset": "dataset_b",
    "seed": 0,
    "methods": "lmjgp_feature_graphfeat,lmjgp_graph_graphfeat,selfcem_feature_graphfeat,selfcem_graph_graphfeat",
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
    "out_dir": "experiments\\synthetic\\graph_local_model_screen_dataset_b_seed0_graphfeat_local_20260518"
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
  "n_rows": 12,
  "n_ok_all": 4,
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
