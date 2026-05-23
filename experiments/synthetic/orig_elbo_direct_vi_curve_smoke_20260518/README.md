# Original ELBO → Direct VI predict — training curve

- Setting: `lh_q5_train500`, seed=0
- Train steps: 900, eval every 150
- Predictor: `_predict_direct_variational_consistent` (ELBO-consistent direct VI)

## Files

- `curve_metrics.csv`: RMSE / CRPS / ELBO at each eval step
- `rmse_crps_vs_step.png`: RMSE and CRPS vs training step
- `summary.json`: run metadata

```json
{
  "args": {
    "setting": "lh_q5_train500",
    "seed": 0,
    "max_test": 40,
    "m1": 2,
    "m2": 40,
    "lr": 0.01,
    "num_steps": 900,
    "eval_every": 150,
    "log_interval": 50,
    "out_dir": "experiments/synthetic/orig_elbo_direct_vi_curve_smoke_20260518"
  },
  "device": "cuda",
  "train_wall_sec": 301.0464393000002,
  "eval_steps": [
    150,
    300,
    450,
    600,
    750,
    900
  ],
  "final_rmse": 810.5253644207476,
  "final_crps": 805.2874755859375
}
```
