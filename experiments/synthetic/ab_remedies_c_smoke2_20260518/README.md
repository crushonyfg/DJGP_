# A/B/C remedies — direct VI predict

Variants run: C2, C3, C4, C5

Key columns in ``eval_records.csv``:
- ``r_mean``: softmax responsibility on inlier branch (neighbour-averaged)
- ``pi_gate_test_mean``: E[sigmoid(omega^T [1,Wx])] at test anchors
- ``delta_T_mean``: mean(T_in - T_out)
- ``rmse_pi1``: RMSE if mixture uses pi=1 (GP-only path)

```json
{
  "args": {
    "setting": "lh_q5_train500",
    "seed": 0,
    "max_test": 30,
    "num_steps": 150,
    "eval_every": 75,
    "lr": 0.01,
    "m1": 2,
    "m2": 40,
    "variants": "C2,C3,C4,C5",
    "no_ab": false,
    "lambda_s": 0.1,
    "sigma_v_start": 0.2,
    "sigma_v_end": 2.0,
    "warmup_frac": 0.5,
    "omega0_init": 2.0,
    "omega0_floor": 0.0,
    "pi_gate_floor": 0.05,
    "lambda_pi_floor": 1.0,
    "force_inlier_steps": 150,
    "u_logit_ceil": -1.0,
    "out_dir": "experiments/synthetic/ab_remedies_c_smoke2_20260518"
  },
  "variants": [
    "C2",
    "C3",
    "C4",
    "C5"
  ],
  "n_records": 8
}
```
