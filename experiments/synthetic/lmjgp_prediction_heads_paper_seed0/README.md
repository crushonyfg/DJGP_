# LMJGP Prediction Head Comparison on Paper Synthetic Data

This compares the same trained transductive LMJGP state with two prediction
heads:

- `sampleW_jumpgp_refit`: `predict_vi`, samples `q(W_*)` and refits JumpGP-CEM.
- `direct_variational_analytic`: `predict_vi_analytic`, uses learned local
  `u_params` directly without JumpGP-CEM refit.

Settings:

```json
{
  "args": {
    "settings": "l2_q2_train500,lh_q5_train500",
    "seed": 0,
    "m1": 2,
    "m2": 40,
    "lmjgp_steps": 300,
    "MC_num": 3,
    "lr": 0.01,
    "out_dir": "experiments\\synthetic\\lmjgp_prediction_heads_paper_seed0"
  },
  "device": "cuda",
  "n_rows": 4
}
```
