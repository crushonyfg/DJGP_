# CEM MAP W0 + Low-Rank Residual LMJGP on Paper Synthetic Data

This experiment uses transductive CEM-EM MAP projections to build a fixed
global projection mean for low-rank residual LMJGP.

```json
{
  "args": {
    "settings": "l2_q2_train1000,lh_q5_train1000",
    "seed": 0,
    "m1": 2,
    "m2": 40,
    "n_pls": 5,
    "n_pca": 5,
    "n_random": 0,
    "cem_outer": 5,
    "cem_m_steps": 80,
    "lmjgp_steps": 300,
    "MC_num": 3,
    "lr": 0.01,
    "lambda_gate": 0.1,
    "lambda_smooth": 0.01,
    "init_C_scale": 0.0,
    "coeff_scale": 1.0,
    "residual_scale": 1.0,
    "out_dir": "experiments\\synthetic\\lmjgp_lowrank_cem_w0_paper_train1000_seed0"
  },
  "device": "cuda",
  "n_rows": 6
}
```
