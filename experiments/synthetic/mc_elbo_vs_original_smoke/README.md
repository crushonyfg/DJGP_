# MC-ELBO vs Original ELBO Comparison

Setting: lh_q5_train200, seed=0, max_test=200

Variants:
1. `orig_elbo_vi_predict` — Original ELBO training → direct VI predict
2. `orig_elbo_sampleW_cem` — Original ELBO training → sample W + JGP CEM
3. `mc_elbo_vi_predict` — MC-ELBO training → per-sample-optimized direct predict
4. `mc_elbo_sampleW_cem` — MC-ELBO training → sample W + JGP CEM

```json
{
  "args": {
    "setting": "lh_q5_train200",
    "seed": 0,
    "max_test": 200,
    "m1": 2,
    "m2": 40,
    "lr": 0.01,
    "orig_steps": 150,
    "mc_steps": 80,
    "mc_S": 2,
    "inner_steps": 8,
    "inner_lr": 0.05,
    "MC_num": 3,
    "out_dir": "experiments/synthetic/mc_elbo_vs_original_smoke"
  },
  "n_rows": 4
}
```
