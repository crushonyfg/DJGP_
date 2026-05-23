# LMJGP Paper Synthetic with X/Y Standardization

This run trains `lmjgp_baseline` and `lmjgp_pls_kl` on standardized X and
standardized Y for paper-style L2/LH train=1000 synthetic data.  Metrics are
reported on the original Y scale.

See `docs/experiments_log.md` for the headline table.

```json
{
  "args": {
    "settings": "l2_q2_train1000,lh_q5_train1000",
    "num_exp": 5,
    "methods": "lmjgp_baseline,lmjgp_pls_kl",
    "lmjgp_steps": 300,
    "MC_num": 3,
    "out_dir": "experiments\\synthetic\\paper_l2_lh_stdxy_lmjgp_train1000_5seeds",
    "resume": false
  },
  "device": "cuda",
  "n_rows": 20
}
```
