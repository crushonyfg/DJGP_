# DeepMahalanobisGP Global Z JumpGP

This folder evaluates global pointwise mapped features from DeepMahalanobisGP:
`z_i = W(x_i)x_i`, followed by ordinary JumpGP in the mapped `Z` space.

It is not the local per-anchor `W(x_*)` projection diagnostic.

```json
{
  "args": {
    "split_dir": "experiments\\synthetic\\deep_mahalanobis_gp_paper_train1000_5seeds",
    "w_dir": "experiments\\synthetic\\deep_mahalanobis_gp_global_z_jgp_train1000_5seeds",
    "out_dir": "experiments\\synthetic\\deep_mahalanobis_gp_global_z_jgp_train1000_5seeds",
    "settings": "l2_q2_train1000,lh_q5_train1000",
    "seed": 0,
    "seeds": "0,1,2,3,4",
    "n": 0,
    "row_normalize_w": false,
    "resume": true
  },
  "device": "cuda",
  "n_rows": 10
}
```

Files:

- `global_z_jgp_metrics.csv`: per-seed metrics.
- `global_z_jgp_aggregate.csv`: mean/std over successful seeds.
- `*_W_train_mean.npy`, `*_W_test_mean.npy`: exported by
  `run_deep_mahalanobis_gp_paper.py` during the matching DMGP retrain.
