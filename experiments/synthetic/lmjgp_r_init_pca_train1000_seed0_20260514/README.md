# LMJGP q(R)/q(V) Initialisation Ablation

Purpose: compare PCA-style initialisation of the LMJGP projection inducing
mean against the existing PLS-KL setup on the paper-style L2/LH train=1000
synthetic protocol.

This folder is additive and does not overwrite the existing paper baseline
results.

Settings:

```json
{
  "args": {
    "num_exp": 1,
    "settings": "l2_q2_train1000,lh_q5_train1000",
    "variants": "pca_unit_kl,pca_dmgp_raw_kl",
    "lmjgp_steps": 300,
    "MC_num": 3,
    "m1": 2,
    "m2": 40,
    "lr": 0.01,
    "warmup_frac": 0.5,
    "aux_max_pairs": 64,
    "log_interval": 100,
    "snapshot_steps": "0,50",
    "out_dir": "experiments\\synthetic\\lmjgp_r_init_pca_train1000_seed0_20260514",
    "resume": false
  },
  "device": "cuda",
  "settings": {
    "l2_q2_train1000": {
      "family": "L2_phantom",
      "N_train": 1000,
      "N_test": 200,
      "d": 2,
      "H": 20,
      "Q_true": 2,
      "Q": 2,
      "n": 25,
      "caseno": 6,
      "noise_std": 2.0,
      "noise_var": 4.0,
      "expansion": "rff",
      "lengthscale": 0.5,
      "kernel_var": 9.0
    },
    "lh_q5_train1000": {
      "family": "LH",
      "N_train": 1000,
      "N_test": 200,
      "d": 5,
      "H": 30,
      "Q_true": 5,
      "Q": 5,
      "n": 35,
      "caseno": 0,
      "noise_std": 2.0,
      "noise_var": 4.0,
      "expansion": "rff",
      "lengthscale": 0.5,
      "kernel_var": 9.0
    }
  },
  "variants": [
    "pca_unit_kl",
    "pca_dmgp_raw_kl"
  ],
  "n_rows": 4,
  "n_ok": 4,
  "n_errors": 0
}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-variant metrics and diagnostics.
- `aggregate.csv`: means/stds over completed seeds.
- `summary.json`: exact runner arguments and setting metadata.
- `metrics_partial.csv`: resumable partial output while the runner is active.
