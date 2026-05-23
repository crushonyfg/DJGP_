# Paper Synthetic L2/LH X-Standardized Baselines

Purpose: corrected baseline comparison on the paper-style synthetic generators,
not the direct low-rank Gaussian-X toy generator.

Settings:

```json
{
  "args": {
    "num_exp": 5,
    "settings": "l2_q2_train1000,lh_q5_train1000",
    "methods": "xgb_bootstrap,dkl_svgp,jgp_pca,lmjgp_baseline,lmjgp_pls_kl,cem_em_transductive_vi_fullcov_ensemble",
    "xgb_boots": 30,
    "dkl_epochs": 80,
    "lmjgp_steps": 300,
    "MC_num": 3,
    "cem_outer": 5,
    "cem_m_steps": 80,
    "cem_vi_steps": 80,
    "cem_vi_lr": 0.005,
    "cem_vi_mc": 2,
    "cem_vi_kl_weight": 1.0,
    "cem_vi_init_std": 0.03,
    "out_dir": "experiments\\synthetic\\paper_l2_lh_xstd_train1000_5seeds",
    "resume": true
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
  "n_rows": 60,
  "n_ok": 60,
  "n_errors": 0
}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-method metrics.
- `aggregate.csv`: means/stds over seeds.
- `summary.json`: exact runner arguments and setting metadata.

All methods use train-only X standardization after the RFF expansion.
