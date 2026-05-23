# High-Noise Orthogonal Synthetic Expansions

Purpose: seed-level stress-test of noisy observed expansions for paper L2/LH
latent synthetic generators.

Settings and args:

```json
{
  "args": {
    "num_exp": 1,
    "seed_start": 0,
    "settings": "l2_linear_noise3_train1000",
    "methods": "xgb_bootstrap",
    "xgb_boots": 2,
    "dkl_epochs": 80,
    "lmjgp_steps": 300,
    "MC_num": 3,
    "standardize_y": true,
    "only_export_dmgp": false,
    "resume": false,
    "out_dir": "experiments\\synthetic\\high_noise_expansions_smoke"
  },
  "device": "cuda",
  "settings": {
    "l2_linear_noise3_train1000": {
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
      "expansion": "linear_noise",
      "lengthscale": 0.5,
      "kernel_var": 9.0,
      "base_setting": "l2_q2_train1000",
      "eta_sigma": 3.0,
      "signal_scale": 1.0,
      "observed_dim": 20
    }
  },
  "exports": [
    {
      "setting": "l2_linear_noise3_train1000",
      "seed": 0,
      "path": "experiments\\synthetic\\high_noise_expansions_smoke\\dmgp_data\\l2_linear_noise3_train1000_seed0.npz",
      "standardize_x": true,
      "standardize_y": true
    }
  ],
  "n_rows": 1,
  "n_ok": 1,
  "n_errors": 0
}
```

Files:

- `metrics.csv`: non-DMGP per-method rows.
- `aggregate.csv`: aggregate over completed seeds.
- `dmgp_data/*.npz`: standardized X/Y exports for DeepMahalanobisGP.
- `dmgp/`: optional output from `run_deep_mahalanobis_gp_paper.py`.

Metrics are in standardized-y units when `standardize_y=true`.
