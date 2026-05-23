# Synthetic Low-Rank Jump X-Standardized Baselines

Purpose: appendix-style comparison of generic baselines and LMJGP on a
representative low-rank jump synthetic benchmark.

Settings:

```json
{
  "args": {
    "num_exp": 5,
    "settings": "moderate",
    "N_test": 200,
    "max_anchors": null,
    "jump_amp": 1.5,
    "noise_std": 0.1,
    "xgb_boots": 30,
    "dkl_epochs": 80,
    "lmjgp_steps": 300,
    "MC_num": 3,
    "methods": "xgb_bootstrap,dkl_svgp,jgp_pca,lmjgp_baseline,lmjgp_pls_kl",
    "out_dir": "experiments\\synthetic\\lowrank_jump_xstd_baselines_moderate_5seeds"
  },
  "device": "cuda",
  "settings": {
    "moderate": {
      "N_train": 1000,
      "D": 50,
      "Q_true": 2,
      "Q": 2,
      "n": 25
    }
  },
  "n_rows": 25,
  "n_ok": 25,
  "n_errors": 0
}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-method metrics.
- `aggregate.csv`: means/stds over seeds.
- `summary.json`: run metadata.

All methods use train-only X standardization.  The generator is imported from
`experiments/synthetic/compare_lmjgp_tricks_synth.py`.
