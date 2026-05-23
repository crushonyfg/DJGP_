# Synthetic Low-Rank Jump X-Standardized Baselines

Purpose: appendix-style comparison of generic baselines and LMJGP on a
representative low-rank jump synthetic benchmark.

Settings:

```json
{
  "args": {
    "num_exp": 1,
    "settings": "smoke",
    "N_test": 30,
    "max_anchors": 30,
    "jump_amp": 1.5,
    "noise_std": 0.1,
    "xgb_boots": 2,
    "dkl_epochs": 2,
    "lmjgp_steps": 5,
    "MC_num": 3,
    "methods": "xgb_bootstrap,dkl_svgp,jgp_pca,lmjgp_baseline",
    "out_dir": "experiments\\synthetic\\lowrank_jump_xstd_baselines_smoke"
  },
  "device": "cuda",
  "settings": {
    "smoke": {
      "N_train": 120,
      "D": 12,
      "Q_true": 2,
      "Q": 2,
      "n": 12
    }
  },
  "n_rows": 4,
  "n_ok": 4,
  "n_errors": 0
}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-method metrics.
- `aggregate.csv`: means/stds over seeds.
- `summary.json`: run metadata.

All methods use train-only X standardization.  The generator is imported from
`experiments/synthetic/compare_lmjgp_tricks_synth.py`.
