# Synthetic Low-Rank Jump X-Standardized Baselines

Purpose: appendix-style comparison of generic baselines and LMJGP on a
representative low-rank jump synthetic benchmark.

Settings:

```json
{
  "args": {
    "num_exp": 1,
    "settings": "smoke",
    "N_test": 12,
    "max_anchors": 12,
    "jump_amp": 1.5,
    "noise_std": 0.1,
    "xgb_boots": 30,
    "dkl_epochs": 80,
    "lmjgp_steps": 300,
    "MC_num": 2,
    "cem_outer": 1,
    "cem_m_steps": 3,
    "cem_vi_steps": 2,
    "cem_vi_lr": 0.005,
    "cem_vi_mc": 2,
    "cem_vi_kl_weight": 1.0,
    "cem_vi_init_std": 0.03,
    "methods": "cem_em_transductive_vi_fullcov_ensemble",
    "out_dir": "experiments\\synthetic\\cem_vi_synthetic_smoke",
    "resume": false
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
  "n_rows": 1,
  "n_ok": 1,
  "n_errors": 0
}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-method metrics.
- `aggregate.csv`: means/stds over seeds.
- `summary.json`: run metadata.

All methods use train-only X standardization.  The generator is imported from
`experiments/synthetic/compare_lmjgp_tricks_synth.py`.
