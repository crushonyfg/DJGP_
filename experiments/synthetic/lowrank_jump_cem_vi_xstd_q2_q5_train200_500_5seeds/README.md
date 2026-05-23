# Synthetic Low-Rank Jump X-Standardized Baselines

Purpose: appendix-style comparison of generic baselines and LMJGP on a
representative low-rank jump synthetic benchmark.

Settings:

```json
{
  "args": {
    "num_exp": 5,
    "settings": "train200,train500,train200_q5,train500_q5",
    "N_test": 200,
    "max_anchors": null,
    "jump_amp": 1.5,
    "noise_std": 0.1,
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
    "methods": "cem_em_transductive_vi_fullcov_ensemble",
    "out_dir": "experiments\\synthetic\\lowrank_jump_cem_vi_xstd_q2_q5_train200_500_5seeds",
    "resume": true
  },
  "device": "cuda",
  "settings": {
    "train200": {
      "N_train": 200,
      "D": 50,
      "Q_true": 2,
      "Q": 2,
      "n": 25
    },
    "train500": {
      "N_train": 500,
      "D": 50,
      "Q_true": 2,
      "Q": 2,
      "n": 25
    },
    "train200_q5": {
      "N_train": 200,
      "D": 50,
      "Q_true": 5,
      "Q": 5,
      "n": 25
    },
    "train500_q5": {
      "N_train": 500,
      "D": 50,
      "Q_true": 5,
      "Q": 5,
      "n": 25
    }
  },
  "n_rows": 20,
  "n_ok": 20,
  "n_errors": 0
}
```

Files:

- `metrics.csv`: per-setting, per-seed, per-method metrics.
- `aggregate.csv`: means/stds over seeds.
- `summary.json`: run metadata.

All methods use train-only X standardization.  The generator is imported from
`experiments/synthetic/compare_lmjgp_tricks_synth.py`.
