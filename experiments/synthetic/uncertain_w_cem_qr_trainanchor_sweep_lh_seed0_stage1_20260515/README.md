# Uncertain-W CEM L2/LH Prediction-Head Ablation

This experiment evaluates the uncertain-W CEM prediction head on paper-style
L2/LH synthetic data.  Base variants use a fixed mean projection from PLS/PCA
and controlled W uncertainty; `qr_mstep_*` and `qr_alt_*` variants train a
sparse-GP `q(R)` M-step through the moment-matched uncertain-W kernel.

Project-wide log entry: `docs/experiments_log.md`, section
`2026-05-15 - uncertain_w_cem_qr_trainanchor_sweeps_20260515`.

Files:

- `metrics.csv`: one row per setting/seed/variant.
- `aggregate.csv`: mean/std over successful rows.
- `summary.json`: exact runner arguments and setting metadata.

```json
{
  "started": "2026-05-15 00:50:57",
  "finished": "2026-05-15 06:19:06",
  "device": "cuda",
  "args": {
    "settings": "lh_q5_train1000",
    "variant": "qr_alt_lr2_refresh10_self_cem2_nstd0.1_kcov0.1",
    "seed_start": 0,
    "num_exp": 1,
    "max_test": 200,
    "n_neighbors_grid": "50,100,200",
    "m_inducing_grid": "100",
    "qr_train_anchor_count_grid": "50,100,200",
    "qr_train_n_neighbors_grid": "",
    "qr_train_anchor_strategy": "kmeans_yvar",
    "projection_init": "pls",
    "lowrank_basis": "pca_residual",
    "lowrank_qr_w_signal_var": 0.1,
    "qr_steps": 40,
    "qr_outer_cycles": 4,
    "qr_beta_kl": 0.05,
    "qr_lr": 0.01,
    "refresh_hyper_steps": 10,
    "refresh_gate_steps": 20,
    "hyper_steps": 30,
    "gate_steps": 60,
    "include_original": true,
    "resume": false,
    "out_dir": "experiments\\synthetic\\uncertain_w_cem_qr_trainanchor_sweep_lh_seed0_stage1_20260515"
  },
  "settings": {
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
  "n_rows": 28,
  "n_ok": 28,
  "n_failed": 0
}
```
