# Uncertain-W CEM L2/LH Prediction-Head Ablation

This experiment evaluates the uncertain-W CEM prediction head on paper-style
L2/LH synthetic data.  Base variants use a fixed mean projection from PLS/PCA
and controlled W uncertainty; `qr_mstep_*` and `qr_alt_*` variants train a
sparse-GP `q(R)` M-step through the moment-matched uncertain-W kernel.

Project log entry:
`docs/experiments_log.md` section
`2026-05-17 - uncertain_w_cem_qr_rankhybrid_lh_seed0_20260517`.

Headline: rank-normalized hybrid anchor retrieval did not provide enough
seed0 signal to expand to 5 seeds.  The best rank-hybrid row was
`lambda=0.40` with `RMSE=665.23`, `CRPS=287.34`, `Cov90=0.785`, and raw
overlap `0.869`.  It modestly improved CRPS over the raw qR/no-qR rows but
did not improve RMSE, and it was much weaker than the previous distance
hybrid seed0 result.

Files:

- `metrics.csv`: one row per setting/seed/variant.
- `aggregate.csv`: mean/std over successful rows.
- `summary.json`: exact runner arguments and setting metadata.

```json
{
  "started": "2026-05-16 22:21:19",
  "finished": "2026-05-16 22:44:29",
  "device": "cuda",
  "args": {
    "settings": "lh_q5_train1000",
    "variants": "qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1,self_cem2_nstd0.1_kcov0.1,qr_alt_hyra0.25_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1,qr_alt_hyra0.4_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1,qr_alt_hyra0.6_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1,qr_alt_hyra0.75_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1",
    "seed_start": 0,
    "num_exp": 1,
    "projection_init": "pls",
    "max_test": 200,
    "n_neighbors": 35,
    "m_inducing": 40,
    "qr_train_anchor_count": 100,
    "qr_train_anchor_strategy": "kmeans_yvar",
    "qr_train_n_neighbors": 35,
    "w_var": 0.02,
    "w_cov_lengthscale": null,
    "hyper_steps": 30,
    "hyper_lr": 0.04,
    "gate_steps": 60,
    "gate_lr": 0.05,
    "gate_l2": 0.001,
    "gate_max_norm": 8.0,
    "gh_points": 20,
    "min_inliers": 2,
    "max_inlier_frac": 0.95,
    "outlier_sigma_mult": 2.5,
    "background_scale_mult": 3.0,
    "min_noise_var": 1e-05,
    "max_noise_var": 10.0,
    "min_signal_var": 0.0001,
    "max_signal_var": 100.0,
    "min_lengthscale": 0.001,
    "max_lengthscale": 20.0,
    "response_seed_frac": 0.25,
    "response_inlier_frac": 0.7,
    "qr_steps": 30,
    "qr_lr": 0.01,
    "qr_beta_kl": 0.05,
    "qr_kl_warmup_steps": 20,
    "qr_composite_temperature": 1.0,
    "qr_init_log_std": -3.0,
    "qr_train_log_std": false,
    "qr_w_lengthscale": null,
    "qr_w_signal_var": 1.0,
    "lowrank_basis": "pca_residual",
    "lowrank_qr_w_signal_var": 0.1,
    "lowrank_basis_scale_l1": 0.0,
    "qr_residual_group_weight": 0.0,
    "qr_residual_group_eps": 1e-08,
    "qr_include_residual": true,
    "qr_grad_clip_norm": 10.0,
    "qr_log_interval": 10,
    "qr_outer_cycles": 4,
    "qr_pairwise_weight": 0.0,
    "qr_pairwise_margin": 1.0,
    "qr_pairwise_same_weight": 1.0,
    "qr_pairwise_cross_weight": 1.0,
    "qr_pairwise_normalize": true,
    "qr_retrieval_pool": 200,
    "qr_retrieval_composite_temperature": 0.0,
    "qr_anchor_pred_weight": 1.0,
    "qr_anchor_pred_var_floor": 1e-06,
    "qr_anchor_pred_exclude_self": true,
    "qr_anchor_self_tol": 1e-10,
    "qr_rank_weight": 0.0,
    "qr_rank_temperature": 0.25,
    "qr_rank_normalize": true,
    "refresh_hyper_steps": 10,
    "refresh_gate_steps": 20,
    "refresh_noise_std_floor": null,
    "refresh_damping": 1.0,
    "refresh_gate_l2": null,
    "refresh_gate_max_norm": null,
    "include_original": true,
    "resume": false,
    "out_dir": "experiments\\synthetic\\uncertain_w_cem_qr_rankhybrid_lh_seed0_20260517"
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
  "variants": [
    "qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1",
    "self_cem2_nstd0.1_kcov0.1",
    "qr_alt_hyra0.25_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1",
    "qr_alt_hyra0.4_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1",
    "qr_alt_hyra0.6_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1",
    "qr_alt_hyra0.75_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1"
  ],
  "n_rows": 7,
  "n_ok": 7,
  "n_failed": 0
}
```
