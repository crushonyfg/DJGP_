# Uncertain-W CEM L2/LH Prediction-Head Ablation

Project log entry:
`docs/experiments_log.md` section
`2026-05-16 - uncertain_w_cem_qr_hybrid_lambda_controls_lh_seed0_20260516`.

Headline: on LH seed0, learned hybrid anchor retrieval survived controls.
Best rows were `lambda=0.40` (RMSE `642.19`, CRPS `275.22`, raw overlap
`0.868`) and the previous `lambda=0.25` run from
`uncertain_w_cem_qr_retrieval_metric_lh_seed0_20260515` (RMSE `643.71`, CRPS
`275.12`, raw overlap `0.915`).  Shuffled (`673.68` / `296.18`) and prior
(`696.10` / `313.86`) controls were clearly worse.

This experiment evaluates the uncertain-W CEM prediction head on paper-style
L2/LH synthetic data.  Base variants use a fixed mean projection from PLS/PCA
and controlled W uncertainty; `qr_mstep_*` and `qr_alt_*` variants train a
sparse-GP `q(R)` M-step through the moment-matched uncertain-W kernel.

Files:

- `metrics.csv`: one row per setting/seed/variant.
- `aggregate.csv`: mean/std over successful rows.
- `summary.json`: exact runner arguments and setting metadata.

```json
{
  "started": "2026-05-16 18:22:54",
  "finished": "2026-05-16 19:08:56",
  "device": "cuda",
  "args": {
    "settings": "lh_q5_train1000",
    "variants": "qr_alt_hya0.1_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1,qr_alt_hya0.15_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1,qr_alt_hya0.2_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1,qr_alt_hya0.3_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1,qr_alt_hya0.4_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1,qr_alt_hya0.0_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1,qr_alt_hya0.25_refresh10_self_cem2_nstd0.1_qrprior_rank0.1_kcov0.1,qr_alt_hya0.25_refresh10_self_cem2_nstd0.1_qrshuf_rank0.1_kcov0.1,qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1,self_cem2_nstd0.1_kcov0.1",
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
    "out_dir": "experiments\\synthetic\\uncertain_w_cem_qr_hybrid_lambda_controls_lh_seed0_20260516"
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
    "qr_alt_hya0.1_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1",
    "qr_alt_hya0.15_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1",
    "qr_alt_hya0.2_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1",
    "qr_alt_hya0.3_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1",
    "qr_alt_hya0.4_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1",
    "qr_alt_hya0.0_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1",
    "qr_alt_hya0.25_refresh10_self_cem2_nstd0.1_qrprior_rank0.1_kcov0.1",
    "qr_alt_hya0.25_refresh10_self_cem2_nstd0.1_qrshuf_rank0.1_kcov0.1",
    "qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1",
    "self_cem2_nstd0.1_kcov0.1"
  ],
  "n_rows": 11,
  "n_ok": 11,
  "n_failed": 0
}
```
