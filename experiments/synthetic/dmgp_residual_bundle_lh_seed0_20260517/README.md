# DeepMahalanobisGP Residual Bundle

This folder stores the direct DeepMahalanobisGP fit plus the arrays needed by
the downstream fixed-W residual uncertain self-CEM head.

Key files:

- `lh_q5_train1000_seed0.npz`: copied paper-style LH split.
- `lh_q5_train1000_seed0_train_fit_mean_std.npy`,
  `lh_q5_train1000_seed0_train_fit_var_std.npy`: full-fit DMGP train
  predictive mean/variance on standardized y.
- `lh_q5_train1000_seed0_train_oof_mean_std.npy`,
  `lh_q5_train1000_seed0_train_oof_var_std.npy`: 3-fold OOF train predictive
  mean/variance on standardized y.
- `lh_q5_train1000_seed0_test_mean_std.npy`,
  `lh_q5_train1000_seed0_test_var_std.npy`: full-fit DMGP test predictive
  mean/variance on standardized y.
- `lh_q5_train1000_seed0_pred_mean.npy`,
  `lh_q5_train1000_seed0_pred_sigma.npy`: full-fit DMGP test prediction on the
  original y scale.
- `lh_q5_train1000_seed0_W_train_mean.npy`,
  `lh_q5_train1000_seed0_W_train_var.npy`,
  `lh_q5_train1000_seed0_W_test_mean.npy`,
  `lh_q5_train1000_seed0_W_test_var.npy`: DMGP projection moments.

Produced by `experiments/synthetic/run_deep_mahalanobis_gp_residual_bundle.py`
with `steps1=300`, `steps2=700`, and `oof_folds=3`.
