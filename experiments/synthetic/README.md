# Synthetic Experiments

Synthetic/LH data generation experiments and hyperparameter sweeps.

- `minibatch_LH.py`: minibatch LH synthetic experiment driver.
- `validation_new_dataset.py`: validation-oriented synthetic/UCI-style experiment driver.
- `run_graph_local_diagnostics.py`: generator-only Stage-0 diagnostics for the
  graph-local DJGP benchmark plan. It creates SBM graph-local synthetic data,
  split-size checks, jump proportions, feature/graph distance correlation, and
  feature-kNN vs graph-kNN overlap diagnostics without training models.
- `run_graph_local_model_screen.py`: seed-level graph-local model screen for
  XGBoost, DKL, LMJGP sampled-W JumpGP-CEM, and fixed-neighborhood self-CEM.
  It compares feature-neighborhood and graph-neighborhood retrieval for the
  local DJGP-style rows.
- `experiment_new.py`: broad synthetic benchmark harness (includes **DKL** and **XGBoost bootstrap** alongside DeepGP/DJGP).
- `baselines_test.py`: `--folder_name` CLI for DKL / XGBoost on a synthetic `dataset.pkl` folder (same return shape as `DJGP_test.py`).
- `compare_synthetic_baselines.py`: appendix-style low-rank jump comparison of
  X-standardized XGB-bootstrap, DKL-SVGP, JGP-PCA, CEM-EM full-covariance VI,
  and LMJGP variants.
- `compare_paper_synthetic_baselines.py`: corrected paper-style L2/LH
  benchmark runner using the historical L2/phantom and LH generators with
  train-only X standardization after RFF expansion.
- `compare_paper_weak_inner_w.py`: train=1000 L2/LH projection-learning
  ablation for weak-inner W objectives, including static PLS, exact local
  GP-NLL pilots, pairwise metric loss, boundary-weighted anchors, and raw vs
  projected final JumpGP neighborhoods.
- `compare_paper_vem_w.py`: conditional-W exact JGP-VEM runner.  It passes a
  low-rank Gaussian projection posterior into local VEM responsibility updates
  and tests MAP-W vs posterior-W, early-stopped VEM, soft responsibilities,
  boundary anchor weighting, and frozen/shared nuisance hyperparameters.
- `compare_paper_vem_w_controls.py`: five-seed soft-gamma VEM-W control runner
  for learned `q(W)`, anchor-shuffled learned `q(W)`, global-mean projection,
  and prior `q(W)` controls.
- `compare_l2_lh_7remedies.py`: focused implementation runner for the seven
  proposed projection-remedy ideas on paper-style L2/LH data, including
  conditional-W exact VEM, soft responsibilities, row-normalized/gate-regularized
  VEM, fixed nuisance, whitened-mean KL warmup, boundary pairwise loss, and a
  shared point-wise `W(x)` projection network.
- `compare_global_wgp_cem_l2_lh.py`: sparse-GP prior over the projection field
  `W(x)` trained by fixed-neighborhood local JGP-CEM composite likelihood.
  Includes anchor-conditioned and pointwise variants plus gamma-floor, coverage,
  broad-background outlier ablations, DeepVMGP-style PCA initialization,
  pairwise warm starts, MAP-`R` training, and q(R) data/KL gradient diagnostics.
- `diagnose_lmjgp_elbo_gradients_paper.py`: training-time ELBO gradient
  diagnostic for paper-style L2/LH, splitting q(V) data gradients from the
  KL pull and reporting responsibility hardening plus boundary-anchor proxies.
- `compare_lmjgp_prediction_heads_paper.py` and
  `compare_lmjgp_qw_controls_paper.py`: diagnostics for the paper-style
  synthetic LMJGP prediction head and `q(W)` controls.
- `compare_K.py`: synthetic K-sensitivity comparison.
- `jgp_alone_highdata.py`: high-dimensional JGP-focused synthetic harness.
- `L2_jgp_alone_highdata.py`: L2-focused high-dimensional JGP-only variant.

Root `minibatch_LH.py`, `validation_new_dataset.py`, `experiment_new.py`, `compare_K.py`, `jgp_alone_highdata.py`, and `L2_jgp_alone_highdata.py` files are compatibility entrypoints.
