# Experiments Log

Cross-experiment UCI small-train synthesis:
`docs/uci_smalltrain_summary.md`.

## 2026-05-23 - `jgp_batch_pca_l2_seed0`

- **Question**: (1) What is the default JGP used in this codebase?
  (2) Does a parallelised batch variant of JGP-CEM give the same
  RMSE/CRPS as the sequential version on the L2 dataset reduced to 2-D
  by PCA, and how much wall-time speedup does parallelism give?
- **New code**:
  - `JumpGaussianProcess/jumpgp.py` — added top-level `_cem_worker` function
    and `JumpGPBatch` class (ProcessPoolExecutor over per-test-point CEM).
  - `jumpgp/__init__.py` — exports `JumpGPBatch`.
  - `experiments/synthetic/compare_jgp_cem_batch_l2_pca.py` — standalone
    runner for this comparison.
- **Runner**:
  `experiments/synthetic/compare_jgp_cem_batch_l2_pca.py`

  ```
  conda activate jumpGP
  cd <repo root>
  python experiments/synthetic/compare_jgp_cem_batch_l2_pca.py ^
      --seed 0 --out_dir experiments/synthetic/jgp_batch_pca_l2_seed0
  ```
- **Settings / provenance**:
  - Dataset family: L2_phantom (`new_highdata_gen.generate_data_phantom`,
    `caseno=6`), RFF expansion (`new_highdata_gen.make_rff_features`).
  - Latent dim d=2, observed dim H=20, PCA target dim=2.
  - Train 1000 / test 200 (setting `l2_q2_train1000`).
  - X standardised with train-only StandardScaler before PCA.
  - y left on raw scale; JGP handles its own offset internally.
  - n_neighbors=25, mode=CEM (linear boundary, L=1).
  - Batch uses `ProcessPoolExecutor(max_workers=None)` (all cores).
- **Artifacts**:
  `experiments/synthetic/jgp_batch_pca_l2_seed0/`
  (`metrics.csv`, `summary.json`, `predictions.npz`, `README.md`).
- **Headline result** (seed=0, sandbox with 2 logical CPUs / 1 physical core):

  | method | wall (s) | RMSE | CRPS |
  |---|---:|---:|---:|
  | jgp_cem_sequential | 7.66 | **0.5089** | **0.2915** |
  | jgp_cem_batch | 9.99 | **0.5089** | **0.2915** |

  Max |mu_seq − mu_batch| = 0.00 (bit-exact identical predictions).
  PCA 20D→2D explained variance = 83.0%.
  Speedup = 0.77× (batch ~30% slower in this run).

- **Status**: Complete. Artifacts generated and verified.
- **Interpretation**:
  - RMSE/CRPS are identical — batch correctness confirmed.
  - Batch is *slower* in a 1-core environment: per-task overhead
    (fork + pickle 25×4-byte neighbor arrays × 200 futures) costs
    ~2.3 s, exceeding any parallelism benefit with only 1 physical core.
  - On a real multi-core machine (e.g. 8 cores) the wall time should
    drop to roughly seq_wall / n_cores, expected ~1–2 s.
- **Follow-ups**: Re-run on a multi-core machine to confirm speedup.
  If speedup > 3× is confirmed, consider making `JumpGPBatch` the
  default in `JumpGP_test.py` for large test sets.

## 2026-05-19 - `selfcem_l2_lh_train200_500_5seeds_20260519`

- **Question**: Fill the missing non-MC self-CEM rows for the paper-style
  synthetic L2/LH benchmark at train sizes 200 and 500, complementing the
  existing train=1000 self-CEM rows in
  `uncertain_w_cem_qr_alt_selected_5seeds_20260514`.
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Settings / provenance**:
  Paper-style synthetic data through the existing runner: L2 low-dimensional
  generator with latent/observed `d=2`, and LH high-dimensional RFF expansion
  with latent `d=5`, observed `H=30`; train sizes `200,500`, test size capped
  at 200 anchors, seeds `0-4`, train-only X standardization, variant
  `self_cem2_nstd0.1_kcov0.1`, no MC-CEM rows.
- **Artifacts**:
  `experiments/synthetic/selfcem_l2_lh_train200_500_5seeds_20260519/`
  (`metrics.csv`, `aggregate.csv`, `summary.json`, `README.md`).
- **Headline result**:

  | setting | RMSE | CRPS | Cov90 | Width90 |
  | --- | ---: | ---: | ---: | ---: |
  | L2 train=200 | 3.224 | 1.694 | 0.741 | 5.698 |
  | L2 train=500 | 2.469 | 1.316 | 0.811 | 5.874 |
  | LH train=200 | 623.706 | 280.360 | 0.761 | 247.177 |
  | LH train=500 | 642.334 | 285.939 | 0.771 | 222.272 |

- **Status**: Completed.  Synthetic self-CEM now has 5-seed rows for
  train=200/500/1000.
- **Follow-ups**: None for this fill-in run.

## 2026-05-19 - `uci_selfcem_train200_1000_5seeds_20260519`

- **Question**: Fill the missing non-MC self-CEM UCI rows at train sizes 200
  and 1000, complementing the existing train=500 run.
- **Runner**:
  `experiments/uci/compare_uncertain_w_cem_train500.py`.
- **Settings / provenance**:
  UCI Wine, Parkinsons, and Appliances with the runner's grouped/small-train
  presets, train sizes `200,1000`, seeds `0-4`, max test anchors `200`,
  variant `self_cem2_nstd0.1_kcov0.1`, no original-CEM rows.
- **Artifacts**:
  `experiments/uci/uncertain_w_cem_train200_1000_selfcem_5seeds_20260519/`
  (`metrics.csv`, `aggregate.csv`, `summary.json`, `README.md`).
- **Headline result**:

  | dataset | train | RMSE | CRPS | Cov90 | Width90 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Wine | 200 | 0.799 | 0.459 | 0.785 | 1.796 |
  | Wine | 1000 | 0.754 | 0.426 | 0.784 | 1.685 |
  | Parkinsons | 200 | 10.185 | 6.589 | 0.548 | 16.198 |
  | Parkinsons | 1000 | 10.490 | 7.166 | 0.421 | 12.872 |
  | Appliances | 200 | 121.315 | 50.254 | 0.727 | 101.056 |
  | Appliances | 1000 | 102.865 | 39.768 | 0.773 | 98.574 |

- **Status**: Completed.  UCI self-CEM now has 5-seed rows for
  train=200/500/1000.
- **Follow-ups**: None for this fill-in run.

## 2026-05-19 - `pdebench_burgers_selfcem_train200_500_1000_5seeds_20260519`

- **Question**: Fill the missing non-MC self-CEM train-size sweep for balanced
  PDEBench Burgers ShockJump at train sizes 200, 500, and 1000.
- **Runner**:
  `experiments/pdebench/compare_burgers_selfcem_train_sizes.py`.
- **Implementation**:
  Added a dedicated self-CEM runner that reuses the processed PDEBench split,
  train-size subsampling, and all/near/far subset metric helpers from the
  existing PDEBench train-size scripts, while calling
  `run_uncertain_w_self_cem` / `uncertain_w_cem_predict_from_labels`.
- **Settings / provenance**:
  `data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000`, balanced
  near-threshold Burgers ShockJump surrogate, train sizes `200,500,1000`,
  seeds `0-4`, `Q=3`, `neighbors=25`, variant
  `self_cem2_nstd0.1_kcov0.1`.
- **Artifacts**:
  `experiments/pdebench/burgers_train200_500_1000_selfcem_5seeds_20260519/`
  (`metrics.csv`, `aggregate.csv`, `summary.json`, `README.md`).
- **Headline result**:

  | train | subset | RMSE | CRPS | Cov90 | Width90 |
  | ---: | --- | ---: | ---: | ---: | ---: |
  | 200 | all | 0.974 | 0.390 | 0.772 | 0.810 |
  | 200 | near | 1.287 | 0.704 | 0.566 | 1.082 |
  | 200 | far | 0.829 | 0.277 | 0.846 | 0.712 |
  | 500 | all | 0.872 | 0.317 | 0.798 | 0.713 |
  | 500 | near | 1.082 | 0.530 | 0.655 | 1.108 |
  | 500 | far | 0.776 | 0.240 | 0.849 | 0.570 |
  | 1000 | all | 0.812 | 0.278 | 0.824 | 0.614 |
  | 1000 | near | 0.898 | 0.380 | 0.760 | 0.775 |
  | 1000 | far | 0.777 | 0.242 | 0.848 | 0.556 |

- **Status**: Completed.  PDEBench Burgers self-CEM now has 5-seed rows for
  train=200/500/1000.
- **Follow-ups**: None for this fill-in run.

## 2026-05-19 - `paper_l2_lh_xstd_lh_pls_kl_seed3_pointfilter_20260519`

- **Question**: Re-run only the known exploding
  `lh_q5_train1000` / `lmjgp_pls_kl` seed 3 row, and record a 5-seed aggregate
  that uses built-in point-level metric filtering rather than manually
  dropping the whole seed.
- **Runner / code path**:
  `experiments/synthetic/compare_paper_synthetic_baselines.py` calling
  `experiments/synthetic/compare_lmjgp_tricks_synth.py`; Gaussian metric
  filtering is implemented in `djgp/evaluation/calibration.py`.
- **Implementation**:
  Added `filtered_gaussian_uq_metrics_dict`, wired the LMJGP PLS-KL runner to
  save `metrics_pointfiltered.csv` / `aggregate_pointfiltered.csv`, and added
  the repository policy that point-level numerical blow-up filtering belongs
  in the metric path.  Existing raw and 4-seed/omit-blowup files are preserved.
- **Settings / provenance**:
  Paper-style LH `lh_q5_train1000`, RFF expansion, latent `d=5`, observed
  `H=30`, train/test `1000/200`, train-only X standardization, method
  `lmjgp_pls_kl`, seed 3 only for the repair run.  The original raw seed 3 row
  reproduced the explosion; point filtering drops one extreme-variance test
  point (`sigma > 1e6`).
- **Artifacts**:
  Repair:
  `experiments/synthetic/paper_l2_lh_xstd_train1000_lh_pls_kl_seed3_pointfilter_20260519/`
  (`metrics_pointfiltered.csv`,
  `metrics_repaired_seed3_pointfiltered_5seeds.csv`,
  `aggregate_repaired_seed3_pointfiltered_5seeds.csv`, `README.md`).
  Reproduction-only old-code seed-3 rerun:
  `experiments/synthetic/paper_l2_lh_xstd_train1000_lh_pls_kl_seed3_rerun_20260519/`.
  Aborted mistaken full-5-seed attempt:
  `experiments/synthetic/paper_l2_lh_xstd_train1000_lh_pls_kl_pointfiltered_5seeds_20260519/`;
  all rows in its `metrics.csv` are `NameError` failures and it must not be
  used.
- **Headline result**:
  The repaired seed 3 point-filtered row drops 1 of 200 test points and gives
  RMSE 635.292 / CRPS 303.823 / Cov90 0.688 / Width90 1144.144.  Combining
  original seeds 0,1,2,4 with this repaired seed 3 gives LH train=1000
  `lmjgp_pls_kl` 5-seed RMSE 581.383 / CRPS 251.499 / Cov90 0.724 /
  Width90 658.616.
- **Status**: Completed as a repaired auxiliary aggregate.  It does not
  overwrite `aggregate_omit_blowup.csv` or the raw aggregate.
- **Follow-ups**: Keep both the raw/omit-seed aggregate and this point-filtered
  repaired aggregate visible in summaries.

## 2026-05-18 - `selfcem_multiinit_lh_seed0_40_20260518`

- **Question**: Before adding any W M-step, can diverse fixed W
  initializations be treated as candidate geometries for self-CEM, then
  selected or mixed by local likelihood?  This tests the idea that W
  initialization should be a discrete geometry/model choice rather than only
  an optimizer starting point.
- **Runner**:
  `experiments/synthetic/compare_selfcem_multiinit_lh.py`.
- **Implementation**:
  Added a fixed-W multi-init screen.  For each test anchor it first builds a
  raw-X candidate pool, reranks that pool under each candidate W, runs one
  batched deterministic-W `run_uncertain_w_self_cem` call over all
  `(initialization, anchor)` pairs, then predicts each candidate head.  It
  reports fixed-init rows, a local-likelihood soft mixture, local-likelihood
  hard selection, a uniform mixture control, and a clearly marked
  `diagnostic_anchor_y_soft_mixture` row that uses true anchor y only as an
  upper-bound diagnostic for future gating/selector work.  No W M-step is run.
- **Settings / provenance**:
  Paper-style LH `lh_q5_train1000` through canonical `_generate_paper_data`:
  LH base generator with RFF expansion, latent dimension `d=5`, observed
  dimension `H=30`, learned `Q=5`, train/test `1000/200`, train-only X
  standardization and train-only y standardization inside the head.  Seed `0`,
  first `40` test anchors, raw candidate pool `120`, final neighbors `35`,
  two self-CEM updates, batched hyperparameter fit with `hyper_steps=3`,
  `gate_steps=4`, and `noise_std_floor=0.1`.  This is a computational pilot,
  not a formal 200-anchor or 5-seed result.  DMGP-W is intentionally excluded.
- **Artifacts**:
  Broad candidate screen:
  `experiments/synthetic/selfcem_multiinit_lh_seed0_40_20260518/`
  (`raw_pls,pls,pca,pairwise,rand0,rand1`).  Core candidate screen:
  `experiments/synthetic/selfcem_multiinit_lh_seed0_40_core_20260518/`
  (`raw_pls,pls,pairwise`).  Smoke:
  `experiments/synthetic/selfcem_multiinit_smoke_20260518/`; do not treat the
  smoke as a reported result.
- **Headline result (LH seed0, first 40 test anchors)**:

  Broad six-candidate screen:

  | variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | fixed PLS self-CEM | **795.03** | **422.78** | 0.700 | 287.55 |
  | fixed raw-pool PLS self-CEM | 815.10 | 459.29 | 0.625 | 310.35 |
  | fixed pairwise self-CEM | 825.22 | 461.95 | 0.650 | 287.13 |
  | fixed PCA self-CEM | 898.24 | 518.96 | 0.650 | 235.61 |
  | fixed random-0 self-CEM | 912.45 | 525.42 | 0.625 | 256.84 |
  | fixed random-1 self-CEM | 945.50 | 562.69 | 0.625 | 184.29 |
  | local-likelihood soft mixture | 943.25 | 561.62 | 0.625 | 186.70 |
  | uniform mixture | 844.41 | 458.55 | 0.650 | 469.60 |
  | diagnostic anchor-y soft mixture | 732.03 | 377.00 | 0.725 | 463.24 |

  Core three-candidate screen:

  | variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | fixed PLS self-CEM | 795.03 | 422.78 | 0.700 | 287.55 |
  | fixed raw-pool PLS self-CEM | 815.10 | 459.29 | 0.625 | 310.35 |
  | fixed pairwise self-CEM | 825.22 | 461.95 | 0.650 | 287.13 |
  | local-likelihood soft mixture | 879.73 | 502.51 | 0.650 | 240.98 |
  | uniform mixture | **790.38** | **421.09** | 0.700 | 455.92 |
  | diagnostic anchor-y soft mixture | 719.48 | 372.24 | 0.725 | 428.93 |

  The fixed-W screen finds real variation across initial geometries, with PLS
  strongest among single candidates in this setup.  The unsupervised local
  marginal-likelihood score is not a good selector here: it prefers narrow
  heads and hurts both soft mixture and hard selection.  However, the
  anchor-y diagnostic mixture is much better, so the promising follow-up is an
  anchor-wise selector/gate trained on train-anchor predictive scores, not a
  more complex qR objective.  If W refinement is added later, it should be a
  residual/proximal update around the selected initialization, i.e.
  penalize `W - W0` or put the qR prior on residual coefficients rather than
  on the absolute W value.

## 2026-05-18 - `self_vem_lh_seed0_40_20260518`

- **Question**: Can the non-MC uncertain-W self-CEM head be softened into a
  self-VEM head, replacing hard CEM partitions with soft inlier
  responsibilities while still marginalising `W` through the same
  anchor-conditioned `W_mu` / `W_var` moments?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation**:
  Added `SelfVEMConfig`, `uncertain_w_vem_update_responsibilities`, and
  `run_uncertain_w_self_vem` in `djgp/projections/uncertain_w_jgp.py`.
  The shared fixed-label kernel fit and prediction helpers now accept
  fractional labels by using responsibilities as weighted GP likelihood
  precisions (`noise_var / responsibility`).  The soft V-step computes
  uncertain-W gate expectations and LOO weighted-GP inlier scores, then updates
  responsibilities with a sigmoid margin instead of thresholding labels.  The
  non-MC q(R) frozen-CEM bound also uses this weighted GP form when supplied
  soft labels, and the runner accepts `qr_mstep_self_vem...` variants for W
  learning through soft partitions.
  The experiment runner accepts variants such as
  `self_vem2_nstd0.1_kcov0.1`.
- **Settings / provenance**:
  Paper-style LH `lh_q5_train1000` through canonical `_generate_paper_data`:
  LH base generator with RFF expansion, latent dimension `d=5`, observed
  dimension `H=30`, learned `Q=5`, train/test `1000/200`, train-only X
  standardization and train-only y standardization inside the head.  Seed `0`,
  first `40` test anchors, raw-X neighborhoods with `n_neighbors=35`, PLS mean
  projection, controlled diagonal W covariance with `kcov=0.1`, two
  self-CEM/VEM outer updates, `hyper_steps=3`, `gate_steps=4`, and
  `noise_std_floor=0.1`.  This is a computational pilot, not a formal
  200-anchor or 5-seed result.
- **Artifacts**:
  `experiments/synthetic/self_vem_lh_seed0_40_20260518/`.
  Smoke path:
  `experiments/synthetic/self_vem_lh_smoke_20260518/`.  q(R) parser/objective
  smoke:
  `experiments/synthetic/self_vem_qr_lh_smoke_20260518/` and
  `experiments/synthetic/self_vem_qr_lh_smoke2_20260518/`.  Do not treat the
  smoke folders as reported results.
- **Headline result (LH seed0, first 40 test anchors)**:

  | variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | original mean-W JumpGP-CEM | 703.02 | 345.09 | 0.500 | 236.11 |
  | hard `self_cem2_nstd0.1_kcov0.1` | **711.04** | **327.01** | 0.775 | 504.60 |
  | soft `self_vem2_nstd0.1_kcov0.1` | 731.94 | 357.16 | 0.775 | 693.59 |

  The initial soft version is functional but not better on this slice.  It
  keeps similar hard-thresholded inlier fraction (`0.68`) but leaves nontrivial
  responsibility entropy (`0.203` nats), widening intervals and worsening CRPS.
  Follow-ups should tune V-step temperature/damping and consider a stronger
  variational objective for the weighted local GP rather than only replacing
  hard labels with precision weights.

## 2026-05-18 - `mc_cem_qr_lh_seed0_20260518`

- **Question**: For the self-CEM/qR line, does a Monte-Carlo CEM M-step that
  refits local self-CEM independently for each sampled `W_j` improve LH
  behavior?  Compare route A (reparameterized sampled `W`, frozen CEM state in
  the M-step) against route B (score-function / REINFORCE reward), plus route-A
  entropy exploration and route-B within-anchor reward standardization.
- **Runner**:
  `experiments/synthetic/compare_mc_cem_qr_lh.py`.
- **Implementation**:
  Added `train_qr_mc_cem_mstep` in
  `djgp/projections/uncertain_w_jgp.py`.  It samples anchor-conditioned
  `W_j ~ q(W_j)` from sparse-GP `q(R)`, runs deterministic-W self-CEM on each
  detached sample, then optimizes either a stop-gradient reparameterized
  frozen-CEM bound (`A_*`) or a score-function objective (`B*`).  Final
  prediction uses sampled deterministic-W self-CEM ensemble.
- **Settings / provenance**:
  Paper-style LH `lh_q5_train1000` through canonical `_generate_paper_data`:
  LH base generator with RFF expansion, latent dimension `d=5`, observed
  dimension `H=30`, learned `Q=5`, train/test `1000/200`, train-only X
  standardization and train-only y standardization inside the head.  Seed `0`,
  first `40` test anchors, `n_neighbors=35`, `m_inducing=40`, PLS `W0`,
  `mc_samples=3`, `mc_steps=3`, `cem_updates=1`, local self-CEM
  `hyper_steps=3`, `gate_steps=4`, final ensemble samples `3`,
  `noise_std_floor=0.1`.  This is a computational pilot, not a formal
  200-anchor or 5-seed result.
- **Artifacts**:
  `experiments/synthetic/mc_cem_qr_lh_seed0_20260518/`.
  Follow-up speed/optimization sweeps:
  `experiments/synthetic/mc_cem_qr_lh_b_steps5_eval_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_lr0002_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_lr0005_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_lr001_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_h1g1_lr001_seed0_20260518/`, and
  `experiments/synthetic/mc_cem_qr_lh_b_h6g8_lr001_seed0_20260518/`.
  Trick ablations requested after the pilot:
  `experiments/synthetic/mc_cem_qr_lh_b_rinit_pca_lr001_h1g1_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_ortho001_lr001_h1g1_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_ortho01_lr001_h1g1_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_early_lr001_h1g1_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_early_eval3_lr001_h1g1_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_ortho10_nonorm_early_lr001_h1g1_seed0_20260518/`, and
  `experiments/synthetic/mc_cem_qr_lh_b_pca_ortho_early_lr001_h1g1_seed0_20260518/`.
  Weak-vs-strong CEM evaluation follow-up:
  `experiments/synthetic/mc_cem_qr_lh_b_final20_lr001_h1g1_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_pca_final20_lr001_h1g1_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_pca_steps5_eval20_lr001_h1g1_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_pca_early_eval20_final20_lr001_h1g1_seed0_20260518/`, and
  `experiments/synthetic/mc_cem_qr_lh_b_pca_early_eval20n3_final20_lr001_h1g1_seed0_20260518/`.
  Smoke/debug folders:
  `experiments/synthetic/mc_cem_qr_lh_smoke_20260518/` and
  `experiments/synthetic/mc_cem_qr_lh_debug_20260518/`; do not treat those as
  reported results.  Batched-hyperopt sanity folders:
  `experiments/synthetic/mc_cem_qr_lh_batched_smoke_20260518/` and
  `experiments/synthetic/mc_cem_qr_lh_seq_smoke_20260518/`.
- **Headline result (LH seed0, first 40 test anchors)**:

  | variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | `B` score function | **710.74** | **333.07** | 0.775 | 541.94 |
  | `A_uniform` stop-grad reparam | 715.81 | 337.76 | 0.775 | 530.34 |
  | `A_softmax` | 716.70 | 337.77 | 0.775 | 533.01 |
  | `A_softmax_entropy` | 716.70 | 337.77 | 0.775 | 533.01 |
  | `B_std` score + reward standardization | 716.86 | 336.94 | 0.775 | 532.26 |
  | same-budget deterministic `baseline_self_cem` | 751.32 | 374.37 | 0.700 | 315.29 |

  All MC-CEM qR variants beat the same-budget deterministic self-CEM baseline
  on this 40-anchor slice.  Route B is the strongest row in this pilot.  Route
  A softmax weighting gives no visible gain over uniform weighting here; the
  entropy bonus is effectively neutral at this small step count; reward
  standardization in B weakens RMSE versus unstandardized B while keeping CRPS
  slightly better than route A.
- **Speed / optimization follow-up**:
  The fixed-label kernel hyperparameter optimizer is now batched for anchor
  mode (`FixedLabelHyperoptConfig.batched=True`, default in the MC runner);
  gate fitting was already batched.  On the same 40-anchor B configuration
  (`mc_steps=3`, `mc_samples=3`, `hyper_steps=3`, `gate_steps=4`, `lr=0.005`),
  training time dropped from the original sequential-hyperopt pilot row
  `37.83` s to `16.13` s with essentially unchanged metrics (`RMSE 710.74`
  -> `710.78`, `CRPS 333.07` -> `333.10`).  A tiny 6-anchor paired smoke gave
  `2.24` s batched vs `2.62` s sequential, confirming the code path without
  relying on the larger pilot.

  Per-step eval for route B with `mc_steps=5` and one-sample eval ensembles was
  noisy rather than monotone: RMSE by step was `715.77, 668.28, 734.85,
  663.81, 708.24` and CRPS was `334.12, 308.72, 356.73, 305.99, 322.67`.
  Thus the MC reward objective is not a stable validation proxy at this sample
  budget; best validation-like rows occurred around steps 2/4, not necessarily
  the final step.

  Route-B learning-rate sweep at `mc_steps=3`, `hyper_steps=3`, `gate_steps=4`
  favored the larger tested step size: `lr=0.002` gave RMSE/CRPS
  `710.93/333.41`, `lr=0.005` gave `710.78/333.10`, and `lr=0.01` gave
  `709.97/331.50`.  With `lr=0.01`, reducing local self-CEM inner work to
  `hyper_steps=1`, `gate_steps=1` was actually best on this slice
  (`708.64/327.83`), while `6/8` was worse (`710.28/331.81`) and narrower.

  Requested trick ablations at the cheap route-B setting
  (`lr=0.01`, `hyper_steps/gate_steps=1/1`) gave the clearest gain from
  qR initialization.  Keeping local self-CEM initialization at PLS but setting
  `--r_init pca` improved RMSE/CRPS to `686.35/318.80` with Cov90 `0.825`,
  versus the PLS/same qR init row `708.64/327.83`.  Soft orthogonality on the
  qR inducing mean did not change the PLS-init final metrics at weights `0.01`
  or `0.1` because the PLS initialization is already nearly semi-orthogonal
  and the score-function training-loss best state often returns to the first
  step.  A stronger non-normalized orthogonality penalty with validation
  early-stopping (`weight=10`, `--no-mc_ortho_normalize`) likewise did not
  improve (`711.28/330.71`).  Validation early stopping itself selected step
  `2` for PLS init but worsened final ensemble metrics (`711.27/330.65`);
  using 3 eval samples did not change the selected state.  Combining PCA init
  with weak orthogonality and early stopping selected step `1` and matched the
  PCA-init-only result.

  A later check decoupled training-CEM strength from eval/final-CEM strength.
  The earlier per-step evals used the runner's weak `hyper_steps/gate_steps=1/1`
  self-CEM.  When final prediction was changed to stronger `20/20` self-CEM
  while keeping weak training, metrics worsened on this slice: PLS/same qR init
  moved from `708.64/327.83` to `712.65/336.09`, and PCA qR init moved from
  `686.35/318.80` to `713.33/342.14`.  Strong `20/20` per-step eval with only
  one W-ensemble sample gave an overly optimistic step-1 estimate
  (`606.43/271.16`), but matching the final ensemble count (`evalN=3`) still
  selected step 1 and final `20/20` remained `713.33/342.14`.  Thus the weak
  CEM result is not simply under-converged; stronger local refits appear to
  sharpen/narrow the ensemble in a way that hurts RMSE/CRPS on this 40-anchor
  pilot.

  A follow-up implemented two additional opt-in controls in the MC-CEM qR
  runner: learnable sparse-GP inducing locations for `q(R)` and a per-local
  learnable constant outlier log-density (`log(1/u)`) for
  `outlier_mode=constant`.  The learnable constant is bounded and anchored to
  the legacy constant with an L2 penalty because an unconstrained fixed-label
  constant density has no finite MLE.  Artifacts:
  `experiments/synthetic/mc_cem_qr_lh_learnZ_constU_smoke_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_pca_learnZ_bg_lr001_h1g1_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_pca_const_fixedU_lr001_h1g1_seed0_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_pca_const_learnU_fixedZ_lr001_h1g1_seed0_20260518/`, and
  `experiments/synthetic/mc_cem_qr_lh_b_pca_learnZ_constU_lr001_h1g1_seed0_20260518/`.

  | route-B PCA qR init setting | RMSE | CRPS | Cov90 | Width90 | mean inducing shift |
  |---|---:|---:|---:|---:|---:|
  | fixed inducing, `background_normal` outlier | 686.35 | 318.80 | 0.825 | 664.19 | - |
  | learn inducing, `background_normal` outlier | 686.48 | 318.83 | 0.825 | 664.33 | 0.0055 |
  | fixed inducing, fixed `constant` outlier | 700.89 | 329.77 | 0.800 | 706.90 | 0.0000 |
  | fixed inducing, learnable per-local `constant` outlier | **662.77** | **308.71** | 0.825 | 724.73 | 0.0000 |
  | learn inducing + learnable per-local `constant` outlier | **662.75** | **308.70** | 0.825 | 724.80 | 0.0055 |

  On this 40-anchor slice, moving the qR inducing locations at
  `inducing_lr_mult=0.1`, `inducing_l2=1e-3` barely moves them and has no
  useful effect by itself.  The learnable constant outlier parameter is the
  meaningful gain here, improving the PCA qR route-B pilot from
  `686.35/318.80` to `662.77/308.71`; adding learnable inducing locations on
  top is neutral.

  The requested 5-seed, full-200-test follow-up of the best 40-anchor setting
  (`B`, PCA qR init, `lr=0.01`, `hyper_steps/gate_steps=1/1`,
  learn-inducing, and learnable per-local `constant` outlier density) is in
  `experiments/synthetic/mc_cem_qr_lh_b_pca_learnZ_constU_test200_5seeds_20260518/`.
  It used the same paper-style LH provenance as above, seeds `0..4`,
  `max_test=200`, `mc_steps=3`, `mc_samples=3`, and final ensemble samples
  `3`.  Aggregate result:

  | setting | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | MC-CEM qR B + PCA init + learnZ + learnable constant outlier | 596.55 +/- 123.23 | 252.59 +/- 78.74 | 0.799 +/- 0.044 | 427.42 +/- 153.72 |

  Per-seed RMSEs were `675.40, 620.00, 556.39, 746.61, 384.36`, and CRPSs
  were `306.26, 253.99, 225.47, 354.79, 122.45`.  The 40-anchor advantage
  therefore does not translate into a clear full-test 5-seed improvement over
  the existing LH seed0-to-4 controls: the stored `self_cem2_nstd0.1_kcov0.1`
  row is `595.68/252.28`, and the stored `qr_alt_refresh10_self_cem2_nstd0.1`
  row is `591.85/250.81`.

  A requested route-A follow-up used `A_uniform`, PCA qR init, `lr=0.01`,
  `hyper_steps/gate_steps=3/5`, learnable per-local `constant` outlier density,
  no learnable qR inducing locations, seeds `0..4`, `max_test=200`,
  `mc_steps=3`, `mc_samples=3`, and final ensemble samples `3`.  Artifacts:
  `experiments/synthetic/mc_cem_qr_lh_a_uniform_pca_constU_h3g5_test200_5seeds_20260518/`.

  | setting | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | MC-CEM qR `A_uniform` + PCA init + learnable constant outlier | 601.40 +/- 130.15 | 255.65 +/- 82.26 | 0.791 +/- 0.046 | 400.79 +/- 135.29 |

  Per-seed RMSEs were `683.47, 616.85, 565.12, 763.83, 377.75`, and CRPSs
  were `308.13, 252.76, 230.29, 366.28, 120.80`.  This route-A setting is
  slightly worse than the corresponding route-B full-test follow-up
  (`596.55/252.59`) and takes longer on average (`106.0` s vs `75.7` s per
  seed), though its intervals are narrower (`Width90 400.79` vs `427.42`).

  Additional seed0/40-anchor ablations added three opt-in controls to
  `compare_mc_cem_qr_lh.py`: full-quadratic local CEM gate boundaries
  (`--gate_basis quadratic`), supervised route-B rewards from training-anchor
  predictive log likelihoods (`--mc_reward_mode supervised_anchor`), and
  post-qR projected-neighborhood reselection (`--reselect_neighbors_after_train
  anchor|pointwise`).  These are pilot diagnostics only, not 5-seed results.
  Artifacts:
  `experiments/synthetic/mc_cem_qr_lh_ab_quadratic_constU_seed0_40_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_supervised_constU_seed0_40_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_reselect_anchor_constU_seed0_40_20260518/`, and
  `experiments/synthetic/mc_cem_qr_lh_b_reselect_pointwise_constU_seed0_40_20260518/`.

  | seed0/40 setting | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | linear B + learnable constant outlier reference | 662.77 | 308.71 | 0.825 | 724.73 |
  | quadratic gate `A_uniform` | 662.86 | **306.65** | 0.825 | 722.81 |
  | quadratic gate B | 700.01 | 329.05 | 0.800 | 705.26 |
  | B supervised training-anchor reward (`reward_anchor_count=200`) | 665.05 | 309.91 | 0.825 | 723.80 |
  | B + final projected-neighborhood reselection, anchor-conditioned | **662.39** | 320.16 | 0.825 | 798.33 |
  | B + final projected-neighborhood reselection, pointwise-projected | 704.77 | 342.85 | 0.750 | 757.19 |

  The clearest small-slice signal is that quadratic route A improves CRPS
  slightly without changing RMSE, while quadratic route B is worse.  The
  supervised route-B reward is close to the transductive reward but does not
  improve this slice.  Projected-neighborhood reselection as currently
  implemented is a final post-training reselection diagnostic, not a per-step
  inner-loop reselection; anchor-conditioned reselection slightly lowers RMSE
  but widens intervals and worsens CRPS, while pointwise-projected reselection
  is clearly worse.

  A later supervised-B long-step check tested `mc_steps=20` and `30` on the
  same seed0/40-anchor setup.  The original MC trainer restores the lowest
  training-loss state by default; for supervised-B the lowest loss was step 1,
  so `mc_steps=20/30` with default restore produced exactly the same metrics as
  the 3-step row (`665.05/309.91`) while taking much longer.  To inspect actual
  later-step states, the runner now has `--mc_restore_state final` as an opt-in
  diagnostic.  Valid artifacts:
  `experiments/synthetic/mc_cem_qr_lh_b_supervised_constU_steps20_seed0_40_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_supervised_constU_steps30_seed0_40_20260518/`,
  `experiments/synthetic/mc_cem_qr_lh_b_supervised_constU_steps20_final2_seed0_40_20260518/`, and
  `experiments/synthetic/mc_cem_qr_lh_b_supervised_constU_steps30_final2_seed0_40_20260518/`.
  The first attempted `steps20_final` and `steps30_final` folders without the
  `final2` suffix failed because `restore_state` was initially attached to the
  wrong config; ignore those failed rows.

  | supervised-B seed0/40 setting | RMSE | CRPS | Cov90 | Width90 | R std median |
  |---|---:|---:|---:|---:|---:|
  | 3 steps, default best-loss restore | 665.05 | 309.91 | 0.825 | 723.80 | 0.0503 |
  | 20 steps, default best-loss restore | 665.05 | 309.91 | 0.825 | 723.80 | 0.0503 |
  | 30 steps, default best-loss restore | 665.05 | 309.91 | 0.825 | 723.80 | 0.0503 |
  | 20 steps, `--mc_restore_state final` | **654.51** | **308.33** | 0.825 | 759.23 | 0.0608 |
  | 30 steps, `--mc_restore_state final` | 660.35 | 313.55 | 0.825 | 735.50 | 0.0672 |

  Thus longer supervised-B training can improve the raw final state at around
  20 steps on this slice, but the gain is modest, expensive, and not monotone.
  The best-loss criterion is misaligned with final RMSE/CRPS for this reward,
  so any larger run should include validation/early stopping or a final-state
  protocol explicitly.
- **Status**:
  Completed one-seed, 40-anchor pilot plus speed/parameter sweeps, and the
  requested 5-seed/full-test follow-up.  The best 40-anchor MC-CEM qR setting is
  competitive on full LH but not better than the strongest existing
  self-CEM/qR controls; do not promote it as a new best LH result without a
  targeted follow-up.

## 2026-05-18 - `uncertain_w_cem_qr_svd_lh_full_seed0_20260518`

- **Question**: Does the promising 40-anchor shared-SVD q(R) mean diagnostic
  survive on the full LH seed0 test set?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Settings / provenance**:
  Same implementation and paper-style LH provenance as
  `uncertain_w_cem_qr_svd_lh_seed0_20260518`, but evaluated all `200` seed0
  test anchors.  Compared free qR mean, shared-SVD qR mean, and no-qR raw
  self-CEM2 with `qr_train_anchor_count=100`, `kmeans_yvar` train anchors,
  `qr_train_n_neighbors=35`, raw retrieval pool `200`, `qr_outer_cycles=4`,
  `refresh10`, self-CEM2, `nstd=0.1`, `rank=0.1`, `kcov=0.1`,
  default `qr_steps=30`, and `m_inducing=40`.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_svd_lh_full_seed0_20260518/`.
- **Headline result (LH seed0, test=200)**:

  | variant | RMSE | CRPS | Cov90 | Width90 | qR ortho penalty |
  |---|---:|---:|---:|---:|---:|
  | no-qR raw self-CEM2 | **659.38** | **288.13** | **0.780** | 267.81 | - |
  | hybrid anchor, free qR mean | 681.80 | 304.36 | 0.775 | 329.78 | 0.01446 |
  | hybrid anchor, shared-SVD qR mean | 727.69 | 341.63 | 0.760 | 388.17 | 0.000407 |

  The full-test result does **not** support the shared-SVD mean as an
  improvement.  Although it enforces the intended structure very strongly
  (orthogonality diagnostic `0.01446 -> 0.000407`, Cayley/Stiefel errors near
  `1e-15`), it worsens RMSE by `+45.89` and CRPS by `+37.27` relative to the
  same free-qR hybrid row, and is also worse than the simpler no-qR self-CEM2
  baseline.  The earlier 40-anchor gain was therefore a slice artifact.
- **Status**:
  Completed full seed0 follow-up.  Do not expand shared-SVD q(R) mean to
  5 seeds in this form.  If continuing structured q(R), the next plausible
  direction is less restrictive: residual shared-SVD around a free/PLS
  component, per-cluster shared factors, or a scale-targeted prior/penalty
  rather than forcing all inducing means into one global singular-vector frame.

## 2026-05-18 - `uncertain_w_cem_qr_svd_lh_seed0_20260518`

- **Question**: Can a structured q(R) mean parameterization remove more of the
  rotation/scale redundancy than a soft Frobenius orthogonality penalty, while
  preserving the self-CEM/qR LH hybrid-anchor signal?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation**:
  Added opt-in `UncertainWQRMstepConfig.mean_parameterization="shared_svd"`.
  In this mode the q(R) mean is constrained to
  `R_l = U diag(s_l) V^T`, with global shared `U in O(Q)` and
  `V in Stiefel(D,Q)`, and positive per-inducing scales `s_l`.  `U` and `V`
  are initialized from the SVD of the starting mean projection and updated by
  Cayley transforms around that initialization; `s_l` is optimized in log
  space.  The diagonal q(R) covariance and existing sparse-GP KL are retained,
  so this is a structured variational-mean control rather than a new manifold
  prior.  The synthetic runner accepts `_svd` before `_kcov`, e.g.
  `qr_alt_hya0.25_refresh10_self_cem2_nstd0.1_rank0.1_svd_kcov0.1`.
- **Settings / provenance**:
  Paper-style LH `lh_q5_train1000` via `_generate_paper_data` (LH base
  generator with RFF expansion, latent dimension `d=5`, observed dimension
  `H=30`, learned `Q=5`, train/test `1000/200`, train-only X standardization
  and train-only y standardization inside the self-CEM head).  Seed `0`,
  evaluated the first `40` test anchors.  Hybrid-anchor qR used
  `qr_train_anchor_count=100`, `kmeans_yvar` anchors,
  `qr_train_n_neighbors=35`, raw retrieval pool `200`, `qr_outer_cycles=4`,
  `refresh10`, self-CEM2, `nstd=0.1`, `rank=0.1`, `kcov=0.1`,
  default `qr_steps=30`, and `m_inducing=40`.  Compared free qR mean,
  shared-SVD qR mean, shared-SVD plus `ortho0.1`, and no-qR raw self-CEM2.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_svd_lh_seed0_20260518/`.
  Smoke check:
  `experiments/synthetic/uncertain_w_cem_qr_svd_lh_smoke_20260518/`.
- **Headline result (LH seed0, test=40)**:

  | variant | RMSE | CRPS | Cov90 | Width90 | qR ortho penalty | qR SVD scale range |
  |---|---:|---:|---:|---:|---:|---:|
  | hybrid anchor, free qR mean | 644.55 | 288.31 | 0.750 | 270.47 | 0.01446 | - |
  | hybrid anchor, shared-SVD qR mean | **642.10** | **284.09** | **0.800** | 460.83 | 0.000407 | 0.961-1.041 |
  | shared-SVD qR mean + `ortho0.1` | 642.10 | **284.09** | **0.800** | 460.83 | 0.000407 | 0.961-1.041 |
  | no-qR raw self-CEM2 | 735.12 | 363.17 | 0.700 | 253.73 | - | - |

  The structured shared-SVD mean is a cleaner intervention than the previous
  soft penalty: it reduces the qR orthogonality diagnostic by about `35x`
  (`0.01446 -> 0.000407`) and modestly improves RMSE/CRPS on this 40-anchor
  LH slice.  The tradeoff is calibration width: `Width90` increases from
  `270.47` to `460.83`, so the CRPS gain may be partly uncertainty-driven.
  Adding the Frobenius penalty on top of shared-SVD has no measurable effect,
  because the structured mean is already near semi-orthogonal.
- **Status**:
  Completed one-seed, 40-anchor LH diagnostic.  Superseded by
  `uncertain_w_cem_qr_svd_lh_full_seed0_20260518`, where the shared-SVD gain
  did not survive on all 200 seed0 test anchors.

## 2026-05-18 - `uncertain_w_cem_qr_ortho_lh_seed0_20260518`

- **Question**: Does adding a direct semi-orthogonality penalty to the learned
  q(R) M-step reduce the rotation/scale redundancy in self-CEM/qR and improve
  the LH hybrid-anchor row?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation**:
  Added `UncertainWQRMstepConfig.ortho_weight`,
  `ortho_target_scale`, and `ortho_normalize`.  When enabled, the q(R) M-step
  adds `ortho_weight * mean_l ||R_l R_l^T - target_scale I||_F^2`
  (normalized by `Q^2` by default) to the loss and records orthogonality
  diagnostics/gradient norms.  The synthetic runner accepts `_ortho<lambda>`
  in qR variants, e.g.
  `qr_alt_hya0.25_refresh10_self_cem2_nstd0.1_rank0.1_ortho0.1_kcov0.1`,
  plus CLI defaults `--qr_ortho_weight`, `--qr_ortho_target_scale`, and
  `--qr_ortho_normalize`.
- **Settings / provenance**:
  Paper-style LH `lh_q5_train1000` via `_generate_paper_data` (LH base
  generator with RFF expansion, latent dimension `d=5`, observed dimension
  `H=30`, learned `Q=5`, train/test `1000/200`, train-only X standardization
  and train-only y standardization inside the self-CEM head).  Seed `0`,
  evaluated the first `40` test anchors.  Hybrid-anchor qR used
  `qr_train_anchor_count=100`, `kmeans_yvar` anchors,
  `qr_train_n_neighbors=35`, raw retrieval pool `200`, `qr_outer_cycles=4`,
  `refresh10`, self-CEM2, `nstd=0.1`, `rank=0.1`, `kcov=0.1`,
  default `qr_steps=30`, and `m_inducing=40`.  Compared no orthogonality
  penalty, `ortho0.1`, `ortho1.0`, and the no-qR raw self-CEM2 baseline.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_ortho_lh_seed0_20260518/`.
  Smoke check:
  `experiments/synthetic/uncertain_w_cem_qr_ortho_lh_smoke_20260518/`.
- **Headline result (LH seed0, test=40)**:

  | variant | RMSE | CRPS | Cov90 | Width90 | qR ortho penalty | qR ortho frob |
  |---|---:|---:|---:|---:|---:|---:|
  | hybrid anchor, no ortho | **644.55** | 288.31 | 0.750 | 270.47 | 0.01446 | 0.5916 |
  | hybrid anchor, `ortho0.1` | 644.67 | **288.05** | 0.750 | 266.23 | 0.01394 | 0.5809 |
  | hybrid anchor, `ortho1.0` | 682.90 | 319.41 | 0.750 | 260.42 | 0.01074 | 0.5089 |
  | no-qR raw self-CEM2 | 735.12 | 363.17 | 0.700 | 253.73 | - | - |

  The penalty behaves mechanically as intended: increasing the weight reduces
  the normalized `||R_l R_l^T - I||_F^2` diagnostic and supplies a nonzero
  qR orthogonality gradient.  However, `ortho0.1` is essentially metric-neutral
  (`+0.12` RMSE, `-0.25` CRPS versus no-ortho), while `ortho1.0` hurts both
  RMSE and CRPS.  This quick LH check does not support expanding the direct
  Frobenius penalty as a mainline improvement without a better scale/target or
  a true structured parameterization.
- **Status**:
  Completed one-seed, 40-anchor LH diagnostic.  Keep the implementation as an
  opt-in diagnostic/control.  Next follow-up, if pursued, should tune
  target scale or implement the proposed structured SVD/Cayley q(R) mean rather
  than simply increasing this penalty.

## 2026-05-18 - `graph_local_model_screen_dataset_b_seed0_20260518`

- **Question**: On the first graph-local hard-jump pilot, how do XGBoost, DKL,
  LMJGP, and fixed-neighborhood self-CEM compare?  For DJGP-style local methods,
  does graph-defined neighborhood retrieval beat feature-kNN retrieval?
- **Runner**:
  `experiments/synthetic/run_graph_local_model_screen.py`.
- **Settings / provenance**:
  New graph-local synthetic family, not a paper L2/LH generator.  Base
  low-dimensional generator is a temporal AR latent state with SBM block means;
  expansion is linear latent-to-observed features; graph is an SBM with
  connected block/ring edges.  Preset `dataset_b`, seed `0`, `n_nodes=300`,
  `n_time=20`, `n_blocks=8`, latent dimension `q=3`, observed dimension
  `D=30`, hard jump enabled, `noise_x=0.5`, `noise_y=0.1`, random train/test
  split with train-only X standardization, subsampled to `max_train=1000` and
  `max_test=80`.  Local methods used `n_neighbors=35`.  LMJGP used
  `train_vi` for `120` steps and `predict_vi` with `MC_num=3`, i.e. sampled
  `W_*` followed by JumpGP-CEM refits.  Self-CEM used PLS `W0`, `2` CEM updates,
  `10` hyperopt steps and `20` gate steps, with neighborhoods fixed after
  feature- or graph-neighborhood retrieval; it did not reselect neighbors by
  `W`.  XGBoost/DKL were run on raw X and graph-augmented features
  `[X, block one-hot, graph-neighbor aggregate X]`.
- **Artifacts**:
  `experiments/synthetic/graph_local_model_screen_dataset_b_seed0_20260518/`.
  Fairness supplement where LMJGP/self-CEM also receive graph-augmented
  features:
  `experiments/synthetic/graph_local_model_screen_dataset_b_seed0_graphfeat_local_20260518/`.
  Smoke/env-check folders created before the main pilot:
  `experiments/synthetic/graph_local_model_screen_smoke_20260518/` and
  `experiments/synthetic/graph_local_model_screen_xgb_envcheck_20260518/`;
  do not treat those as reported results.
- **Headline result (seed0, all 80 test anchors)**:

  | method | RMSE | CRPS | Cov90 | note |
  |---|---:|---:|---:|---|
  | `dkl_graph` | **0.721** | 0.485 | 0.300 | best RMSE, under-covered |
  | `xgb_graph` | 0.765 | **0.455** | 0.613 | best CRPS |
  | `lmjgp_graph` | 0.915 | 0.518 | 0.863 | sampled-W JumpGP-CEM |
  | `lmjgp_feature` | 0.963 | 0.531 | 0.825 | feature-kNN local retrieval |
  | `xgb_raw` | 0.966 | 0.565 | 0.625 | no graph features |
  | `dkl_raw` | 1.004 | 0.722 | 0.125 | no graph features |
  | `selfcem_graph` | 1.025 | 0.586 | 0.688 | fixed graph neighborhoods |
  | `selfcem_feature` | 1.076 | 0.624 | 0.700 | fixed feature neighborhoods |

  Graph information is clearly useful in this pilot.  The graph-feature global
  baselines are strongest overall, but the local DJGP-style comparison also
  moves in the desired direction: `lmjgp_graph` beats `lmjgp_feature` on all
  anchors (`0.915` vs `0.963` RMSE) and on jump anchors (`1.104` vs `1.167`
  RMSE); fixed-neighborhood `selfcem_graph` likewise beats `selfcem_feature`
  overall (`1.025` vs `1.076`) and on jump anchors (`1.246` vs `1.416`).
- **Fairness supplement**:
  Because `xgb_graph`/`dkl_graph` use hand-crafted graph message features
  `[X, block one-hot, graph-neighbor aggregate X]`, an additional run gave the
  same graph-augmented features to LMJGP and self-CEM.  On all anchors,
  `selfcem_feature_graphfeat` reached RMSE/CRPS `0.813/0.467`,
  `lmjgp_feature_graphfeat` `0.838/0.461`, `selfcem_graph_graphfeat`
  `0.852/0.480`, and `lmjgp_graph_graphfeat` `0.954/0.526`.  Thus graph
  features substantially help the local methods and make them competitive with
  `xgb_graph` on CRPS, but `dkl_graph` remains the best RMSE row.  The
  neighborhood comparison becomes mixed: feature-kNN in graph-feature space is
  best on all anchors, while `selfcem_graph_graphfeat` is the best local row on
  jump anchors (`0.890` RMSE / `0.475` CRPS).
- **Status**:
  Completed one-seed pilot.  This supports adding graph-neighborhood retrieval
  to the model screen, but it is not yet the desired DJGP win condition because
  graph-aware global DKL/XGBoost dominate point metrics on this easy seed0
  setting.  Next steps are Dataset E/D stress settings, smaller local sample
  size or higher `D`, and possibly training a graph-aware shared-W objective
  rather than only swapping retrieval neighborhoods.

## 2026-05-18 - `graph_local_stage0_dataset_b_seed0_20260518`

- **Question**: Implement and smoke-test the generator-only Stage-0 diagnostics
  from `docs/graph_local_djgp_experiment_plan.md` before training any
  graph-local DJGP/JGP or baseline models.
- **Runner**:
  `experiments/synthetic/run_graph_local_diagnostics.py`.
- **Settings / provenance**:
  New graph-local synthetic family, not a paper L2/LH generator.  Base
  low-dimensional generator is a temporal AR latent state with SBM block means;
  expansion is linear latent-to-observed features; graph is an SBM with
  connected block/ring edges.  Preset `dataset_b`: `n_nodes=300`, `n_time=20`,
  `n_blocks=8`, latent dimension `q=3`, observed dimension `D=30`,
  train/test sizes are diagnostic masks only (`random`, `future-time`, and
  held-out-region splits), no X/Y standardization in the saved raw data,
  `noise_x=0.5`, `noise_y=0.1`, hard jump enabled, seed `0`.
- **Artifacts**:
  `experiments/synthetic/graph_local_stage0_dataset_b_seed0_20260518/`.
- **Headline result**:
  Generator diagnostics pass the planned Stage-0 sanity checks: `6000`
  node-time samples, jump proportion `0.32`, feature/graph distance
  correlation `0.0046`, feature-kNN/graph-kNN overlap@10 `0.0345` versus
  random reference `0.0334`, signal/noise variance ratio about `89`, and split
  sizes `4239/1761` random, `4200/1800` future-time, `4500/1500`
  held-out-region.
- **Status**:
  Completed generator-only pilot.  Next step is a minimum model screen on
  Dataset A/B/E/D using graph-aware and non-graph baselines; do not treat this
  as a model-performance result.

## 2026-05-18 - `orig_elbo_direct_vi_curve_smoke_20260518`

- **Question**: Track Original ELBO training progress with periodic
  **direct VI predict** (``_predict_direct_variational_consistent``) RMSE/CRPS
  curves instead of a single end-of-run metric.
- **Runner**:
  ``experiments/synthetic/validate_orig_elbo_direct_vi_curve.py``.
- **Settings / provenance**:
  Paper-style LH ``lh_q5_train500``, seed ``0``, ``n_test_eval=40`` (smoke),
  ``num_steps=900``, ``eval_every=150``, ``lr=0.01``, ``m1=2``, ``m2=40``.
- **Artifacts**:
  ``experiments/synthetic/orig_elbo_direct_vi_curve_smoke_20260518/``
  (`curve_metrics.csv`, ``rmse_crps_vs_step.png``, ``summary.json``).
- **Headline result (smoke, direct VI at eval steps 150--900)**:
  RMSE flat ``~810.53``, CRPS flat ``~805.29``; ELBO improves
  ``-24`` → ``-3.0``.  Cov90 ``0`` suggests the direct head uncertainty is
  miscalibrated on this short smoke subset — use ``max_test=200`` for a
  fuller curve before drawing conclusions.
- **Status**:
  Smoke completed.  Re-run with ``--max_test 200`` for the intended full test
  curve.

## 2026-05-17 - `residual_ablation_kcov_predict_fix_lh_smoke_20260517`

- **Question**: Keep ``docs/residual_ablation.md`` Variant A–E aligned with the
  published LH self-CEM2 baseline by applying the same uncertain-W predictive
  variance correction used in ``uncertain_w_cem_predict_from_labels``
  (``correction_weight`` = CLI ``kcov``).  Smoke-test Variant E on LH with a
  DKL global backbone for scalability in ``H=30`` observed dimensions.
- **Runner**:
  ``experiments/synthetic/compare_global_residual_selfcem_lh.py``.
- **Code**: ``djgp/projections/global_residual_selfcem.py`` now passes
  ``correction_weight=float(cfg.kcov)`` into ``_predict_per_anchor_with_extras``
  via ``kernel_vector_covariance_anchor`` (mean diagnostic
  ``mean_kernel_vector_correction``).
- **Settings / provenance**:
  Paper-style LH ``lh_q5_train1000`` via ``_generate_paper_data`` (RFF expansion,
  ``d=5``, ``H=30``, ``Q=5``, train/test ``1000/200``, train-only X/y scaling as
  in the runner).  Seed ``0``, evaluated ``n_test_eval=40``, variant
  ``alt_ensW``, ``alt_outer_cycles=1``, ``ensemble_samples=3``, ``kcov=0.1``
  (default), ``n_neighbors=35``, ``local_noise_std_floor=0.1``, ``cem_updates=2``,
  global backend ``dkl`` with ``global_num_inducing=64``, ``global_epochs=40``,
  ``alt_train_anchor_subsample=200`` (runner default).
- **Artifacts**:
  ``experiments/synthetic/residual_ablation_lh_alt_ensw_dkl_smoke_20260517/``.
- **Headline result (LH seed0, test=40)**:
  ``status=ok``, RMSE ``~750.61``, wall clock ``~363`` s on CUDA (heavy due to
  Variant E local CEM + ensemble + one alternating global refit).
- **Status**:
  Smoke completed; use full ``max_test=200`` and longer epochs when reporting
  against the formal ``self_cem2_nstd0.1_kcov0.1`` LH table.

## 2026-05-17 - `residual_ablation_lh_seed0_dkl_20260517`

- **Question**: After auditing ``experiments/synthetic/compare_global_residual_selfcem_lh.py``
  against ``docs/residual_ablation.md``, does the corrected implementation beat
  the best existing LH seed0 self-CEM log entry at train/test ``1000/200``?
- **Runner**:
  ``experiments/synthetic/compare_global_residual_selfcem_lh.py``.
- **Implementation audit / fix**:
  Variant E (``alt_ensW``) originally sampled deterministic ``W`` but reused the
  same raw-KNN test neighborhood for every sample, so sampled ``W`` could not
  change neighborhoods as described in Section 15 of
  ``docs/residual_ablation.md``.  ``djgp/projections/global_residual_selfcem.py``
  now re-ranks a raw candidate pool (size ``200``) by projected distance for
  each sampled ``W`` before the local self-CEM refit.  The runner/readme text
  was also corrected so ``diagvar`` is described as using per-neighbor
  posterior variances on the local covariance diagonal rather than an
  anchor-mean noise floor.
- **Settings / provenance**:
  Paper-style LH ``lh_q5_train1000`` via ``_generate_paper_data`` (RFF
  expansion, ``d=5``, ``H=30``, ``Q=5``, train/test ``1000/200``, train-only
  X standardization and train-only y standardization inside the residual
  runner).  Seed ``0``.  Variants
  ``diagvar,blockvar,oob_diagvar,alt,alt_ensW`` with the published self-CEM
  defaults ``n_neighbors=35``, ``cem_updates=2``, ``local_noise_std_floor=0.1``,
  ``kcov=0.1``.  Global backend ``dkl`` with ``global_num_inducing=128``,
  ``global_epochs=150``, ``dkl_hidden=64``, ``dkl_latent=8``,
  ``alt_train_anchor_subsample=200``, ``ensemble_samples=5``.
- **Artifacts**:
  ``experiments/synthetic/residual_ablation_lh_seed0_dkl_20260517/``.
- **Headline result (LH seed0, test=200)**:

  | variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | ``oob_diagvar`` | **681.69** | **345.50** | 0.735 | 784.00 |
  | ``diagvar`` | 705.46 | 409.04 | 0.515 | 153.90 |
  | ``blockvar`` | 705.48 | 409.04 | 0.515 | 154.57 |
  | ``alt`` | 741.59 | 409.91 | 0.540 | 153.07 |
  | ``alt_ensW`` | 741.77 | 409.91 | 0.545 | 153.16 |

  Best prior self-CEM-family seed0 log row remains
  ``qr_alt_hya0.25_refresh10_self_cem2_nstd0.1_rank0.1_ens5_kcov0.1`` with
  RMSE ``626.04`` / CRPS ``267.84``; even the strongest residual-ablation row
  here (``oob_diagvar``) is worse by about ``+55.65`` RMSE and ``+77.66`` CRPS.
- **Status**:
  Completed one-seed DKL audit run after the Variant-E neighborhood fix.  The
  current additive global-residual implementation does **not** beat the
  existing best LH seed0 self-CEM result, so do not expand this line before a
  further seed0-only fix materially changes the outcome.

## 2026-05-17 - `dmgp_uncertain_w_cem_head_lh_seed0_20260517`

- **Question**: Starting from a strong global DeepMahalanobisGP fit on the
  paper-style LH seed0 split, can a fixed-W residual uncertain self-CEM head
  improve over direct DMGP by using the DMGP predictive mean/variance as the
  global trend and propagating that variance into local residual self-CEM
  training?  Also, is train-side OOF residualization better than in-sample
  residualization?
- **Runners**:
  - `experiments/synthetic/run_deep_mahalanobis_gp_residual_bundle.py`
  - `experiments/synthetic/compare_dmgp_uncertain_w_cem_head.py`
- **Implementation**:
  Added a two-stage pipeline.  The first stage fits external GPflow-1 DMGP and
  exports direct test prediction, train/test `W` moments, full-fit train
  predictive mean/variance, and optional K-fold OOF train predictive
  mean/variance.  The second stage keeps DMGP `W` fixed, builds raw-X
  neighborhoods, residualizes `y` by DMGP mean, injects DMGP predictive
  variance on the local covariance diagonal, and optimizes only the uncertain
  self-CEM head.
- **Settings / provenance**:
  Paper-style LH `lh_q5_train1000`, generated through the canonical
  `_generate_paper_data` path: LH base generator with RFF expansion, latent
  dimension `d=5`, observed dimension `H=30`, learned `Q=5`, train/test
  `1000/200`, train-only X standardization, and standardized-y training for
  DMGP.  Seed `0`.  DMGP used `m_u=50`, `m_v=25`, `steps1=300`, `steps2=700`,
  `lr=0.01`.  The residual self-CEM head used raw-X `n_neighbors=35`,
  `cem_updates=2`, `kcov=0.1`, `local_noise_std_floor=0.1`, `hyper_steps=10`,
  and `gate_steps=20`.  OOF used `3` folds.
- **Artifacts**:
  - `experiments/synthetic/dmgp_residual_bundle_lh_seed0_20260517/`
  - `experiments/synthetic/dmgp_uncertain_w_cem_head_lh_seed0_20260517/`
- **Headline result (LH seed0, test=200)**:

  | variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | direct DMGP | 617.74 | 302.93 | 0.765 | 677.62 |
  | DMGP + residual uncertain self-CEM (full-fit train residuals) | 617.61 | 302.78 | 0.765 | 698.50 |
  | DMGP + residual uncertain self-CEM (3-fold OOF train residuals) | **615.40** | **300.93** | **0.770** | 700.26 |

  The fixed-W residual head gives a small but real seed0 gain over direct DMGP:
  about `-2.34` RMSE and `-2.00` CRPS for the OOF variant.  The gain is much
  smaller than the direct-DMGP margin over earlier self-CEM-family methods,
  but it is directionally positive and OOF residualization is better than
  in-sample residualization.
- **Status**:
  Completed one-seed pilot.  This is worth a cautious multi-seed follow-up if
  we want to know whether the small OOF gain is robust, but the current effect
  size is modest enough that it should not yet be treated as a new mainline
  winner.

## 2026-05-17 - `uncertain_w_cem_qr_hybrid_ensemble_lh_5seeds_20260517`

- **Question**: Expand the positive LH seed0 sampled-W CEM ensemble signal to
  5 seeds before doing any larger tuning.  Compare the best seed0 candidate
  `hybrid anchor lambda=0.25 + ens5` against its moment-matched uncertain-W
  counterpart and the no-qR raw self-CEM2 baseline.
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Settings / provenance**:
  Paper-style LH synthetic setting `lh_q5_train1000`, generated through
  `_generate_paper_data`: LH base generator with RFF expansion, latent
  dimension `d=5`, observed dimension `H=30`, `Q_true=5`, learned `Q=5`,
  train/test `1000/200`, train-only X standardization, and train-only y
  standardization inside the self-CEM head.  Seeds `0--4`, `n_neighbors=35`,
  `m_inducing=40`, 100 `kmeans_yvar` train anchors,
  `qr_train_n_neighbors=35`, raw retrieval pool `200`, `qr_outer_cycles=4`,
  `refresh10`, self-CEM2, `nstd=0.1`, `rank=0.1`, `kcov=0.1`, ensemble
  samples `N=5`.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_hybrid_ensemble_lh_5seeds_20260517/`.
- **Headline result (LH seeds 0--4, test=200)**:

  | variant | RMSE mean ± std | CRPS mean ± std | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | no-qR raw self-CEM2 | **595.68 ± 120.13** | **252.28 ± 79.55** | 0.787 | 238.43 |
  | hybrid anchor `lambda=0.25` + sampled-W CEM ens5 | 604.53 ± 97.23 | 256.40 ± 69.91 | 0.783 | 274.36 |
  | hybrid anchor `lambda=0.25` moment uncertain-W | 611.81 ± 97.89 | 260.54 ± 69.78 | 0.784 | 263.90 |
  | original mean-W JumpGP CEM | 616.69 ± 103.09 | 272.39 ± 76.82 | 0.362 | 166.34 |

- **Interpretation**:
  The sampled-W deterministic CEM ensemble improves the hybrid qR retrieval
  row over the moment-matched uncertain-W head (`611.81/260.54` ->
  `604.53/256.40`), so the seed0 ensemble effect is not purely accidental.
  However, it still does not beat the simpler no-qR raw self-CEM2 baseline on
  5 seeds (`595.68/252.28`), and earlier 5-seed train-anchor raw qR was also
  around `590.71/250.42`.  The current conclusion is therefore: ensemble helps
  the hybrid learned-W line, but learned hybrid retrieval is still not the
  best robust LH method.
- **Status**:
  Completed 5 seeds.  Do not claim the hybrid ensemble as a new 5-seed winner;
  use it as evidence that final sampled-W CEM prediction helps, while the
  retrieval/training objective remains the bottleneck.

## 2026-05-17 - `uncertain_w_cem_qr_hybrid_ensemble_lh_seed0_20260517`

- **Question**: Test whether the strong LH seed0 hybrid retrieval signal is
  missing LMJGP's main advantage: final-time sampled-W ensembling.  Training
  remains the retrieval-aware hybrid self-CEM/qR flow, but final prediction
  samples W from the learned q(W), runs deterministic-W self-CEM for each
  sample, and moment-matches the sampled means/variances.  This is CEM with
  `Var(W)=0` per sample, not the uncertain-W moment-matched head.
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation changes**:
  Added optional qR variant suffix `_ens<N>` after `_rank...`, e.g.
  `qr_alt_hya0.25_refresh10_self_cem2_nstd0.1_rank0.1_ens5_kcov0.1`.
  The final test anchors are expanded to `N * n_test` rows, sampled W is
  passed as deterministic `W_mu_anchor` with zero `W_var_anchor`, self-CEM is
  refit on the expanded batch, and the sampled predictive moments are
  mixture-matched.  Added ensemble timing and within/between/total sigma
  diagnostics.
- **Settings / provenance**:
  Paper-style LH synthetic setting `lh_q5_train1000`, generated through
  `_generate_paper_data`: LH base generator with RFF expansion, latent
  dimension `d=5`, observed dimension `H=30`, `Q_true=5`, learned `Q=5`,
  train/test `1000/200`, train-only X standardization, and train-only y
  standardization inside the self-CEM head.  Seed `0`, `n_neighbors=35`,
  `m_inducing=40`, 100 `kmeans_yvar` train anchors,
  `qr_train_n_neighbors=35`, raw retrieval pool `200`, `qr_outer_cycles=4`,
  `refresh10`, self-CEM2, `nstd=0.1`, `rank=0.1`, `kcov=0.1`, ensemble
  samples `N=5`.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_hybrid_ensemble_lh_seed0_20260517/`
  and smoke check
  `experiments/synthetic/uncertain_w_cem_qr_hybrid_ensemble_smoke_20260517/`.
- **Headline result (LH seed0, test=200)**:

  | variant | RMSE | CRPS | Cov90 | Width90 | raw overlap |
  |---|---:|---:|---:|---:|---:|
  | hybrid anchor `lambda=0.25` + sampled-W CEM ens5 | **626.04** | **267.84** | 0.805 | 386.95 | 0.915 |
  | hybrid anchor `lambda=0.25` moment uncertain-W | 643.71 | 275.12 | 0.800 | 353.56 | 0.915 |
  | hybrid anchor `lambda=0.40` + sampled-W CEM ens5 | 641.41 | 277.93 | 0.805 | 356.30 | 0.868 |
  | hybrid anchor `lambda=0.40` moment uncertain-W | 642.19 | 275.22 | 0.800 | 327.77 | 0.868 |
  | no-qR raw self-CEM2 | 664.32 | 292.17 | 0.775 | 267.01 | - |
  | original mean-W JumpGP CEM | 684.30 | 322.57 | 0.530 | 247.96 | - |

- **Interpretation**:
  Sampled-W deterministic CEM ensembling is a real positive signal for the
  conservative hybrid retrieval setting.  At `lambda=0.25`, ens5 improves the
  moment head by `17.67` RMSE and `7.28` CRPS on LH seed0, and is the best
  current seed0 hybrid/self-CEM row.  The gain is not uniform: `lambda=0.40`
  gets similar RMSE but worse CRPS than its moment head, suggesting the
  ensemble helps most when retrieval remains highly raw-local (`overlap=0.915`)
  and W only makes a mild correction.
- **Status**:
  Completed LH seed0.  This passes the user's seed0 threshold for considering
  further work, but should still be treated as a pilot because the earlier
  non-ensemble hybrid line did not generalize across 5 seeds.

## 2026-05-17 - `lmjgp_uncertain_w_cem_head_lh_seed0_20260517`

- **Question**: Compare the reverse head swap: train the original transductive
  LMJGP q(W), then replace its final sampled-W JumpGP-CEM head with the
  moment-matched uncertain-W self-CEM head.
- **Runner**:
  `experiments/synthetic/compare_lmjgp_uncertain_w_cem_head_lh.py`.
- **Implementation changes**:
  Added a narrow runner that trains LMJGP via
  `compare_lmjgp_prediction_heads_paper._train_lmjgp_state`, evaluates the
  original `predict_vi` sampled-W JumpGP-CEM head, and then evaluates two
  uncertain-W self-CEM heads on raw-X neighborhoods: raw q(W) moments and a
  row-normalized moment approximation.  The CEM head standardizes y using
  train-only statistics and rescales metrics to original y units.
- **Settings / provenance**:
  Paper-style LH synthetic setting `lh_q5_train1000`, generated through
  `_generate_paper_data`: LH base generator with RFF expansion, latent
  dimension `d=5`, observed dimension `H=30`, `Q_true=5`, learned `Q=5`,
  train/test `1000/200`, train-only X standardization.  LMJGP was trained on
  raw y as in the baseline, with `m1=2`, `m2=40`, `n_neighbors=35`,
  `lmjgp_steps=300`, `MC_num=5`, seed `0`.  The uncertain-W CEM head used
  train-only y standardization, self-CEM2, `hyper_steps=10`, `gate_steps=20`,
  `kcov=0.1`.
- **Artifacts**:
  `experiments/synthetic/lmjgp_uncertain_w_cem_head_lh_seed0_20260517/` and
  smoke check
  `experiments/synthetic/lmjgp_uncertain_w_cem_head_smoke_20260517/`.
- **Headline result (LH seed0, test=200)**:

  | variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | LMJGP sampled-W JumpGP-CEM refit (`MC=5`) | 753.14 | 5.23e12 | 0.780 | 7.36e13 |
  | LMJGP q(W) -> uncertain self-CEM, raw moments | 850.41 | 447.25 | 0.725 | 553.33 |
  | LMJGP q(W) -> uncertain self-CEM, row-normalized moments | 840.31 | inf | 1.000 | inf |

- **Interpretation**:
  LMJGP-trained q(W) is not a good moment-matched uncertain-W CEM posterior in
  this seed0 run.  The learned q(W) mean nearly collapses
  (`mu_w_norm_mean ~= 9e-13`) while posterior standard deviation remains
  around `0.316`, so the original LMJGP head mostly benefits from sampled,
  row-normalized random projections plus CEM refits.  Passing those moments
  directly into the uncertain-W CEM head either performs poorly (raw moments)
  or explodes under the crude row-normalized moment approximation.
- **Status**:
  Completed LH seed0.  Do not expand this LMJGP -> uncertain-W CEM head as
  currently implemented; the useful direction is instead the hybrid
  self-CEM/qR training with sampled-W deterministic CEM ensemble at inference.

## 2026-05-17 - `projected_dataset_transductive_selfcem_lh_seed0_20260517`

- **Question**: Rename the ambiguous "global-Z" line as projected-dataset
  self-CEM, and replace the training-anchor supervised objective with a
  transductive-X objective: after each M-step, map train/test to
  `z_i = W(x_i)x_i`, fit uncertain-W/self-CEM on test anchors' projected
  neighborhoods, then update W using the fixed projected neighborhoods and
  CEM partitions.  This uses `X_test` but not `y_test` during training.
- **Runner**:
  `experiments/synthetic/compare_projected_dataset_selfcem_l2_lh.py`
  (thin entrypoint over
  `experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py`).
- **Implementation changes**:
  Added `pd_*` projected-dataset variants:
  `pd_raw_selfcem`, `pd_pca_selfcem`, `pd_pls_selfcem`,
  `pd_static_transductive`, `pd_lowrank_transductive`,
  `pd_lowrank_transductive_shuffled`, and `pd_lowrank_random`.  The
  transductive M-step fits test-anchor CEM states in projected space, then
  updates `W_theta` with fixed-label local GP marginal likelihood on selected
  inlier training responses plus a small fixed-gate label loss.  No test
  response is used by the transductive objective.
- **Settings / provenance**:
  Paper-style LH synthetic setting `lh_q5_train1000`, generated through
  `_generate_paper_data`: LH base generator with RFF expansion, latent
  dimension `d=5`, observed dimension `H=30`, `Q_true=5`, learned `Q=5`,
  train/test `1000/200`, train-only X standardization, and train-only y
  standardization inside the self-CEM head.  Seed `0`, `n_neighbors=35`,
  all 200 test anchors used transductively, `epochs=4`, `inner_steps=20`,
  `anchor_batch=64`, `lr_w=1e-3`, `weight_decay=1e-4`, `w_l2=1e-4`,
  `lowrank_r=8`, `hidden=64`, `init_mode=pls`, self-CEM2,
  `self_cem_hyper_steps=10`, `self_cem_gate_steps=20`,
  `self_cem_noise_std_floor=0.1`, `self_cem_kcov=0.1`,
  `transductive_gate_weight=0.1`.
- **Artifacts**:
  `experiments/synthetic/projected_dataset_transductive_selfcem_lh_seed0_20260517/`
  and smoke check
  `experiments/synthetic/projected_dataset_transductive_selfcem_smoke_20260517/`.
- **Headline result (LH seed0, test=200)**:

  | variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | previous hybrid anchor `lambda=0.40` | **642.19** | **275.22** | 0.800 | 327.77 |
  | previous hybrid anchor `lambda=0.25` | 643.71 | 275.12 | 0.800 | 353.56 |
  | previous no-qR raw self-CEM2 | 664.32 | 292.17 | 0.775 | 267.01 |
  | `pd_static_transductive` | 687.38 | 315.21 | 0.750 | 260.38 |
  | `pd_pls_selfcem` | 700.34 | 329.04 | 0.735 | 303.97 |
  | `pd_lowrank_random` | 700.36 | 329.10 | 0.745 | 306.41 |
  | `pd_lowrank_transductive` | 793.51 | 403.29 | 0.685 | 249.47 |
  | `pd_lowrank_transductive_shuffled` | 826.05 | 439.00 | 0.670 | 271.27 |
  | `pd_raw_selfcem` | 908.69 | 511.97 | 0.625 | 146.25 |
  | `pd_pca_selfcem` | 906.08 | 516.24 | 0.645 | 300.53 |

- **Interpretation**:
  The transductive projected-dataset objective improves over the static PLS
  projected self-CEM baseline (`700.34/329.04` -> `687.38/315.21`) when W is a
  single global/static projection, but it still clearly loses to the current
  best LH seed0 uncertain-W/self-CEM/qR candidates around `642--664` RMSE and
  `275--292` CRPS.  The input-dependent lowrank transductive W remains much
  worse than its PLS/random control, though it beats the shuffled lowrank row.
  The useful signal is therefore not competitive with the existing main line.
- **Status**:
  Completed LH seed0.  The attempted 5-seed expansion was immediately stopped
  after user interruption; do not use any partial 5-seed artifacts.  Do not
  expand this transductive projected-dataset version to 5 seeds unless a
  follow-up seed0 setting first beats the existing self-CEM/qR baselines.

## 2026-05-17 - `global_z_selfcem_supervised_lh_seed0_20260517`

- **Question**: Test the clarified global-Z self-CEM alternating idea: after a
  W M-step, reproject the whole dataset as `z_i = W(x_i) x_i`, select new
  neighborhoods in global `Z` space, refit the current uncertain-W self-CEM
  head on those neighborhoods, then use the refreshed neighborhoods and CEM
  partitions in the next supervised anchor predictive M-step plus
  regularization.
- **Runner**:
  `experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py`.
- **Implementation changes**:
  Added self-CEM global-Z variants distinct from the older surrogate-only
  global-Z rows: `raw_selfcem`, `pca_selfcem`, `pls_selfcem`,
  `static_selfcem_supervised`, `lowrank_selfcem_supervised`,
  `lowrank_selfcem_shuffled`, and `lowrank_selfcem_random`.  The new training
  loop alternates: global `Z`-KNN refresh, self-CEM refit in `Z` space with an
  identity deterministic W inside the head, then a fixed-neighborhood /
  fixed-label M-step minimizing leave-anchor-out GP predictive NLL through
  `W_theta`.
- **Settings / provenance**:
  Paper-style LH synthetic setting `lh_q5_train1000`, generated through
  `_generate_paper_data`: LH base generator with RFF expansion, latent
  dimension `d=5`, observed dimension `H=30`, `Q_true=5`, learned `Q=5`,
  train/test `1000/200`, train-only X standardization, and train-only y
  standardization inside the self-CEM head.  Seed `0`, `n_neighbors=35`,
  100 `kmeans_yvar` train anchors, `anchor_batch=64`, `epochs=4`,
  `inner_steps=20`, `lr_w=1e-3`, `weight_decay=1e-4`, `w_l2=1e-4`,
  `lowrank_r=8`, `hidden=64`, `init_mode=pls`, self-CEM2,
  `self_cem_hyper_steps=10`, `self_cem_gate_steps=20`,
  `self_cem_noise_std_floor=0.1`, `self_cem_kcov=0.1`.
- **Artifacts**:
  `experiments/synthetic/global_z_selfcem_supervised_lh_seed0_20260517/` and
  smoke check
  `experiments/synthetic/global_z_selfcem_supervised_smoke_20260517/`.
- **Headline result (LH seed0, test=200)**:

  | variant | RMSE | CRPS | Cov90 | Width90 | train raw overlap |
  |---|---:|---:|---:|---:|---:|
  | PLS global-Z self-CEM | **700.34** | **329.04** | 0.735 | 303.97 | - |
  | lowrank random / PLS init | 700.36 | 329.10 | 0.745 | 306.41 | - |
  | static self-CEM supervised W | 744.15 | 356.20 | 0.735 | 259.76 | 0.434 |
  | lowrank self-CEM supervised W | 801.86 | 411.39 | 0.725 | 547.08 | 0.393 |
  | lowrank self-CEM shuffled W | 896.03 | 505.43 | 0.655 | 432.03 | 0.419 |
  | PCA global-Z self-CEM | 906.08 | 516.24 | 0.645 | 300.53 | - |
  | raw-X self-CEM identity head | 908.69 | 511.97 | 0.625 | 146.25 | - |
  | raw-X JumpGP CEM | 931.63 | 544.15 | 0.270 | 23.25 | - |

- **Interpretation**:
  This exact self-CEM alternating global-Z version does not pass the seed0
  expansion gate.  The best row is simply PLS global-Z self-CEM, essentially
  tied with the lowrank random/zero-residual control.  Both learned W M-step
  rows make performance worse than the PLS initialization.  The input-dependent
  learned row beats the shuffled control (`801.86/411.39` vs `896.03/505.43`),
  so some spatial binding signal exists, but it is not useful enough for
  regression neighborhoods and is dominated by the static PLS geometry.  This
  suggests the supervised M-step is moving global neighborhoods too far from
  the good PLS-local geometry rather than improving it.
- **Status**:
  Completed LH seed0 only.  Do not expand this self-CEM global-Z alternating
  version to 5 seeds unless a follow-up seed0 variant first beats the PLS
  global-Z self-CEM and lowrank-random controls.

## 2026-05-16 - `global_z_supervised_jgp_lh_seed0_20260516`

- **Question**: Following the alternative direction in
  `docs/global_z_supervised_jgp_plan_cn.md`, can the entire dataset be
  reprojected through ``z_i = W_theta(x_i) x_i`` with a JGP-supervised anchor
  predictive objective, so that JumpGP runs in the resulting global ``Z``-space
  KNN instead of raw-X KNN or hybrid raw/W reranking?  Is the supervised global
  ``Z``-space better than PCA/PLS/DMGP static-projection baselines, and does
  the spatial binding ``x -> W(x)`` survive controls?
- **Runner**:
  `experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py`.
- **Implementation changes**:
  New self-contained runner.  Architectures: `static` (single global
  ``W_0 \in R^{Q x D}``) and `lowrank` (``W_0 + C_theta(x) V^T`` with a small
  MLP for ``C_theta`` and ``V`` from PCA-residual basis).  Training uses
  alternating E/M cycles: E-step refreshes leave-anchor-out ``Z``-KNN, M-step
  takes inner gradient steps on a batched differentiable local RBF GP NLL
  surrogate over the current neighborhoods.  Final test prediction uses
  JumpGP CEM in the learned ``Z``-space (no GP surrogate at inference).
  Controls: `raw_jgp`, `pca_jgp`, `pls_jgp`, `static_supervised`,
  `lowrank_supervised`, `lowrank_shuffled` (trained then inputs permuted to
  break the ``x -> W(x)`` map at inference), `lowrank_random` (no training,
  zero-initialised residual MLP so it collapses to PLS init).
- **Settings / provenance**:
  Paper-style LH synthetic setting `lh_q5_train1000`, generated through
  `_generate_paper_data`: LH base generator with RFF expansion, latent
  dimension `d=5`, observed dimension `H=30`, `Q_true=5`, learned `Q=5`,
  train/test `1000/200`, train-only X standardization (no y standardization).
  Seed `0`, `n_neighbors=35`, all 1000 training points as anchors,
  `anchor_batch=128`, `epochs=100` (outer Z-KNN refreshes), `inner_steps=20`
  (gradient steps per refresh), `lr_w=1e-3`, `lr_gp=1e-2`,
  `weight_decay=1e-4`, `w_l2=1e-4`, `lowrank_r=8`, `hidden=64`,
  `init_mode=pls`, GP surrogate initial `(ell, sf, sn)=(1.0, 1.0, 0.1)`.
- **Artifacts**:
  `experiments/synthetic/global_z_supervised_jgp_lh_seed0_20260516/` and smoke
  `experiments/synthetic/global_z_supervised_jgp_smoke_20260516/`.
- **Headline result (LH seed0, test=200)**:

  | variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | static_supervised (global linear W_0, JGP-supervised) | **673.32** | **307.70** | 0.575 | 170.04 |
  | dmgp_global_z_mean_jgp (LH seed0, prior run) | 737.67 | 350.86 | 0.510 | 18.87 |
  | lowrank_random (PLS init, no training) | 765.20 | 391.70 | 0.490 | 261.51 |
  | pls_jgp | 766.68 | 393.27 | 0.490 | 256.13 |
  | lowrank_supervised | 849.74 | 459.77 | 0.265 | 11.64 |
  | lowrank_shuffled | 883.74 | 498.38 | 0.575 | 394.94 |
  | pca_jgp | 907.29 | 525.32 | 0.545 | 242.65 |
  | raw_jgp | 932.30 | 544.04 | 0.275 | 23.94 |

  For external context (different prediction head, not directly comparable):
  the best previously logged LH seed0 uncertain-W self-CEM / qR variants sit
  at `RMSE \approx 642--664`, `CRPS \approx 275--292`, `Cov90 \approx 0.78--0.80`.
- **Interpretation**:
  Supervised global ``Z = W_theta(x) x`` is a real direction on LH seed0.
  The simplest variant, a single global linear ``W_0`` trained end-to-end with
  the leave-anchor-out RBF GP NLL surrogate, beats every static-projection
  baseline by a large margin: it improves over PLS by ~93 RMSE / ~86 CRPS and
  over DMGP's global pointwise mapping by ~64 RMSE / ~43 CRPS.  This is the
  first clean positive signal for the "use ``W(x)x`` as the JGP input space"
  hypothesis after several rounds of hybrid retrieval and W-structure
  ablations failed to stabilize.  The lowrank input-dependent variant
  overfits: it drives training NLL to ~0.23 vs static's ~5.0, but test RMSE
  goes up to 849 because the per-point ``C_theta(x)`` perturbation concentrates
  Z neighbourhoods, collapses local JumpGP uncertainty (Width90=11.64), and
  damages predictive coverage (Cov90=0.265).  The shuffled-W control beats
  random JGP/PCA but loses to the learned variant (RMSE 884 vs 850, CRPS 498
  vs 460), so the ``x -> W(x)`` spatial binding is non-trivially present but
  not the dominant gain.  ``lowrank_random`` confirms the zero-init residual
  collapses to PLS at training step 0 (RMSE 765.20 vs PLS 766.68).
- **Status**:
  Completed seed0.  **Do not** launch the 5-seed expansion before adding
  three follow-ups for seed0:
  1. Reduce lowrank capacity (smaller `hidden`, larger `weight_decay`, or
     explicit residual L2) to recover the static gain — current overfitting is
     interpretable.
  2. Inflate the JumpGP predictive width to fix the Width90=11.64 / Cov90=0.265
     collapse on `lowrank_supervised` (e.g. report alternative head or add
     local noise floor on the JGP side).
  3. Run on `l2_q2_train1000` seed0 to confirm the direction generalises to
     the second paper synthetic family before any 5-seed table.
- **Follow-ups**:
  Do not return to hybrid raw+W retrieval (`compare_uncertain_w_cem_l2_lh.py`)
  until the global-Z supervised direction has either been falsified on
  `l2_q2_train1000` seed0 or its `static_supervised` win has been
  stress-tested across at least one additional setting.  Both directions
  should not run in parallel because they target the same neighbourhood-shape
  question and would dilute the signal.

## 2026-05-17 - `uncertain_w_cem_qr_rankhybrid_lh_seed0_20260517`

- **Question**: Following
  `docs/neighborhood_retrieval_hypothesis_experiment_plan_cn.md`, does
  rank-normalized hybrid retrieval stabilize the learned-W/raw-X mixing that
  failed to generalize in the previous distance-hybrid 5-seed check?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation changes**:
  Added rank-hybrid retrieval variants `qr_alt_hyra<lambda>_*` and
  `qr_alt_hyrpw<lambda>_*`.  Inside each fixed raw candidate pool, raw and W
  distances are converted to fractional ranks before mixing:
  `d = lambda rank(d_W) + (1 - lambda) rank(d_raw)`.  Added diagnostics
  `test_wknn_rank_mixing` and `qr_train_wknn_rank_mixing_final`.
- **Settings / provenance**:
  Paper-style LH synthetic setting `lh_q5_train1000`, generated through
  `_generate_paper_data`: LH base generator with RFF expansion, latent
  dimension `d=5`, observed dimension `H=30`, `Q_true=5`, learned `Q=5`,
  train/test `1000/200`, train-only X standardization and local-head y
  standardization.  Seed `0`, `n_neighbors=35`, `m_inducing=40`, 100
  `kmeans_yvar` train anchors, fixed raw candidate pool 200,
  `qr_outer_cycles=4`, `refresh_hyper_steps=10`, `refresh_gate_steps=20`,
  `qr_beta_kl=0.05`, self-CEM2, `nstd=0.1`, `kcov=0.1`, and rank weight
  `0.1` for rank-hybrid variants.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_rankhybrid_lh_seed0_20260517/`
  and smoke check
  `experiments/synthetic/uncertain_w_cem_qr_rankhybrid_smoke_20260517/`.
- **Headline result (LH seed0, test=200)**:

  | variant | RMSE | CRPS | Cov90 | Width90 | raw overlap |
  |---|---:|---:|---:|---:|---:|
  | previous distance hybrid anchor lambda=0.40 | **642.19** | **275.22** | 0.800 | 327.77 | 0.868 |
  | rank hybrid anchor lambda=0.40 | 665.23 | 287.34 | 0.785 | 359.59 | 0.869 |
  | no-qR raw self-CEM2 | 664.32 | 292.17 | 0.775 | 267.01 | - |
  | train-anchor raw qR | 666.15 | 296.90 | 0.790 | 372.89 | - |
  | rank hybrid anchor lambda=0.25 | 703.20 | 322.94 | 0.765 | 332.98 | 0.910 |
  | rank hybrid anchor lambda=0.60 | 742.07 | 347.61 | 0.735 | 404.83 | 0.813 |
  | rank hybrid anchor lambda=0.75 | 735.19 | 341.50 | 0.745 | 348.36 | 0.760 |
  | original mean-W CEM | 684.30 | 322.57 | 0.530 | 247.96 | - |

- **Interpretation**:
  Rank-hybrid anchor retrieval gives at most a modest seed0 CRPS improvement
  over raw qR/no-qR baselines at `lambda=0.40`, but it does not improve RMSE
  and is much weaker than the previous distance-hybrid seed0 result.  Larger
  rank-hybrid weights again hurt strongly.  This is not enough positive signal
  to justify the planned 5-seed expansion.
- **Status**:
  Completed seed0 screen only.  Do not launch the 5-seed rank-hybrid expansion
  unless a follow-up seed0 variant beats the raw baselines and the previous
  distance-hybrid seed0 candidate clearly.

## 2026-05-17 - `uncertain_w_cem_qr_wstruct_lh_seed0_20260517`

- **Question**: Do the W-structure hypotheses in
  `docs/w_learning_ablation_plan_cn.md` improve the current best LH
  uncertain-W self-CEM candidate
  `qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1`?  Specifically: low-rank
  residual q(R), overcomplete residual bases, residual-column group penalty,
  learned basis scales, and pairwise warm-start W0.
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation changes**:
  Added `pairwise` W0, `pls_pca` / `overcomplete` residual bases,
  `_rg<lambda>` variant parsing for residual-column group penalty, optional
  `_scale` learned low-rank basis scales, and diagnostics for residual usage
  and learned scale.  The existing low-rank residual moment path was kept as a
  full row covariance `V diag(var(C_q)) V^T`; no diagonalization bug was found
  there.  Added unit coverage in `tests/test_uncertain_w_jgp.py`.
- **Settings / provenance**:
  Paper-style LH synthetic setting `lh_q5_train1000`, generated through
  `_generate_paper_data`: LH base generator with RFF expansion, latent
  dimension `d=5`, observed dimension `H=30`, `Q_true=5`, learned `Q=5`,
  train/test `1000/200`, train-only X standardization and local-head y
  standardization.  Seed `0`, `n_neighbors=35`, `m_inducing=40`,
  `qr_outer_cycles=4`, `refresh_hyper_steps=10`, `refresh_gate_steps=20`,
  `qr_beta_kl=0.05`, self-CEM2, `nstd=0.1`, `kcov=0.1`.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_wstruct_lh_seed0_20260517/`
  (PLS W0 + overcomplete V),
  `experiments/synthetic/uncertain_w_cem_qr_wstruct_pairw0_lh_seed0_20260517/`
  (pairwise W0 + overcomplete V), and
  `experiments/synthetic/uncertain_w_cem_qr_wstruct_smoke_20260517/`.
- **Headline result (LH seed0, PLS W0 + overcomplete V)**:

  | variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | lowrank r4 group `1e-3` shuffled | 662.08 | **291.29** | 0.770 | 236.23 |
  | lowrank r4 group `1e-3` prior | 667.46 | 291.59 | 0.780 | 220.97 |
  | current best self-CEM/qR baseline | **658.55** | 291.74 | 0.765 | 225.94 |
  | lowrank r2 learned | 680.03 | 307.43 | 0.750 | 216.17 |
  | lowrank r8 + scale + group `1e-3` learned | 677.65 | 308.73 | 0.740 | 200.36 |
  | lowrank r4 learned | 689.04 | 314.80 | 0.730 | 199.39 |
  | lowrank r4 group `1e-3` learned | 689.40 | 315.01 | 0.730 | 199.36 |
  | original mean-W CEM | 684.30 | 322.57 | 0.530 | 247.96 |

  Pairwise W0 was much worse overall: original mean-W CEM was
  `RMSE=876.80, CRPS=486.21`; the pairwise-W0 current self-CEM/qR baseline was
  `RMSE=888.05, CRPS=491.03`; learned low-rank variants were also worse
  (`CRPS=497.40` for r4 group `1e-3`, `504.40` for r8+scale).
- **Interpretation**:
  No large improvement was observed, so the planned 5-seed expansion was not
  launched.  The strongest PLS-W0 learned low-rank variants are worse than the
  current self-CEM/qR baseline, and the prior/shuffled low-rank controls are
  slightly better than the learned low-rank rows.  This does not support a
  stable learned spatial W-binding claim for these W-structure changes.  The
  residual group penalty at `1e-4`/`1e-3` barely changes r4 and does not rescue
  it; learned basis scale shrinks the r8 basis mean to about `0.71` but remains
  worse than baseline.
- **Status**:
  Completed seed0 structural check.  Do not promote these W-structure variants
  to 5-seed LH results unless a new W0/V selection rule beats both baseline and
  prior/shuffled controls on seed0 first.

## 2026-05-16 - `uncertain_w_cem_qr_hybrid_anchor_5seeds_20260516`

- **Question**: Does the LH seed0 hybrid-anchor retrieval gain survive a small
  5-seed validation against original CEM, train-anchor raw qR, and no-qR raw
  self-CEM?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Settings / provenance**:
  Paper-style LH synthetic setting `lh_q5_train1000`, generated through the
  repository paper synthetic runner: LH base generator with RFF expansion,
  latent dimension `d=5`, observed dimension `H=30`, `Q_true=5`, learned
  `Q=5`, train/test `1000/200`, train-only X standardization and local-head y
  standardization.  Seeds `0..4`, `n_neighbors=35`, `m_inducing=40`, 100
  `kmeans_yvar` train anchors, fixed raw candidate pool 200, `qr_outer_cycles=4`,
  `refresh_hyper_steps=10`, `refresh_gate_steps=20`, `qr_beta_kl=0.05`,
  self-CEM2, `nstd=0.1`, `kcov=0.1`, and rank weight `0.1` for hybrid variants.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_hybrid_anchor_5seeds_20260516/`.
- **Headline result (LH, 5 seeds, test=200)**:

  | variant | RMSE mean | RMSE std | CRPS mean | CRPS std | Cov90 | Width90 |
  |---|---:|---:|---:|---:|---:|---:|
  | train-anchor raw qR | **590.71** | 123.98 | **250.42** | 79.35 | 0.789 | 285.26 |
  | no-qR raw self-CEM2 | 595.68 | 120.13 | 252.28 | 79.55 | 0.787 | 238.43 |
  | hybrid anchor λ=0.25 | 611.81 | 97.89 | 260.54 | 69.78 | 0.784 | 263.90 |
  | hybrid anchor λ=0.40 | 623.56 | 92.25 | 269.48 | 69.38 | 0.780 | 270.10 |
  | original mean-W CEM | 616.69 | 103.09 | 272.39 | 76.82 | 0.362 | 166.34 |

- **Interpretation**:
  The seed0 hybrid retrieval improvement does not survive this 5-seed check.
  Hybrid λ=0.25 and λ=0.40 still improve uncertainty calibration and CRPS over
  original mean-W CEM, but both are worse on average than train-anchor raw qR
  and no-qR raw self-CEM.  Per-seed, λ=0.25 is competitive on seeds 0 and 1,
  but degrades on seeds 2, 3, and especially 4 relative to raw neighborhoods.
  High raw-overlap alone is therefore not a sufficient guardrail.  The current
  evidence should be interpreted as: learned W retrieval can help on some LH
  seeds, and controls show the learned spatial binding is real on seed0, but
  the hybrid retrieval rule is not robust enough to become a default.
- **Status**:
  Completed.  Do not promote hybrid-anchor retrieval as the main candidate
  without a seed-robust gating/selection rule.  If this line continues, next
  work should diagnose which seeds/regimes benefit and use a validation
  criterion to decide whether to apply W retrieval per seed/anchor, rather than
  fixing a global λ.

## 2026-05-16 - `uncertain_w_cem_qr_hybrid_lambda_controls_lh_seed0_20260516`

- **Question**: Is the hybrid anchor retrieval gain concentrated at
  `lambda=0.25`, and does it survive learned/prior/shuffled/raw-only controls?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Settings / provenance**:
  Same paper-style LH seed0 protocol as
  `uncertain_w_cem_qr_retrieval_metric_lh_seed0_20260515`: LH generator with
  RFF expansion, latent dimension `d=5`, observed dimension `H=30`, train/test
  `1000/200`, train-only X standardization and local-head y standardization,
  `n_neighbors=35`, `m_inducing=40`, 100 `kmeans_yvar` train anchors, fixed
  raw candidate pool 200, `qr_outer_cycles=4`, `refresh_hyper_steps=10`,
  `refresh_gate_steps=20`, `qr_beta_kl=0.05`, self-CEM2, `nstd=0.1`,
  `kcov=0.1`, and anchor-ranking weight `0.1` where present.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_hybrid_lambda_controls_lh_seed0_20260516/`.
- **Headline result (LH seed0, test=200)**:

  | variant | RMSE | CRPS | Cov90 | Width90 | raw-overlap |
  |---|---:|---:|---:|---:|---:|
  | hybrid anchor λ=0.40 learned | **642.19** | 275.22 | 0.800 | 327.77 | 0.868 |
  | hybrid anchor λ=0.25 learned | 643.71 | **275.12** | 0.800 | 353.56 | 0.915 |
  | hybrid anchor λ=0.30 learned | 658.48 | 286.50 | 0.775 | 370.05 | 0.900 |
  | hybrid anchor λ=0.00 learned/raw retrieval | 656.57 | 289.66 | 0.785 | 363.47 | 1.000 |
  | no-qR raw self-CEM2 | 664.32 | 292.17 | 0.775 | 267.01 | n/a |
  | train-anchor raw qR | 666.15 | 296.90 | 0.790 | 372.89 | n/a |
  | hybrid anchor λ=0.25 shuffled | 673.68 | 296.18 | 0.775 | 316.07 | 0.911 |
  | hybrid anchor λ=0.25 prior | 696.10 | 313.86 | 0.775 | 470.56 | 1.000 |
  | original mean-W CEM | 684.30 | 322.57 | 0.530 | 247.96 | n/a |

- **Interpretation**:
  The learned hybrid signal survives controls.  λ=0.25 and λ=0.40 are the two
  strong seed0 candidates, both substantially better than prior, shuffled,
  raw-only retrieval (`λ=0`), train-anchor raw qR, and no-qR raw self-CEM.
  The λ curve is not monotone: `0.1/0.15/0.2` are weak, `0.3` is moderate,
  and `0.4` is as strong as `0.25`.  The useful regime still satisfies high
  raw locality (`raw-overlap >= 0.868`), consistent with W acting as a local
  re-ranker rather than a replacement for raw neighborhoods.
- **Status**:
  Completed seed0 λ/control check.  Proceeding to a minimal 5-seed check with
  original CEM, train-anchor raw qR, no-qR raw self-CEM2, hybrid anchor λ=0.25,
  and hybrid anchor λ=0.40.

## 2026-05-15 - `uncertain_w_cem_qr_retrieval_metric_lh_seed0_20260515`

- **Question**: After learned W-KNN beat prior/shuffled controls but hurt RMSE
  relative to raw/fixed neighborhoods, is the problem the retrieval metric
  being too aggressive?  Compare anchor-conditioned W-KNN, pointwise W-KNN,
  and raw/W hybrid retrieval on LH seed0.
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation changes**:
  Extended retrieval-aware `qr_alt` variants with `pwwknn` pointwise
  retrieval, `hya<lambda>` hybrid anchor retrieval, and `hypw<lambda>` hybrid
  pointwise retrieval.  Hybrid distances are mixed after per-anchor median
  normalization of raw and W distances.  Added an efficient pointwise q(R)
  pool-distance helper for
  `E[||W(x_i)x_i - W(x_a)x_a||^2]` that avoids materializing the full
  `[T,Q,D,pool,pool]` covariance, and recorded raw-topk overlap plus selected
  inlier fraction diagnostics.
- **Settings / provenance**:
  Paper-style LH synthetic setting `lh_q5_train1000`, generated through the
  repository paper synthetic runner (`_generate_paper_data` preset): base LH
  generator with RFF high-dimensional expansion, latent dimension `d=5`,
  observed dimension `H=30`, `Q_true=5`, learned `Q=5`, train/test sizes
  `1000/200`, `n_neighbors=35`, `m_inducing=40`, `max_test=200`, train-only X
  standardization and local-head y standardization.  q(R) used 100 train
  anchors selected by `kmeans_yvar`, fixed raw candidate pool size 200,
  `qr_outer_cycles=4`, `refresh_hyper_steps=10`, `refresh_gate_steps=20`,
  `qr_beta_kl=0.05`, leave-anchor-out predictive q(R) loss, and rank weights
  from the variant names.
- **Artifacts**:
  Main matrix:
  `experiments/synthetic/uncertain_w_cem_qr_retrieval_metric_lh_seed0_20260515/`.
  Smoke:
  `experiments/synthetic/uncertain_w_cem_qr_retrieval_metric_smoke_20260515/`.
- **Headline result (LH seed0, test=200)**:

  | retrieval | variant | RMSE | CRPS | Cov90 | Width90 | raw-overlap |
  |---|---|---:|---:|---:|---:|---:|
  | hybrid anchor λ=0.25 | `qr_alt_hya0.25_..._rank0.1` | **643.71** | **275.12** | 0.800 | 353.56 | 0.915 |
  | hybrid pointwise λ=0.5 | `qr_alt_hypw0.5_..._rank0.1` | 678.56 | 296.43 | 0.770 | 302.19 | 0.808 |
  | anchor W-KNN rank0.1 | `qr_alt_wknn_..._rank0.1` | 697.02 | 313.89 | 0.775 | 414.52 | 0.657 |
  | pointwise W-KNN rank0.1 | `qr_alt_pwwknn_..._rank0.1` | 747.36 | 354.55 | 0.730 | 344.84 | 0.574 |
  | previous raw/fixed qR baseline | `qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1` | 657.75 | 290.75 | 0.775 | 225.96 | n/a |
  | previous no-qR raw self-CEM2 | `self_cem2_nstd0.1_kcov0.1` | 659.38 | 288.13 | 0.780 | 267.81 | n/a |

- **Interpretation**:
  Pure pointwise retrieval is too aggressive: overlap drops to about `0.57`
  and both RMSE/CRPS degrade.  Pure anchor W-KNN also changes too many
  neighbors and underperforms raw/fixed neighborhoods.  The clear winner is
  hybrid anchor λ=0.25, which keeps high raw locality (`0.915` overlap) while
  letting W make a small supervised retrieval correction; this improves both
  RMSE and CRPS over the previous raw/fixed qR and no-qR raw self-CEM baselines
  on this seed.  Larger W weights (`λ=0.5/0.75`) lose the RMSE gain.
- **Status**:
  Completed seed0 metric ablation.  Best current LH candidate is
  `qr_alt_hya0.25_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1`.  Next
  follow-up should be a small multi-seed confirmation and, if stable, a
  tighter λ sweep around `0.15-0.35` before treating it as a new default.

## 2026-05-15 - `uncertain_w_cem_train500_selfcem_5seeds_20260515`

- **Question**: After synthetic tuning showed no large qR gain from larger
  neighborhoods or train-anchor qR, does the self-contained uncertain-W CEM
  prediction head improve UCI train=500/test=200 performance enough to compare
  with XGBoost/DKL/LMJGP baselines?
- **Runner**:
  `experiments/uci/compare_uncertain_w_cem_train500.py`.
- **Implementation changes**:
  Added a UCI runner for uncertain-W CEM prediction heads.  It reuses the
  synthetic uncertain-W CEM/qR implementation, follows the existing UCI
  train-size/test-anchor subsampling convention, uses grouped-subject
  Parkinsons splits by default, standardizes X on train only, standardizes y
  internally for the local head, and reports metrics on original y scale.
- **Settings**:
  Main run:
  `--datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --train_sizes 500 --num_exp 5 --max_test_anchors 200 --variants
  self_cem2_nstd0.05_kcov0.1,self_cem2_nstd0.1_kcov0.1,
  self_cem2_nstd0.15_kcov0.1 --include_original`.  Neighborhoods use
  dataset defaults (`n=25` for Wine/Parkinsons, `n=35` for Appliances).
  Seed0 head sweep:
  `experiments/uci/uncertain_w_cem_train500_seed0_head_sweep_20260515/`;
  fixed Parkinsons rerun:
  `experiments/uci/uncertain_w_cem_train500_seed0_head_sweep_parkinsons_fix_20260515/`.
  A Parkinsons seed0 qR-lowrank check used
  `qr_alt_lr2_refresh10_self_cem2_nstd0.15_kcov0.1`.
- **Artifacts**:
  Main 5-seed run:
  `experiments/uci/uncertain_w_cem_train500_selfcem_5seeds_20260515/`.
  Same train-size XGBoost/DKL baseline rerun:
  `experiments/uci/smalltrain_xgb_dkl_train500_5seeds_20260515/`.
  qR Parkinsons check:
  `experiments/uci/uncertain_w_cem_train500_parkinsons_qr_seed0_20260515/`.
  Smoke:
  `experiments/uci/uncertain_w_cem_train500_smoke_20260515/`.
- **Headline result (5 seeds, train=500, test=200)**:

  | dataset | method | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | Wine | original mean-W CEM | 0.808 | 0.471 | 0.714 | 1.536 |
  | Wine | self-CEM `nstd0.05` | 0.785 | 0.450 | 0.757 | 1.664 |
  | Wine | self-CEM `nstd0.10` | 0.786 | 0.450 | 0.759 | 1.680 |
  | Wine | self-CEM `nstd0.15` | **0.785** | **0.449** | **0.761** | 1.696 |
  | Parkinsons grouped | original mean-W CEM | 10.291 | 7.076 | 0.406 | 12.540 |
  | Parkinsons grouped | self-CEM `nstd0.05` | 10.281 | 6.825 | 0.468 | 14.488 |
  | Parkinsons grouped | self-CEM `nstd0.10` | 10.270 | 6.813 | 0.466 | 14.543 |
  | Parkinsons grouped | self-CEM `nstd0.15` | **10.235** | **6.770** | **0.475** | 14.680 |
  | Appliances | original mean-W CEM | 113.173 | 48.249 | 0.631 | 94.272 |
  | Appliances | self-CEM `nstd0.05` | **109.282** | 43.335 | 0.755 | 100.256 |
  | Appliances | self-CEM `nstd0.10` | 109.429 | 43.227 | 0.770 | 101.265 |
  | Appliances | self-CEM `nstd0.15` | 109.727 | **43.195** | **0.783** | 105.082 |

- **Comparison to existing train=500 projection baselines**:
  A same-train-size XGBoost/DKL rerun gave:
  Wine XGB/DKL `0.420/0.598` CRPS, Parkinsons XGB/DKL `7.037/4.935`,
  Appliances XGB/DKL `43.366/52.370`.  Against these baselines, self-CEM
  beats DKL on Wine and Appliances, beats XGB on Parkinsons CRPS, is close to
  XGB on Appliances CRPS but has worse RMSE, and is behind XGB on Wine.
  Against `repaired_projection_variants_train200_500_5seeds`, self-CEM is
  competitive on Wine but still behind LMJGP/CEM-EM projection variants
  (`Wine` best existing CRPS about `0.420`).  It is better on Appliances than
  the listed LMJGP/projection rows (`43.2` CRPS vs `44.8+`, and lower RMSE),
  but it is not competitive on grouped Parkinsons (`6.77` CRPS vs existing
  LMJGP-lowrank `3.78`).  The Parkinsons qR-lowrank seed0 check was worse
  than self-CEM (`7.84` CRPS), so qR was not expanded there.
- **Status**:
  Completed 5-seed UCI self-CEM benchmark.  Best single default is
  `self_cem2_nstd0.15_kcov0.1` for CRPS/UQ.  The method is a useful
  probabilistic prediction-head improvement over original CEM, especially on
  Appliances, but it does not replace LMJGP/MDGP-style representation learning
  on Parkinsons.

## 2026-05-15 - `uncertain_w_cem_qr_trainanchor_sweeps_20260515`

- **Question**: Can larger neighborhoods, more qR inducing points, or
  train-set cluster anchors for qR training produce a much larger improvement
  than the previous default uncertain-W qR alternating CEM configuration?
- **Runner**:
  `experiments/synthetic/sweep_uncertain_w_cem_qr_trainanchors.py`, which
  reuses `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation changes**:
  Added an inductive train-anchor qR path to
  `compare_uncertain_w_cem_l2_lh.py`: when `--qr_train_anchor_count > 0`,
  q(R) is trained on train-set anchors selected by k-means/random/high-local-y
  variance, then q(W) is induced at test anchors for the final uncertain-W CEM
  prediction head.  Added `--qr_train_anchor_count`,
  `--qr_train_anchor_strategy`, and `--qr_train_n_neighbors`, plus coverage
  diagnostics for train-anchor and test-anchor neighborhoods.  Added a sweep
  driver that makes variant names unique per `n/M/A/an` tuple.
- **Settings**:
  LH seed0, 200 test points, low-rank r2 candidate
  `qr_alt_lr2_refresh10_self_cem2_nstd0.1_kcov0.1`.
  Train-anchor sweep:
  `n_neighbors_grid=50,100,200`, `m_inducing=100`,
  `qr_train_anchor_count=50,100,200`, `qr_train_n_neighbors` matched to the
  same grid, `qr_train_anchor_strategy=kmeans_yvar`.  Test-anchor qR sweep:
  `n_neighbors_grid=50,100,200`, `m_inducing=100,200`,
  `qr_train_anchor_count=0`.
- **Artifacts**:
  Train-anchor qR sweep:
  `experiments/synthetic/uncertain_w_cem_qr_trainanchor_sweep_lh_seed0_stage1_20260515/`.
  Test-anchor qR neighborhood/M sweep:
  `experiments/synthetic/uncertain_w_cem_qr_testanchor_nm_sweep_lh_seed0_20260515/`.
  Smokes:
  `experiments/synthetic/uncertain_w_cem_qr_trainanchors_smoke_20260515/` and
  `experiments/synthetic/uncertain_w_cem_qr_trainanchor_sweep_smoke_20260515/`.
- **Headline result (LH seed0, 200 test points)**:

  | sweep | best/representative variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | previous default lowrank r2 (`n=35,M=40,test anchors`) | from `uncertain_w_cem_qr_lowrank_seed0_20260514` | 653.830 | **288.111** | 0.755 | 215.906 |
  | train-anchor qR | `n=50,M=100,A=100,anchor-n=100` | 686.220 | 310.845 | 0.765 | 349.2 |
  | train-anchor qR | `n=50,M=100,A=200,anchor-n=50` | 683.819 | 312.583 | 0.755 | 327.5 |
  | test-anchor qR | `n=50,M=200` | 675.798 | 300.197 | 0.775 | 282.8 |
  | test-anchor qR | `n=50,M=100` | 686.218 | 308.623 | 0.770 | 285.1 |
  | original mean-W CEM | baseline in these sweeps | 684.300 | 322.573 | 0.530 | 248.0 |

- **Diagnostics**:
  Larger neighborhoods are not a free improvement.  `n=50` improves over
  original CEM but is still worse than the previous default `n=35`; `n=100`
  and `n=200` generally degrade CRPS and become slow.  Increasing inducing
  points from `M=100` to `M=200` helps the `n=50` test-anchor sweep, but still
  does not recover the earlier `n=35,M=40` result.  Train-anchor qR gives high
  train coverage (`~0.99-1.00` for many settings) but worse prediction,
  indicating that coverage alone does not make the qR objective align with
  the final test-anchor CEM head.
- **Status**:
  Completed synthetic hyperparameter probe.  Do not expand large-neighborhood
  or train-anchor qR settings to 5 seeds.  Keep the default small-neighborhood
  self-CEM/qR settings for synthetic, and prefer the simpler self-CEM head for
  UCI.

## 2026-05-15 - `uncertain_w_cem_qr_retrieval_lh_seed0_20260515`

- **Question**: Does making uncertain `W(x)` control neighbor retrieval, rather
  than only the local kernel/gate inside a fixed raw-X neighborhood, make the
  learned spatial projection field identifiable on paper-style LH?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation changes**:
  Added retrieval-aware `qr_alt_wknn_*` variants.  Train anchors use a fixed
  raw-X candidate pool `C_a` and select the top-k neighbors by
  `E_q ||W(x_a)(x_i-x_a)||^2`; the same W-KNN retrieval is used for final test
  anchors.  For W-KNN variants the q(R) M-step defaults to a supervised
  leave-anchor-out predictive NLL (`--qr_anchor_pred_weight 1.0`) and disables
  the selected-neighborhood CEM bound
  (`--qr_retrieval_composite_temperature 0.0`) to avoid incomparable
  selected-set likelihoods.  Added optional anchor-centric retrieval ranking
  loss via `_rank{weight}` variant suffix.  The ranking loss pulls current
  self-CEM inliers ahead of outliers in anchor-to-neighbor expected projected
  distance.  Existing raw/fixed variants are unchanged.
- **Settings**:
  Main LH seed0 W-KNN grid:
  `--settings lh_q5_train1000 --variants
  qr_alt_wknn_refresh10_self_cem2_nstd0.1_kcov0.1,
  qr_alt_wknn_refresh10_self_cem2_nstd0.1_rank0.01_kcov0.1,
  qr_alt_wknn_refresh10_self_cem2_nstd0.1_rank0.05_kcov0.1,
  qr_alt_wknn_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1,
  qr_alt_wknn_refresh10_self_cem2_nstd0.1_qrprior_kcov0.1,
  qr_alt_wknn_refresh10_self_cem2_nstd0.1_qrshuf_kcov0.1,
  self_cem2_nstd0.1_kcov0.1 --seed_start 0 --num_exp 1 --max_test 200
  --n_neighbors 35 --m_inducing 40 --qr_train_anchor_count 100
  --qr_train_anchor_strategy kmeans_yvar --qr_train_n_neighbors 35
  --qr_retrieval_pool 200 --qr_outer_cycles 4 --refresh_hyper_steps 10
  --refresh_gate_steps 20 --qr_beta_kl 0.05`.  A same-code current raw/fixed
  baseline was run separately without train-anchor/W-KNN:
  `--variants qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1`.
  Smoke used `max_test=5`, `qr_train_anchor_count=8`, `n_neighbors=8`, and
  `qr_retrieval_pool=20`.
  Synthetic provenance: paper-style LH/RFF generator from
  `compare_paper_synthetic_baselines.py` (`new_highdata_gen_utils.generate_data`
  with `caseno=0`, latent dimension 5, observed dimension 30, train=1000,
  test=200); train-only X standardization; local head uses standardized y;
  metrics are on original y scale.
- **Artifacts**:
  Main W-KNN grid:
  `experiments/synthetic/uncertain_w_cem_qr_retrieval_lh_seed0_20260515/`.
  Same-code raw/fixed current baseline:
  `experiments/synthetic/uncertain_w_cem_qr_retrieval_current_lh_seed0_20260515/`.
  Smoke:
  `experiments/synthetic/uncertain_w_cem_qr_retrieval_smoke_20260515/`.
- **Headline result (seed 0, 200 test points)**:

  | variant | RMSE | CRPS | Cov90 | Width90 | notes |
  |---|---:|---:|---:|---:|---|
  | raw/fixed current qR | **657.749** | 290.751 | 0.775 | 225.956 | same-code current baseline, no W-KNN |
  | original mean-W CEM | 681.846 | 319.431 | 0.530 | 248.028 | main grid baseline |
  | no-qR raw self-CEM2 | 659.377 | 288.126 | 0.780 | 267.810 | raw-KNN head-only control |
  | W-KNN learned, no rank | 737.891 | 343.638 | 0.750 | 344.627 | W changes retrieval but hurts badly |
  | W-KNN + rank0.01 | 736.870 | 344.867 | 0.745 | 372.164 | too weak/unstable |
  | W-KNN + rank0.05 | 676.638 | 293.374 | 0.775 | 378.897 | partial recovery |
  | W-KNN + rank0.1 | 662.527 | **282.749** | **0.790** | 406.767 | best CRPS; RMSE worse than raw current |
  | W-KNN prior | 694.686 | 311.247 | 0.775 | 463.649 | isotropic prior retrieval equals raw top-k ordering |
  | W-KNN shuffled | 717.969 | 329.891 | 0.745 | 389.621 | learned qR with anchor-W bindings shuffled |

- **Diagnostics**:
  W-KNN retrieval is active: learned variants have test raw-topk overlap around
  `0.67-0.68`, while prior W-KNN has overlap `1.0`.  Without ranking, letting
  W select neighbors is harmful.  The anchor-ranking signal is active and
  improves monotonically over this small grid; `rank0.1` beats prior and
  shuffled W-KNN by `28.5` and `47.1` CRPS, respectively, and beats no-qR raw
  self-CEM2 by `5.4` CRPS.  However, it does so with much wider intervals
  (`Width90=406.8`) and slightly worse RMSE than raw/fixed current qR.  This is
  evidence that spatial W binding can matter once W controls retrieval, but the
  current ranking/predictive objective is not yet a clean point-prediction win.
- **Status**:
  Completed LH seed0 retrieval-aware probe.  Do not promote without a 5-seed
  run and calibration/width analysis.  The next useful checks are a smaller
  rank grid around `0.07-0.15`, lower interval-width/noise calibration, and
  whether the rank0.1 gain holds against shuffled controls over multiple seeds.

## 2026-05-14 - `uncertain_w_cem_qr_lowrank_seed0_20260514`

- **Question**: Does a centered low-rank residual projection prior
  `W(x)=W0+C(x)V^T` improve the weak alternating uncertain-W CEM/q(R)
  candidate by shrinking the full `Q x D` projection-function space?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation changes**:
  Added anchor-conditioned low-rank residual q(R) support.  `W0` is the fixed
  base projection, `V` is a fixed orthonormal residual basis, and q(R) now
  parameterizes the sparse-GP coefficient field `C(x)` with shape `M x Q x r`
  for variants named `qr_alt_lr{r}_...`.  The uncertain-W kernel path now
  supports full per-row covariance matrices, so the expected SE kernel, gate
  moments, kernel-vector covariance correction, self-CEM label updates, local
  hyperparameter fitting, and q(R) M-step use the closed-form covariance
  induced by `V diag(var(C_q(x))) V^T` instead of diagonalizing the residual.
  Added CLI controls `--lowrank_basis` and `--lowrank_qr_w_signal_var`.
- **Settings**:
  Main seed0 grid:
  `--settings l2_q2_train1000,lh_q5_train1000 --variants
  qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1,
  qr_alt_lr1_refresh10_self_cem2_nstd0.1_kcov0.1,
  qr_alt_lr2_refresh10_self_cem2_nstd0.1_kcov0.1,
  qr_alt_lr4_refresh10_self_cem2_nstd0.1_kcov0.1,
  qr_alt_lr2_refresh10_self_cem2_nstd0.1_qrprior_kcov0.1,
  qr_alt_lr2_refresh10_self_cem2_nstd0.1_qrshuf_kcov0.1,
  self_cem2_nstd0.1_kcov0.1 --seed_start 0 --num_exp 1 --max_test 200
  --m_inducing 40 --qr_steps 40 --qr_outer_cycles 4
  --refresh_hyper_steps 10 --refresh_gate_steps 20 --qr_beta_kl 0.05
  --lowrank_qr_w_signal_var 0.1`.  LH-only sensitivity runs used
  `lowrank_qr_w_signal_var=0.02` and `0.5`.  Synthetic provenance matches
  the paper-style L2 phantom/RFF and LH/RFF generators used by the recent
  uncertain-W CEM comparisons: train=1000, test=200, train-only X
  standardization, local y standardization, and original-scale metrics.
- **Artifacts**:
  Main grid:
  `experiments/synthetic/uncertain_w_cem_qr_lowrank_seed0_20260514/`.
  Signal sensitivity:
  `experiments/synthetic/uncertain_w_cem_qr_lowrank_lrsig002_seed0_20260514/`
  and
  `experiments/synthetic/uncertain_w_cem_qr_lowrank_lrsig05_seed0_20260514/`.
  Parser/gradient smoke:
  `experiments/synthetic/uncertain_w_cem_qr_lowrank_smoke_20260514/`.
- **Headline result (seed 0, 200 test points)**:

  | setting | variant | RMSE | CRPS | Cov90 | Width90 | qR KL | R_mu data/KL |
  |---|---|---:|---:|---:|---:|---:|---:|
  | L2 train=1000 | original CEM | 2.254 | **1.270** | 0.800 | 5.853 | - | - |
  | L2 train=1000 | full WGP qR | 2.261 | 1.281 | 0.770 | 5.633 | 48.06 | 0.688 |
  | L2 train=1000 | lowrank r1 | 2.253 | 1.274 | 0.775 | 5.615 | 6.76 | 0.729 |
  | L2 train=1000 | lowrank r2 | 2.257 | 1.276 | 0.770 | 5.603 | 13.56 | 0.775 |
  | L2 train=1000 | lowrank r4 | 2.272 | 1.292 | 0.765 | 5.594 | 26.55 | 0.645 |
  | L2 train=1000 | lowrank r2 prior | 2.258 | 1.275 | 0.780 | 5.654 | - | - |
  | L2 train=1000 | lowrank r2 shuffled | **2.247** | 1.268 | 0.775 | 5.617 | 13.56 | - |
  | L2 train=1000 | no-qR head | 2.266 | 1.279 | 0.770 | 5.668 | - | - |
  | LH train=1000 | original CEM | 684.300 | 322.573 | 0.530 | 247.958 | - | - |
  | LH train=1000 | full WGP qR | 658.548 | 291.744 | 0.765 | 225.938 | 53.87 | 8.260 |
  | LH train=1000 | lowrank r1 | 655.381 | 290.446 | 0.760 | 218.694 | 1.70 | 2.695 |
  | LH train=1000 | lowrank r2 | **653.830** | 288.111 | 0.755 | 215.906 | 3.30 | 1.814 |
  | LH train=1000 | lowrank r4 | 667.126 | 300.707 | 0.755 | 211.853 | 6.89 | 1.669 |
  | LH train=1000 | lowrank r2 prior | 672.549 | 299.240 | 0.765 | 235.513 | - | - |
  | LH train=1000 | lowrank r2 shuffled | 656.563 | **286.823** | 0.765 | 212.835 | 3.30 | - |
  | LH train=1000 | no-qR head | 664.321 | 292.167 | 0.775 | 267.008 | - | - |

- **Signal sensitivity (LH r2)**:

  | lowrank qR signal variance | RMSE | CRPS | Cov90 | Width90 | R_mu data/KL |
  |---:|---:|---:|---:|---:|---:|
  | 0.02 | 662.071 | 294.877 | 0.760 | 207.677 | 0.867 |
  | 0.10 | 653.830 | 288.111 | 0.755 | 215.906 | 1.814 |
  | 0.50 | 653.648 | 288.011 | 0.755 | 226.325 | 6.433 |

- **Diagnostics**:
  The low-rank residual prior is a useful structural restriction on LH:
  `r=1/2` improves over the full WGP qR candidate and clearly beats the
  low-rank prior/no-qR head.  The KL is also much smaller, which confirms the
  residual coefficient field concentrates the qR optimization.  However, the
  shuffled low-rank control has the best LH CRPS in this seed, so this run does
  not yet prove a meaningful spatial binding `x -> C(x)`.  `r=4` is worse,
  consistent with reintroducing too much residual freedom.  On L2, the method
  remains essentially neutral and does not beat original CEM.
- **Status**:
  Completed seed0 structural probe.  Do not promote low-rank residual qR to a
  5-seed main result yet.  The next useful step is to improve the residual
  basis or warm start so learned low-rank qR beats its shuffled control; until
  then, the fair interpretation is "better constrained qR/head refinement on
  LH" rather than demonstrated learned spatial projection field.

## 2026-05-14 - `uncertain_w_cem_qr_pairwise_seed0_20260514`

- **Question**: Does adding a direct pairwise boundary-geometry auxiliary to
  the uncertain-W q(R) M-step improve the selected weak alternating CEM
  candidate?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation changes**:
  Added optional pairwise-augmented q(R) M-step support in
  `djgp.projections.uncertain_w_jgp.train_qr_uncertain_w_cem_mstep`.  The
  auxiliary uses expected squared projected distances under q(W): inlier-inlier
  pairs are pulled together, inlier-outlier pairs are pushed apart with a
  margin, and outlier-outlier pairs are ignored.  Distances are normalized by
  the current batch median by default.  Runner variants can set the weight with
  `_pair{weight}`, e.g.
  `qr_alt_refresh10_self_cem2_nstd0.1_pair0.05_kcov0.1`.  Added CLI controls
  `--qr_pairwise_weight`, `--qr_pairwise_margin`,
  `--qr_pairwise_same_weight`, `--qr_pairwise_cross_weight`, and
  `--qr_pairwise_normalize`.
- **Settings**:
  Main seed0 grid:
  `--settings l2_q2_train1000,lh_q5_train1000 --variants
  qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1,
  qr_alt_refresh10_self_cem2_nstd0.1_pair0.005_kcov0.1,
  qr_alt_refresh10_self_cem2_nstd0.1_pair0.01_kcov0.1,
  qr_alt_refresh10_self_cem2_nstd0.1_pair0.05_kcov0.1 --seed_start 0
  --num_exp 1 --max_test 200 --m_inducing 40 --qr_steps 40
  --qr_outer_cycles 4 --refresh_hyper_steps 10 --refresh_gate_steps 20
  --qr_beta_kl 0.05`.  A resumed LH-only strong-weight check added
  `pair0.2`, `pair0.5`, and `pair1.0`.  Synthetic provenance matches
  `compare_paper_synthetic_baselines.py`: paper L2 phantom/RFF
  (`d=2,H=20,Q=2`) and LH/RFF (`d=5,H=30,Q=5`) with
  `N_train=1000,N_test=200`; train-only X standardisation; local uncertain-W
  CEM uses standardised train `y`; metrics are on original y scale.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_pairwise_seed0_20260514/` with
  `metrics.csv`, `aggregate.csv`, `summary.json`, and `README.md`.
  Parser/gradient smoke:
  `experiments/synthetic/uncertain_w_cem_qr_pairwise_smoke_20260514/`.
- **Headline result (seed 0, 200 test points)**:

  | setting | variant | RMSE | CRPS | Cov90 | Width90 | pair/data grad |
  |---|---|---:|---:|---:|---:|---:|
  | L2 train=1000 | no pair | 2.261 | 1.281 | 0.770 | 5.633 | 0.000 |
  | L2 train=1000 | `pair0.005` | 2.262 | 1.282 | 0.770 | 5.634 | 0.017 |
  | L2 train=1000 | `pair0.01` | 2.263 | 1.282 | 0.770 | 5.633 | 0.035 |
  | L2 train=1000 | `pair0.05` | 2.268 | 1.286 | 0.770 | 5.636 | 0.171 |
  | LH train=1000 | no pair | 658.548 | **291.744** | 0.765 | 225.938 | 0.000 |
  | LH train=1000 | `pair0.005` | 658.561 | 291.763 | 0.770 | 225.984 | 0.003 |
  | LH train=1000 | `pair0.01` | 658.567 | 291.779 | 0.770 | 226.036 | 0.006 |
  | LH train=1000 | `pair0.05` | 658.533 | 291.847 | 0.770 | 226.179 | 0.032 |
  | LH train=1000 | `pair0.2` | **658.285** | 291.870 | 0.770 | 225.056 | 0.124 |
  | LH train=1000 | `pair0.5` | 662.778 | 295.512 | 0.760 | 223.528 | 0.289 |
  | LH train=1000 | `pair1.0` | 670.094 | 301.838 | 0.755 | 225.046 | 0.504 |

- **Diagnostics**:
  The pairwise term is active and differentiable, but it does not improve the
  selected candidate.  Small weights are too weak on LH (`pair/data` gradient
  ratio up to `0.032` for `pair0.05`) and leave prediction essentially
  unchanged.  Stronger weights start to affect qR (`0.124` for `pair0.2`,
  `0.504` for `pair1.0`) but CRPS worsens; `pair0.2` gives only a negligible
  RMSE gain while hurting CRPS, and `pair0.5/1.0` clearly degrade prediction.
  L2 also degrades monotonically as pairwise weight increases.
- **Status**:
  Completed seed0 pairwise-augmented qR probe.  Do not include this auxiliary
  in the main candidate.  If revisiting pairwise learning, it likely needs a
  better target than local hard CEM labels, such as boundary-weighted pairs
  derived from response discontinuity or a warm-start pretraining phase rather
  than a simultaneous M-step penalty.

## 2026-05-14 - `uncertain_w_cem_qr_controls_seed0_20260514`

- **Question**: For the selected weak alternating uncertain-W CEM candidate,
  is the LH gain attributable to a learned q(R) projection field, or mostly to
  the uncertain-W CEM head/noise-floor refresh?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation changes**:
  Added backward-compatible control modifiers for `qr_alt` variants:
  `_qrprior` bypasses q(R) training and uses the GP prior W moments,
  `_qrmeanonly` keeps only the learned q(R) mean path by disabling qR residual
  variance, and `_qrshuf` learns q(R) but randomly permutes anchor-to-q(W)
  moment bindings before refresh/prediction.  Added `self_cem*_nstd*_kcov*`
  variants as no-qR head-only controls with the same local noise floor.
- **Settings**:
  Controls:
  `--settings l2_q2_train1000,lh_q5_train1000 --variants
  qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1,
  qr_alt_refresh10_self_cem2_nstd0.1_qrprior_kcov0.1,
  qr_alt_refresh10_self_cem2_nstd0.1_qrmeanonly_kcov0.1,
  qr_alt_refresh10_self_cem2_nstd0.1_qrshuf_kcov0.1,
  self_cem2_nstd0.1_kcov0.1 --seed_start 0 --num_exp 1 --max_test 200
  --m_inducing 40 --qr_steps 40 --qr_outer_cycles 4 --refresh_hyper_steps 10
  --refresh_gate_steps 20 --qr_beta_kl 0.05`.
  Noise-floor sensitivity:
  `--variants qr_alt_refresh10_self_cem2_nstd0.05_kcov0.1,
  qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1,
  qr_alt_refresh10_self_cem2_nstd0.15_kcov0.1,
  qr_alt_refresh10_self_cem2_nstd0.2_kcov0.1` with the same settings.
  Synthetic provenance matches `compare_paper_synthetic_baselines.py`: paper
  L2 phantom/RFF (`d=2,H=20,Q=2`) and LH/RFF (`d=5,H=30,Q=5`) with
  `N_train=1000,N_test=200`; train-only X standardisation; local uncertain-W
  CEM uses standardised train `y`; metrics are on original y scale.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_controls_seed0_20260514/`,
  `experiments/synthetic/uncertain_w_cem_qr_nstd_sensitivity_seed0_20260514/`,
  and parser smoke
  `experiments/synthetic/uncertain_w_cem_qr_controls_smoke_20260514/`.
  A user-aborted 5-seed control attempt left
  `experiments/synthetic/uncertain_w_cem_qr_controls_5seeds_20260514/` with
  only one original-CEM row; ignore that partial folder.
- **Headline controls (seed 0, 200 test points)**:

  | setting | variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | L2 train=1000 | original mean-W JumpGP-CEM | 2.254 | 1.270 | 0.800 | 5.853 |
  | L2 train=1000 | learned qR | 2.261 | 1.281 | 0.770 | 5.633 |
  | L2 train=1000 | prior qR | 2.253 | 1.273 | 0.785 | 5.717 |
  | L2 train=1000 | mean-only qR | 2.268 | 1.283 | 0.770 | 5.616 |
  | L2 train=1000 | shuffled qR | **2.250** | 1.271 | 0.775 | 5.659 |
  | L2 train=1000 | no-qR self-CEM2+nstd | 2.266 | 1.279 | 0.770 | 5.668 |
  | LH train=1000 | original mean-W JumpGP-CEM | 684.300 | 322.573 | 0.530 | 247.958 |
  | LH train=1000 | learned qR | **658.548** | **291.744** | 0.765 | 225.938 |
  | LH train=1000 | prior qR | 699.346 | 315.558 | 0.765 | 191.528 |
  | LH train=1000 | mean-only qR | 680.509 | 310.304 | 0.735 | 206.187 |
  | LH train=1000 | shuffled qR | 665.795 | 294.699 | 0.765 | 251.880 |
  | LH train=1000 | no-qR self-CEM2+nstd | 664.321 | 292.167 | **0.775** | 267.008 |

- **Noise-floor sensitivity (seed 0, LH)**:

  | nstd | RMSE | CRPS | Cov90 | Width90 | mean noise var |
  |---:|---:|---:|---:|---:|---:|
  | 0.05 | **653.209** | **287.562** | 0.760 | 170.201 | 0.0079 |
  | 0.10 | 658.548 | 291.744 | 0.765 | 225.938 | 0.0154 |
  | 0.15 | 658.452 | 292.129 | 0.785 | 287.488 | 0.0285 |
  | 0.20 | 657.729 | 292.170 | **0.795** | 349.628 | 0.0457 |

- **Diagnostics**:
  Seed0 controls support a real qR contribution on LH: learned qR beats prior
  qR by `23.8` CRPS and mean-only qR by `18.6` CRPS.  However, shuffled qR and
  no-qR head-only are close (`294.7` and `292.2` CRPS vs learned `291.7`), so
  the gain is not purely explained by a strongly learned spatial projection
  field.  It is better described as learned qR plus uncertain-W CEM/noise-floor
  refinement.  L2 controls again show no learned-qR benefit.  Noise-floor
  sensitivity is smooth: larger floors monotonically improve LH coverage but
  widen intervals and slightly hurt CRPS; seed0's sharpest CRPS is `nstd0.05`,
  while the 5-seed selected `nstd0.1` remains a more conservative UQ default.
- **Status**:
  Completed seed0 controls and seed0 noise-floor sensitivity.  Do not claim
  learned qR alone explains the whole improvement.  The next method step, if
  pushing representation learning, should be a pairwise-augmented qR M-step;
  otherwise this line should be positioned as an improved probabilistic
  uncertain-W CEM head with some learned-qR contribution.

## 2026-05-14 - `uncertain_w_cem_qr_alt_selected_5seeds_20260514`

- **Question**: Does the balanced alternating uncertain-W q(R)/CEM candidate
  selected from seed0 hold up over 5 seeds on paper-style L2/LH?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Settings**:
  `--settings l2_q2_train1000,lh_q5_train1000 --variants
  qr_mstep_self_cem1_kcov0.1,qr_alt_refresh10_self_cem2_kcov0.1,
  qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1 --seed_start 0 --num_exp 5
  --max_test 200 --m_inducing 40 --qr_steps 40 --qr_outer_cycles 4
  --refresh_hyper_steps 10 --refresh_gate_steps 20 --qr_beta_kl 0.05`.
  Synthetic provenance matches `compare_paper_synthetic_baselines.py`: paper
  L2 phantom/RFF (`d=2,H=20,Q=2`) and LH/RFF (`d=5,H=30,Q=5`) with
  `N_train=1000,N_test=200`; train-only X standardisation; local uncertain-W
  CEM uses standardised train `y`; metrics are on original y scale.  The
  no-refresh q(R) baseline uses `qr_steps=40` to match the total q(R) update
  budget of `refresh10 x 4 outer cycles`.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_alt_selected_5seeds_20260514/`
  with `metrics.csv`, `aggregate.csv`, `summary.json`, and `README.md`.
- **Headline result (5 seeds, 200 test points)**:

  | setting | variant | RMSE mean/std | CRPS mean/std | Cov90 mean/std | Width90 mean/std |
  |---|---|---:|---:|---:|---:|
  | L2 train=1000 | original mean-W JumpGP-CEM | **2.502 / 0.198** | **1.376 / 0.111** | 0.770 / 0.038 | 5.765 / 0.135 |
  | L2 train=1000 | `qr_mstep_self_cem1_kcov0.1` | 2.514 / 0.185 | 1.382 / 0.100 | 0.753 / 0.029 | 5.543 / 0.117 |
  | L2 train=1000 | `qr_alt_refresh10_self_cem2_kcov0.1` | 2.518 / 0.188 | 1.387 / 0.105 | 0.757 / 0.029 | 5.536 / 0.147 |
  | L2 train=1000 | `qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1` | 2.511 / 0.182 | 1.382 / 0.101 | 0.755 / 0.027 | 5.538 / 0.144 |
  | LH train=1000 | original mean-W JumpGP-CEM | 616.693 / 103.089 | 272.389 / 76.821 | 0.362 / 0.167 | 166.340 / 57.037 |
  | LH train=1000 | `qr_mstep_self_cem1_kcov0.1` | 597.883 / 122.215 | 254.738 / 80.809 | 0.533 / 0.166 | 228.554 / 67.219 |
  | LH train=1000 | `qr_alt_refresh10_self_cem2_kcov0.1` | 593.937 / 122.674 | 254.004 / 81.678 | 0.456 / 0.202 | 128.330 / 41.256 |
  | LH train=1000 | `qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1` | **591.847 / 118.548** | **250.806 / 78.398** | **0.781 / 0.050** | 194.979 / 60.730 |

- **Diagnostics**:
  The 5-seed result confirms the seed0 conclusion with a stronger UQ signal.
  On LH, `nstd0.1` has the best mean RMSE and CRPS while increasing Cov90 from
  `0.456` for unfloored refresh to `0.781`, and from `0.362` for original CEM
  to `0.781`.  The unfloored refresh still improves mean prediction but is too
  sharp: Width90 is only `128.3`.  The noise-floor candidate keeps a narrower
  interval than no-refresh q(R) (`195.0` vs `228.6`) while giving better CRPS
  and much better Cov90.  Final mean local noise variance on LH is close to
  the floor (`0.0122` vs floor `0.01`), so a future focused calibration probe
  should record per-anchor floor-hit rate and test `nstd0.15/0.2`.
- **Status**:
  Completed.  Current main candidate for LH-style hard-boundary synthetic data
  is `qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1`.  L2 remains a no-gain
  setting: original CEM is still best on average, and qR/refresh variants are
  acceptable only in the sense that they do not severely harm L2.  Next step
  should be learned-qR controls (`prior qR`, `mean-only qR`, shuffled qR) on
  the selected candidate, not a wider variant grid.

## 2026-05-14 - `uncertain_w_cem_qr_alt_uqfix_l2_lh_seed0_20260514`

- **Question**: Can the best weak alternating uncertain-W q(R)/CEM variant
  avoid LH under-coverage by adding a refresh-time noise floor, local
  hyperparameter/gate damping, and a modest kcov grid?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation changes**:
  Added backward-compatible `qr_alt` variant modifiers:
  `_nstd{x}` applies a refresh-time noise standard-deviation floor on the
  standardized-y scale (`noise_var >= x^2`), and `_damp{a}` blends refreshed
  local mean, kernel hyperparameters, and gate with the previous state.  Added
  CLI controls `--refresh_noise_std_floor`, `--refresh_damping`,
  `--refresh_gate_l2`, and `--refresh_gate_max_norm`.  Existing variant names
  and previous result folders are unchanged.
- **Settings**:
  `--settings l2_q2_train1000,lh_q5_train1000 --variants
  qr_alt_refresh10_self_cem1_kcov0.1,qr_alt_refresh10_self_cem2_kcov0.1,
  qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1,
  qr_alt_refresh10_self_cem2_nstd0.1_damp0.5_kcov0.1,
  qr_alt_refresh10_self_cem2_nstd0.1_damp0.5_kcov0.25 --max_test 200
  --m_inducing 40 --qr_outer_cycles 4 --refresh_hyper_steps 10
  --refresh_gate_steps 20 --qr_beta_kl 0.05`.  Synthetic provenance matches
  `compare_paper_synthetic_baselines.py`: paper L2 phantom/RFF
  (`d=2,H=20,Q=2`) and LH/RFF (`d=5,H=30,Q=5`) with
  `N_train=1000,N_test=200`; train-only X standardisation; local uncertain-W
  CEM uses standardised train `y`; metrics are on original y scale.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_alt_uqfix_l2_lh_seed0_20260514/`
  with `metrics.csv`, `aggregate.csv`, `summary.json`, and `README.md`.
  A parser/runner smoke is in
  `experiments/synthetic/uncertain_w_cem_qr_alt_uqfix_smoke_20260514/`.
- **Headline result (seed 0, 200 test points)**:

  | setting | variant | RMSE | CRPS | Cov90 | Width90 | mean noise var | final data/KL `R_mu` |
  |---|---|---:|---:|---:|---:|---:|---:|
  | L2 train=1000 | original mean-W JumpGP-CEM | 2.254 | 1.270 | 0.800 | 5.853 |  |  |
  | L2 train=1000 | `qr_alt_refresh10_self_cem2_kcov0.1` | 2.270 | 1.287 | 0.770 | 5.634 | 0.183 | 0.639 |
  | L2 train=1000 | `qr_alt_refresh10_self_cem2_nstd0.1_damp0.5_kcov0.25` | **2.252** | 1.274 | 0.780 | 5.705 | 0.187 | 0.731 |
  | LH train=1000 | original mean-W JumpGP-CEM | 684.300 | 322.573 | 0.530 | 247.958 |  |  |
  | LH train=1000 | `qr_alt_refresh10_self_cem1_kcov0.1` | 652.828 | 289.338 | 0.695 | 166.970 | 0.005 | 10.189 |
  | LH train=1000 | `qr_alt_refresh10_self_cem2_kcov0.1` | **649.545** | **285.215** | 0.685 | 128.394 | 0.004 | 9.944 |
  | LH train=1000 | `qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1` | 653.137 | 287.530 | 0.775 | 226.271 | 0.015 | 8.175 |
  | LH train=1000 | `qr_alt_refresh10_self_cem2_nstd0.1_damp0.5_kcov0.1` | 656.353 | 289.408 | 0.785 | 285.135 | 0.015 | 7.719 |
  | LH train=1000 | `qr_alt_refresh10_self_cem2_nstd0.1_damp0.5_kcov0.25` | 656.353 | 289.068 | **0.790** | 297.478 | 0.015 | 7.719 |

- **Diagnostics**:
  The refresh noise floor is the main calibration lever.  On LH, the
  `nstd0.1` floor raises final mean local noise variance from `0.0043` to
  `0.0154`, widening Width90 from `128.4` to `226.3` and improving Cov90 from
  `0.685` to `0.775`, with CRPS only moving from `285.2` to `287.5`.
  Damping further widens intervals and lowers gate norm (`3.25` to `2.91`) but
  gives up more RMSE/CRPS.  Increasing kcov from `0.1` to `0.25` is secondary:
  it changes LH CRPS by about `-0.34` and Width90 by about `+12.3` for the
  damped candidate.  L2 remains essentially a no-gain setting; the damped
  floor variant matches original RMSE but not original CRPS.
- **Status**:
  Completed seed-0 UQ-fix probe.  Best LH CRPS is still the unfloored
  `refresh10_self_cem2_kcov0.1`, but it under-covers.  Best balanced LH UQ is
  `qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1`: it keeps CRPS near `287.5`
  while lifting Cov90 to `0.775`.  The damped `kcov0.25` candidate is more
  conservative and may be useful if coverage is prioritized over sharpness.

## 2026-05-14 - `uncertain_w_cem_qr_alt_l2_lh_seed0_20260514`

- **Question**: Does periodic weak C-step refresh close the loop between the
  uncertain-W q(R) M-step and local CEM partition/hyperparameters better than a
  single fixed-C-step q(R) update?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation changes**:
  Added `qr_alt_refresh{m}_self_cem{k}_kcov{lambda}` variants.  Each variant
  alternates q(R) M-step segments with weak self-CEM refreshes using the current
  q(R)-induced q(W) moments.  Refresh diagnostics record label-change rate,
  inlier rate, and local hyperparameter drift per refresh.  Existing
  `qr_mstep_*` variants remain unchanged.
- **Settings**:
  Main run:
  `--settings l2_q2_train1000,lh_q5_train1000 --variants
  qr_mstep_fixed_label_kcov0,qr_mstep_self_cem1_kcov0.1,
  qr_alt_refresh10_self_cem1_kcov0.1,qr_alt_refresh10_self_cem2_kcov0.1
  --max_test 200 --m_inducing 40 --qr_steps 40 --qr_outer_cycles 4
  --refresh_hyper_steps 10 --refresh_gate_steps 20 --qr_beta_kl 0.05`.
  A resumed append added `qr_alt_refresh5_self_cem1_kcov0.1` with
  `--qr_outer_cycles 8`, so total qR steps are also 40.  Synthetic provenance
  matches `compare_paper_synthetic_baselines.py`: paper L2 phantom/RFF
  (`d=2,H=20,Q=2`) and LH/RFF (`d=5,H=30,Q=5`) with
  `N_train=1000,N_test=200`; train-only X standardisation; local uncertain-W
  CEM uses standardised train `y`; metrics are on original y scale.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_alt_l2_lh_seed0_20260514/` with
  `metrics.csv`, `aggregate.csv`, `summary.json`, and `README.md`.  A tiny
  smoke is in `experiments/synthetic/uncertain_w_cem_qr_alt_smoke/`.
- **Headline result (seed 0, 200 test points)**:

  | setting | variant | RMSE | CRPS | Cov90 | Width90 | mean refresh label-change | final data/KL `R_mu` |
  |---|---|---:|---:|---:|---:|---:|---:|
  | L2 train=1000 | original mean-W JumpGP-CEM | **2.254** | **1.270** | 0.800 | 5.853 |  |  |
  | L2 train=1000 | `qr_mstep_self_cem1_kcov0.1` | 2.258 | 1.276 | 0.775 | 5.622 |  | 0.804 |
  | L2 train=1000 | `qr_alt_refresh10_self_cem1_kcov0.1` | 2.262 | 1.279 | 0.780 | 5.628 | 0.010 | 0.703 |
  | L2 train=1000 | `qr_alt_refresh10_self_cem2_kcov0.1` | 2.270 | 1.287 | 0.770 | 5.634 | 0.010 | 0.639 |
  | L2 train=1000 | `qr_alt_refresh5_self_cem1_kcov0.1` | 2.292 | 1.304 | 0.765 | 5.565 | 0.004 | 1.388 |
  | LH train=1000 | original mean-W JumpGP-CEM | 684.300 | 322.573 | 0.530 | 247.958 |  |  |
  | LH train=1000 | `qr_mstep_self_cem1_kcov0.1` | 666.913 | 298.407 | 0.710 | 287.751 |  | 5.229 |
  | LH train=1000 | `qr_alt_refresh10_self_cem1_kcov0.1` | 652.828 | 289.338 | **0.695** | 166.970 | 0.022 | 10.189 |
  | LH train=1000 | `qr_alt_refresh10_self_cem2_kcov0.1` | **649.545** | **285.215** | 0.685 | 128.394 | 0.022 | 9.944 |
  | LH train=1000 | `qr_alt_refresh5_self_cem1_kcov0.1` | 667.632 | 299.856 | 0.570 | 86.176 | 0.011 | 19.066 |

- **Diagnostics**:
  Refresh-10 is the first variant that clearly improves the LH qR result:
  self-CEM1 improves CRPS from `298.4` to `289.3`, and self-CEM2 reaches
  `285.2`.  Label changes are small after the first refresh: LH
  `refresh10/self-CEM1` changes `6.0%`, `0.5%`, and `0.2%` across the three
  refreshes, so the loop is doing weak refinement rather than wholesale
  repartitioning.  Refresh-5 is too aggressive: LH Width90 collapses to `86.2`
  and Cov90 drops to `0.57`, with local noise shrinking toward the lower bound.
- **Status**:
  Completed seed-0 alternating-EM probe.  Conclusion: periodic weak refresh is
  worth keeping, with `refresh10_self_cem2` best on LH seed0; refresh5
  over-tightens.  L2 still does not beat original mean-W CEM.  The next useful
  step is a 5-seed run of `qr_alt_refresh10_self_cem1/2` only if the LH
  coverage/width tradeoff is acceptable, or adding a noise floor / pairwise
  auxiliary before 5 seeds if calibration is the priority.

## 2026-05-14 - `uncertain_w_cem_qr_mstep_l2_lh_seed0_20260514`

- **Question**: If the uncertain-W CEM local state is fixed after
  fixed-label/self-CEM hyperparameter and gate learning, does a sparse-GP
  `q(R)` M-step through the moment-matched uncertain-W kernel improve L2/LH
  prediction?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py` with
  `qr_mstep_*` variants backed by
  `djgp.projections.uncertain_w_jgp.train_qr_uncertain_w_cem_mstep`.
- **Implementation changes**:
  Added a differentiable hard-CEM M-step objective
  `q(R) -> q(W) -> E[K(W)] -> log N(y_I; m, E[K]_II + sigma^2 I)` plus
  Gauss-Hermite gate terms and `KL(q(R)||p(R))`.  The local labels,
  hyperparameters, mean constants, and gates are held fixed during the M-step,
  matching the EM separation in `docs/uncertain_w_djgp_method.md`.  Existing
  baseline interfaces and previous result folders are not overwritten.
- **Settings**:
  MAP-R run:
  `--settings l2_q2_train1000,lh_q5_train1000 --variants
  self_cem1_kcov0.1,qr_mstep_fixed_label_kcov0,
  qr_mstep_self_cem1_kcov0.1 --max_test 200 --m_inducing 40 --qr_steps 30
  --qr_beta_kl 0.05`.
  Trainable-variance check:
  `--variants qr_mstep_self_cem1_kcov0.1 --qr_train_log_std` with the same
  data settings.  Synthetic provenance matches
  `compare_paper_synthetic_baselines.py`: paper L2 phantom/RFF
  (`d=2,H=20,Q=2`) and LH/RFF (`d=5,H=30,Q=5`) with
  `N_train=1000,N_test=200`; train-only X standardisation; the local
  uncertain-W CEM objective uses standardised train `y`, final metrics are on
  original y scale.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_qr_mstep_l2_lh_seed0_20260514/` with
  `metrics.csv`, `aggregate.csv`, `summary.json`, and `README.md`.
  Trainable `R_log_std` check is in
  `experiments/synthetic/uncertain_w_cem_qr_mstep_trainstd_l2_lh_seed0_20260514/`.
  Smoke artifacts are in `experiments/synthetic/uncertain_w_cem_qr_smoke/`.
- **Headline result (seed 0, 200 test points)**:

  | setting | variant | RMSE | CRPS | Cov90 | Width90 | final data/KL `R_mu` |
  |---|---|---:|---:|---:|---:|---:|
  | L2 train=1000 | original mean-W JumpGP-CEM | **2.254** | **1.270** | 0.800 | 5.853 |  |
  | L2 train=1000 | `self_cem1_kcov0.1` | 2.263 | 1.276 | 0.775 | 5.642 |  |
  | L2 train=1000 | `qr_mstep_fixed_label_kcov0` | 2.274 | 1.282 | 0.805 | 5.903 | 0.933 |
  | L2 train=1000 | `qr_mstep_self_cem1_kcov0.1` | 2.258 | 1.276 | 0.775 | 5.622 | 0.623 |
  | LH train=1000 | original mean-W JumpGP-CEM | 684.300 | 322.573 | 0.530 | 247.958 |  |
  | LH train=1000 | `self_cem1_kcov0.1` | 668.091 | 299.940 | 0.690 | 281.725 |  |
  | LH train=1000 | `qr_mstep_fixed_label_kcov0` | **665.084** | 307.213 | 0.710 | 483.007 | 7.282 |
  | LH train=1000 | `qr_mstep_self_cem1_kcov0.1` | 665.602 | **297.389** | 0.715 | 287.140 | 7.313 |

- **Variance-path check**:
  Enabling `--qr_train_log_std` barely changes prediction for
  `qr_mstep_self_cem1_kcov0.1` (L2 RMSE/CRPS `2.258/1.276`, LH
  `665.600/297.384`).  Final `R_log_std` data/KL ratio is small on L2
  (`0.038`) and moderate on LH (`0.705`), so the mean path is the main active
  path in this seed-0 run.
- **Status**:
  Completed seed-0 q(R) M-step probe.  Conclusion: the M-step is technically
  working and improves LH over self-CEM1, but gains are modest and L2 does not
  improve over original mean-W CEM.  The trainable variance path is present but
  not yet translating into meaningful UQ gains.  Follow-up: use better
  non-CEM/boundary initial labels or integrate q(R) training with periodic
  weak C-step refresh before considering 5 seeds.

## 2026-05-14 - `uncertain_w_cem_l2_lh_seed0_20260514`

- **Question**: After the uncertain-W CEM prediction head passed the small
  sanity check, does it remain stable on paper-style L2/LH when it learns local
  kernel hyperparameters, gates, and one hard CEM label update?
- **Runner**:
  `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`.
- **Implementation changes**:
  Added a sidecar runner only.  It does not change existing LMJGP, WGP-CEM, or
  JumpGP interfaces.  The runner fixes a mean projection from PLS/PCA, injects a
  controlled uncertain-W covariance, then evaluates `fixed_label_hyperopt`,
  `self_cem1`, and non-CEM label initialisation controls with label diagnostics.
- **Settings**:
  `--settings l2_q2_train1000,lh_q5_train1000 --variants
  fixed_label_hyperopt_kcov0,self_cem1_kcov0.05,self_cem1_kcov0.1,
  self_cem1_response_kcov0.05 --max_test 200 --projection_init pls
  --w_var 0.02 --hyper_steps 30 --gate_steps 60`.
  Synthetic provenance matches `compare_paper_synthetic_baselines.py`: paper L2
  phantom/RFF (`d=2,H=20,Q=2`) and LH/RFF (`d=5,H=30,Q=5`) with
  `N_train=1000,N_test=200`; train-only X standardisation; the uncertain-W CEM
  local head uses standardised train `y` and metrics are reported back on the
  original y scale.
- **Artifacts**:
  `experiments/synthetic/uncertain_w_cem_l2_lh_seed0_20260514/` with
  `metrics.csv`, `aggregate.csv`, `summary.json`, and `README.md`.  Smoke
  artifacts are in `experiments/synthetic/uncertain_w_cem_smoke/` and
  `experiments/synthetic/uncertain_w_cem_pointwise_smoke/`.
- **Headline result (seed 0, 200 test points)**:

  | setting | variant | RMSE | CRPS | Cov90 | Width90 | label change |
  |---|---|---:|---:|---:|---:|---:|
  | L2 train=1000 | original mean-W JumpGP-CEM | **2.254** | **1.270** | 0.800 | 5.853 | 0.000 |
  | L2 train=1000 | `fixed_label_hyperopt_kcov0` | 2.266 | 1.277 | 0.810 | 5.928 | 0.000 |
  | L2 train=1000 | `self_cem1_kcov0.05` | 2.263 | 1.276 | 0.775 | 5.632 | 0.026 |
  | L2 train=1000 | `self_cem1_kcov0.1` | 2.263 | 1.276 | 0.775 | 5.642 | 0.026 |
  | L2 train=1000 | `self_cem1_response_kcov0.05` | 2.466 | 1.558 | 0.540 | 3.753 | 0.139 |
  | LH train=1000 | original mean-W JumpGP-CEM | 684.300 | 322.573 | 0.530 | 247.958 | 0.000 |
  | LH train=1000 | `fixed_label_hyperopt_kcov0` | 671.688 | 311.614 | 0.680 | 422.645 | 0.000 |
  | LH train=1000 | `self_cem1_kcov0.05` | **668.091** | 300.170 | 0.690 | 262.780 | 0.060 |
  | LH train=1000 | `self_cem1_kcov0.1` | **668.091** | **299.940** | 0.690 | 281.725 | 0.060 |
  | LH train=1000 | `self_cem1_response_kcov0.05` | 796.361 | 416.404 | 0.665 | 287.759 | 0.126 |

- **Diagnostics**:
  CEM-initialised `self_cem1` changes only a small fraction of labels (`2.6%`
  on L2, `6.0%` on LH), so the current gain is a conservative weak-update
  effect rather than a full replacement of CEM initialisation.  Response-only
  initialisation changes more labels but is much worse, which means the
  non-CEM initialisation path still needs work.  The pointwise code path was
  smoke-tested with `pointwise_self_cem1_kcov0.05` and ran successfully.
- **Status**:
  Completed seed-0 prediction-head probe.  Conclusion: fixed-label hyperopt is
  stable, one self-CEM update is safe and helps LH CRPS, but the method still
  depends on mean-W CEM initial labels for good performance.  Next follow-up is
  either a 5-seed run of CEM-initialised `self_cem1_kcov0.1` or a stronger
  response/pairwise initialisation before moving this head into a q(R) M-step.

## 2026-05-14 - `lmjgp_r_init_pca_train1000_seed0_20260514`

- **Question**: If the existing LMJGP baseline / `pls_kl` path is left
  untouched, what happens when only the global projection inducing mean
  `V_params['mu_V']` (paper `R`/`q(R)` mean) is initialised from PCA instead
  of PLS/random?
- **Runner**:
  `experiments/synthetic/compare_lmjgp_r_init_paper.py`.
- **Implementation changes**:
  Added a sidecar runner only.  No existing baseline interface or result
  folder is overwritten.  The runner reuses the paper L2/LH generator and the
  existing LMJGP training helpers, then locally swaps the initial
  `V_params['mu_V']`.
- **Variants**:
  `pca_unit_kl` = PCA components repeated over inducing inputs and calibrated
  so induced `E[W]` row norms start near one, matching the current PLS path;
  `pca_dmgp_raw_kl` = DeepMahalanobisGP-style
  `PCA components * 10 / ptp(X @ PCA.T)^2` with no post-interpolation
  calibration.  Smoke also checked `pca_dmgp_kl` and `sir_kl`.
- **Settings**:
  `--num_exp 1 --settings l2_q2_train1000,lh_q5_train1000 --variants
  pca_unit_kl,pca_dmgp_raw_kl --lmjgp_steps 300 --MC_num 3 --log_interval
  100`.  Synthetic provenance matches
  `compare_paper_synthetic_baselines.py`: paper L2 phantom/RFF
  (`d=2,H=20,Q=2`) and LH/RFF (`d=5,H=30,Q=5`) with
  `N_train=1000,N_test=200`; train-only X standardisation; original y scale.
- **Artifacts**:
  `experiments/synthetic/lmjgp_r_init_pca_train1000_seed0_20260514/` with
  `metrics.csv`, `aggregate.csv`, `summary.json`, and `README.md`.  Short
  40-step screen is in
  `experiments/synthetic/lmjgp_r_init_pca_smoke_20260514/`.
- **Headline result (seed 0, 200 test points, 300 LMJGP steps)**:

  | setting | method | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | L2 train=1000 | existing `lmjgp_baseline` | **2.247** | **1.265** | 0.820 | 6.198 |
  | L2 train=1000 | existing `lmjgp_pls_kl` | 2.290 | 1.295 | 0.825 | 6.158 |
  | L2 train=1000 | `pca_unit_kl` | 2.293 | 1.283 | 0.820 | 6.208 |
  | L2 train=1000 | `pca_dmgp_raw_kl` | 2.310 | 1.285 | 0.830 | 6.237 |
  | LH train=1000 | existing `lmjgp_baseline` | 757.807 | 385.980 | 0.765 | 921.402 |
  | LH train=1000 | existing `lmjgp_pls_kl` | 738.602 | 370.929 | 0.770 | 1022.070 |
  | LH train=1000 | `pca_unit_kl` | 749.739 | 380.186 | 0.785 | 1168.401 |
  | LH train=1000 | `pca_dmgp_raw_kl` | **712.753** | **349.212** | 0.845 | 1169.023 |

- **Diagnostics**:
  PCA initialisation does move the induced projection mean: final median
  `||mu_W||_F` is about `1.54/1.56` on L2 and `2.11/2.06` on LH for
  `pca_unit_kl`/`pca_dmgp_raw_kl`.  The DMGP-scaled raw PCA starts much smaller
  (`snapshot0_median_frob_mu_W` `0.129` on L2, `0.380` on LH) but optimisation
  grows it to the same scale by step 300.
- **Status**: Completed seed-0 probe.  Conclusion: PCA init is not a clear
  replacement for the existing LMJGP baseline/PLS-KL.  It does not beat
  baseline on L2.  On LH, DMGP-style raw PCA improves seed-0 RMSE/CRPS over
  PLS-KL but mostly by producing much wider predictive intervals; this needs a
  5-seed run only if we decide the LH width/coverage tradeoff is worth
  promoting.

## 2026-05-14 - `global_wgp_cem_selected_pca_map_5seeds_20260514`

- **Question**: After adding DeepMahalanobisGP-style PCA initialization,
  MAP-`R` training, more anchors, random-anchor coverage, and pairwise
  warm-start, do the best global WGP-CEM variants hold up over 5 seeds on the
  full 200-test paper L2/LH benchmarks with 5 global `R` samples?
- **Runner**:
  `experiments/synthetic/compare_global_wgp_cem_l2_lh.py`.
- **Implementation changes**:
  `djgp/projections/global_wgp_cem.py` now accepts direct `init_R_mu`, can
  freeze `R_log_std`, and can train with mean-only `R` samples for MAP-style
  updates.  The runner now supports DeepVMGP-style PCA `R_mu` initialization,
  optional VMGP-style scaled PCA, pairwise pointwise-`W(x)` warm start, and
  k-means + high-local-variance + random training anchor selection.
- **Selected variants**:
  `anchor_pca_background_map_soft` = anchor-conditioned `W(c_a)` with PCA
  repeated over `R_mu`, broad fixed normal outlier background, soft labels, and
  MAP-`R`; `pointwise_pairwarm_map_soft` = pointwise `W(x_i)` with pairwise
  boundary warm-started inducing `R_mu`, soft labels, and MAP-`R`.
- **Settings**:
  `--settings l2_q2_train1000,lh_q5_train1000 --variants
  anchor_pca_background_map_soft,pointwise_pairwarm_map_soft --seed_start 0
  --num_exp 5 --anchor_count 96 --boundary_anchor_frac 0.45
  --random_anchor_frac 0.20 --steps 60 --max_test 200 --eval_mc 5
  --m_inducing 50 --n_candidate 100 --pairwarm_steps 120`.
  Paper-style L2 phantom/RFF (`d=2,H=20,Q=2`) and LH/RFF
  (`d=5,H=30,Q=5`) generators are used with `N_train=1000,N_test=200`. X is
  standardized from train only; the WGP-CEM training objective uses standardized
  train `y`; final JumpGP predictions and metrics are on original `y` scale.
- **Artifacts**:
  `experiments/synthetic/global_wgp_cem_selected_pca_map_5seeds_20260514/`
  with `metrics.csv`, `aggregate.csv`, `summary.json`, `README.md`, and
  per-setting/seed/variant `history_*.csv` optimization traces. The seed-0
  three-variant screen is in
  `experiments/synthetic/global_wgp_cem_selected_pca_map_seed0_20260514/`.
- **Headline result (5 seeds, 200 test points, 5 `R` samples)**:

  | setting | variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | L2 train=1000 | `anchor_pca_background_map_soft` | 2.515 ± 0.131 | 1.361 ± 0.069 | 0.811 | 6.239 |
  | L2 train=1000 | `pointwise_pairwarm_map_soft` | **2.474 ± 0.032** | **1.345 ± 0.037** | 0.822 | 6.573 |
  | LH train=1000 | `anchor_pca_background_map_soft` | **585.945 ± 122.513** | 249.980 ± 87.895 | 0.584 | 341.350 |
  | LH train=1000 | `pointwise_pairwarm_map_soft` | 587.757 ± 72.023 | **245.503 ± 56.301** | 0.566 | 421.660 |

- **Comparison**:
  The selected global WGP-CEM variants do not beat the existing external
  DeepMahalanobisGP 5-seed aggregate (`L2 RMSE=2.329, CRPS=1.304`; `LH
  RMSE=513.638, CRPS=216.633`) and do not beat the previous pairwise localC
  LH CRPS (`228.244`).  They are nevertheless much healthier than the first
  seed-0 sparse-WGP CEM attempt on coverage/visited fraction and show that
  pairwise warm-start is the only selected variant that materially helps LH.
- **Diagnostics**:
  Mean-gradient signal is not the bottleneck: final data/scaled-KL ratio for
  `R_mu` averages `0.54`/`2.02` on L2 anchor/pointwise and `2.60`/`5.10` on LH
  anchor/pointwise.  Training anchor coverage improves to about `0.906` of
  train points with 96 mixed anchors, compared with about `0.45-0.59` in the
  earlier 32-anchor run.
- **Status**: Completed requested 5-seed selected-variant run. Conclusion:
  DMGP-style PCA init and MAP-`R` stabilize L2, and pairwise warm start is the
  right LH direction, but local JGP-CEM composite fine-tuning still does not
  recover DMGP-level LH performance.

## 2026-05-14 - `global_wgp_cem_l2_lh_withdiag_20260514`

- **Question**: Can the proposed "global sparse-GP prior over `W(x)` + local
  JGP-CEM composite likelihood" repair the long original LMJGP gradient chain,
  and which outlier/coverage variants help on paper-style L2/LH?
- **Runner**:
  `experiments/synthetic/compare_global_wgp_cem_l2_lh.py`, backed by
  `djgp/projections/global_wgp_cem.py`.
- **Implemented variants**:
  `anchor_uniform` = anchor-conditioned `W(c_a)` and fixed uniform outlier
  likelihood; `anchor_floor` adds a soft gamma floor; `anchor_coverage` adds
  the coverage penalty; `anchor_background` replaces uniform outlier scoring
  with broad fixed normal `p0(y)`; `pointwise_coverage` and
  `pointwise_background` use pointwise `W(x_i)` embeddings with the same
  outlier controls.
- **Settings**:
  `--settings l2_q2_train1000,lh_q5_train1000 --variants
  anchor_uniform,anchor_floor,anchor_coverage,anchor_background,
  pointwise_coverage,pointwise_background --anchor_count 32 --steps 40
  --max_test 60 --eval_mc 2 --m_inducing 32 --n_candidate 70`.
  Paper-style generators are used via
  `compare_paper_synthetic_baselines._generate_paper_data`: L2 phantom/RFF
  (`d=2,H=20,Q=2`) and LH/RFF (`d=5,H=30,Q=5`), both with
  `N_train=1000,N_test=200`. X is standardized from train only. The CEM
  composite objective uses standardized train `y`; final JumpGP predictions and
  metrics are on the original `y` scale. Training neighborhoods are fixed in
  raw standardized X; prediction samples global `R`, recomputes latent space,
  reselects neighbors, and refits JumpGP-CEM.
- **Artifacts**:
  `experiments/synthetic/global_wgp_cem_l2_lh_withdiag_20260514/` with
  `metrics.csv`, `aggregate.csv`, `summary.json`, `README.md`, and per-variant
  `history_<setting>_seed0_<variant>.csv` gradient traces. Smoke artifacts are
  in `experiments/synthetic/global_wgp_cem_smoke/`. The earlier partial folder
  `experiments/synthetic/global_wgp_cem_l2_lh_20260514/` is superseded by the
  `withdiag` folder.
- **Headline result (seed 0, first 60 test points)**:

  | setting | best variant | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | L2 train=1000 | `pointwise_coverage` | 2.226 | 1.309 | 0.800 | 6.611 |
  | LH train=1000 | `anchor_background` | 726.659 | 355.291 | 0.650 | 480.149 |

  Full rows:

  | setting | variant | RMSE | CRPS | Cov90 | Width90 | final data/KL grad ratio (`R_mu`, `R_log_std`) |
  |---|---|---:|---:|---:|---:|---:|
  | L2 | `anchor_uniform` | 2.435 | 1.392 | 0.767 | 6.172 | 0.958, 0.202 |
  | L2 | `anchor_floor` | 2.403 | 1.382 | 0.783 | 6.118 | 0.961, 0.206 |
  | L2 | `anchor_coverage` | 2.403 | 1.382 | 0.783 | 6.118 | 0.961, 0.206 |
  | L2 | `anchor_background` | 2.374 | 1.361 | 0.767 | 6.105 | 0.923, 0.195 |
  | L2 | `pointwise_coverage` | 2.226 | 1.309 | 0.800 | 6.611 | 0.702, 0.401 |
  | L2 | `pointwise_background` | 2.231 | 1.313 | 0.783 | 6.604 | 0.817, 0.432 |
  | LH | `anchor_uniform` | 736.180 | 363.450 | 0.633 | 443.712 | 10.150, 0.815 |
  | LH | `anchor_floor` | 741.858 | 365.249 | 0.633 | 448.576 | 10.188, 0.818 |
  | LH | `anchor_coverage` | 741.858 | 365.249 | 0.633 | 448.576 | 10.188, 0.818 |
  | LH | `anchor_background` | 726.659 | 355.291 | 0.650 | 480.149 | 10.163, 0.815 |
  | LH | `pointwise_coverage` | 762.047 | 387.548 | 0.617 | 371.413 | 11.815, 0.803 |
  | LH | `pointwise_background` | 743.006 | 373.375 | 0.617 | 373.771 | 11.684, 0.793 |
- **Optimization diagnostics**:
  The new objective produces strong `R_mu` data gradients: final data/scaled-KL
  ratio is near parity on L2 (`0.70-0.96`) and much larger than KL on LH
  (`10-12`). `R_log_std` remains weaker (`0.20-0.43` on L2, `0.79-0.82` on
  LH). The coverage penalty at `coverage_min=0.15` is inactive for all rows:
  visited-point soft coverage is already above the threshold and
  `soft_coverage_below_min_fraction=0`; only 45.3% of L2 train points and 59.0%
  of LH train points are visited by the 32 fixed neighborhoods, so this penalty
  cannot affect unvisited points without broader anchor/neighborhood coverage.
- **Status**: Completed seed-0 ablation. Main takeaways: pointwise GP-`W(x)` is
  the best L2 variant, anchor-conditioned + broad background is best on LH, and
  the sparse-GP CEM objective fixes the weak `R_mu` gradient symptom but not
  uncertainty calibration or incomplete training-point coverage. Follow-up:
  raise anchor count/neighborhood coverage, use stronger coverage or union-style
  coverage, and run the best rows at 5 seeds.

## 2026-05-14 - `l2_lh_7remedies_20260514`

- **Question**: If the seven proposed projection-gradient remedies are
  implemented as concrete runnable methods, which ones help on paper-style
  L2/LH synthetic data under one focused protocol?
- **Runner**:
  `experiments/synthetic/compare_l2_lh_7remedies.py`.
- **Implemented methods**:
  `p1_exact_vem` = conditional-W exact local covariance VEM with trainable
  shared kernel/noise nuisance;
  `p2_soft_vem` = soft responsibility weak decoder (`temperature 5 -> 3`,
  damping `0.3`, fixed nuisance);
  `p3_norm_gate_vem` = row-normalized low-rank W plus stronger gate L2
  (`1e-2`);
  `p4_fixed_nuisance_vem` = exact VEM with fixed shared kernel/noise nuisance;
  `p5_whitened_kl` = original LMJGP-style PLS init + KL warmup with whitened
  mean parameterization `mu_V = L_K eta`;
  `p6_pairwise_boundary` = direct boundary-weighted pairwise metric loss;
  `p7_global_wx` = shared point-wise network `z(x)=W(x)x` trained by
  boundary-weighted pairwise loss and evaluated by JumpGP on point-wise
  projected coordinates.
- **Settings**:
  `--settings l2_q2_train1000,lh_q5_train1000 --methods
  p1_exact_vem,p2_soft_vem,p3_norm_gate_vem,p4_fixed_nuisance_vem,
  p5_whitened_kl,p6_pairwise_boundary,p7_global_wx --max_anchors 40
  --n_candidate 60 --n_vem 40 --pair_neighbors 40 --steps 40
  --mc_train 1 --mc_gamma 1 --eval_mc 1 --out_dir
  experiments/synthetic/l2_lh_7remedies_20260514`.
  Paper-style L2 phantom/RFF (`d=2,H=20,Q=2`) and LH/RFF
  (`d=5,H=30,Q=5`) generators, `N_train=1000`, first 40 test anchors only,
  train-only X standardization, and no Y standardization for reported metrics.
  VEM/pairwise objectives use standardized training y internally where needed
  and transform predictions back to original y scale for metrics.
- **Artifacts**:
  `experiments/synthetic/l2_lh_7remedies_20260514/` with `metrics.csv`,
  `aggregate.csv`, `summary.json`, and `README.md`. Smoke runs are in
  `experiments/synthetic/l2_lh_7remedies_smoke*`.
- **Headline result**:

  | setting | best RMSE/CRPS in this run | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | L2 train=1000, 40 anchors | `p3_norm_gate_vem` | 2.681 | 1.528 | 0.775 | 6.412 |
  | LH train=1000, 40 anchors | `p7_global_wx` | 535.090 | 216.075 | 0.475 | 138.000 |

  Full rows:

  | setting | method | RMSE | CRPS | Cov90 | Width90 | key diagnostic |
  |---|---|---:|---:|---:|---:|---|
  | L2 | `p1_exact_vem` | 2.961 | 1.683 | 0.775 | 5.890 | gamma H `0.101` |
  | L2 | `p2_soft_vem` | 2.710 | 1.536 | 0.775 | 6.430 | gamma H `0.582` |
  | L2 | `p3_norm_gate_vem` | 2.681 | 1.528 | 0.775 | 6.412 | gamma H `0.582` |
  | L2 | `p4_fixed_nuisance_vem` | 2.780 | 1.573 | 0.725 | 5.695 | gamma H `0.117` |
  | L2 | `p5_whitened_kl` | 2.984 | 1.674 | 0.725 | 5.706 | train loss `27.38` |
  | L2 | `p6_pairwise_boundary` | 2.806 | 1.563 | 0.700 | 6.013 | pair loss `0.336` |
  | L2 | `p7_global_wx` | 3.036 | 1.735 | 0.700 | 5.811 | pair loss `0.330` |
  | LH | `p1_exact_vem` | 732.130 | 366.061 | 0.475 | 217.898 | gamma H `0.057` |
  | LH | `p2_soft_vem` | 719.254 | 351.996 | 0.475 | 92.125 | gamma H `0.549` |
  | LH | `p3_norm_gate_vem` | 719.203 | 351.217 | 0.450 | 83.542 | gamma H `0.549` |
  | LH | `p4_fixed_nuisance_vem` | 761.486 | 396.717 | 0.500 | 139.352 | gamma H `0.056` |
  | LH | `p5_whitened_kl` | 859.340 | 477.394 | 0.475 | 455.948 | train loss `35.80` |
  | LH | `p6_pairwise_boundary` | 823.295 | 444.725 | 0.550 | 391.067 | pair loss `0.354` |
  | LH | `p7_global_wx` | 535.090 | 216.075 | 0.475 | 138.000 | pair loss `0.135` |
- **Interpretation**:
  This focused run confirms the earlier gradient diagnosis in an implemented
  method matrix. Soft responsibilities materially improve hard exact VEM on
  both L2 and LH, and stronger gate regularization is a small additional L2
  gain. Fixed nuisance without softening remains hard (`gamma` entropy near
  default VEM) and underperforms soft VEM. The whitened-mean KL warmup row did
  not help at this short 40-step/40-anchor budget. Direct boundary pairwise
  was not competitive in this small slice, but the shared point-wise `W(x)`
  implementation was the clear LH winner, supporting the idea that sharing a
  projection function across points shortens the gradient path. Coverage for
  the sharp LH winners remains poor, so any promotion needs calibration or a
  posterior/ensemble version.
- **Status**: Completed focused seed-0 implementation run. It is not a
  replacement for the existing 5-seed tables; next run should scale the
  promising rows (`p2`, `p3`, `p7`) to 200 anchors and 5 seeds.

## 2026-05-14 - `l2_lh_elbo_gradient_diagnostics_20260514`

- **Question**: During the original LMJGP ELBO optimization, how large are the
  q(V) data gradients relative to the direct KL pull, and does PLS+KL warmup
  materially change that ratio on paper-style L2/LH synthetic data?
- **Runner**:
  `experiments/synthetic/diagnose_lmjgp_elbo_gradients_paper.py`, backed by
  `djgp/diagnostics/elbo_gradients.py`.
- **Settings**:
  `--settings l2_q2_train1000,lh_q5_train1000 --variants baseline,pls_kl
  --max_anchors 80 --num_steps 80 --diag_steps 0,1,5,10,25,50,80
  --out_dir experiments/synthetic/l2_lh_elbo_gradient_diagnostics_20260514`.
  Paper-style generators are used through
  `compare_paper_synthetic_baselines._generate_paper_data`: L2 phantom/RFF
  (`d=2,H=20,Q=2,N_train=1000,N_test=200`) and LH/RFF
  (`d=5,H=30,Q=5,N_train=1000,N_test=200`). Train-only X standardization is
  applied, Y is not standardized, and only the first 80 test anchors are used
  for this gradient diagnostic. Boundary anchors are proxied by the top 25%
  local-neighborhood response variance.
- **Artifacts**:
  `experiments/synthetic/l2_lh_elbo_gradient_diagnostics_20260514/` with
  `gradient_history.csv`, `gradient_latest.csv`, `summary.json`, and
  `README.md`. A tiny smoke run lives in
  `experiments/synthetic/l2_lh_elbo_gradient_diagnostics_smoke/`.
- **Headline result**:
  At step 80, baseline `mu_V` data/KL-gradient ratios are below one:
  L2 `0.804`, LH `0.657`. The variance-proxy ratios are much weaker:
  L2 `0.261`, LH `0.103`. PLS+KL warmup keeps the mean ratio near parity
  at step 80 (L2 `1.135`, LH `1.019`) and improves the variance-proxy ratio
  (L2 `0.839`, LH `0.449`) but does not fix log-sum-exp hardening:
  responsibility entropy is near zero and hard responsibility fraction is
  `0.991` on L2 and `1.0` on LH for both variants. The KL sign check is
  consistent with ELBO subtracting KL: `kl_mu_pull_dot_scaled` is negative in
  all latest rows (e.g. L2 baseline `-18.7`, LH baseline `-55.6`), so this
  diagnostic does not support a KL sign bug.
- **Interpretation**:
  The optimization trace supports the weak-gradient-chain diagnosis. KL warmup
  helps the mean path, especially early, but the variance path remains
  data-weak and the local responsibility term hardens almost immediately.
  Boundary-proxy anchors contribute more than their 25% count but not enough
  to dominate (`mu` split share about `0.39-0.46` at step 80), so simple
  boundary weighting may help but is unlikely to solve the ELBO bottleneck by
  itself.
- **Status**: Completed focused one-seed gradient diagnostic. Use existing
  prediction artifacts for performance conclusions about the seven proposed
  remedies; this run is diagnostic-only and capped at 80 anchors.

## 2026-05-13 - `high_noise_expansions_seed0`

- **Question**: If the paper L2/LH latent generators are embedded through
  high-noise orthogonal-mixing observed expansions, do XGBoost, DKL, LMJGP,
  LMJGP+PLS-KL, or DeepMahalanobisGP handle the noisy high-dimensional input
  best when both `X` and `Y` are train-standardized?
- **Runner**:
  `experiments/synthetic/compare_high_noise_expansions.py` for
  XGB/DKL/LMJGP rows; `experiments/synthetic/run_deep_mahalanobis_gp_paper.py`
  in the `dmgp_tf1` environment for DeepMahalanobisGP.
- **Settings**:
  Seed `0`, `N_train=1000`, `N_test=200`, `X` and `Y` standardized with
  train-only statistics, metrics reported in standardized-`Y` units.
  The latent `z,y` come from the paper L2 phantom and LH generators.  The
  observed expansions are:
  `linear_noise`: `x = Q [A z, eta]`;
  `nonlinear_noise`: `x = Q [z, z^2, sin(pi z), eta]`;
  `eta ~ N(0, 3^2 I)`, with random orthogonal `Q`.
- **Artifacts**:
  `experiments/synthetic/high_noise_expansions_seed0/`.
  `aggregate_with_dmgp.csv` is the combined comparison table; `dmgp_data/`
  contains the standardized splits exported for the external DMGP runner.
- **Headline result**:

  | setting | best RMSE | best CRPS | notes |
  |---|---|---|---|
  | L2 linear-noise | DMGP 1.031 / XGB 1.031 tie | DMGP 0.555 | LMJGP rows trail on RMSE but are close on CRPS. |
  | L2 nonlinear-noise | DMGP 0.952 | DMGP 0.527 | DMGP clearly improves over XGB 0.987 / 0.564. |
  | LH linear-noise | XGB 2.324 | XGB 1.379 | DMGP is close in RMSE but slightly worse CRPS. |
  | LH nonlinear-noise | XGB 2.314 | XGB 1.373, LMJGP+PLS-KL 1.374 tie | DMGP is worse than XGB/LMJGP+PLS-KL. |

  Full seed-0 table:

  | setting | method | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | L2 linear | XGB | 1.031 | 0.580 | 0.720 | 1.810 |
  | L2 linear | DKL | 1.329 | 0.892 | 0.245 | 0.526 |
  | L2 linear | LMJGP | 1.109 | 0.606 | 0.795 | 2.317 |
  | L2 linear | LMJGP+PLS-KL | 1.119 | 0.598 | 0.795 | 2.383 |
  | L2 linear | DeepMahalanobisGP | 1.031 | 0.555 | 0.745 | 1.854 |
  | L2 nonlinear | XGB | 0.987 | 0.564 | 0.680 | 1.712 |
  | L2 nonlinear | DKL | 1.067 | 0.706 | 0.250 | 0.536 |
  | L2 nonlinear | LMJGP | 1.091 | 0.585 | 0.790 | 2.209 |
  | L2 nonlinear | LMJGP+PLS-KL | 1.064 | 0.576 | 0.810 | 2.442 |
  | L2 nonlinear | DeepMahalanobisGP | 0.952 | 0.527 | 0.700 | 1.803 |
  | LH linear | XGB | 2.324 | 1.379 | 0.650 | 1.790 |
  | LH linear | DKL | 2.398 | 1.481 | 0.500 | 0.507 |
  | LH linear | LMJGP | 2.383 | 1.398 | 0.660 | 1.540 |
  | LH linear | LMJGP+PLS-KL | 2.440 | 1.464 | 0.625 | 1.289 |
  | LH linear | DeepMahalanobisGP | 2.349 | 1.400 | 0.635 | 1.340 |
  | LH nonlinear | XGB | 2.314 | 1.373 | 0.640 | 1.747 |
  | LH nonlinear | DKL | 2.473 | 1.542 | 0.500 | 0.464 |
  | LH nonlinear | LMJGP | 2.436 | 1.448 | 0.620 | 1.160 |
  | LH nonlinear | LMJGP+PLS-KL | 2.373 | 1.374 | 0.640 | 1.223 |
  | LH nonlinear | DeepMahalanobisGP | 2.406 | 1.426 | 0.635 | 1.275 |

- **Status**: Completed seed-0 pilot. This is not a 5-seed table. The next
  useful follow-up is a 5-seed run for the two most informative settings:
  `l2_nonlinear_noise3_train1000` and `lh_nonlinear_noise3_train1000`.

## 2026-05-13 - `paper_l2_lh_stdxy_lmjgp_train1000_5seeds`

- **Question**: Does standardizing both X and Y improve LMJGP baseline / PLS-KL
  on the paper-style synthetic train=1000 table, making the preprocessing
  protocol closer to DeepMahalanobisGP?
- **Runner**: `experiments/synthetic/compare_lmjgp_stdxy_paper.py`.
- **Settings**:
  `--settings l2_q2_train1000,lh_q5_train1000 --num_exp 5 --methods
  lmjgp_baseline,lmjgp_pls_kl --lmjgp_steps 300 --MC_num 3 --out_dir
  experiments/synthetic/paper_l2_lh_stdxy_lmjgp_train1000_5seeds`.
  Train-only X standardization and train-only Y standardization.  Metrics are
  reported on the original Y scale by multiplying RMSE/CRPS/width by the
  training Y standard deviation; coverage is unchanged.
- **Artifacts**:
  `experiments/synthetic/paper_l2_lh_stdxy_lmjgp_train1000_5seeds/`.
- **Headline result**: Metrics below use
  `aggregate_omit_blowup.csv`, because LH PLS-KL seed 3 numerically exploded
  in the raw aggregate (`RMSE=171531.710`, `CRPS=8.612e18`,
  `Width90=1.212e20`).

  | setting | method | seeds | RMSE | CRPS | Cov90 | Width90 |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | L2 Q=2 train=1000 | LMJGP baseline | 5 | 2.417 | 1.320 | 0.805 | 6.134 |
  | L2 Q=2 train=1000 | LMJGP PLS-KL | 5 | 2.445 | 1.338 | 0.801 | 6.058 |
  | LH Q=5 train=1000 | LMJGP baseline | 5 | 596.501 | 259.528 | 0.626 | 539.874 |
  | LH Q=5 train=1000 | LMJGP PLS-KL | 4 | 565.536 | 238.784 | 0.643 | 372.880 |

  Standardizing Y did not improve the LMJGP baseline relative to the previous
  X-only-standardized paper synthetic run (`L2 2.404/1.315`, `LH
  595.848/258.690` for RMSE/CRPS).  After omitting the LH PLS-KL blow-up,
  PLS-KL is also close to the X-only run (`LH 567.906/238.418`) but has lower
  coverage here (`0.643` vs `0.733`).  DeepMahalanobisGP remains the stronger
  5-seed row on these two settings (`L2 2.329/1.304`, `LH 513.638/216.633`).
- **Status**: Completed.

## 2026-05-13 - `deep_mahalanobis_gp_global_z_jgp_train1000_5seeds`

- **Question**: If DeepMahalanobisGP learns a pointwise metric/projection field,
  how does ordinary JumpGP perform after globally mapping the full train/test
  datasets as `z_i = W(x_i)x_i`?
- **Runner**:
  1. `experiments/synthetic/run_deep_mahalanobis_gp_paper.py` retrained DMGP
     in a new output directory and saved train/test-side `W(x)` arrays.
  2. `experiments/synthetic/evaluate_dmgp_global_z_jgp_paper.py` mapped every
     train/test point to `Z` with the posterior mean `W(x)` and fit JumpGP in
     the global `Z` space.
- **Settings**:
  `--settings l2_q2_train1000,lh_q5_train1000 --seeds 0,1,2,3,4`.
  Paper-style L2/LH RFF synthetic data; train-only X standardization; DMGP
  trains on standardized Y but the final JumpGP-on-Z metrics use original-scale
  `y_train/y_test`.  L2 uses `n=25`; LH uses `n=35`.
- **Artifacts**:
  `experiments/synthetic/deep_mahalanobis_gp_global_z_jgp_train1000_5seeds/`.
- **Headline result**:

  | setting | method | seeds | RMSE | CRPS | Cov90 | Width90 |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | L2 Q=2 train=1000 | DeepMahalanobisGP direct | 5 | 2.328 | 1.304 | 0.746 | 5.211 |
  | L2 Q=2 train=1000 | DMGP global `W(x)x` + JGP | 5 | 2.408 | 1.337 | 0.784 | 5.744 |
  | LH Q=5 train=1000 | DeepMahalanobisGP direct | 5 | 513.626 | 216.653 | 0.803 | 408.047 |
  | LH Q=5 train=1000 | DMGP global `W(x)x` + JGP | 5 | 596.000 | 251.931 | 0.282 | 20.849 |

  The global mapped-space JumpGP head does not improve over the direct DMGP
  predictive head.  On L2 it is close to LMJGP baseline (`2.404/1.315`) but
  worse than direct DMGP.  On LH it has comparable RMSE to LMJGP baseline but
  severely under-covers because the mapped `Z`-space JumpGP intervals are very
  narrow (`Cov90=0.282`, `Width90=20.849`).  This suggests that the pointwise
  DMGP metric does not transfer cleanly into a single global `Z` dataset for
  JumpGP, especially in the LH setting; the local per-anchor metric use and the
  direct DMGP posterior are materially different.
- **Status**: Completed.

## 2026-05-14 - `deep_mahalanobis_gp_paper_train200_500_5seeds`

- **Question**: How does the external DeepMahalanobisGP baseline perform in the
  smaller paper-style synthetic regimes (`train=200/500`, `test=200`) on L2/LH?
- **Runner**:
  1. `experiments/synthetic/export_paper_synthetic_for_dmgp.py`.
  2. `experiments/synthetic/run_deep_mahalanobis_gp_paper.py`.
- **Settings**:
  `--settings l2_q2_train200,l2_q2_train500,lh_q5_train200,lh_q5_train500
  --seeds 0,1,2,3,4 --steps1 300 --steps2 700 --w_samples 0`.
  Paper-style L2/LH RFF synthetic data; train-only X standardization; DMGP
  trains on standardized Y and reports metrics on the original Y scale.
- **Artifacts**:
  `experiments/synthetic/deep_mahalanobis_gp_paper_train200_500_5seeds/`.
- **Headline result**:

  | setting | seeds | RMSE | CRPS | Cov90 | Width90 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | L2 Q=2 train=200 | 5 | 2.493 | 1.358 | 0.807 | 6.008 |
  | L2 Q=2 train=500 | 5 | 2.226 | 1.212 | 0.822 | 5.432 |
  | LH Q=5 train=200 | 5 | 563.732 | 248.094 | 0.778 | 419.547 |
  | LH Q=5 train=500 | 5 | 567.613 | 240.735 | 0.806 | 389.173 |

  DMGP is very strong on L2 in the small-data regime: it beats the existing
  XGB/DKL/JGP-PCA/LMJGP/CEM rows at both train sizes on RMSE and CRPS.  On LH,
  train=200 is competitive but not uniformly best: CEM-VI has lower RMSE/CRPS
  on the 4 successful seeds, while DMGP has much better coverage.  On LH
  train=500, DKL has the best RMSE (`538.039`) but under-covers (`0.606`);
  DMGP has the best CRPS among the comparable rows and coverage near `0.806`.
- **Status**: Completed.  `conda run` returned nonzero after the long TF1 run,
  but all 20 per-seed rows, `dmgp_aggregate.csv`, and `dmgp_summary.json` were
  written successfully.

## 2026-05-13 - `vem_em_uci_pdebench_5seeds`

- **Question**: How does the current best conditional-W VEM-EM projection
  variant perform on UCI train=200/500/1000 and the balanced PDEBench-Burgers
  surrogate over five seeds?
- **Runner**:
  `experiments/vem/compare_vem_em_uci_pdebench.py`.
- **Settings**:
  `--benchmarks uci,pdebench --uci_train_sizes 200,500,1000 --num_exp 5
  --n_candidate 100 --n_vem 50 --steps 80 --eval_mc 2
  --pdebench_dataset_folder
  data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000
  --pdebench_Q 3 --pdebench_n 25 --resume`.
  UCI uses the current best-preset split/scaling policy: Wine `std_x`,
  Appliances `std_x,std_y`, and Parkinsons subject-grouped `std_x`.
  The method row is `vem_em_learned_qw`: soft-gamma conditional-W VEM,
  posterior `q(W)` samples, and projected-space final JumpGP neighborhoods.
- **Artifacts**:
  `experiments/vem/vem_em_uci_pdebench_5seeds/`.
- **Headline result**:

  | benchmark | train | dataset/subset | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---|---:|---:|---:|---:|
  | UCI | 200 | Wine | 0.848 | 0.501 | 0.678 | 1.589 |
  | UCI | 200 | Parkinsons grouped | 10.340 | 7.060 | 0.428 | 14.216 |
  | UCI | 200 | Appliances | 114.343 | 49.088 | 0.698 | 106.149 |
  | UCI | 500 | Wine | 0.770 | 0.447 | 0.719 | 1.530 |
  | UCI | 500 | Parkinsons grouped | 10.418 | 7.162 | 0.383 | 12.597 |
  | UCI | 500 | Appliances | 108.455 | 44.188 | 0.739 | 110.088 |
  | UCI | 1000 | Wine | 0.770 | 0.437 | 0.737 | 1.537 |
  | UCI | 1000 | Parkinsons grouped | 9.931 | 6.830 | 0.374 | 11.422 |
  | UCI | 1000 | Appliances | 110.568 | 44.225 | 0.717 | 96.282 |
  | PDEBench | - | all | 0.768 | 0.255 | 0.782 | 0.589 |
  | PDEBench | - | near-threshold | 0.843 | 0.344 | 0.723 | 0.792 |
  | PDEBench | - | far-from-threshold | 0.739 | 0.222 | 0.803 | 0.516 |

  Relative to the existing best-preset UCI tables, this VEM-EM variant is not
  competitive: Wine/Appliances remain better with XGB or residual LMJGP rows,
  and Parkinsons grouped remains best with DKL/LMJGP. On PDEBench, VEM-EM is
  worse than the existing LMJGP row (`all RMSE=0.714, CRPS=0.240`) and worse
  than XGB on RMSE (`0.694`), but it improves over DKL and JGP-PCA on the
  all/near/far CRPS rows.
- **Status**: Completed 5 seeds. No CEM-EM-style learned/global/prior controls
  were run in this UCI/PDEBench pass; this is the learned posterior q(W) row
  only.

## 2026-05-13 - `deep_mahalanobis_gp_paper_train1000_5seeds`

- **Question**: Does the seed-0 DeepMahalanobisGP advantage on LH hold over
  five paired seeds against the existing paper-style synthetic train=1000
  baseline table?
- **Runners**:
  `experiments/synthetic/export_paper_synthetic_for_dmgp.py`,
  `experiments/synthetic/run_deep_mahalanobis_gp_paper.py`.
- **Settings**:
  Paper-style `l2_q2_train1000,lh_q5_train1000`, seeds `0,1,2,3,4`,
  `N_test=200`, train-only X standardization, DeepMahalanobisGP trains on
  standardized Y and reports original-scale metrics.  External implementation
  uses the `dmgp_tf1` environment from the seed-0 diagnostic.  No `W` controls
  or projected-JGP-on-DeepVMGP-W diagnostics are run in this 5-seed aggregate.
- **Artifacts**:
  `experiments/synthetic/deep_mahalanobis_gp_paper_train1000_5seeds/`.
- **Headline result**:

  | setting | method | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | L2 train=1000 | DeepMahalanobisGP | **2.329** | **1.304** | 0.746 | 5.211 |
  | LH train=1000 | DeepMahalanobisGP | **513.638** | **216.633** | 0.803 | 408.064 |

  Compared to the existing 5-seed paper-synthetic train=1000 table,
  DeepMahalanobisGP is the best RMSE/CRPS method on both L2 and LH in this
  direct-prediction aggregate.  On L2, it is only slightly ahead of LMJGP
  baseline in CRPS (`1.304` vs `1.315`) and has lower coverage (`0.746` vs
  `0.805`).  On LH, the seed-0 advantage holds over 5 seeds: DeepMahalanobisGP
  improves over LMJGP+PLS-KL (`RMSE=567.906, CRPS=238.418, Cov90=0.733`) and
  CEM-EM VI (`584.181, 249.989, 0.546`).  Caveat: DeepMahalanobisGP follows
  its original notebook and trains on standardized Y; the DJGP/CEM table did
  not y-standardize these synthetic runs.
- **Status**: Completed 5 seeds. No `W` controls or DeepVMGP-W + JGP
  diagnostics were run for this aggregate.

## 2026-05-13 - `vem_w_softgamma_controls_train1000_5seeds`

- **Question**: Does the best current soft-gamma conditional-W VEM variant
  hold up over five seeds, and do learned projection samples beat
  anchor-shuffled, global-mean, and prior-`q(W)` controls?
- **Runner**: `experiments/synthetic/compare_paper_vem_w_controls.py`.
- **Settings**:
  `--num_exp 5 --settings l2_q2_train1000,lh_q5_train1000 --controls
  learned_qw,anchor_shuffled,global_mean,prior_qw --n_candidate 100
  --n_vem 50 --steps 80 --eval_mc 2 --resume --out_dir
  experiments/synthetic/vem_w_softgamma_controls_train1000_5seeds`.
  This trains `post_vem1_boundary` with soft responsibilities
  (`temperature 5 -> 3`, `gamma_damping=0.3`), posterior-W prediction, and
  projected final JumpGP neighborhoods.  Controls are evaluated from the same
  trained posterior.
- **Control definitions**:
  `learned_qw` samples learned `q(W_j)`;
  `anchor_shuffled` permutes learned samples across anchors;
  `global_mean` collapses learned samples to a sample-wise global projection;
  `prior_qw` samples coefficient residuals from the prior around learned `W0`.
- **Artifacts**:
  `experiments/synthetic/vem_w_softgamma_controls_train1000_5seeds/` with
  `metrics.csv`, `metrics_pointfiltered.csv`, `aggregate.csv`,
  `aggregate_pointfiltered.csv`, `aggregate_omit_blowup.csv`,
  `aggregate_common_seed_omit_lh_seed4.csv`, `summary.json`, `README.md`, and
  `histories/*.json`.  `metrics_pointfiltered.csv` replaces the LH seed-4
  rows after recomputing metrics with per-point extreme-variance filtering
  (`max_sigma_for_metrics=1e6`).  A smoke run is in
  `experiments/synthetic/vem_w_controls_smoke/`.
- **Headline, raw aggregate (5 seeds unless noted)**:

  | setting | control | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | L2 train=1000 | `prior_qw` | 2.480 | 1.351 | 0.799 | 6.112 |
  | L2 train=1000 | `learned_qw` | 2.483 | 1.356 | 0.794 | 5.966 |
  | L2 train=1000 | `anchor_shuffled` | 2.516 | 1.365 | 0.806 | 6.188 |
  | L2 train=1000 | `global_mean` | 2.572 | 1.394 | 0.799 | 5.837 |
  | LH train=1000 | `prior_qw` | 567.132 | 235.594 | 0.612 | 436.264 |
  | LH train=1000 | `global_mean` | 572.334 | 234.560 | 0.417 | 147.990 |
  | LH train=1000 | `anchor_shuffled` | 580.284 | 239.700 | 0.470 | 322.262 |
  | LH train=1000 | `learned_qw` | 564.740 | `8.7e27` | 0.510 | `1.2e29` |

- **Point-filtered aggregate**:
  LH `learned_qw` seed 4 had one predictive-variance blow-up point
  (`max sigma before filter = 3.72e31`).  The corrected per-seed metric drops
  that single point and keeps 199/200 test points.  In
  `aggregate_pointfiltered.csv`, the 5-seed LH learned result is RMSE 564.949
  / CRPS 232.103 / Cov90 0.510 / Width90 249.214, with exactly 1 dropped
  point over 1000 LH learned test predictions.  The corresponding LH controls
  are: `prior_qw` RMSE 567.132 / CRPS 235.594 / Cov90 0.612 / Width90
  436.264; `global_mean` RMSE 572.334 / CRPS 234.560 / Cov90 0.417 /
  Width90 147.990; `anchor_shuffled` RMSE 580.284 / CRPS 239.700 / Cov90
  0.470 / Width90 322.262.
- **Comparison to existing train=1000 baselines**:
  On L2, soft-gamma VEM-W controls do not beat the existing best baselines:
  DKL has RMSE 2.375, LMJGP baseline has CRPS 1.315, and XGB has RMSE 2.436.
  The learned VEM-W posterior is essentially tied with prior-`q(W)`, which is
  not evidence of useful learned `q(W)` geometry.  On LH after point filtering,
  learned `q(W)` is the best RMSE/CRPS among the VEM-W controls and improves
  over the existing filtered `lmjgp_pls_kl` CRPS 238.4, CEM-EM CRPS 250.0, and
  DKL CRPS 268.5, but coverage remains weaker than LMJGP/XGB.
- **Interpretation**:
  With per-point filtering, the LH result is meaningfully more favorable to
  learned `q(W)` than the original row-level blow-up interpretation: one bad
  point was dominating the aggregate.  The learned posterior still does not
  clearly help on L2, and LH coverage is still too low, so the result supports
  robust point-level variance filtering plus calibration rather than an
  unqualified learned-`q(W)` claim.
- **Status**: Completed 5-seed learned/control run.

## 2026-05-13 - `deep_mahalanobis_gp_paper_train1000_seed0`

- **Question**: How does the external DeepMahalanobisGP repository compare to
  LMJGP/CEM-EM on paper-style L2/LH synthetic data, and does its learned
  intermediate projection `W(x)` contain useful signal when plugged into our
  projected JumpGP head?
- **Runners**:
  `experiments/synthetic/export_paper_synthetic_for_dmgp.py`,
  `experiments/synthetic/run_deep_mahalanobis_gp_paper.py`,
  `experiments/synthetic/evaluate_dmgp_w_with_jgp_paper.py`.
- **External implementation**:
  `C:/Users/yxu59/files/spring2026/park/external/DeepMahalanobisGP` original
  GPflow-1 code.  A separate conda env `dmgp_tf1` was created with
  `python=3.7`, `tensorflow==1.15.0`, `gpflow==1.5.1`,
  `tensorflow-probability==0.8.0`, and `protobuf==3.20.3`.
- **Settings**:
  Paper-style `l2_q2_train1000,lh_q5_train1000`, seed 0, train-only X
  standardization, DeepMahalanobisGP trains on standardized Y following its
  original notebook and reports metrics after inverse-scaling to original Y.
  Initial run uses `m_u=50`, `m_v=25`, `steps1=300`, `steps2=700`, `Q=2/5`,
  and saves `W_mean`, `W_var`, and three `W` samples for test anchors.
- **Artifacts**:
  `experiments/synthetic/deep_mahalanobis_gp_paper_train1000_seed0/`.
- **Headline result**:

  Direct external DeepMahalanobisGP predictive metrics, one seed:

  | setting | RMSE | CRPS | Cov90 | Width90 | train sec |
  |---|---:|---:|---:|---:|---:|
  | L2 train=1000 | 2.222 | 1.264 | 0.765 | 5.157 | 176.1 |
  | LH train=1000 | 617.886 | 303.121 | 0.765 | 677.836 | 489.7 |

  DeepMahalanobisGP `W` plugged into projected JumpGP-CEM:

  | setting | method | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | L2 | `W_mean + JGP` | 2.354 | 1.344 | 0.805 | 5.878 |
  | L2 | `W_samples + JGP` | 2.220 | 1.256 | 0.845 | 6.102 |
  | L2 | feature-permuted `W_mean + JGP` | 2.277 | 1.281 | 0.830 | 6.033 |
  | L2 | random isotropic `W + JGP` | 2.259 | 1.265 | 0.840 | 6.203 |
  | LH | `W_mean + JGP` | 751.062 | 366.943 | 0.465 | 111.684 |
  | LH | `W_samples + JGP` | 650.442 | 298.224 | 0.765 | 796.443 |
  | LH | feature-permuted `W_mean + JGP` | 839.594 | 446.719 | 0.640 | 585.636 |
  | LH | random isotropic `W + JGP` | 783.803 | 401.833 | 0.755 | 824.254 |

  Direct seed-0 comparison: on L2, DeepMahalanobisGP direct prediction
  (`RMSE=2.222, CRPS=1.264`) is essentially tied with seed-0 LMJGP baseline
  (`2.247, 1.265`) and better than seed-0 CEM-EM VI (`2.285, 1.291`), but
  under-covers more. On LH seed 0, DeepMahalanobisGP direct prediction
  (`RMSE=617.886, CRPS=303.121`) is better than the same-seed LMJGP baseline
  (`757.807, 385.980`), PLS-KL (`738.602, 370.929`), and CEM-EM VI
  (`808.700, 416.207`).  It is only worse than the best 5-seed aggregate
  LMJGP/CEM-EM results, which is not a paired seed-level comparison.  Its
  sampled `W` plus our JGP head helps substantially relative to its mean `W`,
  suggesting useful projection uncertainty; however on L2 the
  feature-permuted/random controls remain comparable, so the learned feature
  alignment is weak. On LH, feature permutation hurts and sampled `W` beats
  random, so there is some signal in the learned stochastic projection family,
  but the mean `W` alone gives very narrow, under-covering JGP intervals.
- **Status**: Completed one-seed diagnostic.

## 2026-05-13 - `vem_w_paper_train1000_seed0`

- **Question**: Can a more faithful conditional-W exact JGP-VEM objective
  address the prior-like LMJGP `q(W)` issue better than the earlier pairwise
  proxy ablation?
- **Runner**: `experiments/synthetic/compare_paper_vem_w.py`, backed by
  `djgp/projections/conditional_vem.py`.
- **Implementation**:
  A low-rank diagonal Gaussian posterior `q(C_j)` induces `q(W_j)`.
  Training alternates: sample `W_j`; compute exact local GP posterior
  `q(f_j | W_j, gamma_j)`; update `gamma_j` by conditional-W VEM; maximize a
  Monte Carlo collapsed soft JGP bound over `q(W_j)` with KL annealing, anchor
  smoothness, and weak logistic gates.  This is not yet the paper's global
  `q(R)` GP posterior, but it is the first repository implementation where
  posterior `W` samples enter local VEM updates directly.
- **Settings**:
  Main focused run:
  `--num_exp 1 --settings l2_q2_train1000,lh_q5_train1000 --methods
  map_vem1,map_vem5,post_vem1_boundary --n_candidate 100 --n_vem 50
  --steps 80 --mc_train 2 --mc_gamma 2 --eval_mc 2 --prediction_heads
  mean,posterior --final_neighbor_modes raw,projected --resume --out_dir
  experiments/synthetic/vem_w_paper_train1000_seed0`.
  Additional nuisance-ablation rows used `post_vem1_trainhyp_boundary`.
  Additional soft-responsibility run:
  `--temperature 5 --temperature_final 3 --gamma_damping 0.3 --out_dir
  experiments/synthetic/vem_w_softgamma_paper_train1000_seed0`.
- **Artifacts**:
  `experiments/synthetic/vem_w_paper_train1000_seed0/`,
  `experiments/synthetic/vem_w_softgamma_paper_train1000_seed0/`, and
  `experiments/synthetic/vem_w_smoke/`, each with CSV/JSON outputs; the main
  VEM-W directories also include `histories/*.json`.
- **Headline default-VEM seed 0**:

  | setting | method | head/final | RMSE | CRPS | Cov90 | Width90 | gamma mean / entropy |
  |---|---|---|---:|---:|---:|---:|---:|
  | L2 train=1000 | `map_vem1` | mean/projected | 2.259 | 1.282 | 0.830 | 5.843 | 0.930 / 0.077 |
  | L2 train=1000 | `post_vem1_boundary` | mean/raw | 2.273 | 1.304 | 0.800 | 5.831 | 0.929 / 0.079 |
  | LH train=1000 | `post_vem1_boundary` | posterior/projected | 654.547 | 291.399 | 0.695 | 434.704 | 0.944 / 0.034 |
  | LH train=1000 | `map_vem1` | mean/raw | 708.014 | 322.218 | 0.555 | 120.236 | 0.947 / 0.030 |

- **Headline soft-gamma seed 0**:

  | setting | method | head/final | RMSE | CRPS | Cov90 | Width90 | gamma mean / entropy |
  |---|---|---|---:|---:|---:|---:|---:|
  | L2 train=1000 | `post_vem1_boundary` | posterior/projected | 2.212 | 1.268 | 0.820 | 5.960 | 0.692 / 0.571 |
  | L2 train=1000 | `post_vem1_boundary` | posterior/raw | 2.228 | 1.271 | 0.825 | 6.082 | 0.692 / 0.571 |
  | LH train=1000 | `post_vem1_boundary` | posterior/projected | 645.607 | 281.449 | 0.705 | 388.936 | 0.752 / 0.523 |
  | LH train=1000 | `map_vem1` | mean/projected | 704.895 | 326.596 | 0.610 | 122.792 | 0.759 / 0.517 |

- **Interpretation**:
  The true conditional-W VEM implementation confirms a key hypothesis from
  the discussion: even one-step VEM can become too hard (`gamma` near 0.94,
  entropy near 0.03 on LH), so local latent variables still absorb much of the
  signal.  The soft-gamma variant is materially better, especially for L2
  (`posterior/projected` RMSE 2.212 and CRPS 1.268, better than the seed-0
  LMJGP baseline RMSE 2.247 and close in CRPS 1.265) and modestly better on
  LH than default VEM-W.  Posterior-W prediction helps in LH by widening
  intervals and improving coverage, while mean-W often remains under-dispersed.
  Releasing shared kernel/noise hyperparameters (`trainhyp`) did not help in
  the focused seed-0 run and tended to re-harden `gamma`, consistent with the
  nuisance-absorption concern.
- **Status**: Completed implementation, smoke, default-VEM seed-0 run,
  soft-gamma seed-0 run, and shared-nuisance ablation.  Next useful step is a
  5-seed run of only the soft-gamma `post_vem1_boundary` posterior/projected
  variant plus the strongest baseline rows.

## 2026-05-13 - `lmjgp_pls_kl_qw_controls_paper_train1000_seed0`

- **Question**: Does the PLS-KL LMJGP variant learn useful projection
  information beyond a PLS-biased/random projection family?
- **Runner**:
  `experiments/synthetic/compare_lmjgp_pls_kl_qw_controls_paper.py`.
- **Settings**:
  `--settings l2_q2_train1000,lh_q5_train1000 --seed 0 --lmjgp_steps 300
  --MC_num 3 --m1 2 --m2 40 --out_dir
  experiments/synthetic/lmjgp_pls_kl_qw_controls_paper_train1000_seed0`.
  Paper-style L2/LH generators, train-only X standardization, `N_test=200`.
- **Controls**:
  PLS-KL learned `q(W)`, prior, moment-matched isotropic, feature-permuted,
  anchor-shuffled, global-mean, zero-mean learned sigma, and deterministic
  frozen PLS projection.
- **Artifacts**:
  `experiments/synthetic/lmjgp_pls_kl_qw_controls_paper_train1000_seed0/`.
- **Headline result**:
  | setting | control | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | L2 train=1000 | PLS-KL learned `q(W)` | 2.223 | 1.252 | 0.830 | 6.153 |
  | L2 train=1000 | prior | 2.199 | 1.247 | 0.845 | 6.250 |
  | L2 train=1000 | moment-matched isotropic | 2.199 | 1.248 | 0.845 | 6.233 |
  | L2 train=1000 | feature-permuted | 2.234 | 1.256 | 0.845 | 6.249 |
  | L2 train=1000 | anchor-shuffled | 2.259 | 1.269 | 0.840 | 6.054 |
  | L2 train=1000 | global-mean | 2.329 | 1.315 | 0.795 | 5.825 |
  | L2 train=1000 | frozen PLS | 2.230 | 1.259 | 0.800 | 5.847 |
  | LH train=1000 | PLS-KL learned `q(W)` | 766.469 | 398.739 | 0.800 | 986.924 |
  | LH train=1000 | prior | 766.254 | 393.868 | 0.790 | 965.285 |
  | LH train=1000 | moment-matched isotropic | 761.964 | 389.949 | 0.790 | 957.595 |
  | LH train=1000 | feature-permuted | 787.938 | 410.064 | 0.770 | 945.760 |
  | LH train=1000 | anchor-shuffled | 732.925 | 368.708 | 0.805 | 976.049 |
  | LH train=1000 | global-mean | 767.498 | 381.869 | 0.700 | 520.701 |
  | LH train=1000 | zero-mean learned sigma | 758.973 | 387.928 | 0.780 | 991.196 |
  | LH train=1000 | frozen PLS | **693.986** | **330.653** | 0.460 | 184.076 |

- **Interpretation**: PLS-KL learned `q(W)` does not beat prior/isotropic
  controls on L2 and is not better than the zero-mean/prior family on LH.  The
  deterministic frozen PLS projection is the strongest LH point/CRPS result but
  badly under-covers, so the PLS direction itself carries signal while the
  learned stochastic PLS-KL field mostly adds broad random-projection
  uncertainty rather than a better `x -> W(x)` map.  Feature permutation hurts
  LH somewhat, but anchor-shuffling improves it, so there is little evidence in
  this seed for a reliable learned anchor-specific projection field.
- **Status**: Completed one-seed diagnostic.

## 2026-05-13 - `weak_inner_w_paper_train1000_seed0`

- **Question**: Does giving `W` a more direct supervised geometry signal
  improve paper-style L2/LH train=1000 synthetic performance versus current
  LMJGP/CEM-EM variants, where learned `q(W)` often looks prior-like?
- **Runner**: `experiments/synthetic/compare_paper_weak_inner_w.py`.
- **Implementation**: Practical weak-inner ablation, not full stochastic
  q(W)-averaged JGP-VEM.  The runner learns deterministic row-normalized
  projections and evaluates all variants through the same projected
  JumpGP-CEM refit head.  Variants:
  `pls_static`, exact-local-RBF-GP-NLL pilots (`gp_localC`,
  `gp_pairwise_boundary_localC`, seed 0 only), and the 5-seed selected
  pairwise metric variants (`pairwise_localC`,
  `pairwise_boundary_localC`).  `localC` means per-anchor low-rank residual
  coefficients around PLS/PCA `W0`; `boundary` weights anchors by local
  response variance.
- **Settings**:
  `--num_exp 5 --settings l2_q2_train1000,lh_q5_train1000 --methods
  pls_static,pairwise_localC,pairwise_boundary_localC --n_candidate 100
  --pair_neighbors 100 --final_neighbor_modes raw,projected --steps 200
  --resume --out_dir experiments/synthetic/weak_inner_w_paper_train1000_seed0`.
  Initial seed-0 pilot also included `gp_localC` and
  `gp_pairwise_boundary_localC`.  Both settings use paper-style RFF-expanded
  L2/LH generators and train-only X standardization.
- **Artifacts**:
  `experiments/synthetic/weak_inner_w_paper_train1000_seed0/`
  with `metrics.csv`, `metrics_partial.csv`, `aggregate.csv`,
  `summary.json`, and `README.md`.  Two transient error rows remain in
  `metrics.csv` from a failed LH seed-2 retry, but the successful retry
  completed the missing row; `aggregate.csv` filters to `status=ok`.  A small
  runner-validation smoke artifact is also present at
  `experiments/synthetic/weak_inner_w_smoke/`.
- **Headline selected variants, 5 seeds (RMSE / CRPS / Cov90 / Width90)**:

  | setting | method | final neighbors | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---|---:|---:|---:|---:|
  | L2 train=1000 | `pairwise_boundary_localC` | raw | 2.447 | 1.336 | 0.791 | 5.812 |
  | L2 train=1000 | `pairwise_localC` | raw | 2.455 | 1.338 | 0.795 | 5.822 |
  | L2 train=1000 | `pls_static` | raw | 2.486 | 1.364 | 0.777 | 5.800 |
  | LH train=1000 | `pairwise_boundary_localC` | raw | 570.119 | 228.244 | 0.376 | 155.504 |
  | LH train=1000 | `pairwise_localC` | raw | 594.169 | 246.715 | 0.390 | 172.589 |
  | LH train=1000 | `pls_static` | raw | 618.112 | 273.515 | 0.350 | 142.552 |

- **Comparison to existing train=1000 baselines**:
  On L2, pairwise local projections improve over static PLS and are close to
  existing LMJGP/CEM-EM UQ metrics, but do not beat DKL RMSE (2.375) or
  LMJGP CRPS (1.315) in the 5-seed baseline table.  On LH, the boundary
  pairwise variant gives the best CRPS among compared methods
  (`228.2` vs filtered `lmjgp_pls_kl` `238.4`, CEM-EM `251.5`, DKL `254.2`)
  but has poor coverage because intervals are much narrower.
- **Interpretation**: A direct pairwise metric signal is more useful than the
  GP-NLL weak-inner proxy in this pilot.  The projected-neighbor refit did not
  consistently help; raw final neighborhoods were better for the strongest
  5-seed variants.  The result supports trying a calibrated pairwise/VEM
  hybrid, but the current pairwise variants need a UQ calibration or ensemble
  layer before they are paper-ready.
- **Status**: Completed selected 5-seed run plus seed-0 GP-NLL pilot.

## 2026-05-13 - `lmjgp_qw_anchor_controls_paper_train1000_seed0`

- **Question**: Do the paper-synthetic LMJGP projection samples contain useful
  anchor-specific structure `x -> W(x)` beyond feature-axis alignment and
  marginal projection scale?
- **Runner**: `experiments/synthetic/compare_lmjgp_qw_controls_paper.py`
  after adding `anchor_shuffled_learned_qw` and `global_mean_learned_qw`.
- **Settings**:
  `--settings l2_q2_train1000,lh_q5_train1000 --seed 0 --out_dir
  experiments/synthetic/lmjgp_qw_anchor_controls_paper_train1000_seed0`.
  Paper-style L2/LH generators, train-only X standardization, `N_test=200`,
  `lmjgp_steps=300`, `MC_num=3`.
- **Artifacts**:
  `experiments/synthetic/lmjgp_qw_anchor_controls_paper_train1000_seed0/`
  with `metrics.csv`, `metrics_partial.csv`, `summary.json`, and `README.md`.
- **Headline (one seed, RMSE / CRPS / Cov90 / Width90)**:

  | setting | control | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | L2 train=1000 | learned `q(W)` | 2.271 | 1.270 | 0.845 | 6.141 |
  | L2 train=1000 | feature-permuted | 2.187 | 1.239 | 0.845 | 6.207 |
  | L2 train=1000 | anchor-shuffled | 2.260 | 1.267 | 0.845 | 6.093 |
  | L2 train=1000 | global-mean | 2.260 | 1.272 | 0.865 | 6.335 |
  | LH train=1000 | learned `q(W)` | 786.661 | 406.215 | 0.760 | 984.083 |
  | LH train=1000 | feature-permuted | 762.486 | 385.736 | 0.765 | 867.146 |
  | LH train=1000 | anchor-shuffled | 791.681 | 406.015 | 0.760 | 820.870 |
  | LH train=1000 | global-mean | 802.350 | 420.570 | 0.710 | 869.379 |

- **Interpretation**: On L2, anchor-shuffling and global-mean controls are
  essentially tied with learned `q(W)`, so this seed does not show meaningful
  evidence that the learned `x -> W(x)` assignment matters. On LH, global-mean
  degrades, suggesting some anchor-to-anchor sample variation helps, but
  anchor-shuffling remains close to learned `q(W)`, so the learned spatial
  assignment is still weak. This complements the feature-permutation result:
  learned feature alignment is weak, and learned anchor alignment is at best
  modest in this seed.
- **Status**: Completed one-seed diagnostic.

## 2026-05-13 - `lmjgp_qw_controls_paper_train1000_seed0`

- **Question**: On paper-style L2/LH synthetic data, does learned LMJGP
  `q(W_*)` improve the projected JumpGP prediction head over controls:
  prior samples, moment-matched isotropic samples, feature-permuted learned
  samples, and zero-mean learned-variance samples?
- **Runner**: `experiments/synthetic/compare_lmjgp_qw_controls_paper.py`.
- **Settings**:
  `--settings l2_q2_train1000,lh_q5_train1000 --seed 0 --lmjgp_steps 300
  --MC_num 3 --m1 2 --m2 40`.  Both settings use paper-style generation,
  RFF-expanded X, train-only X standardization, and 200 test anchors.  All
  controls use the same prediction head: sampled `W` -> projected JumpGP-CEM
  refit -> mixture aggregation.
- **Artifacts**:
  `experiments/synthetic/lmjgp_qw_controls_paper_train1000_seed0/`
  - `metrics_partial.csv` while running.
  - `metrics.csv`, `summary.json`, `README.md` when complete.
- **Headline (one seed, RMSE / CRPS / Cov90 / Width90)**:

  | setting | control | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | L2 train=1000 | learned `q(W)` | 2.271 | 1.270 | 0.845 | 6.141 |
  | L2 train=1000 | prior `q(W)` | 2.252 | 1.265 | 0.840 | 6.121 |
  | L2 train=1000 | moment-matched isotropic | 2.253 | 1.267 | 0.840 | 6.118 |
  | L2 train=1000 | feature-permuted learned | 2.187 | 1.239 | 0.845 | 6.207 |
  | L2 train=1000 | zero-mean learned sigma | 2.258 | 1.268 | 0.850 | 6.128 |
  | LH train=1000 | learned `q(W)` | 786.661 | 406.215 | 0.760 | 984.083 |
  | LH train=1000 | prior `q(W)` | 786.661 | 406.215 | 0.760 | 984.083 |
  | LH train=1000 | moment-matched isotropic | 779.515 | 401.721 | 0.760 | 1013.155 |
  | LH train=1000 | feature-permuted learned | 762.486 | 385.736 | 0.765 | 867.146 |
  | LH train=1000 | zero-mean learned sigma | 786.661 | 406.215 | 0.760 | 984.083 |

- **Interpretation**: In this one-seed diagnostic, learned `q(W)` is not better
  than prior/isotropic/permuted controls.  On LH, `mu_W` is essentially zero
  (`mean row norm ≈ 9e-13`) and `sigma_W` is constant (`≈ 0.316`), so learned,
  prior, and zero-mean learned-sigma controls collapse to the same row-normalized
  random projection distribution.  On L2, learned `mu_W` is nonzero, but the
  controls are still comparable or slightly better, so the downstream JumpGP
  head is not relying on useful learned directions in this seed.
- **Status**: Completed one-seed diagnostic.

## 2026-05-13 - `lmjgp_prediction_heads_paper_seed0`

- **Question**: For the same trained transductive LMJGP state on paper-style
  L2/LH synthetic data, how different is the default two-stage prediction
  head (`q(W_*)` samples + JumpGP-CEM refit) from a direct variational
  posterior predictive head using the learned local `u_params`?
- **Runner**: `experiments/synthetic/compare_lmjgp_prediction_heads_paper.py`.
- **Settings**:
  `--settings l2_q2_train500,lh_q5_train500 --seed 0 --lmjgp_steps 300
  --MC_num 3 --m1 2 --m2 40`.  Both settings use paper-style generation,
  RFF-expanded X, train-only X standardization, and 200 test anchors.
- **Artifacts**:
  `experiments/synthetic/lmjgp_prediction_heads_paper_seed0/`
  - `metrics.csv`, `summary.json`, `README.md`.
- **Implementation note**: The legacy `predict_vi_analytic` helper has stale
  shape/scale conventions (`U_logit` broadcasts to a matrix and raw
  `Sigma_u` is not PSD-transformed).  The runner therefore uses a local
  `direct_variational_consistent` head that mirrors the training ELBO
  conventions: PSD Cholesky `Sigma_u`, `sigma_k` for kernel expectations, and
  observation noise added to the inlier predictive variance.
- **Headline**:

  | setting | head | RMSE | CRPS | Cov90 | Width90 |
  |---|---|---:|---:|---:|---:|
  | L2 train=500 | `sampleW_jumpgp_refit` | 2.667 | 1.352 | 0.860 | 6.279 |
  | L2 train=500 | `direct_variational_consistent` | 4.763 | 4.210 | 0.015 | 0.655 |
  | LH train=500 | `sampleW_jumpgp_refit` | 707.283 | 363.224 | 0.825 | 1372.425 |
  | LH train=500 | `direct_variational_consistent` | 823.726 | 819.030 | 0.000 | 0.141 |

- **Interpretation**: The direct variational head is dramatically
  under-dispersed and worse in RMSE/CRPS.  The empirical performance of the
  current LMJGP pipeline is therefore coming mainly from the sampled projection
  plus JumpGP-CEM refit prediction head, not from the local `u_params`
  posterior being a reliable standalone predictive posterior.
- **Status**: Completed one-seed diagnostic.

## 2026-05-13 - `paper_l2_lh_xstd_train1000_5seeds`

- **Question**: Extend the corrected paper-style synthetic comparison to
  `N_train=1000` for both L2/phantom (`Q_true=2`) and LH (`Q_true=5`).
- **Runner**: `experiments/synthetic/compare_paper_synthetic_baselines.py`.
- **Settings**:
  `--num_exp 5 --settings l2_q2_train1000,lh_q5_train1000
  --out_dir experiments/synthetic/paper_l2_lh_xstd_train1000_5seeds --resume`.
  Methods are the runner defaults:
  `xgb_bootstrap,dkl_svgp,jgp_pca,lmjgp_baseline,lmjgp_pls_kl,cem_em_transductive_vi_fullcov_ensemble`.
  Both datasets use 200 test points and train-only X standardization after the
  RFF expansion.  L2 uses `generate_data_phantom` with `caseno=6`, latent
  dimension 2, observed dimension 20, fitted `Q=2`, `n=25`; LH uses
  `new_highdata_gen_utils.generate_data`, latent dimension 5, observed
  dimension 30, fitted `Q=5`, `n=35`.
- **Artifacts**:
  `experiments/synthetic/paper_l2_lh_xstd_train1000_5seeds/`
  - `metrics_partial.csv` is updated during the run.
  - `metrics.csv`, `aggregate.csv`, `summary.json`, and `README.md` are
    written when the run completes.
  - `run_stdout.log` and `run_stderr.log` capture the detached run.
- **Headline (RMSE / CRPS / Cov90 / Width90)**:

  | setting | best RMSE | best CRPS / UQ note |
  |---|---|---|
  | L2 train=1000 | `dkl_svgp` RMSE 2.375, closely followed by `lmjgp_baseline` RMSE 2.404 and XGB 2.436 | `lmjgp_baseline` CRPS 1.315, Cov90 0.805 |
  | LH train=1000 | `dkl_svgp` RMSE 528.9 | `cem_em_transductive_vi_fullcov_ensemble` CRPS 250.0; XGB has higher Cov90 0.755 |

- **Numerical status**: 60/60 rows succeeded.  However, `lh_q5_train1000`
  `lmjgp_pls_kl` seed 3 has a severe predictive variance blow-up
  (`CRPS=2.16e16`, `Width90=3.03e17`), so its aggregate CRPS/width is not
  interpretable without a pre-specified omit/winsorization rule.
- **Filtered outputs**:
  - `aggregate_omit_blowup.csv` omits only the exploded
    `lh_q5_train1000` / seed 3 / `lmjgp_pls_kl` row.  In that file,
    `lmjgp_pls_kl` on LH train=1000 becomes RMSE 567.9 / CRPS 238.4 /
    Cov90 0.733 / Width90 537.2 over 4 seeds.
  - `aggregate_common_seed_omit_lh_seed3.csv` omits seed 3 for all LH
    train=1000 methods, giving a strict common-seed comparison over 4 seeds.
- **Status**: Completed.

## 2026-05-12 - `paper_l2_lh_xstd_train200_500_5seeds`

- **Question**: Correct the synthetic-baseline provenance error by rerunning
  the train=200/500 comparison on the paper-style L2/phantom and LH synthetic
  data generators rather than the direct `lowrank_jump` toy generator.
- **Runner**: `experiments/synthetic/compare_paper_synthetic_baselines.py`.
- **Settings**:
  `--num_exp 5 --settings l2_q2_train200,l2_q2_train500,lh_q5_train200,lh_q5_train500
  --out_dir experiments/synthetic/paper_l2_lh_xstd_train200_500_5seeds --resume`.
  Methods are the runner defaults:
  `xgb_bootstrap,dkl_svgp,jgp_pca,lmjgp_baseline,lmjgp_pls_kl,cem_em_transductive_vi_fullcov_ensemble`.
  All rows use 200 test points and train-only X standardization after the
  Appendix-C RFF expansion.  L2 uses `generate_data_phantom` with `caseno=6`,
  latent `d=2`, observed `H=20`, fitted `Q=2`, `n=25`; LH uses
  `new_highdata_gen_utils.generate_data`, latent `d=5`, observed `H=30`,
  fitted `Q=5`, `n=35`.
- **Artifacts**:
  `experiments/synthetic/paper_l2_lh_xstd_train200_500_5seeds/`
  - `metrics_partial.csv` is updated during the run.
  - `metrics.csv`, `aggregate.csv`, `summary.json`, and `README.md` are
    written when the run completes.
  - `run_stdout.log` and `run_stderr.log` capture the detached overnight run.
- **Headline (RMSE / CRPS / Cov90 / Width90)**:

  | setting | strongest point result | strongest CRPS/UQ result |
  |---|---|---|
  | L2 train=200 | `xgb_bootstrap` RMSE 2.685 | `lmjgp_pls_kl` CRPS 1.474, Cov90 0.806 |
  | L2 train=500 | `dkl_svgp` RMSE 2.339 | `lmjgp_pls_kl` CRPS 1.280, Cov90 0.845 |
  | LH train=200 | `cem_em_transductive_vi_fullcov_ensemble` RMSE 550.6 / CRPS 230.3 but only 4 successful seeds | CEM has best CRPS; coverage is low (0.546) |
  | LH train=500 | `dkl_svgp` RMSE 538.0 | `cem_em_transductive_vi_fullcov_ensemble` CRPS 254.4, Cov90 0.526 |

- **Numerical status**: 119/120 rows succeeded.  The failed row is
  `lh_q5_train200`, seed 3, `cem_em_transductive_vi_fullcov_ensemble`
  (`ValueError: arange: cannot compute length`), so that aggregate uses 4
  successful seeds.  A manual retry was started and then interrupted before it
  wrote new outputs; the reported files remain the 02:35 completed outputs.
- **Status**: Completed with one failed CEM-EM row.

## 2026-05-12 - `lowrank_jump_cem_vi_xstd_q2_q5_train200_500_5seeds`

- **Question**: How does the strongest repaired CEM-EM posterior variant,
  `cem_em_transductive_vi_fullcov_ensemble`, behave on the same synthetic
  small-train tables for `Q_true=2` and `Q_true=5`?
- **Runner**: `experiments/synthetic/compare_synthetic_baselines.py`.
- **Settings**:
  `--num_exp 5 --settings train200,train500,train200_q5,train500_q5
  --N_test 200 --methods cem_em_transductive_vi_fullcov_ensemble
  --cem_outer 5 --cem_m_steps 80 --cem_vi_steps 80 --MC_num 3`.
  All settings use `D=50`, `n=25`, `jump_amp=1.5`, `noise_std=0.1`;
  X is standardized using training-set statistics.  CEM is transductive over
  test anchors but does not use test labels.  This is the direct
  `lowrank_jump` generator from `compare_lmjgp_tricks_synth.py`, not the
  paper's historical L2/LH expansion generator.
- **Artifacts**:
  `experiments/synthetic/lowrank_jump_cem_vi_xstd_q2_q5_train200_500_5seeds/`
  - `metrics.csv`, `aggregate.csv`, `summary.json`, `README.md`.
  - The first run timed out after writing partial results; `--resume` completed
    the remaining rows.  `run_resume.log` contains the final resumed command
    log; `run.log` is empty because redirected live output was buffered.
- **Headline (RMSE / CRPS / Cov90 / Width90)**:

  | train | Q_true | RMSE | CRPS | Cov90 | Width90 |
  |---:|---:|---:|---:|---:|---:|
  | 200 | 2 | 1.470 | 0.875 | 0.694 | 2.802 |
  | 500 | 2 | 1.377 | 0.820 | 0.654 | 2.531 |
  | 200 | 5 | 1.735 | 1.102 | 0.501 | 2.372 |
  | 500 | 5 | 1.628 | 1.002 | 0.553 | 2.344 |

- **Comparison to existing synthetic baselines**:
  For `Q_true=2`, CEM-EM VI is competitive in CRPS but not a clear winner:
  at train=200, LMJGP PLS+KL remains better (`1.428 / 0.817 / 0.905 /
  4.242`), while CEM is close to XGB in CRPS but has better coverage.  At
  train=500, CEM has the best CRPS by a hair (`0.820` vs LMJGP baseline
  `0.821`) and better RMSE than LMJGP, but XGB/DKL are stronger RMSE baselines.
  For `Q_true=5`, CEM does not improve the main table: XGB/DKL remain stronger
  point baselines and LMJGP has better CRPS/coverage.
- **Projection diagnostic note**:
  `median_capture_ratio_top_Q_truth` is `1.0` for CEM here because each CEM
  projection matrix has rank at most the fitted `Q`, so this capture-ratio
  diagnostic is not informative for comparing CEM rank-Q projections.  The
  subspace cosine is modest (`0.41-0.49` for Q=2 and `0.34-0.39` for Q=5),
  suggesting CEM's rank-Q subspace is not tightly aligned with the true
  generator subspace despite the rank diagnostic saturating.
- **Numerical status**: Completed 20 rows with no errors.
- **Status**: Completed.  Recommendation: include as an appendix row if we
  want to show the repaired CEM posterior ensemble is implemented; it is not a
  synthetic headline winner except for near-tied CRPS at Q=2, train=500.

## 2026-05-12 - `lowrank_jump_qtrue5_xstd_baselines_train200_500_5seeds`

- **Question**: If the synthetic latent jump structure has true dimension
  `Q_true=5` rather than `Q_true=2`, do the same X-standardized
  XGB/DKL/JGP-PCA/LMJGP comparisons change in the `N_train=200/500` regimes?
- **Runner**: `experiments/synthetic/compare_synthetic_baselines.py`.
- **Settings**:
  `--num_exp 5 --settings train200_q5,train500_q5 --N_test 200
  --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300
  --methods xgb_bootstrap,dkl_svgp,jgp_pca,lmjgp_baseline,lmjgp_pls_kl`.
  Both presets use `D=50`, `Q_true=5`, fitted `Q=5`, `n=25`,
  `jump_amp=1.5`, `noise_std=0.1`; X is standardized using training-set
  statistics before every method.  This is the direct `lowrank_jump`
  generator from `compare_lmjgp_tricks_synth.py`, not the paper's historical
  L2/LH expansion generator.
- **Artifacts**:
  `experiments/synthetic/lowrank_jump_qtrue5_xstd_baselines_train200_500_5seeds/`
  - `metrics.csv`, `aggregate.csv`, `summary.json`, `README.md`.
  - `run.log` is empty because the first long run redirected no live output;
    `run_resume.log` contains the resumed run log.  The final resume run
    exited successfully after all 50 rows were complete.
- **Headline (RMSE / CRPS / Cov90 / Width90)**:

  | train | method | RMSE | CRPS | Cov90 | Width90 |
  |---:|---|---:|---:|---:|---:|
  | 200 | XGB-bootstrap | 1.532 | 0.963 | 0.539 | 2.595 |
  | 200 | DKL-SVGP | 1.592 | 1.176 | 0.168 | 0.831 |
  | 200 | JGP-PCA | 2.007 | 1.289 | 0.521 | 2.993 |
  | 200 | LMJGP baseline | 1.687 | 0.978 | 0.783 | 4.263 |
  | 200 | LMJGP PLS+KL | 1.721 | 1.003 | 0.749 | 3.841 |
  | 500 | XGB-bootstrap | 1.481 | 0.928 | 0.504 | 2.342 |
  | 500 | DKL-SVGP | 1.468 | 1.055 | 0.213 | 0.778 |
  | 500 | JGP-PCA | 1.990 | 1.264 | 0.545 | 3.135 |
  | 500 | LMJGP baseline | 1.645 | 0.946 | 0.805 | 4.331 |
  | 500 | LMJGP PLS+KL | 1.647 | 0.948 | 0.780 | 4.008 |

- **Projection diagnostic headline**:
  PLS+KL again improves alignment diagnostics but does not improve predictive
  metrics.  Truth-subspace capture improves from `0.232` to `0.687` at
  train=200 and from `0.235` to `0.689` at train=500; median effective rank of
  `A` drops from about `40` to about `9`.  Despite this, LMJGP PLS+KL is not
  better than the LMJGP baseline in RMSE/CRPS on this Q=5 setting.
- **Interpretation**:
  Increasing the true latent dimension makes the synthetic task harder for
  local projection methods.  XGB is the best train=200 point/CRPS baseline, and
  DKL has the best train=500 RMSE but remains severely under-covered.  LMJGP
  keeps better coverage than XGB/DKL but no longer reaches the near-90%
  coverage seen in the Q=2 table, and its intervals remain much wider.
- **Numerical status**: Completed 50 rows with no errors.  The first run hit a
  command timeout after writing partial results; the runner now supports
  `--resume`, and the resumed run completed the table.
- **Status**: Completed.  This is a useful appendix stress test showing that
  higher true latent dimension weakens the LMJGP calibration advantage and that
  better projection recovery alone is not sufficient for better prediction.

## 2026-05-12 - `lowrank_jump_xstd_baselines_train200_500_5seeds`

- **Question**: In the same representative synthetic low-rank jump setting as
  the previous `N_train=1000` appendix table, do XGB/DKL/JGP-PCA/LMJGP behave
  differently in smaller training regimes (`N_train=200` and `500`)?
- **Runner**: `experiments/synthetic/compare_synthetic_baselines.py`.
- **Settings**:
  `--num_exp 5 --settings train200,train500 --N_test 200 --xgb_boots 30
  --dkl_epochs 80 --lmjgp_steps 300
  --methods xgb_bootstrap,dkl_svgp,jgp_pca,lmjgp_baseline,lmjgp_pls_kl`.
  Both presets use `D=50`, `Q_true=2`, fitted `Q=2`, `n=25`,
  `jump_amp=1.5`, `noise_std=0.1`; X is standardized using training-set
  statistics before every method.  This is the direct `lowrank_jump`
  generator from `compare_lmjgp_tricks_synth.py`, not the paper's historical
  L2/LH expansion generator.
- **Artifacts**:
  `experiments/synthetic/lowrank_jump_xstd_baselines_train200_500_5seeds/`
  - `metrics.csv`, `aggregate.csv`, `summary.json`, `README.md`.
- **Headline (RMSE / CRPS / Cov90 / Width90)**:

  | train | method | RMSE | CRPS | Cov90 | Width90 |
  |---:|---|---:|---:|---:|---:|
  | 200 | XGB-bootstrap | 1.369 | 0.877 | 0.507 | 2.370 |
  | 200 | DKL-SVGP | 1.459 | 1.049 | 0.210 | 0.734 |
  | 200 | JGP-PCA | 1.569 | 0.928 | 0.880 | 4.544 |
  | 200 | LMJGP baseline | 1.473 | 0.851 | 0.901 | 4.392 |
  | 200 | LMJGP PLS+KL | 1.428 | 0.817 | 0.905 | 4.242 |
  | 500 | XGB-bootstrap | 1.321 | 0.853 | 0.486 | 2.210 |
  | 500 | DKL-SVGP | 1.201 | 0.832 | 0.257 | 0.721 |
  | 500 | JGP-PCA | 1.526 | 0.898 | 0.887 | 4.476 |
  | 500 | LMJGP baseline | 1.421 | 0.821 | 0.898 | 4.386 |
  | 500 | LMJGP PLS+KL | 1.431 | 0.827 | 0.878 | 4.263 |

- **Interpretation**:
  At `N_train=200`, XGB has the best RMSE, while LMJGP PLS+KL has the best
  CRPS and coverage.  At `N_train=500`, DKL has the best RMSE and LMJGP
  baseline is marginally best CRPS.  DKL remains severely under-covered in
  both regimes; XGB also under-covers but less severely.  LMJGP/JGP-PCA
  provide near-nominal coverage with much wider intervals.  The PLS+KL
  projection prior substantially improves truth-subspace capture at both train
  sizes (`0.613` vs `0.192` at train=200; `0.584` vs `0.173` at train=500),
  but predictive gains are modest.
- **Numerical status**: Completed 50 rows with no errors.
- **Status**: Completed.  Suitable as an appendix small-train synthetic
  baseline table; main narrative should emphasize calibration/uncertainty
  rather than uniform RMSE dominance.

## 2026-05-12 - `lowrank_jump_xstd_baselines_moderate_5seeds`

- **Question**: On a representative synthetic low-rank jump benchmark, how do
  the paper's generic real-data baselines compare with JGP-PCA and LMJGP when
  all methods use train-only X standardization?
- **Runner**: `experiments/synthetic/compare_synthetic_baselines.py`.
- **Settings**:
  `--num_exp 5 --settings moderate --N_test 200 --xgb_boots 30
  --dkl_epochs 80 --lmjgp_steps 300
  --methods xgb_bootstrap,dkl_svgp,jgp_pca,lmjgp_baseline,lmjgp_pls_kl`.
  The `moderate` preset is `N_train=1000`, `N_test=200`, `D=50`,
  `Q_true=2`, fitted `Q=2`, `n=25`, `jump_amp=1.5`, `noise_std=0.1`.
  X is standardized using training-set statistics before every method.  This is
  the direct `lowrank_jump` generator from `compare_lmjgp_tricks_synth.py`,
  not the paper's historical L2/LH expansion generator.
- **Artifacts**:
  `experiments/synthetic/lowrank_jump_xstd_baselines_moderate_5seeds/`
  - `metrics.csv`, `aggregate.csv`, `summary.json`, `README.md`.
  - Smoke output:
    `experiments/synthetic/lowrank_jump_xstd_baselines_smoke/`.
- **Headline (RMSE / CRPS / Cov90 / Width90)**:

  | method | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | DKL-SVGP | 1.066 | 0.702 | 0.318 | 0.720 |
  | XGB-bootstrap | 1.325 | 0.853 | 0.480 | 2.133 |
  | LMJGP PLS+KL | 1.439 | 0.824 | 0.880 | 4.278 |
  | LMJGP baseline | 1.441 | 0.826 | 0.904 | 4.375 |
  | JGP-PCA | 1.561 | 0.918 | 0.880 | 4.503 |

- **Projection diagnostic headline**:
  `lmjgp_pls_kl` substantially increased truth-subspace capture relative to
  the random-init baseline (`median_capture_ratio_top_Q_truth` mean `0.584`
  versus `0.173`, and `median_eff_rank_A` mean `5.43` versus `33.86`), but the
  predictive RMSE/CRPS barely changed.  This suggests the projection diagnostic
  can improve without immediately translating into point-prediction gains in
  the current local JumpGP prediction pipeline.
- **Interpretation**:
  DKL is the strongest point/CRPS baseline on this representative synthetic
  setting but severely under-covers.  LMJGP and JGP-PCA have much higher
  coverage, mostly through wider intervals.  This is suitable as an appendix
  sanity table, not as evidence that LMJGP dominates generic black-box
  baselines on all synthetic settings.
- **Status**: Completed representative 5-seed table.  The optional hard
  preset (`D=100`, `N_train=500`) was not launched in this pass because the
  moderate table took about 42 minutes with two LMJGP variants; run it only if
  the appendix needs a harder second synthetic point.

## 2026-05-12 - `repaired_projection_variants_paper_test200_2methods`

- **Question**: For a paper-facing table comparable with the existing
  XGBoost/DKL/base-LMJGP small-train UCI results, how do the two selected
  repaired variants perform with 200 test anchors?
- **Runner**: `experiments/uci/run_repaired_projection_variants.py`.
- **Settings**:
  `--datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --train_sizes 200,500 --num_exp 5 --max_test_anchors 200
  --max_train_anchors 128 --n -1 --lmjgp_steps 300 --cem_vi_steps 80
  --n_outer 5 --n_m_steps 80 --MC_num 3 --paper_smalltrain_preset 1
  --cem_output_variants vi --methods "cem_transductive_vi,lowrank_cem_mean_cem_svd"`.
  The split/scaling policy matches the paper small-train summary: Wine random
  row + `std_x`, Appliances random row + `std_x,std_y`, and Parkinsons
  subject-grouped + `std_x`.
- **Artifacts**:
  `experiments/uci/repaired_projection_variants_paper_test200_2methods/`
  - `metrics.csv`, `aggregate.csv`, `summary.json`, `run.log`.
  - `aggregate_paper_named.csv` renames
    `lmjgp_lowrank_cem_mean_cem_svd` to
    `lmjgp_transductive_low_rank_cem_residuals`.
  - `aggregate_with_baselines.csv` appends these two methods to
    `experiments/uci/smalltrain_multiseed_bestpreset_summary/aggregate_train200_500_bestpresets.csv`.
- **Headline (RMSE / CRPS / Cov90 / Width90)**:

  | dataset | train | CEM transductive VI | LMJGP low-rank CEM residuals |
  |---|---:|---:|---:|
  | Wine | 200 | 0.850 / 0.514 / 0.652 / 1.559 | 0.779 / 0.440 / 0.852 / 2.194 |
  | Wine | 500 | 0.839 / 0.491 / 0.684 / 1.478 | 0.750 / 0.423 / 0.826 / 2.046 |
  | Parkinsons | 200 | 10.022 / 6.472 / 0.570 / 18.093 | 9.288 / 5.746 / 0.678 / 21.374 |
  | Parkinsons | 500 | 9.921 / 6.588 / 0.491 / 15.702 | 9.143 / 5.723 / 0.652 / 19.821 |
  | Appliances | 200 | 111.753 / 46.877 / 0.681 / 112.814 | 111.669 / 46.352 / 0.751 / 126.719 |
  | Appliances | 500 | 108.911 / 45.685 / 0.685 / 120.184 | 107.386 / 43.513 / 0.764 / 138.959 |

- **Comparison to existing baseline summary**:
  These two repaired methods are not uniformly table winners.  In
  `aggregate_with_baselines.csv`, XGBoost remains best RMSE on Wine and
  Appliances train=500, DKL remains best on grouped Parkinsons, and base
  LMJGP variants remain strongest CRPS on Wine/Appliances.  The low-rank CEM
  residual variant is the stronger of the two repaired methods and is closest
  to existing LMJGP baselines on Wine/Appliances, but it is not enough to
  displace DKL on Parkinsons.
- **Numerical status**:
  60 metric rows were produced.  One CEM transductive VI row on Wine
  train=200 seed=3 omitted a single non-finite/exploding prediction according
  to the existing numeric-filter policy; no other rows required omission.
- **Status**: Completed.  Paper recommendation: include at most the low-rank
  CEM residual as an ablation/appendix row, and use CEM transductive VI mainly
  as evidence that a cleaner CEM posterior ensemble is implemented rather than
  as a headline UCI winner.

## 2026-05-11 - `repaired_projection_variants_train200_500_5seeds`

- **Question**: After fixing supervised LMJGP reparameterization, adding a
  full-anchor-covariance VI posterior for CEM-EM coefficients, and adding
  low-rank residual LMJGP variants with CEM-derived `W0`/`V`, which repaired
  projection variants are strongest on the UCI preset splits?
- **Runner**: `experiments/uci/run_repaired_projection_variants.py`.
- **Settings**:
  `--datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --train_sizes 200,500 --num_exp 5 --max_test_anchors 64
  --max_train_anchors 64 --n -1 --lmjgp_steps 300 --cem_vi_steps 80
  --n_outer 5 --n_m_steps 80 --MC_num 3 --scaling_preset uci_empirical`.
  Scaling preset: Wine=`std_x`, Parkinsons=raw, Appliances=`std_x,std_y`;
  Wine/Parkinsons use `n=25`, Appliances uses `n=35`.
- **Artifacts**:
  `experiments/uci/repaired_projection_variants_train200_500_5seeds/`
  - `metrics.csv`, `metrics_partial.csv`, `aggregate.csv`, `summary.json`,
    `README.md`.
  - Smoke: `experiments/uci/repaired_projection_variants_smoke/`.
- **Methods included**:
  CEM-EM supervised/transductive with GP-conditional, diagonal-Laplace, and
  full-covariance VI coefficient ensembles; low-rank residual LMJGP with
  PLS/PCA `W0,V`, CEM-mean `W0` + PLS/PCA `V`, and CEM-mean `W0` + CEM-SVD
  `V`; corrected supervised LMJGP with differentiable
  `W_a = mu_a + L_a eps`.
- **Best RMSE / CRPS by dataset and train size**:

  | dataset | train | best RMSE | best CRPS |
  |---|---:|---|---|
  | Wine | 200 | LMJGP supervised reparam: 0.756 | low-rank PLS/PCA: 0.423 |
  | Wine | 500 | CEM supervised VI fullcov: 0.749 | low-rank CEM-mean + PLS/PCA: 0.420 |
  | Parkinsons | 200 | low-rank PLS/PCA: 8.131 | low-rank PLS/PCA: 4.681 |
  | Parkinsons | 500 | low-rank CEM-mean + PLS/PCA: 7.021 | low-rank CEM-mean + PLS/PCA: 3.776 |
  | Appliances | 200 | low-rank CEM-mean + CEM-SVD: 113.483 | low-rank CEM-mean + CEM-SVD: 47.358 |
  | Appliances | 500 | CEM transductive VI fullcov: 105.146 | CEM transductive VI fullcov: 43.514 |

- **CEM-EM ensemble headline**:
  Full-covariance VI is consistently useful on Parkinsons and often better
  than GP-conditional/Laplace.  At train=500, supervised VI improves
  Parkinsons from GP-cond `7.623 / 4.230` to `7.185 / 3.866`; transductive VI
  improves Appliances from GP-cond `113.284 / 49.522` to `105.146 / 43.514`.
  Wine gains are smaller but supervised VI is the best Wine train=500 RMSE.
- **Low-rank residual headline**:
  CEM-derived `W0`/`V` helps in some regimes but is not uniformly best.
  CEM-mean + PLS/PCA is strongest on Parkinsons train=500, while CEM-mean +
  CEM-SVD is strongest on Appliances train=200.  Plain PLS/PCA remains best on
  Parkinsons train=200 and has the best Wine train=200 CRPS.
- **Numerical status**:
  `metrics.csv` has 300 rows (3 datasets x 2 train sizes x 5 seeds x 10
  variants).  No prediction rows required omitting non-finite/exploding points.
  The VI sampler needed a PSD fallback for near-PSD query covariances; this is
  implemented in `djgp/projections/vi_cem.py`.
- **Status**: Completed 5-seed train=200/500 pilot.  Current recommendation:
  keep CEM full-covariance VI as the cleanest CEM posterior ensemble; treat
  low-rank residual LMJGP as promising but basis-dependent; keep supervised
  LMJGP reparameterized version as the correct implementation.

## 2026-05-11 - `cemem_projected_refit_pilot_train500_seed0`

- **Question**: Does CEM-EM improve if projection learning uses a larger raw-X
  candidate set and the final JumpGP refits on a projected-space small
  neighborhood? Does a simple cluster-anchor shared-W version stabilize the
  projection field?
- **Runner**: `experiments/uci/compare_cemem_projected_refit.py`.
- **Settings**:
  One seed, train=500, max test anchors=200, current small-train split/scaling
  policy, `n_candidate=100`, dataset-specific `n_final` from tuned presets
  (Wine/Parkinsons 25, Appliances 35), `n_clusters=64`, `n_outer=5`,
  `n_m_steps=80`. Variants: `cemem_small_current`,
  `cemem_large_raw_final`, `cemem_large_projected_final`, and
  `cluster_anchor_projected_final`.
- **Artifacts**:
  `experiments/uci/cemem_projected_refit_pilot_train500_seed0/`
  - `metrics.csv`, `aggregate.csv`, `summary.json`, `README.md`.
  - `comparison_same_split_seed0_train500.csv` and
    `comparison_same_split_seed0_train500_summary.json` compare the new rows
    against the current same-split bestpreset seed-0 baselines.
- **Headline (same split, seed 0)**:

  | dataset | best new variant | new RMSE / CRPS | best existing same-split row | existing RMSE / CRPS |
  |---|---|---:|---|---:|
  | Wine Quality | `cemem_large_raw_final` | 0.780 / 0.479 | `lmjgp_supervised_residual_ridge` | 0.697 / 0.395 |
  | Parkinsons Telemonitoring | `cemem_large_raw_final` | 10.083 / 6.904 | `dkl_svgp` | 7.106 / 4.105 |
  | Appliances Energy Prediction | `cemem_small_current` RMSE, `cemem_large_raw_final` CRPS | 107.241 / 42.348 | `xgb_bootstrap` RMSE, `lmjgp_supervised` CRPS | 99.977 / 38.792 |

  Large-candidate learning is the best new CEM-EM variant on Wine/Parkinsons
  and improves Appliances CRPS relative to small-current, but it does not beat
  the current best same-split baselines. Projected-small final refit and the
  simple cluster-anchor variant are not better in this pilot.
- **Status**: Completed one-seed pilot. Follow-up only if pursuing CEM-EM:
  try label-aware/regime-aware cluster anchors or tune `n_candidate`/CEM
  regularization; do not promote the current projected-refit variant to the
  main table.

## 2026-05-11 - `cem_laplace_lowrank_lmjgp_1seed`

- **Question**: Does a diagonal-Laplace coefficient posterior make supervised
  CEM-EM a more complete projection ensemble, and does fixed-mean low-rank
  residual VI improve transductive LMJGP over the full mean-zero projection
  field?
- **Runners**:
  `experiments/uci/compare_projection_uncertainty_controls.py`,
  `experiments/uci/compare_lowrank_residual_lmjgp.py`, and a small
  supervised-gradient smoke run through
  `experiments/uci/compare_inductive_projection.py`.
- **Settings**:
  `--datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --max_total_train 500 --max_test_anchors 64 --max_train_anchors 64
  --n -1 --lmjgp_steps 300 --MC_num 3 --scaling_preset uci_empirical`.
  CEM-EM used `--n_outer 5 --n_m_steps 80`.  Scaling preset:
  Wine=`std_x`, Parkinsons=raw, Appliances=`std_x,std_y`; Wine/Parkinsons
  use `n=25`, Appliances uses `n=35`.
- **Artifacts**:
  `experiments/uci/projection_uncertainty_controls_laplace_1seed/`
  (`metrics.csv`, `aggregate.csv`, `summary.json`, `README.md`),
  `experiments/uci/lowrank_residual_lmjgp_1seed/`
  (`metrics.csv`, `summary.json`, `README.md`), and supervised gradient
  smoke output in `experiments/uci/lmjgp_supervised_grad_smoke/`.
- **CEM-EM headline (RMSE / CRPS / Cov90 / Width90)**:

  | dataset | MAP mean-C | GP-cond ensemble | diagonal-Laplace ensemble |
  |---|---:|---:|---:|
  | Wine | 0.738 / 0.418 / 0.797 / 1.551 | 0.662 / 0.384 / 0.766 / 1.945 | 0.705 / 0.404 / 0.828 / 1.978 |
  | Parkinsons | 7.491 / 4.202 / 0.625 / 11.113 | 7.410 / 4.219 / 0.672 / 13.487 | 7.211 / 3.936 / 0.719 / 13.597 |
  | Appliances | 109.994 / 42.399 / 0.656 / 99.903 | 97.640 / 37.360 / 0.844 / 144.338 | 91.826 / 38.032 / 0.875 / 167.511 |

  The diagonal-Laplace posterior improves Parkinsons and Appliances over the
  MAP point estimate, and raises coverage over the plain GP-conditional
  ensemble.  On Wine, plain GP conditioning has the best RMSE/CRPS while
  diagonal-Laplace gives the highest coverage.  The Laplace curvature median is
  finite on all three datasets, so the implementation is numerically usable in
  this pilot.
- **Low-rank residual LMJGP headline (RMSE / CRPS / Cov90 / Width90)**:

  | dataset | full transductive LMJGP | low-rank residual PLS/PCA |
  |---|---:|---:|
  | Wine | 0.715 / 0.414 / 0.766 / 1.995 | 0.660 / 0.384 / 0.797 / 2.023 |
  | Parkinsons | 6.602 / 3.549 / 0.828 / 17.716 | 7.151 / 3.981 / 0.891 / 22.339 |
  | Appliances | 104.662 / 41.479 / 0.875 / 163.835 | 108.633 / 42.727 / 0.844 / 168.770 |

  The fixed PLS/PCA mean plus low-rank residual coefficient field helps Wine,
  but hurts Parkinsons and Appliances in this first one-seed setting.  This
  suggests the low-rank residual prior is viable, but `W0`, basis rank, and
  residual scale need validation instead of a fixed one-size preset.
- **Supervised LMJGP gradient smoke**:
  with `max_total_train=120`, 12 supervised anchors, 8 test anchors, and 5
  steps on Wine, the predictive-only loss decreased from `3.1809` to `3.1460`
  in one optimizer step.  Gradient norms were nonzero:
  `||grad_mu_V||=4.13e-02`, `||grad_sigma_V||=1.07e-04`,
  `||grad_Z||=1.06e-01`, confirming the reparameterized
  `W_a = mu_a + L_a eps` path carries gradients into `q(V)`.
- **Status**: Completed seed-0 implementation pilot.  The CEM-EM diagonal
  Laplace posterior is worth keeping as the theoretically cleaner CEM ensemble.
  The low-rank residual LMJGP implementation is correct enough for ablation,
  but should be tuned over `W0`, `V`, rank, and residual scale before drawing
  conclusions.

## 2026-05-11 - `projection_uncertainty_controls_1seed` (CEM-EM ensemble and LMJGP triviality controls)

- **Question**: On UCI one-seed slices, does supervised CEM-EM benefit from a
  projection-coefficient ensemble, and does learned LMJGP `q(W)` beat
  prior/isotropic/direction-destroyed projection controls?
- **Runner**: `experiments/uci/compare_projection_uncertainty_controls.py`.
- **Settings**:
  `--datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --max_total_train 500 --max_test_anchors 64 --max_train_anchors 64
  --n -1 --lmjgp_steps 300 --n_outer 5 --n_m_steps 80 --MC_num 3
  --scaling_preset uci_empirical`.  Scaling preset: Wine=`std_x`,
  Parkinsons=raw, Appliances=`std_x,std_y`.  Wine/Parkinsons use `n=25`;
  Appliances uses `n=35`.
- **Artifacts**:
  `experiments/uci/projection_uncertainty_controls_1seed/`
  - `metrics.csv`, `aggregate.csv`, `summary.json`, `README.md`.
  - Smoke: `experiments/uci/projection_controls_smoke/`.
- **CEM-EM headline (RMSE / CRPS / Cov90 / Width90)**:

  | dataset | MAP mean-C | GP-cond ensemble | GP-cond var x3 |
  |---|---:|---:|---:|
  | Wine | 0.738 / 0.418 / 0.797 / 1.551 | 0.662 / 0.384 / 0.766 / 1.945 | 0.697 / 0.408 / 0.812 / 1.968 |
  | Parkinsons | 7.491 / 4.202 / 0.625 / 11.113 | 7.410 / 4.219 / 0.672 / 13.487 | 7.246 / 4.056 / 0.672 / 13.347 |
  | Appliances | 109.994 / 42.399 / 0.656 / 99.903 | 97.640 / 37.360 / 0.844 / 144.338 | 111.041 / 43.668 / 0.922 / 209.798 |

  The empirical-Bayes GP-conditional coefficient ensemble improves RMSE/CRPS on
  Wine and Appliances and raises coverage on Parkinsons/Appliances. Inflating
  coefficient variance improves Parkinsons and coverage, but hurts Appliances
  by making intervals too wide. This supports adding projection uncertainty to
  CEM-EM, but variance scale needs validation calibration. This is not a
  Laplace posterior yet.
- **LMJGP control headline (RMSE / CRPS / Cov90 / Width90)**:

  | dataset | learned `q(W)` | prior unit rows | isotropic matched | permuted features | zero-mean learned sigma |
  |---|---:|---:|---:|---:|---:|
  | Wine | 0.706 / 0.400 / 0.812 / 1.991 | 0.724 / 0.423 / 0.844 / 2.096 | 0.701 / 0.402 / 0.797 / 1.937 | 0.667 / 0.382 / 0.859 / 2.041 | 0.681 / 0.386 / 0.859 / 1.959 |
  | Parkinsons | 6.591 / 3.473 / 0.875 / 16.513 | 7.427 / 4.109 / 0.828 / 18.502 | 6.809 / 3.769 / 0.828 / 18.083 | 6.726 / 3.693 / 0.844 / 17.727 | 7.194 / 3.874 / 0.828 / 18.338 |
  | Appliances | 93.453 / 38.641 / 0.891 / 162.324 | 87.482 / 36.363 / 0.891 / 189.022 | 93.622 / 38.038 / 0.875 / 186.486 | 107.063 / 41.766 / 0.844 / 167.554 | 100.025 / 38.852 / 0.875 / 171.300 |

  Wine does not support a learned-direction story: permuted/zero-mean controls
  slightly beat learned `q(W)`. Parkinsons does support a nontrivial `q(W)`:
  learned `q(W)` is best among the controls. Appliances is mixed: prior unit
  rows have the best RMSE/CRPS, but feature permutation hurts, so direction
  information may matter but is not robustly better than random-projection
  smoothing in this one seed.
- **Status**: Completed one-seed pilot. Follow-up: implement a real
  diagonal-Laplace or variational coefficient posterior for CEM-EM, and repeat
  LMJGP controls over 5 seeds before making a paper claim about learned
  directions.

## 2026-05-11 - `pdebench_projection_matrix_diag_balanced_q3n25`

- **Question**: On the balanced PDEBench-Burgers ShockJump surrogate where
  LMJGP has strong CRPS, do the learned projection matrices themselves carry
  response-aware or spatially meaningful structure?
- **Runner**: `experiments/pdebench/diagnose_projection_matrices.py`.
- **Settings**:
  `--dataset-folder data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000
  --num_exp 1 --max_test_anchors 128 --max_train_anchors 128 --Q 3
  --n 25 --lmjgp_steps 300 --n_outer 5 --n_m_steps 80 --mc_diag 16`.
  Methods: `lmjgp_transductive`, `lmjgp_supervised`,
  `cem_em_transductive`, `cem_em_supervised`, `pls_only`, and `pca_train`.
  The processed PDEBench split already contains train-only standardized `X`
  and `y`. Diagnostics do not run final JumpGP prediction.
- **Artifacts**:
  `experiments/pdebench/projection_matrix_diag_balanced_q3n25/`
  - `summary.csv`, `aggregate.csv`, `per_anchor.csv`, `summary.json`,
    `README.md`.
  - Smoke: `experiments/pdebench/projection_matrix_diag_smoke/`.
- **Headline (seed 0, 128 test anchors; 28 near-threshold / 100 far)**:

  | method | feature eff. count | spatial COM | pair-distance response delta | KNN-y delta |
  |---|---:|---:|---:|---:|
  | LMJGP transductive | 127.98 / 128 | 0.500 | -0.028 | +0.003 |
  | LMJGP supervised | 127.98 / 128 | 0.500 | -0.028 | +0.003 |
  | CEM-EM transductive | 106.57 / 128 | 0.417 | +0.003 | +0.004 |
  | CEM-EM supervised | 112.85 / 128 | 0.447 | -0.008 | 0.000 |
  | PLS-only | 117.47 / 128 | 0.494 | -0.034 | +0.001 |
  | PCA-train | 105.30 / 128 | 0.347 | -0.036 | +0.006 |

  The LMJGP projection posterior does not show feature selection or
  response-aware geometry here: row-normalized posterior samples spread weight
  nearly uniformly over all 128 spatial coordinates, the spatial weight
  center-of-mass is exactly central, and projected local distances are slightly
  less aligned with `|Delta y|` than raw-X distances. `E[W]` is essentially
  zero, but the metric moment is nonzero because of posterior variance, so this
  is not just an `E[W]` diagnostic artifact. CEM-EM transductive is less diffuse
  and has a weak positive pair-distance response delta, especially near the
  threshold (`+0.016`), but the effect is modest. The supervised LMJGP run
  reported `mu_V/sigma_V` gradient norms of zero in its predictive gradient
  check and produced diagnostics indistinguishable from transductive LMJGP.
- **Status**: Completed seed-0 diagnostic. Current interpretation: PDEBench
  LMJGP's predictive advantage is not explained by learning a clearly
  response-aware projection matrix; it is more likely coming from local
  neighborhoods, JumpGP behavior, projection sampling/uncertainty aggregation,
  or random-projection-like smoothing rather than an interpretable learned `W`.

## 2026-05-11 - `projection_matrix_std_diagnostics` (standardized-UCI projection matrix diagnostics)

- **Question**: After standardizing UCI inputs, do LMJGP and CEM-EM
  transductive/supervised variants learn informative local projection matrices,
  or do the learned `W` matrices still look uninformative as in the earlier raw
  scale diagnostic?
- **Runner**: `experiments/uci/diagnose_projection_matrices_std.py`.
- **Settings**:
  `--datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --num_exp 1 --max_total_train 500 --max_test_anchors 64
  --max_train_anchors 64 --n -1 --lmjgp_steps 300 --n_outer 5
  --n_m_steps 80 --mc_diag 16 --standardize_x 1
  --standardize_y_mode uci_empirical`. X is standardized for all three
  datasets; y is standardized only for Appliances, matching the current UCI
  empirical preset. Wine/Parkinsons use `n=25`; Appliances uses `n=35`.
  Diagnostics are computed on induced test-anchor projections and do not run the
  final JumpGP prediction pass.
- **Artifacts**:
  `experiments/uci/projection_matrix_std_diagnostics/`
  - `summary.csv`, `aggregate.csv`, `per_anchor.csv`, `summary.json`,
    `README.md`.
  - Smoke: `experiments/uci/projection_matrix_std_smoke/`.
- **Headline (seed 0)**:

  | dataset | method | feature eff. count | pair-distance response delta | anchor-KNN response delta |
  |---|---|---:|---:|---:|
  | Wine | LMJGP transductive | 10.95 / 11 | -0.051 | 0.000 |
  | Wine | LMJGP supervised | 11.00 / 11 | -0.046 | 0.000 |
  | Wine | CEM-EM transductive | 10.23 / 11 | +0.041 | 0.000 |
  | Wine | CEM-EM supervised | 10.79 / 11 | -0.013 | 0.000 |
  | Parkinsons | LMJGP transductive | 17.90 / ~19 | -0.027 | +0.511 |
  | Parkinsons | LMJGP supervised | 17.99 / ~19 | -0.023 | +0.457 |
  | Parkinsons | CEM-EM transductive | 14.98 / ~19 | +0.091 | -0.130 |
  | Parkinsons | CEM-EM supervised | 16.04 / ~19 | -0.000 | +0.193 |
  | Appliances | LMJGP transductive | 26.99 / 27 | -0.029 | -0.005 |
  | Appliances | LMJGP supervised | 26.99 / 27 | -0.029 | -0.005 |
  | Appliances | CEM-EM transductive | 19.34 / 27 | +0.056 | -0.044 |
  | Appliances | CEM-EM supervised | 23.19 / 27 | +0.001 | 0.000 |

  `feature eff. count` is the entropy effective number of input features in
  the metric diagonal; values near the input dimension mean diffuse/no feature
  selection. `pair-distance response delta` is projected pairwise-distance vs
  `|Delta y|` correlation minus the same raw-X correlation inside local
  neighborhoods; positive is better. `anchor-KNN response delta` is
  projected-nearest-neighbor mean `|y_i-y_anchor|` minus raw-X KNN; negative is
  better.

  Standardizing X did not make LMJGP projections clearly response-aware on these
  UCI slices. LMJGP row-normalized posterior samples still spread weight almost
  uniformly over features, especially Appliances where transductive and
  supervised diagnostics are nearly identical. CEM-EM transductive is the only
  variant that consistently moves the projection geometry in the expected
  direction on Parkinsons and Appliances; supervised CEM interpolation is
  weaker. For LMJGP, `E[W]` alone is not the right diagnostic, but even the
  prediction-aligned row-normalized samples do not show much response-aware
  structure.
- **Status**: Completed seed-0 diagnostic. Recommended follow-up is not more
  standardization; it is to test stronger supervised signals or response-aware
  neighborhood/global-residual mechanisms if projection learning on tabular UCI
  is important for the paper.

## 2026-05-11 - `pdebench_burgers_jgp_pca_q3n25`

- **Question**: On the balanced PDEBench-Burgers ShockJump surrogate, does a
  deterministic `JGP-PCA` baseline beat LMJGP when both use the same latent
  dimension and neighborhood size?
- **Runner**: `experiments/pdebench/compare_pdebench_burgers.py`.
- **Settings**:
  `--dataset-folder data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000
  --num_exp 5 --Q 3 --neighbors 25 --include_xgb 0 --include_dkl 0
  --include_lmjgp 0 --include_cemem 0 --include_jgp_pca 1`. The processed
  dataset already contains train-only standardized `X_train/X_test`, so no raw-X
  rerun was performed. PCA is fit on the training inputs only.
- **Artifacts**: `results/`
  - `pdebench_burgers_balanced_delta01_n2000_jgp_pca_q3_n25_seeds0to4.{json,pkl}`.
  - `pdebench_burgers_balanced_delta01_n2000_jgp_pca_vs_lmjgp_q3_n25_seeds0to4_summary.{json,csv}`.
- **Headline (mean +/- population std over seeds 0-4)**:

  | subset | JGP-PCA RMSE | LMJGP RMSE | JGP-PCA CRPS | LMJGP CRPS |
  |---|---:|---:|---:|---:|
  | all | 1.025 +/- 0.000 | 0.714 +/- 0.005 | 0.404 +/- 0.000 | 0.240 +/- 0.003 |
  | near-threshold | 1.421 +/- 0.000 | 0.877 +/- 0.023 | 0.813 +/- 0.000 | 0.377 +/- 0.016 |
  | far-from-threshold | 0.837 +/- 0.000 | 0.644 +/- 0.008 | 0.256 +/- 0.000 | 0.191 +/- 0.003 |

  `JGP-PCA` is deterministic on this fixed split, hence the zero seed-to-seed
  std. It explains about 96.3% of standardized input variance with `Q=3`, but
  it is worse than LMJGP on RMSE and CRPS for all/near/far subsets.
- **Status**: Completed. Use `JGP-PCA` as an appendix baseline; it does not
  improve the PDEBench-Burgers result.

## 2026-05-11 - `xgb_conformal_quantile_smalltrain_5seeds` (conformalized quantile XGBoost)

- **Question**: Does a stronger uncertainty baseline, conformalized quantile
  XGBoost (CQR), fix the undercoverage of naive bootstrap XGBoost on the UCI
  small-train runs?
- **Runner**: `experiments/uci/compare_xgb_conformal_quantile.py`.
- **Settings**:
  `--num_exp 5 --train_sizes 200,500,1000
  --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --out_dir experiments/uci/xgb_conformal_quantile_smalltrain_5seeds`.
  Split conformal CQR uses 80% proper training / 20% calibration inside each
  small training set. Lower/upper XGBoost quantile models are fit on the proper
  training split, conformal corrections are computed on calibration rows, and a
  separate squared-error XGBoost model fit on all training rows provides RMSE.
  CQR intervals are direct conformal intervals, not Gaussian posteriors, so CRPS
  is intentionally not reported.
- **Artifacts**:
  `experiments/uci/xgb_conformal_quantile_smalltrain_5seeds/`
  - `metrics.csv`, `aggregate.csv`, `xgb_cqr_vs_bootstrap.csv`,
    `summary.json`, `README.md`, and `run.log`.
  - Smoke: `experiments/uci/xgb_conformal_quantile_smoke/`.
- **Headline (mean over 5 seeds, CQR Cov90 / Width90)**:

  | train | Wine | Parkinsons grouped | Appliances |
  |---:|---|---|---|
  | 200 | 0.960 / 3.793 | 0.831 / 23.703 | 0.900 / 269.672 |
  | 500 | 0.910 / 2.736 | 0.707 / 19.015 | 0.946 / 343.492 |
  | 1000 | 0.915 / 2.591 | 0.649 / 17.409 | 0.905 / 205.004 |

  Compared with naive bootstrap XGBoost, CQR raises coverage substantially:
  Wine Cov90 goes from about `0.65-0.70` to `0.91-0.96`, Parkinsons from
  `0.19-0.32` to `0.65-0.83`, and Appliances from `0.88` to `0.90-0.95`.
  The cost is much wider intervals, especially Wine and Appliances. Parkinsons
  grouped still undercovers at train=500/1000, consistent with calibration rows
  coming from training subjects while test rows are unseen subjects.
- **Status**: Completed. Use as an appendix robustness baseline for interval
  calibration, not as a direct CRPS/posterior baseline.

## 2026-05-11 - `smalltrain_jgp_projection_train500_5seeds` (appendix projection-JGP baselines)

- **Question**: Do classical deterministic projection + JumpGP baselines
  (`JGP-PCA`, `JGP-SIR`) change the small-train UCI conclusions, or is the
  modern baseline comparison (XGBoost/DKL/LMJGP) sufficient for the main table?
- **Runner**: `experiments/uci/compare_smalltrain_jgp_projection.py`.
- **Settings**:
  `--num_exp 5 --train_sizes 500
  --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --methods "jgp_pca,jgp_sir"
  --out_dir experiments/uci/smalltrain_jgp_projection_train500_5seeds`.
  Wine/Appliances use random-row splits; Parkinsons uses the subject-grouped
  split. PCA/SIR are fit on training rows only. Appliances uses train-only
  target standardization for fitting and reports original-scale metrics.
- **Artifacts**:
  `experiments/uci/smalltrain_jgp_projection_train500_5seeds/`
  - `metrics.csv`, `aggregate.csv`, `diagnostics.csv`, `summary.json`,
    `README.md`, and `run.log`.
  - Smoke: `experiments/uci/smalltrain_jgp_projection_smoke/`.
- **Headline (train=500, mean +/- population std over 5 seeds)**:

  | dataset | method | RMSE | CRPS | Cov90 |
  |---|---|---:|---:|---:|
  | Wine Quality | JGP-PCA | 0.839 +/- 0.028 | 0.478 +/- 0.016 | 0.768 |
  | Wine Quality | JGP-SIR | 0.803 +/- 0.039 | 0.469 +/- 0.028 | 0.752 |
  | Parkinsons Telemonitoring | JGP-PCA | 10.202 +/- 0.905 | 6.914 +/- 1.016 | 0.452 |
  | Parkinsons Telemonitoring | JGP-SIR | 10.158 +/- 0.317 | 6.933 +/- 0.466 | 0.432 |
  | Appliances Energy Prediction | JGP-PCA | 113.204 +/- 6.567 | 49.100 +/- 3.935 | 0.615 |
  | Appliances Energy Prediction | JGP-SIR | 120.179 +/- 11.293 | 51.661 +/- 7.403 | 0.631 |

  These deterministic projection-JGP baselines are weaker than the best
  XGBoost/DKL/LMJGP rows at train=500 and do not change the small-train
  conclusions. They are appropriate for an appendix sanity table, not the main
  comparison.
- **Status**: Completed 5 seeds.

## 2026-05-11 - `dkl_sanity_parkinsons_grouped_seed0` (DKL implementation sanity check)

- **Question**: Does the DKL-SVGP baseline use train-only scaling and
  original-scale CRPS correctly, and why does Parkinsons grouped `std_x` hurt
  DKL so much?
- **Runner**: `experiments/uci/dkl_sanity_check.py`.
- **Settings**:
  `--dataset "Parkinsons Telemonitoring" --train_sizes 200,500,1000
  --seed 0 --epochs 80
  --out_dir experiments/uci/dkl_sanity_parkinsons_grouped_seed0`.
  The diagnostic mirrors `djgp.baselines.run_dkl_svgp_regression` but records
  train-only scaler checks, y-scaling, learned lengthscales, likelihood noise,
  latent feature scale, and predictive sigma scale.
- **Artifacts**:
  `experiments/uci/dkl_sanity_parkinsons_grouped_seed0/`
  - `diagnostics.csv`, `summary.json`, `README.md`, and `run.log`.
  - Smoke: `experiments/uci/dkl_sanity_smoke/`.
- **Headline**:
  The implementation uses train-only X scaling when enabled, and DKL's internal
  y-standardization uses training `y` only before mapping predictions back to
  original scale. The bad `std_x` result is not a leakage/scale-reporting bug.
  Instead, `std_x` collapses the learned latent feature scale
  (`latent_train_std_mean` about `0.24-0.31` versus `15-26` raw) and predictive
  sigma (`1.48-2.04` versus `4.69-4.88` raw), causing severe undercoverage and
  worse CRPS/RMSE under the current DKL hyperparameters.
- **Status**: Completed seed-0 sanity check. Keep raw Parkinsons DKL in the
  consolidated comparison unless DKL is retuned for standardized inputs.

## 2026-05-11 - `parkinsons_grouped_stdx_xgb_dkl_probe` (X-standardization check for XGB/DKL)

- **Question**: Since grouped Parkinsons LMJGP improves with `std_x`, should
  XGBoost bootstrap and DKL-SVGP also use `std_x` instead of the existing raw
  grouped preset?
- **Runner**: `experiments/uci/compare_smalltrain_multiseed_grouped.py`.
- **Settings**:
  Seed 0 only; Parkinsons Telemonitoring subject-grouped split; methods
  `xgb_bootstrap,dkl_svgp`; train sizes 200, 500, and 1000;
  `--parkinsons_standardize_x 1 --parkinsons_standardize_y 0`;
  `--xgb_boots 30 --dkl_epochs 80`. Raw comparison rows are from
  `experiments/uci/smalltrain_multiseed_bestpreset_summary/metrics_bestpresets.csv`.
- **Artifacts**:
  - Summary:
    `experiments/uci/parkinsons_grouped_stdx_xgb_dkl_probe_summary/`.
  - Probe runs:
    `experiments/uci/parkinsons_grouped_stdx_xgb_dkl_probe_train200_seed0/`,
    `experiments/uci/parkinsons_grouped_stdx_xgb_dkl_probe_train500_seed0/`,
    `experiments/uci/parkinsons_grouped_stdx_xgb_dkl_probe_train1000_seed0/`.
- **Headline (seed 0, delta = std_x - raw)**:

  | train | method | raw RMSE | std_x RMSE | delta RMSE | raw CRPS | std_x CRPS | delta CRPS |
  |---:|---|---:|---:|---:|---:|---:|---:|
  | 200 | XGBoost | 7.828 | 7.828 | 0.000 | 5.263 | 5.263 | 0.000 |
  | 200 | DKL-SVGP | 6.642 | 9.989 | +3.347 | 3.855 | 7.337 | +3.482 |
  | 500 | XGBoost | 8.438 | 8.438 | 0.000 | 6.110 | 6.110 | 0.000 |
  | 500 | DKL-SVGP | 7.106 | 10.055 | +2.949 | 4.105 | 8.000 | +3.895 |
  | 1000 | XGBoost | 9.015 | 9.015 | 0.000 | 7.022 | 7.022 | 0.000 |
  | 1000 | DKL-SVGP | 7.021 | 10.112 | +3.091 | 4.076 | 8.260 | +4.184 |

  XGBoost is unchanged, as expected for tree splits under feature rescaling.
  DKL-SVGP is much worse with `std_x` under the current hyperparameters, so the
  consolidated Parkinsons policy should remain: raw for XGBoost/DKL, `std_x`
  only for LMJGP-family rows.
- **Status**: Completed seed-0 probe.

## 2026-05-10 - `smalltrain_multiseed_bestpreset_summary` (UCI train-size summary CSVs)

- **Question**: Put the 5-seed train=200/500/1000 UCI results in one place so
  downstream tables do not require searching across multiple artifact folders.
- **Runner**: aggregation script over existing `aggregate.csv` and
  `metrics.csv` files from the UCI small-train runs.
- **Synthesis**: See `docs/uci_smalltrain_summary.md`.
- **Artifacts**:
  `experiments/uci/smalltrain_multiseed_bestpreset_summary/`
  - `aggregate_bestpresets.csv` and `metrics_bestpresets.csv` - latest summary
    over train sizes 200, 500, and 1000.
  - `aggregate_train200_500_bestpresets.csv` and
    `metrics_train200_500_bestpresets.csv` - 200/500 snapshot.
  - `aggregate_train200_500_1000_bestpresets.csv` and
    `metrics_train200_500_1000_bestpresets.csv` - explicit 200/500/1000
    snapshot.
- **Preset policy**: Wine=`std_x,n=25`; Appliances=`std_x,std_y,n=35`;
  Parkinsons is subject-grouped, with XGBoost/DKL kept from the raw grouped
  comparison and LMJGP-family rows replaced by the grouped `std_x` LMJGP runs.
- **Status**: Completed. Use this folder as the current CSV source of truth for
  UCI small-train tables.

## 2026-05-10 - `smalltrain_multiseed_train1000_bestpresets` (UCI 1000-train comparison)

- **Question**: Extend the train=200/500 multi-seed UCI comparison to
  train=1000, using the updated Parkinsons policy: LMJGP-family methods use
  subject-grouped `std_x` and no raw LMJGP rerun is needed.
- **Runner**: `experiments/uci/compare_smalltrain_multiseed_grouped.py`.
- **Synthesis**: See `docs/uci_smalltrain_summary.md`.
- **Settings**:
  5 seeds, `--max_total_train 1000 --max_test_anchors 200
  --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80
  --lmjgp_steps 300 --MC_num 3`. Wine/Appliances were run together with all
  six methods. Parkinsons was split into raw XGBoost/DKL and `std_x` LMJGP
  runs.
- **Artifacts**:
  - `experiments/uci/smalltrain_multiseed_train1000_wine_appliances_5seeds/`
  - `experiments/uci/parkinsons_grouped_raw_xgb_dkl_train1000_5seeds/`
  - `experiments/uci/parkinsons_grouped_stdx_lmjgp_train1000_5seeds/`
  - Consolidated into
    `experiments/uci/smalltrain_multiseed_bestpreset_summary/`.
- **Headline (mean +/- population std over 5 seeds)**:

  | dataset | best RMSE | RMSE | best CRPS | CRPS |
  |---|---|---:|---|---:|
  | Wine Quality | XGBoost | 0.689 +/- 0.015 | XGBoost | 0.397 +/- 0.016 |
  | Parkinsons Telemonitoring (grouped) | DKL-SVGP | 7.860 +/- 1.036 | DKL-SVGP | 4.926 +/- 0.876 |
  | Appliances Energy Prediction | XGBoost | 96.244 +/- 8.165 | transductive LMJGP + ridge residual | 41.055 +/- 3.730 |

  No train=1000 row required numeric omission. At train=1000, XGBoost is best
  on Wine, DKL remains best on grouped Parkinsons, and Appliances keeps the
  same tradeoff: XGBoost is best RMSE while LMJGP residual is slightly best
  CRPS.
- **Status**: Completed 5 seeds.

## 2026-05-10 - `parkinsons_grouped_stdx_lmjgp` (Parkinsons grouped LMJGP scaling check)

- **Question**: The original Parkinsons no-standardization preset was chosen
  under random-row splitting. Under subject-grouped splitting, does LMJGP
  improve if `X` or both `X,y` are standardized?
- **Runner**: `experiments/uci/compare_smalltrain_multiseed_grouped.py`.
- **Synthesis**: See `docs/uci_smalltrain_summary.md`.
- **Settings**:
  First, a one-seed train=200 probe compared raw, `std_x`, and `std_x,std_y`
  for `lmjgp_transductive`, `lmjgp_transductive_residual_ridge`,
  `lmjgp_supervised`, and `lmjgp_supervised_residual_ridge`.
  Because `std_x` was best in the probe, follow-up runs used
  `--parkinsons_standardize_x 1 --parkinsons_standardize_y 0`,
  `--num_exp 5`, `--max_test_anchors 200`, `--max_train_anchors 128`,
  `--lmjgp_steps 300`, and `--MC_num 3` at train sizes 200 and 500.
- **Artifacts**:
  - Probe: `experiments/uci/parkinsons_grouped_scaling_probe_200_stdx/` and
    `experiments/uci/parkinsons_grouped_scaling_probe_200_stdxy/`.
  - Five-seed runs:
    `experiments/uci/parkinsons_grouped_stdx_lmjgp_train200_5seeds/` and
    `experiments/uci/parkinsons_grouped_stdx_lmjgp_train500_5seeds/`.
- **Headline (mean +/- population std over 5 seeds, LMJGP only)**:

  | train | best LMJGP RMSE | RMSE | best LMJGP CRPS | CRPS |
  |---:|---|---:|---|---:|
  | 200 | supervised LMJGP (`std_x`) | 8.912 +/- 0.757 | supervised LMJGP (`std_x`) | 5.447 +/- 0.598 |
  | 500 | transductive LMJGP (`std_x`) | 8.937 +/- 1.010 | transductive LMJGP (`std_x`) | 5.513 +/- 0.813 |

  Compared with the raw grouped preset, `std_x` improves non-residual LMJGP
  substantially at both train sizes. Ridge residuals are not helpful for
  Parkinsons once `X` is standardized. DKL-SVGP remains the best method overall
  on the grouped split, but `std_x` LMJGP beats XGBoost bootstrap on RMSE/CRPS
  in the existing train=200 and train=500 comparisons.
- **Status**: Completed. For Parkinsons grouped LMJGP reporting, prefer
  `std_x` over the earlier raw preset.

## 2026-05-10 - `smalltrain_multiseed_grouped_train500_5seeds_with_residual` (tuned 500-train UCI comparison)

- **Question**: With 500 labeled training rows, do ridge-residual LMJGP variants
  improve the tuned transductive/supervised LMJGP comparison against XGBoost
  bootstrap and DKL-SVGP?
- **Runner**: `experiments/uci/compare_smalltrain_multiseed_grouped.py`.
- **Synthesis**: See `docs/uci_smalltrain_summary.md`.
- **Settings**:
  `--num_exp 5
  --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --max_total_train 500 --max_test_anchors 200 --max_train_anchors 128
  --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3
  --out_dir experiments/uci/smalltrain_multiseed_grouped_train500_5seeds_with_residual`.
  Presets are Wine=`std_x,n=25`, Parkinsons=`raw,n=25,subject_grouped`,
  Appliances=`std_x,std_y,n=35`.
- **Artifacts**:
  `experiments/uci/smalltrain_multiseed_grouped_train500_5seeds_with_residual/`
  - `metrics.csv`, `metrics_raw.csv`, `aggregate.csv`, `summary.json`,
    `run.log`, and `README.md`.
- **Headline (mean +/- population std over 5 seeds)**:

  | dataset | best RMSE | RMSE | best CRPS | CRPS |
  |---|---|---:|---|---:|
  | Wine Quality | XGBoost | 0.720 +/- 0.018 | transductive LMJGP | 0.416 +/- 0.020 |
  | Parkinsons Telemonitoring (grouped) | DKL-SVGP | 7.860 +/- 1.030 | DKL-SVGP | 4.934 +/- 0.886 |
  | Appliances Energy Prediction | XGBoost | 98.477 +/- 9.591 | supervised LMJGP | 42.076 +/- 4.194 |

  No rows required numeric omission. Ridge residuals improve LMJGP RMSE on
  Wine and Appliances, but generally worsen or do not improve CRPS because the
  intervals become less favorable.
- **Status**: Completed 5 seeds. Follow-up: run the same method set at
  `max_total_train=1000` if the small-train trend needs a third point.

## 2026-05-10 - `smalltrain_multiseed_grouped_5seeds_with_residual` (tuned 200-train UCI comparison with residuals)

- **Question**: Complete the tuned 200-train multi-seed comparison by adding
  ridge-residual variants for both transductive and supervised LMJGP, while
  preserving the original non-residual methods.
- **Runner**: `experiments/uci/compare_smalltrain_multiseed_grouped.py`.
- **Synthesis**: See `docs/uci_smalltrain_summary.md`.
- **Settings**:
  5 seeds, `--max_total_train 200 --max_test_anchors 200
  --max_train_anchors 128 --xgb_boots 30 --dkl_epochs 80
  --lmjgp_steps 300 --MC_num 3`. Presets are Wine=`std_x,n=25`,
  Parkinsons=`raw,n=25,subject_grouped`, Appliances=`std_x,std_y,n=35`.
  This combined artifact uses the original four-method run, the residual-only
  run, and a clean rerun of Wine seed 0 `lmjgp_transductive`.
- **Artifacts**:
  `experiments/uci/smalltrain_multiseed_grouped_5seeds_with_residual/`
  - `metrics.csv`, `aggregate.csv`, `summary.json`, and `README.md`.
  - Source runs:
    `experiments/uci/smalltrain_multiseed_grouped_5seeds/`,
    `experiments/uci/smalltrain_multiseed_grouped_5seeds_residual_only/`,
    `experiments/uci/smalltrain_multiseed_grouped_5seeds_wine_transductive_seed0_filter/`.
- **Headline (mean +/- population std over 5 seeds)**:

  | dataset | best RMSE | RMSE | best CRPS | CRPS |
  |---|---|---:|---|---:|
  | Wine Quality | XGBoost | 0.751 +/- 0.037 | supervised LMJGP + ridge residual | 0.429 +/- 0.013 |
  | Parkinsons Telemonitoring (grouped) | DKL-SVGP | 7.814 +/- 1.167 | DKL-SVGP | 4.923 +/- 0.990 |
  | Appliances Energy Prediction | supervised LMJGP + ridge residual | 102.981 +/- 5.667 | supervised LMJGP | 44.979 +/- 4.876 |

  The Wine seed 0 transductive LMJGP variance explosion from the original run
  was not reproduced by a clean rerun (`RMSE=0.752`, `CRPS=0.419`), so the
  combined aggregate uses the clean row. Residual LMJGP improves Appliances
  RMSE at train=200, but non-residual supervised LMJGP remains better on CRPS.
- **Status**: Completed 5 seeds; this supersedes
  `smalltrain_multiseed_grouped_5seeds` for train=200 reporting.

## 2026-05-10 - `smalltrain_multiseed_grouped_5seeds` (tuned 200-train UCI main comparison)

- **Question**: With the tuned small-train settings, how do XGBoost bootstrap,
  DKL-SVGP, transductive LMJGP, and supervised LMJGP compare over multiple
  seeds, using a subject-grouped Parkinsons split?
- **Runner**: `experiments/uci/compare_smalltrain_multiseed_grouped.py`.
- **Synthesis**: See `docs/uci_smalltrain_summary.md`.
- **Settings**:
  `--num_exp 5
  --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128
  --xgb_boots 30 --dkl_epochs 80 --lmjgp_steps 300 --MC_num 3
  --out_dir experiments/uci/smalltrain_multiseed_grouped_5seeds`.
  Presets are Wine=`std_x,n=25`, Parkinsons=`raw,n=25,subject_grouped`,
  Appliances=`std_x,std_y,n=35`; no residual variants are included.
- **Artifacts**: `experiments/uci/smalltrain_multiseed_grouped_5seeds/`
  - `metrics.csv`, `aggregate.csv`, `summary.json`, `run.log`, and `README.md`.
  - `experiments/uci/smalltrain_multiseed_grouped_smoke/` - Wine-only smoke run.
- **Headline (mean +/- population std over 5 seeds)**:

  | dataset | best RMSE | RMSE | best CRPS | CRPS |
  |---|---|---:|---|---:|
  | Wine Quality | XGBoost | 0.751 +/- 0.037 | supervised LMJGP | 0.437 +/- 0.013 |
  | Parkinsons Telemonitoring (grouped) | DKL-SVGP | 7.814 +/- 1.167 | DKL-SVGP | 4.923 +/- 0.990 |
  | Appliances Energy Prediction | XGBoost | 104.002 +/- 5.667 | supervised LMJGP | 44.979 +/- 4.876 |

  Parkinsons grouped split had zero subject overlap for every seed. Wine
  `lmjgp_transductive` seed 0 had a numerical predictive-variance explosion
  (RMSE normal at 0.760, but CRPS/width enormous), so its mean CRPS/width in
  `aggregate.csv` should not be used without that caveat; its RMSE aggregate is
  still interpretable.
- **Status**: Superseded by
  `smalltrain_multiseed_grouped_5seeds_with_residual`, which adds residual
  variants and replaces the non-reproducible Wine seed 0 variance explosion
  with a clean rerun.

## 2026-05-10 - `parkinsons_grouped_split_200` (subject-grouped Parkinsons small-train comparison)

- **Question**: Under a stricter Parkinsons Telemonitoring split with no
  subject overlap between train and test, how do XGBoost, supervised LMJGP,
  ridge-residual LMJGP, and supervised CEM-EM compare?
- **Runner**: `experiments/uci/compare_parkinsons_grouped_split.py`.
- **Synthesis**: See `docs/uci_smalltrain_summary.md`.
- **Settings**:
  `--max_total_train 200 --max_test_anchors 200 --max_train_anchors 128
  --n 25 --lmjgp_steps 300 --n_outer 5 --n_m_steps 80 --MC_num 3
  --xgb_boots 30 --out_dir experiments/uci/parkinsons_grouped_split_200`.
  Split is by `subject#`: 33 train subjects, 9 test subjects, zero subject
  overlap. Parkinsons is left unstandardized to match the empirical UCI preset.
- **Artifacts**: `experiments/uci/parkinsons_grouped_split_200/`
  - `metrics.csv`, `predictions.csv`, `summary.json`, `run.log`, and `README.md`.
  - `experiments/uci/parkinsons_grouped_split_smoke/` - tiny smoke run.
- **Headline (one seed)**:

  | method | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|
  | XGBoost | 7.828 | 5.263 | 0.430 | 10.527 |
  | supervised LMJGP | 9.063 | 5.707 | 0.675 | 21.714 |
  | ridge-residual LMJGP | 8.710 | 5.445 | 0.680 | 21.200 |
  | supervised CEM-EM | 9.543 | 6.339 | 0.570 | 17.958 |

  The subject-grouped split is substantially harder than the historical random
  row split. Ridge residuals improve supervised LMJGP but do not overtake
  XGBoost. Supervised CEM-EM is worst in this seed. XGBoost has the best point
  accuracy and CRPS but low 90% coverage; LMJGP variants are wider and better
  covered.
- **Status**: Completed one-seed grouped-split diagnostic. Follow-ups: repeat
  over several subject seeds and consider reporting Parkinsons as a grouped
  split sensitivity rather than only a random-row UCI benchmark.

## 2026-05-09 - `smalltrain_lmjgp_remedies` (LMJGP remedy ablations on 200-train UCI)

- **Question**: Can low-cost remedies for supervised LMJGP address the
  small-train UCI failure modes seen in PC1/tail diagnostics: local smoothness
  mismatch, rare-regime support, and systematic/global trend misses?
- **Runner**: `experiments/uci/try_lmjgp_remedies.py`.
- **Synthesis**: See `docs/uci_smalltrain_summary.md`.
- **Settings**:
  `--datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128
  --neighbors auto --lmjgp_steps 300 --MC_num 3 --xgb_boots 30
  --scaling_preset uci_empirical --out_dir experiments/uci/smalltrain_lmjgp_remedies`.
  The auto neighborhood uses Wine/Parkinsons=`25`, Appliances=`35`; scaling is
  Wine=`std_x`, Parkinsons=raw, and Appliances=`std_x,std_y`. The new variants
  are experiment-only and do not overwrite the existing transductive/supervised
  method implementations.
- **Artifacts**: `experiments/uci/smalltrain_lmjgp_remedies/`
  - `metrics.csv`, `predictions.csv`, `tail_contribution.csv`,
    `true_y_quantile_metrics.csv`, `summary.json`, `run.log`, and `README.md`.
  - `experiments/uci/smalltrain_lmjgp_remedies_smoke/` - tiny Wine-only smoke
    test of the same runner.
- **Headline (one seed)**:

  | dataset | XGBoost RMSE | base supervised LMJGP RMSE | best remedy RMSE | best remedy | best remedy CRPS |
  |---|---:|---:|---:|---|---:|
  | Wine Quality | 0.728 | 0.754 | 0.750 | hybrid PLS neighborhood | 0.423 |
  | Parkinsons Telemonitoring | 5.558 | 8.427 | 7.989 | hybrid PLS neighborhood | 4.600 |
  | Appliances Energy Prediction | 103.260 | 109.017 | 102.740 | ridge residual LMJGP | 42.840 |

  Matérn local-GP diagnostics changed RMSE only modestly relative to a plain
  local-GP RBF diagnostic; this supports treating Matérn as a kernel ablation,
  not the main fix. Ridge residual LMJGP slightly beat XGBoost RMSE on
  Appliances in this seed and reduced the `y>q90` RMSE from 323.3 (base
  supervised LMJGP) to 298.6, but CRPS remained worse than XGBoost and
  supervised-base LMJGP. Hybrid PLS neighborhoods improved Parkinsons over
  base LMJGP but remained far behind XGBoost.
- **Status**: Completed one-seed remedy diagnostic. Follow-ups: confirm
  Appliances residual improvement over multiple seeds; tune the global mean
  model and response-aware neighborhood separately; keep Matérn as a secondary
  local-kernel ablation rather than a primary method change.

## 2026-05-09 - `smalltrain_pca_viz` (PC1 visualization and LMJGP step/neighborhood sweep)

- **Question**: For the 200-train UCI setting, what do XGBoost and supervised
  LMJGP predictions look like along the first principal component, and are
  200 LMJGP steps / neighborhood size 35 reasonable?
- **Runner**: `experiments/uci/visualize_smalltrain_pca_predictions.py`.
- **Synthesis**: See `docs/uci_smalltrain_summary.md`.
- **Settings**:
  `--datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128
  --neighbors "25,35" --steps "200,300,500" --xgb_boots 30
  --scaling_preset uci_empirical --out_dir experiments/uci/smalltrain_pca_viz`.
  The preset uses Wine=`std_x`, Parkinsons=raw, and Appliances=`std_x,std_y`;
  metrics and plots are on the original response scale.
- **Artifacts**: `experiments/uci/smalltrain_pca_viz/`
  - `metrics.csv` - XGBoost plus LMJGP supervised step/neighborhood sweep.
  - `predictions.csv` - PC1, test truth, predictive mean/sigma, and 90% range.
  - `summary.json` - split metadata, PCA variance, metrics, figure paths.
  - `figures/*_pc1_response.png` and `figures/*_pc1_predictions.png`.
  - `README.md` - command and headline table.
  - `experiments/uci/smalltrain_pca_viz_smoke/` - tiny Wine-only smoke test.
- **Headline (one seed)**:

  | dataset | XGBoost RMSE | best LMJGP config | best LMJGP RMSE | XGBoost CRPS | best LMJGP CRPS |
  |---|---:|---|---:|---:|---:|
  | Wine Quality | 0.728 | `n=25, steps=300` | 0.751 | 0.433 | 0.420 |
  | Parkinsons Telemonitoring | 5.558 | `n=25, steps=300` | 7.773 | 3.393 | 4.399 |
  | Appliances Energy Prediction | 103.260 | `n=35, steps=300` | 105.065 | 41.900 | 40.622 |

  In this seed, 300 LMJGP steps is best among 200/300/500 for all datasets.
  The smaller neighborhood (`n=25`) helps Wine and Parkinsons, but Appliances
  still prefers `n=35`. XGBoost remains best on RMSE, while LMJGP supervised
  has better CRPS on Wine and Appliances.
- **Status**: Completed one-seed diagnostic. Follow-ups: multi-seed confirmation
  and optional early stopping on supervised-anchor predictive validation loss.

## 2026-05-09 - `smalltrain_tail_diagnostics` (UCI residual-tail and calibration diagnostics)

- **Question**: Do the PC1 visualizations hide the error sources behind the
  RMSE/coverage gap between XGBoost and supervised LMJGP?
- **Runner**: `experiments/uci/diagnose_smalltrain_tail.py`, consuming
  `experiments/uci/smalltrain_pca_viz/predictions.csv`.
- **Settings**:
  `--viz_dir experiments/uci/smalltrain_pca_viz
  --out_dir experiments/uci/smalltrain_tail_diagnostics`. Uses the same
  200-train / 200-test / seed-0 split and empirical scaling as
  `smalltrain_pca_viz`.
- **Artifacts**: `experiments/uci/smalltrain_tail_diagnostics/`
  - `tail_contribution.csv`, `true_y_quantile_metrics.csv`,
    `sigma_bin_calibration.csv`, `neighbor_high_y_support.csv`,
    `largest_errors.csv`, `summary.json`.
  - `figures/*_residual_diagnostics.png`.
  - `README.md`.
- **Headline**:
  Appliances RMSE is dominated by rare high-response misses: the top 10%
  residuals explain 89% of XGBoost SSE and 91% of best-LMJGP SSE. In the
  `y>q90` group, coverage is only 0.10 for XGBoost and 0.15 for best LMJGP;
  LMJGP has wider intervals but still centers far below the largest responses.
  Parkinsons is more of a systematic response-structure miss: the top 10%
  residuals explain only about 37-40% of SSE, but LMJGP misses high-y points
  by pulling means toward the center.
- **Status**: Completed one-seed diagnostic. Follow-ups: residual-by-subject
  for Parkinsons and tail-aware/local-neighbor variants for Appliances.

## 2026-05-09 - `pdebench_burgers_balanced_delta01_q3n25_seeds0to4`

- **Question**: On the near-threshold-balanced PDEBench-Burgers ShockJump
  surrogate, do the main public-data baselines keep the same ordering across
  several random seeds?
- **Runner**: `experiments/pdebench/compare_pdebench_burgers.py`.
- **Settings**:
  `--dataset-folder data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000
  --Q 3 --neighbors 25 --lmjgp_steps 300 --include_xgb 1 --include_dkl 1
  --include_lmjgp 1 --include_cemem 0`. Seed 0 was run first; seeds 1-4 were
  run with `--seed_start 1 --num_exp 4`. The processed split uses
  near-threshold balancing at `delta=0.1` with 325/1600 near train points and
  106/400 near test points.
- **Artifacts**: `results/`
  - `pdebench_burgers_balanced_delta01_n2000_xgb_dkl_lmjgp_q3_n25_seed0.{json,pkl}`.
  - `pdebench_burgers_balanced_delta01_n2000_xgb_dkl_lmjgp_q3_n25_seeds1to4.{json,pkl,out.log,err.log}`.
  - `pdebench_burgers_balanced_delta01_n2000_xgb_dkl_lmjgp_q3_n25_seeds0to4_summary.{json,csv}`.
- **Headline (mean +/- population std over seeds 0-4)**:

  | subset | best RMSE | RMSE | best CRPS | CRPS |
  |---|---:|---:|---:|---:|
  | all | `xgb_bootstrap` | 0.694 +/- 0.006 | `lmjgp_predict_vi` | 0.240 +/- 0.003 |
  | near-threshold | `xgb_bootstrap` | 0.833 +/- 0.017 | `lmjgp_predict_vi` | 0.377 +/- 0.016 |
  | far-from-threshold | `xgb_bootstrap` | 0.636 +/- 0.004 | `lmjgp_predict_vi` | 0.191 +/- 0.003 |

  XGBoost remains best for point RMSE. LMJGP is consistently best for CRPS,
  including the near-threshold subset, but its RMSE does not beat XGBoost.
- **Status**: Completed seeds 0-4 without CEM-EM. Follow-up is to run the same
  balanced split with CEM-EM defaults once runtime budget is acceptable.

## 2026-05-09 - `smalltrain_200_xgb` (UCI very-small-train vs XGBoost)

- **Question**: With only 200 labeled training rows and 200 test rows, do
  LMJGP/CEM-EM variants gain more ground against XGBoost than in the 500-train
  pilot?
- **Runner**: `experiments/uci/compare_inductive_projection.py`.
- **Synthesis**: See `docs/uci_smalltrain_summary.md`.
- **Settings**:
  `--num_exp 1 --methods "lmjgp_transductive,cem_em_transductive,lmjgp_supervised,cem_em_supervised,xgboost"
  --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --max_total_train 200 --max_test_anchors 200 --max_train_anchors 128
  --neighbor_pool train --n 35 --m2 40 --lmjgp_steps 200 --n_outer 5
  --n_m_steps 80 --MC_num 3 --lmjgp_supervised_train_mc 2
  --lmjgp_supervised_grad_check 0 --xgb_boots 30 --scaling_preset uci_empirical`.
  The empirical scaling preset uses Wine=`std_x`, Parkinsons=raw, and
  Appliances=`std_x,std_y`; Appliances metrics are reported on the original
  target scale.
- **Artifacts**: `experiments/uci/inductive_projection_compare/`
  - `smalltrain_200_1seed_200test_xgb.csv` / `.json`.
- **Headline (one seed)**:

  | dataset | best RMSE method | RMSE | best CRPS method | CRPS |
  |---|---:|---:|---:|---:|
  | Wine Quality | `xgboost` | 0.728 | `lmjgp_supervised` | 0.427 |
  | Parkinsons Telemonitoring | `xgboost` | 5.558 | `xgboost` | 3.393 |
  | Appliances Energy Prediction | `xgboost` | 103.260 | `lmjgp_supervised` | 39.833 |

  XGBoost remains best on RMSE, but the strict supervised LMJGP objective now
  has the best CRPS on Wine and Appliances. The gain is uncertainty/calibration
  oriented rather than point-prediction oriented.
- **Status**: Completed one-seed pilot. Follow-up is multi-seed small-train
  sweeps and possibly retuning `max_train_anchors` for the 200-train case.

## 2026-05-09 - `smalltrain_500_xgb` (UCI small-train vs XGBoost)

- **Question**: With only 500 labeled training rows and 200 test rows, do
  transductive/supervised LMJGP and CEM-EM gain an advantage over the existing
  XGBoost bootstrap baseline?
- **Runner**: `experiments/uci/compare_inductive_projection.py`.
- **Settings**:
  `--num_exp 1 --methods "lmjgp_transductive,cem_em_transductive,lmjgp_supervised,cem_em_supervised,xgboost"
  --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --max_total_train 500 --max_test_anchors 200 --max_train_anchors 128
  --neighbor_pool train --n 35 --m2 40 --lmjgp_steps 200 --n_outer 5
  --n_m_steps 80 --MC_num 3 --lmjgp_supervised_train_mc 2
  --lmjgp_supervised_grad_check 0 --xgb_boots 30 --scaling_preset uci_empirical`.
  The empirical scaling preset uses Wine=`std_x`, Parkinsons=raw, and
  Appliances=`std_x,std_y`; Appliances metrics are reported on the original
  target scale.
- **Artifacts**: `experiments/uci/inductive_projection_compare/`
  - `smalltrain_500_1seed_200test_xgb.csv` / `.json`.
- **Headline (one seed)**:

  | dataset | best RMSE method | RMSE | best CRPS method | CRPS |
  |---|---:|---:|---:|---:|
  | Wine Quality | `xgboost` | 0.698 | `lmjgp_transductive` | 0.399 |
  | Parkinsons Telemonitoring | `xgboost` | 4.636 | `xgboost` | 2.741 |
  | Appliances Energy Prediction | `xgboost` | 99.977 | `lmjgp_supervised` | 38.906 |

  XGBoost is still strongest on RMSE in this 500-train pilot. The DJGP family
  remains competitive on distributional metrics for Wine and Appliances, where
  LMJGP variants beat XGBoost on CRPS but use much wider intervals.
- **Status**: Completed one-seed pilot. Follow-up is multi-seed small-train
  sweeps (`500` and `1000` rows) if the paper needs a robustness claim.

## 2026-05-09 - `inductive_projection_compare` (UCI supervised/inductive projection pilot)

- **Question**: Can we replace test-anchor transductive projection fitting with
  labeled training-anchor projection learning, then induce `W_*` / `C_*` at true
  test inputs by GP conditioning, and compare against the existing LMJGP and
  CEM-EM transductive versions?
- **Runner**: `experiments/uci/compare_inductive_projection.py`.
- **Settings (current medium UCI run)**:
  `--num_exp 1 --datasets "Wine Quality,Parkinsons Telemonitoring,Appliances Energy Prediction"
  --max_test_anchors 64 --max_train_anchors 64 --n 35 --m2 40
  --lmjgp_steps 200 --n_outer 5 --n_m_steps 80 --MC_num 3
  --lmjgp_supervised_train_mc 2 --scaling_preset uci_empirical`.
  Default split policy uses `D_base` as the test-time local-neighborhood pool,
  `D_anchor` only for supervised projection learning, and `D_test` only for
  final metrics. The scaling preset uses Wine=`std_x`, Parkinsons=raw, and
  Appliances=`std_x,std_y`. CEM supervised uses `anchor_pred_var_floor=0.05`
  to avoid overconfident single-anchor NLL blow-ups.
- **Artifacts**: `experiments/uci/inductive_projection_compare/`
  - `uci4_methods_empstd_1seed_64anchors.csv` / `.json` - current 64-anchor
    medium run.
  - `pilot_3uci_strict_lmjgp_preset.csv` / `.json` - tiny strict-LMJGP pilot.
  - `smoke.csv` / `.json` - tiny four-method smoke run.
  - `pilot_3uci_stdy_varfloor.csv`, `pilot_3uci.csv`, `pilot_3uci_stdy.csv` -
    pre-strict-LMJGP or pre-varfloor diagnostic pilots.
  - `README.md` - command and artifact notes.
- **Headline (64 test anchors, 64 training anchors, one seed)**:

  | dataset | best RMSE method | RMSE | best CRPS method | CRPS |
  |---|---:|---:|---:|---:|
  | Wine Quality | `cem_em_supervised` | 0.519 | `cem_em_supervised` | 0.273 |
  | Parkinsons Telemonitoring | `cem_em_supervised` | 2.267 | `cem_em_supervised` | 0.390 |
  | Appliances Energy Prediction | `lmjgp_transductive` | 55.489 | `lmjgp_transductive` | 19.659 |

  Under the empirical scaling preset, supervised CEM-EM is strongest on Wine
  and Parkinsons in this one-seed run, while LMJGP transductive remains best on
  Appliances. The earlier 12-test-anchor strict-LMJGP pilot had supervised
  LMJGP best on the tiny slices, so the ordering is not yet stable enough for a
  paper conclusion.
- **Gradient/leakage audit**: the runner now records `base_anchor_disjoint`,
  `anchor_label_used_as_region_observation=false`, `kl_scaled_by_n_anchors=true`,
  and a predictive-only q(V) gradient check. On
  `smoke_lmjgp_supervised_gradcheck`, predictive-only loss decreased
  `2.4256 -> 2.3798`, with `mu_V_grad_norm=0.113` and
  `sigma_V_grad_norm=4.21e-4`. The 64-anchor run also produced nonzero q(V)
  gradients and one-step predictive-only loss decreases for all three datasets.
- **Status**: Live implementation/pilot. Follow-ups: run at least 3-10 seeds
  with 128-200 anchors, tune `lambda_anchor_pred` / `anchor_pred_var_floor`,
  and tune `lmjgp_supervised_kl_weight` / training-MC for the strict supervised
  LMJGP objective.

Single source of truth for every experimental run that lives somewhere under
`experiments/`. Each entry records:

- the **purpose** (what question we tried to answer),
- the **runner** script and key CLI flags,
- the **artifacts folder** with raw outputs (CSV / JSON / logs / plots),
- the **headline result** in one or two lines,
- the **status** (live / superseded / abandoned) and any follow-up.

When you start a new experiment that produces persistent outputs you should
append an entry below, even if the result was negative. When an existing
experiment is superseded by a newer one, mark it `superseded by ...` instead of
deleting the entry.

Add a corresponding `README.md` inside the artifacts folder if the run has more
than ~3 files; the entry below should at least point to where outputs live.

---

## 2026-05-08 — `cem_em_park_grid` (Parkinsons CEM-EM hyperparameter sweep)

- **Question**: After fixing scale/normalization issues in CEM-aligned EM, does
  it actually beat PLS-only and the existing LMJGP-VI baseline on a real UCI
  dataset (Parkinsons Telemonitoring)? What are the best `(α, λ_smooth)`?
- **Runner**: `experiments/uci/compare_cem_em_projection.py` (looped via
  `experiments/uci/cem_em_park_grid/run_grid.ps1`).
- **Settings shared across grid**:
  `--n 35 --m1 2 --max_anchors 128 --n_outer 5 --n_m_steps 80 --lr 0.01
  --lambda_gate 0.1 --lambda_scale 0 --row_normalize_W 1
  --normalize_smoothness 1 --track_best_W 1`.
  Sweep grid: `coeff_scale (α) ∈ {1, 3, 10}` × `lambda_smooth ∈ {1e-4, 1e-3, 1e-2}`.
  Plus a winner re-run with `--include_lmjgp 1 --lmjgp_steps 300`.
- **Artifacts**: `experiments/uci/cem_em_park_grid/`
  - `a{α}_l{λ}.{csv,json,log}` — one triple per grid cell (9 cells).
  - `best_with_lmjgp.{csv,json,log}` — winning config vs LMJGP-VI.
  - `summary.md` — markdown table produced by `summarize.py`.
  - `summarize.py` — aggregator (run from repo root).
  - `run_grid.ps1` — driver that produced the grid.
- **Headline (128 anchors, n=35)**:
  Winner is `α=1, λ_smooth=0.01, n_outer=5`.
  | method | RMSE | CRPS | Wall (s) | cov95 | w95 |
  |---|---|---|---|---|---|
  | `pls_only` | 7.141 | 3.940 | 16.5 | 0.703 | 13.71 |
  | `lmjgp_predict_vi` | 4.493 | 1.728 | 70.5 | 0.922 | 10.62 |
  | `cem_em_coeff` (best) | **3.580** | **0.722** | 39.9 | 0.711 | 1.42 |
  CEM-EM beats LMJGP-VI by 20% RMSE / 58% CRPS at lower wall time.
- **Diagnostics on winner**: `eff_rank_WWT=2.82` (Q=5), `anchor_dispersion=1.15`,
  `cosine_to_PLS=0.30` — local W really moved off PLS, is multi-direction.
- **Caveats**: Only 128 anchors (Parkinsons full test = 5875), only 1 seed.
  Single anchor σ blow-up observed at `α=3, λ=0.01` (CRPS spikes to 1e23 while
  RMSE stays normal); winner config does not exhibit it. Coverage at 95% is a
  bit underconfident (0.71) — predictive widths are very narrow because
  JumpGP-CEM gives inlier-only posteriors.
- **Status**: Live. Conclusions are inputs to the cross-method comparison
  (`compare_3way_10seeds`, see entry below). Document of the implementation:
  [`docs/cem_em_projection.md`](cem_em_projection.md).

---

## 2026-05-13 - `lmjgp_lowrank_cem_w0_paper_train1000_seed0`

- **Question**: On the paper-style L2/LH synthetic benchmarks, does using the
  transductive CEM-EM MAP projection as a fixed `W0` improve low-rank residual
  LMJGP over the earlier PLS/PCA-mean low-rank variants and the baseline
  learned/prior `q(W)` controls?
- **Runner**:
  `experiments/synthetic/compare_lmjgp_lowrank_cem_w0_paper.py`.
- **Settings**:
  `--settings l2_q2_train1000,lh_q5_train1000 --seed 0 --test_size 200
  --std_x 1 --std_y 0 --cem_outer 5 --cem_m_steps 80 --lmjgp_steps 300
  --MC_num 3`.
  Synthetic provenance: paper-style generators imported from
  `experiments/synthetic/compare_paper_synthetic_baselines.py`; this is not the
  earlier toy `lowrank_jump` generator.
- **Artifacts**:
  `experiments/synthetic/lmjgp_lowrank_cem_w0_paper_train1000_seed0/`.
- **Methods**:
  `cem_em_transductive_map`,
  `lmjgp_lowrank_cem_mean_pls_pca`,
  `lmjgp_lowrank_cem_mean_cem_svd`.
  The low-rank residual variants use fixed global
  `W0 = mean_j W_j^{CEM-MAP}` rather than an anchor-specific mean function.
- **Headline result**:
  | setting | method | RMSE | CRPS | Cov90 | Width90 |
  |---|---:|---:|---:|---:|---:|
  | L2, train=1000 | `cem_em_transductive_map` | 2.293 | 1.304 | 0.795 | 5.908 |
  | L2, train=1000 | `lmjgp_lowrank_cem_mean_pls_pca` | 2.272 | 1.282 | 0.825 | 6.232 |
  | L2, train=1000 | `lmjgp_lowrank_cem_mean_cem_svd` | **2.202** | **1.259** | 0.815 | 6.218 |
  | LH, train=1000 | `cem_em_transductive_map` | 835.580 | 446.318 | 0.460 | 127.299 |
  | LH, train=1000 | `lmjgp_lowrank_cem_mean_pls_pca` | **729.353** | **364.386** | 0.775 | 938.690 |
  | LH, train=1000 | `lmjgp_lowrank_cem_mean_cem_svd` | 762.406 | 393.325 | 0.785 | 1077.121 |
  On L2, using `W0=mean(CEM-MAP W)` plus a CEM-SVD residual basis gives the
  best one-seed RMSE/CRPS among this small set and modestly improves over the
  earlier seed-0 learned/prior `q(W)` controls. On LH, both CEM-mean low-rank
  variants improve over CEM-MAP itself and over the earlier seed-0 learned
  `q(W)` head (`RMSE=786.661, CRPS=406.215`); PLS/PCA is the sharper/better
  CEM-mean basis in this run, while CEM-SVD is wider and worse on CRPS.
- **Status**: Completed one-seed probe. Follow-up only if this variant is
  promoted: run 5 seeds and compare to the same train/test protocol used by
  the paper synthetic baseline table.

---

## 2026-05-17 - `uci_dmgp_smalltrain_5seeds_20260517`

- **Question**: Do we have Deep Mahalanobis GP (DMGP) results on the UCI
  small-train benchmark at train sizes 200/500/1000, and how do they compare
  with the existing XGBoost, DKL, and LMJGP best-preset summary?
- **Runners**:
  - `experiments/uci/export_smalltrain_dmgp_splits.py`
  - `experiments/dmgp/run_deep_mahalanobis_gp_npz.py`
  - `experiments/dmgp/combine_with_baselines.py`
- **Settings**:
  `--train_sizes 200,500,1000 --num_exp 5 --max_test_anchors 200`.
  DMGP used `steps1=300`, `steps2=700`, `m_u=50`, `m_v=25`, and original-scale
  reporting after standardized-y training. Wine and Appliances use the
  random-row UCI split policy; Parkinsons uses the subject-grouped split. The
  comparison baseline is
  `experiments/uci/smalltrain_multiseed_bestpreset_summary/aggregate_train200_500_1000_bestpresets.csv`.
- **Artifacts**:
  `experiments/uci/dmgp_smalltrain_5seeds_20260517/`.
  Key files: `manifest.json`, `metrics.csv`, `aggregate.csv`, `summary.json`,
  `combined/aggregate_with_dmgp.csv`, and `combined/best_by_metric.csv`.
- **Headline result**:
  DMGP is competitive only on Wine. It is best on Wine train=200 for both RMSE
  and CRPS, best on Wine train=500 for CRPS and only `+0.005` RMSE behind XGB,
  and `+0.022` RMSE / `+0.011` CRPS behind XGB on Wine train=1000. On
  Parkinsons, DKL is best across train sizes and DMGP is worse by roughly
  `+0.86` to `+1.17` RMSE and `+0.57` to `+0.90` CRPS. On Appliances, the best
  methods are XGB/LMJGP variants; DMGP is worse by roughly `+4.33` to `+7.42`
  RMSE and `+0.40` to `+2.88` CRPS.
- **Status**: Completed. DMGP UCI coverage is now filled for 5 seeds and train
  sizes 200/500/1000. Follow-up only if DMGP should be tuned specifically for
  the Parkinsons/Appliances regimes.

---

## 2026-05-17 - `pdebench_burgers_train200_500_1000_dmgp_5seeds_20260517`

- **Question**: Fill the missing PDEBench Burgers ShockJump train-size sweep for
  Deep Mahalanobis GP and compare it with XGBoost, DKL, and LMJGP at
  train sizes 200/500/1000 over 5 seeds.
- **Runners**:
  - `experiments/pdebench/compare_burgers_train_sizes.py`
  - `experiments/pdebench/export_burgers_dmgp_splits.py`
  - `experiments/dmgp/run_deep_mahalanobis_gp_npz.py`
  - `experiments/dmgp/combine_with_baselines.py`
- **Settings**:
  Processed fixed-test split
  `data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000`;
  `--train_sizes 200,500,1000 --num_exp 5 --Q 3 --neighbors 25
  --lmjgp_steps 300`. DMGP used `steps1=300`, `steps2=700`, `m_u=50`,
  `m_v=25`. Metrics are reported for `all`, `near_threshold`, and
  `far_from_threshold` test subsets.
- **Artifacts**:
  - Baseline folder:
    `experiments/pdebench/burgers_train200_500_1000_xgb_dkl_lmjgp_5seeds_20260517/`.
  - DMGP folder:
    `experiments/pdebench/dmgp_burgers_train200_500_1000_5seeds_20260517/`.
  - Combined tables:
    `experiments/pdebench/dmgp_burgers_train200_500_1000_5seeds_20260517/combined/`.
- **Headline result**:
  On the full test set, DMGP has the best RMSE at train=200 (`0.830`) and
  train=1000 (`0.700`), while XGB is slightly better at train=500 (`0.724` vs
  DMGP `0.731`). LMJGP has the best full-set CRPS at all train sizes
  (`0.352`, `0.293`, `0.262`), with DMGP trailing by about `+0.027` to
  `+0.040` CRPS. On the `near_threshold` subset, DMGP has the best RMSE for all
  train sizes and the best CRPS at train=500/1000; train=200 near-threshold
  CRPS is essentially tied but XGB is lower by `0.006`. On the
  `far_from_threshold` subset, LMJGP/XGB remain better than DMGP, especially on
  CRPS.
- **Status**: Completed. PDEBench Burgers now has 5-seed train=200/500/1000
  comparisons for XGB, DKL, LMJGP, and DMGP.
