# Repository Map

The repository was reorganized on **2026-05-23** into an `src/` layout with `pip install -e .` discovery. Everything below reflects the post-reorg state. Historical notes about which files were latest / important pre-reorg follow at the bottom.

## Top-Level Layout

```
src/                   # importable packages (declared in pyproject.toml)
experiments/           # experiment harnesses, organized by area
scripts/               # standalone driver scripts
notebooks/             # Jupyter notebooks (exploratory / analysis / legacy)
tests/                 # smoke and import tests
docs/                  # research docs, papers/, figures/
data/                  # raw input datasets
results/               # generated experiment artifacts
legacy/                # frozen historical code
```

## src/ — Active Code

| Module | Purpose |
| --- | --- |
| `src/djgp/variational.py` | Full-batch variational DJGP: `qW_from_qV`, `compute_ELBO`, `train_vi`, `predict_vi`. |
| `src/djgp/minibatch.py` | Minibatch variational DJGP. |
| `src/djgp/active_learning.py` | Active-learning wrapper + JGP-based acquisition. |
| `src/djgp/acquisition_metrics.py` | Acquisition metric helpers. |
| `src/djgp/projections/` | CEM-EM projection, low-rank factorization, projected JumpGP, uncertain-W JGP, … |
| `src/djgp/baselines/` | DKL, global sparse GP, XGBoost bootstrap baselines. |
| `src/djgp/diagnostics/` | ELBO gradient and projection-posterior diagnostics. |
| `src/djgp/evaluation/` | Calibration, CRPS. |
| `src/djgp/jumpgp_bridge.py` | Bridges DJGP gates to JumpGP. |
| `src/djgp/sir.py` | Sliced Inverse Regression utilities. |
| `src/jumpgp/` | Thin facade over `JumpGaussianProcess`. |
| `src/JumpGaussianProcess/` | Canonical vendored Jump GP source. |
| `src/shared/utils.py`, `src/shared/utils1.py` | Shared utilities used across the codebase (Pyro/HMC/Gibbs, JGP wrappers). |
| `src/shared/djgp_runner.py` | Main DJGP CLI/library (was root `DJGP_test.py`). |
| `src/shared/jumpgp_runner.py` | Main JGP CLI/library (was root `JumpGP_test.py`). |
| `src/shared/djgp_validation.py` | DJGP validation variant (was root `DJGP_test_validation.py`). |
| `src/shared/deepgp.py` | Deep GP baseline (was root `DeepGP_test.py`). |
| `src/shared/maxmin_design.py` | Max-min design utility. |
| `src/data_gen/synthetic.py` | Low-dimensional latent generator (was `data_generate.py`). |
| `src/data_gen/highdata.py`, `src/data_gen/highdata_utils.py` | LH high-dimensional generators. |
| `src/data_gen/autoencoder.py` | AE-based dimensionality reduction (was `AE.py`). |
| `src/data_gen/lh_autoencoder.py`, `src/data_gen/uci_autoencoder.py` | AE wrappers for LH and UCI. |
| `src/data_gen/analysis.py` | Dataset-level analysis (was `dataset_analysis.py`). |

## experiments/ — Experiment Harnesses

Unchanged from pre-reorg structure:

- `experiments/uci/` — UCI benchmarks (Wine Quality, Parkinsons, Appliances).
- `experiments/synthetic/` — synthetic / LH experiments (many `compare_*`, `diagnose_*`, results folders).
- `experiments/erosion/` — erosion / K-sensitivity benchmarks.
- `experiments/oht/` — OHT benchmark.
- `experiments/active_learning/` — synthetic active-learning drivers.
- `experiments/diagnostics/` — projection-posterior W diagnostics.
- `experiments/pdebench/` — PDEBench Burgers surrogate experiments.
- `experiments/dmgp/` — Deep Mahalanobis GP experiments.
- `experiments/vem/` — VEM/EM comparison.

## scripts/ — Driver Scripts

- `scripts/try_new_JGP.py`
- `scripts/simulate_case_d_linear.py`
- `scripts/scripts_lh_l2_seed0_jgp_on_z.py`
- `scripts/scripts_lh_l2_seed0_overlap_only.py`
- `scripts/data/` — data preparation drivers (PDEBench downloads, etc.)
- `scripts/smoke/run_smoke.ps1`

## notebooks/

- `notebooks/exploratory/` — `try_new_JGP.ipynb`, `UCIdataset_try.ipynb`, `new_dataset_test.ipynb`, `erosion_exp.ipynb`, `new_dataGen.ipynb`.
- `notebooks/analysis/` — `summary_table.ipynb`, `visualize.ipynb`.
- `notebooks/legacy/` — `test0618.ipynb`, `test_VI_utils.ipynb`, `test_VI_utils_new.ipynb`.

## results/

- `results/uci/` — UCI baseline / lmjgp tricks / metric variants pickles.
- `results/synthetic/` — DJGP active-learning .npz files, synth lmjgp tricks pickle.
- `results/lh/` — LH seed-0 diagnostic summary, overlap report, CEM training JSON.
- `results/oht/` — `result_OHT_4d.mat`.
- `results/diagnostics/` — `_diag_test.json`, `diag_wine.json`.
- `results/tables/` — Excel summaries (`UCIdataset_results`, `model_metrics_summary`, …).
- `results/plots/pca_nonstationary_plots/`.
- `results/pdebench/` — pre-reorg PDEBench results (moved from `results/` flat).

## data/

- `data/OHTDataset.mat` — OHT benchmark input.
- `data/housing.arff` — Boston housing reference.

## legacy/

Frozen historical code. See `legacy/README.md` for the per-folder breakdown:

- `legacy/compat_shims/` — 22 root-level shims that used to forward to `experiments/<area>/` or `djgp.*`. Kept for external callers with hardcoded old import paths.
- `legacy/early_models/` — BNNJGP, NNJGP, NN_test, NNJGP_test, SIR_GP.
- `legacy/early_djgp/` — DJGP_CV, DJGP_CV_modified.
- `legacy/early_jumpgp/` — JumpGP_test_CV, JumpGP_test_local.
- `legacy/old_vi_utils/` — VI_utils_gpu_acc_UZ_qumodified_cor, minibatch_check_VI_utils.
- `legacy/old_experiments/` — experiment.py, draft.py.
- `legacy/old_plotting/` — deal_fig.py, replot.py, plot_res.py.
- `legacy/old_data_gen/` — new_data_gen.py, test_datagen.py.
- `legacy/r_code/` — main_OHT4D_TGP_R1.R, OHTDataset.Rmd, .Rhistory.
- `legacy/jumpgp_code_py/` — older vendored Jump GP copy.
- `legacy/reorg_2026-05-23/` — one-shot reorganization scripts (`_reorg_rewrite_imports.py`, `_reorg_rewrite_notebooks.py`) kept for reference.

## Pre-Reorg Importance Notes (Historical)

These notes were the original `repository_map.md` content and refer to **pre-reorg paths**. Use them to understand which files were the latest research iteration when looking at older commits.

| Pre-reorg file | Last modified observed | Role | Post-reorg location |
| --- | ---: | --- | --- |
| `djgp/variational.py` | 2025-10-09 | Latest variational DJGP. | `src/djgp/variational.py` |
| `djgp/active_learning.py` | 2025-10-09 | Active-learning wrapper. | `src/djgp/active_learning.py` |
| `DJGP_test.py` | 2025-10-09 | DJGP CLI. | `src/shared/djgp_runner.py` |
| `JumpGP_test.py` | 2025-10-09 | JGP CLI. | `src/shared/jumpgp_runner.py` |
| `experiments/active_learning/run_synth_al4jgp.py` | 2025-10-10 | AL synthetic driver. | unchanged |
| `experiments/uci/new_dataset.py` | 2025-09-29 | UCI benchmark. | unchanged |
| `JumpGaussianProcess/` | 2025-09-25 | Vendored canonical JGP. | `src/JumpGaussianProcess/` |
| `experiments/erosion/erosion_exp_alone.py` | 2025-08-05 | Erosion orchestrator. | unchanged |
| `experiments/synthetic/minibatch_LH.py` | 2025-08-05 | LH minibatch driver. | unchanged |
| `dataset_analysis.py` | 2025-08-05 | Dataset analysis. | `src/data_gen/analysis.py` |
| `utils.py`, `utils1.py` | 2025-05-07 | Shared utils. | `src/shared/utils.py`, `src/shared/utils1.py` |
| `VI_utils_gpu_acc_UZ_qumodified_cor.py` | 2025-05-21 | Older VI utils. | `legacy/old_vi_utils/` |
| `data_generate.py` | 2025-05-21 | Early synth generator. | `src/data_gen/synthetic.py` |
| `new_highdata_gen.py`, `new_highdata_gen_utils.py` | various | LH high-dim generators. | `src/data_gen/highdata.py`, `src/data_gen/highdata_utils.py` |
| `LHDataset_AE.py`, `AE.py`, `UCIdataset_autoencoder.py` | 2025-08–09 | AE expansion utilities. | `src/data_gen/lh_autoencoder.py`, `src/data_gen/autoencoder.py`, `src/data_gen/uci_autoencoder.py` |
| `JumpGP_code_py/` | 2025-05–07 | Older vendored JGP copy. | `legacy/jumpgp_code_py/` |

## Import Rewrite Cheat Sheet

For anyone reading pre-reorg code or commits:

| Pre-reorg import | Post-reorg import |
| --- | --- |
| `from utils import X` | `from shared.utils import X` |
| `from utils1 import X` | `from shared.utils1 import X` |
| `from data_generate import X` | `from data_gen.synthetic import X` |
| `from new_highdata_gen import X` | `from data_gen.highdata import X` |
| `from new_highdata_gen_utils import X` | `from data_gen.highdata_utils import X` |
| `from AE import X` | `from data_gen.autoencoder import X` |
| `from LHDataset_AE import X` | `from data_gen.lh_autoencoder import X` |
| `from UCIdataset_autoencoder import X` | `from data_gen.uci_autoencoder import X` |
| `from dataset_analysis import X` | `from data_gen.analysis import X` |
| `from DeepGP_test import X` | `from shared.deepgp import X` |
| `from DJGP_test_validation import X` | `from shared.djgp_validation import X` |
| `from maxmin_design import X` | `from shared.maxmin_design import X` |
| `from DJGP_test import X` | `from shared.djgp_runner import X` |
| `from JumpGP_test import X` | `from shared.jumpgp_runner import X` |
| `from check_VI_utils_gpu_acc_UZ_qumodified_cor import X` | `from djgp.variational import X` |
| `from active_djgp_acquisition import X` | `from djgp.active_learning import X` |
| `from calculate_bias_and_variance import X` | `from djgp.acquisition_metrics import X` |
| `from new_minibatch_try import X` | `from djgp.minibatch import X` |

## ISM-LMJGP (Identifiable Structured-Metric LMJGP)

Transductive local regressor. For each test anchor `x_t`, fit a JumpGP-style local
GP in a learned low-dim projected space `z = diag(exp(eta_t/2)) U^T x`, where `U` is a
global orthonormal subspace (shared) and `eta_t` a per-anchor diagonal log-metric drawn
from a sparse GP. The **analytic** variant marginalises the projection posterior `q(W)`
into the local GP in closed form (Bayesian-GPLVM ψ-statistics) — no Monte-Carlo sampling
of `W` in training. A robust per-neighbour inlier/outlier mixture (collapsed `logsumexp`)
handles contamination, and an optional **data-driven heteroscedastic observation noise**
(`obs_noise_prior_mode="data_driven"`) anchors each anchor's noise prior at a
nearest-neighbour-difference estimate of the local residual scale — this fixes the
analytic head's undercoverage on datasets where the fixed prior is too tight
(Wine/Parkinsons/LH); keep `fixed` where coverage already comes from `var_f`
(L2, Appliances, PDEBench-Burgers). Predict with the analytic head (data-driven noise) or
the sample-W CEM head; select the best of K subspace inits by unsupervised
head-disagreement, mix top-1/2/4.

- Method code: `src/djgp/projections/structured_metric_lmjgp.py`
  (depends on `src/djgp/projections/uncertain_w_jgp.py`).
- Full method, tricks, and 5-seed results (UCI Wine/Parkinsons/Appliances, synthetic
  L2/LH, PDEBench Burgers): `docs/ism_lmjgp_analytic.md`.
- Runners (experiment harnesses; may need experiment-tree import paths wired up):
  `experiments/synthetic/_exp_uci_hetnoise.py` (UCI),
  `experiments/synthetic/_exp_pdebench_hetnoise.py` + `_exp_pdebench_baselines_npz.py`
  (PDEBench), `experiments/synthetic/_exp_u_ensemble_5seed.py` (synthetic L2/LH, with
  inducing/minibatch/local-m ablation knobs), `experiments/uci/_exp_baselines_npz.py`
  (XGB/DKL on exported npz).
