# PDEBench Burgers Surrogate

PDEBench is kept outside this repository at `../external/PDEBench` to avoid a nested Git repository inside DJGP.

The full PDEBench Burgers download is about 93GB. A single viscosity shard is still about 8GB, so use one shard first.

## Download One Shard

```powershell
conda run -n jumpGP python scripts\data\download_pdebench_burgers.py --nu 0.01 --yes-large-download
```

This writes raw HDF5 under `data/raw/pdebench/`, which is ignored by Git.
The downloader uses the `certifi` CA bundle from the active environment and writes incomplete downloads as `.part` files for resume. If Windows/Conda certificate verification still fails, update the environment certificates:

```powershell
conda update -n jumpGP certifi ca-certificates openssl
```

Use `--insecure-skip-tls-verify` only as a last resort on a trusted network.

## Convert To DJGP Format

Install `h5py` in the `jumpGP` environment if needed:

```powershell
conda install -n jumpGP h5py
```

Then convert a downloaded shard:

```powershell
conda run -n jumpGP python scripts\data\prepare_pdebench_burgers.py `
  --hdf5 data\raw\pdebench\1D\Burgers\Train\1D_Burgers_Sols_Nu0.01.hdf5 `
  --output-folder data\processed\pdebench_burgers_shockjump `
  --qoi shock_jump `
  --D 128 `
  --max-samples 2000 `
  --t0 0 `
  --t-final -1 `
  --near-threshold-delta 0.05
```

The output is `data/processed/pdebench_burgers_shockjump/dataset.pkl`, compatible with `DJGP_test.py` and `JumpGP_test.py`.
The dataset keeps `X_train`, `Y_train`, `X_test`, and `Y_test` unchanged for existing loaders, with conversion metadata stored under `args`.

## Inspect Processed Dataset

```powershell
conda run -n jumpGP python scripts\data\inspect_pdebench_burgers.py `
  --dataset-folder data\processed\pdebench_burgers_shockjump `
  --out results\pdebench_inspection\summary.json `
  --save-plots
```

This prints shape, QoI, split, standardization, y histogram, shock-location histogram, and near-threshold counts. Plots are written to `results/pdebench_inspection/` when `--save-plots` is enabled.

## Run Baseline Comparison

```powershell
conda run -n jumpGP python -m experiments.pdebench.compare_pdebench_burgers `
  --dataset-folder data\processed\pdebench_burgers_shockjump `
  --num_exp 1 `
  --max_anchors 200 `
  --lmjgp_steps 200 `
  --out results\pdebench_burgers_compare.pkl `
  --out_json results\pdebench_burgers_compare.json
```

The runner reports rows in this order:

```text
[rmse, mean_crps, wall_sec, coverage_90, width_90, coverage_95, width_95]
```

By default it runs XGB bootstrap, DKL-SVGP, LMJGP `predict_vi`, and CEM-EM projection variants. Disable expensive variants with flags such as `--no-include_dkl` or `--no-include_cemem`.

## Current Comparable Results

The current paper-facing PDEBench performance table should use the
near-threshold-balanced Burgers ShockJump surrogate:

`data/processed/pdebench_burgers_shockjump_balanced_delta01_n2000`

Comparable setting:

- train/test: `1600 / 400`.
- seeds: `0-4`.
- `Q=3`, `neighbors=25`, `lmjgp_steps=300`.
- Processed dataset already contains train-only standardized `X_train/X_test`
  and response values used by the runner.
- Near-threshold balancing at `delta=0.1`; test split has 106 near-threshold
  and 294 far-from-threshold points.

Primary artifacts:

- `results/pdebench_burgers_balanced_delta01_n2000_xgb_dkl_lmjgp_q3_n25_seeds0to4_summary.csv`
  for XGB-bootstrap, DKL-SVGP, and LMJGP `predict_vi`.
- `results/pdebench_burgers_balanced_delta01_n2000_jgp_pca_vs_lmjgp_q3_n25_seeds0to4_summary.csv`
  for the deterministic JGP-PCA appendix baseline.

Format below is `RMSE / CRPS / Cov90 / Width90`.

### All Test Points

| method | result |
|---|---:|
| XGB-bootstrap | 0.694 / 0.308 / 0.922 / 1.822 |
| DKL-SVGP | 0.897 / 0.425 / 0.741 / 1.282 |
| LMJGP predict_vi | 0.714 / 0.240 / 0.832 / 0.960 |
| JGP-PCA | 1.025 / 0.404 / 0.708 / 0.758 |

### Near Threshold

| method | result |
|---|---:|
| XGB-bootstrap | 0.833 / 0.400 / 0.853 / 1.890 |
| DKL-SVGP | 1.274 / 0.726 / 0.502 / 1.325 |
| LMJGP predict_vi | 0.877 / 0.377 / 0.798 / 1.613 |
| JGP-PCA | 1.421 / 0.813 / 0.594 / 1.499 |

### Far From Threshold

| method | result |
|---|---:|
| XGB-bootstrap | 0.636 / 0.275 / 0.946 / 1.797 |
| DKL-SVGP | 0.706 / 0.317 / 0.827 / 1.267 |
| LMJGP predict_vi | 0.644 / 0.191 / 0.844 / 0.724 |
| JGP-PCA | 0.837 / 0.256 / 0.748 / 0.491 |

Main reading:

- XGB-bootstrap is the strongest RMSE baseline on all/near/far subsets.
- LMJGP `predict_vi` is the strongest CRPS method on all/near/far subsets and
  has substantially sharper intervals than XGB while maintaining reasonable
  coverage.
- The near-threshold subset is the most important stress test: XGB still has
  the best point error, but LMJGP has the best distributional score.
- JGP-PCA is clearly worse than LMJGP, so a deterministic PCA projection does
  not explain the PDEBench gain.
- There is not yet a five-seed PDEBench table for the repaired UCI variants
  (`cem_em_transductive_vi_fullcov_ensemble` or low-rank CEM residual LMJGP).
  Existing CEM-EM PDEBench files are older seed-0 diagnostics and should not be
  mixed into the comparable table above.

## Projection Diagnostics

Projection-matrix diagnostics are separate from predictive performance. The
current diagnostic artifact is:

`experiments/pdebench/projection_matrix_diag_balanced_q3n25/`

Setting:

- Same balanced dataset family, seed 0 diagnostic.
- `max_test_anchors=128`, `max_train_anchors=128`, `Q=3`, `n=25`.
- Methods: LMJGP transductive/supervised, CEM-EM transductive/supervised,
  PLS-only, and PCA-train.
- Diagnostics do not run final JumpGP prediction.

Headline:

- LMJGP projection posterior samples are nearly diffuse over all 128 spatial
  coordinates: effective feature count is about `128 / 128`, spatial center of
  mass is central, and `E[W]` is essentially zero.
- CEM-EM projections are less diffuse and show weak local response-alignment
  signals, but the effect is modest.
- Interpretation: PDEBench LMJGP's predictive CRPS advantage is not currently
  explained by a clearly interpretable learned projection matrix. It more
  plausibly comes from local neighborhoods, JumpGP behavior, projection
  sampling/uncertainty aggregation, or random-projection-like smoothing.

## QoIs

- `probe`: `u(T, x_0)`.
- `shock_location`: `argmax_x |d u(T,x) / dx|`.
- `shock_jump`: `s + jump_size * I(s > jump_threshold) + noise`.
- `right_energy`: mean final-time energy to the right of the probe.
