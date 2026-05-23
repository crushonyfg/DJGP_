# Diagnostics (projection posterior)

Appendix-style checks for **local projections** \(q(W_j)\) in LMJGP / DJGP. The script reports four complementary groups; \(E[W]\approx 0\) does **not** imply a degenerate Mahalanobis metric when posterior variance is nontrivial.

## What gets exported

1. **`EW_raw`** — Frobenius norm, condition number, and effective rank of **\(E[W_j]\)**. When \(E[W]\) is (near) zero, \(\kappa\) and effective rank are **`null` in JSON** (stored as `nan` internally), not `0`, to avoid misreading a “healthy” condition number.

2. **`metric_E_WtW`** — Moments **\(A_j = E[W_j^\top W_j]\)** and **\(B_j = E[W_j W_j^\top]\)** (from `mu_W` and full `cov_W`: mean outer product plus covariance terms). Summaries: trace / effective rank (side \(A\) = input space, side \(B\) = latent rows), top and smallest positive eigenvalue of \(B\).

3. **`kernel_metric_A`** — Local RBF Gram matrices built from **expected squared Mahalanobis distance** \(\delta^\top A_j \delta\). Reports \(\kappa(K)\), \(\kappa(K+\mathrm{jitter}\,I)\), and \(\kappa(K+\mathrm{jitter}\,I+\sigma_{\mathrm{noise}}^2 I)\) (the last aligns with a noisy GP covariance). Contrast with **`kernel_projected_Z_from_EW_only`** per anchor ( \(Z=X E[W]^\top\) ), which can collapse when \(E[W]\approx 0\).

4. **`sampled_W_row_normalized`** / **`kernel_sampled_row_normalized`** — Monte Carlo draws from \(q(W)\) with the **same row-normalization as `predict_vi`**, plus kernel condition numbers on projected \(Z\). Per-anchor **median** \(\kappa\) over MC draws is in `per_anchor_flat.kappa_K_sampled_norm_plus_noise`.

Nested **`per_anchor`** rows include `EW_raw`, `metric_moment_matrices`, `kernel_metric_A_expected_sqdist`, `kernel_projected_Z_from_EW_only`, optional `pairwise_dist_corr_true_z` (synthetic with known \(Z\)), and `kernel_sampled_row_normalized_local`. **`per_anchor_flat`** narrows fields for scatter plots vs. absolute prediction error.

## Run

Canonical commands live in the module docstring of:

`experiments/diagnostics/run_projection_w_diagnostics.py`

Implementation:

- `djgp/diagnostics/projection_posterior.py` — metrics and `compute_full_projection_diagnostics`.

**Tip:** use `--max_anchors K` to subsample test points; full UCI test sets make `predict_vi` slow because it runs local JumpGP per anchor with a progress bar.
