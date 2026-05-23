# CEM-aligned EM Projection Learning

This document records the design, implementation, key bug fixes, and current
recommended defaults for the CEM-aligned EM projection learning module
(`djgp.projections.cem_em_projection`). It is intended as a single reference
for anyone returning to the work later — read this first, then the source.

## 1. Why a CEM-aligned EM at all?

Our existing variational learner (`djgp.variational`) trains a global
projection posterior `q(V)` and a per-anchor `q(u_j)` by maximising an ELBO.
At test time, prediction relies on a Monte-Carlo over `q(V)` (`predict_vi`):
each MC draw is fed into `JumpGP_LD` (the CEM-fit local model), and the
predictions are averaged. That training/prediction pipeline has two
mismatches with what we actually want at test time:

1. The training objective (ELBO) is *not* the same as the JumpGP-CEM
   marginal likelihood that drives the per-anchor predictive mean/variance,
   so improving ELBO does not necessarily improve test-time CRPS.
2. The single global posterior `q(V)` is forced to compromise across all
   anchors; per-anchor adaptation only enters via `q(u_j)`, which controls
   the gate, not the per-anchor projection direction.

The CEM-aligned EM directly maximises the same JumpGP-CEM marginal
likelihood that the predictor uses, while letting the projection differ
per-anchor through a low-rank perturbation of a shared base.

## 2. Model

Given anchors `j = 1, …, T`, latent dimension `Q`, input dimension `D`, and a
fixed input-side basis `V ∈ R^{D × Rv}`:

```
W_j  =  rowNorm( W_0  +  α · C_j V^T )                 ∈ R^{Q × D}
```

- `W_0 ∈ R^{Q × D}` — shared base projection (PLS-initialised).
- `C_j ∈ R^{Q × Rv}` — per-anchor coefficient (small, learnable).
- `V ∈ R^{D × Rv}` — fixed basis (PLS + PCA + random columns).
- `α ∈ R_+` — perturbation scale (`coeff_scale`).
- `rowNorm` — divide each row of the resulting matrix by its L2 norm.

`Rv ≪ D`, so `T·Q·Rv ≪ T·Q·D`: this is the "low-rank tensor factorisation"
ablation, but with row normalisation and a perturbation scale.

When `V = I_D` and `learn_W0=True` we recover the unconstrained per-anchor
`W_j` model (`cem_em_free`), used as an upper bound.

## 3. Algorithm: outer EM with frozen-CEM-internal M-step

Each outer iteration performs:

1. **C-step (per anchor, no autograd)**: Run `JumpGP_LD` (CEM mode) on the
   neighbourhood projected by the current `W_j`. Cache the CEM internals:
   `r_j` (inlier indicators), `w_j` (gate weights), `logθ_j` (kernel
   hyperparams), `m_s_j` (gate offsets).
2. **M-step (joint, autograd through `W_j`)**: Optimise

   ```
   L(W_0, {C_j})  =
        − mean_j  log p(y_inliers_j | Z_inliers_j; logθ_j frozen)
        + λ_g · mean_j  BCE(σ(w_j · Z_j + m_s_j), r_j)
        + λ_s · Smooth({C_j})
        + λ_r · RowNormPenalty({W_j})            # disabled when row_normalize_W=1
   ```

   for `n_m_steps` Adam steps; the cached CEM internals are treated as
   constants during this phase (it is the GEM/CEM step where the latent
   assignments are fixed before maximising over the parameters).

3. **Smoothness penalty** (graph-Laplacian on `{C_j}` weighted by an RBF on
   `X_anchors`):

   ```
   Smooth = (Σ_{j,j'} K_{jj'} ‖C_j − C_{j'}‖_F²)  /  Σ_{j,j'} K_{jj'}
   ```

   The denominator (`normalize_by_graph_weight=True`) makes the term
   `O(1)` regardless of the number of anchors `T`. Without it the penalty
   scales `O(T²)` and easily dominates the rest of the loss for large `T`.

4. **Best-W tracking**: `track_best_W=True` keeps the `W` that achieved
   the lowest M-step loss across outer iterations. Final returned `W` is
   the best one, not the last one.

## 4. Hyperparameter / numerical lessons (the bugs we hit)

In rough chronological order, these are the four fixes that made CEM-EM
actually work on real-world UCI data; do **not** revert any of them
without understanding why they exist.

### 4.1  Row-normalize `W` (and `W_0_init`)

If `X` is unstandardised (Wine, Parkinsons) the PLS-initialised rows can
have norm `‖W_pls,q‖ ≈ 10²–10⁴`. The original "row-norm penalty"
`(‖W_q‖² − target)²` did not actually constrain `W` — its target was set
from `W_pls` itself, so the penalty just kept `W` near the huge PLS scale.

The fix is structural, not regularisation:

- The forward pass divides every row of `W_0 + α C_j V^T` by its L2 norm
  (`row_normalize=True`).
- **Crucially**, `W_0_init` is also row-normalised inside
  `CoeffProjection.__init__` when `row_normalize=True`. Otherwise raw-X
  PLS rows of magnitude `~10⁴` make `α · C_j V^T` (which has magnitude
  `~1`) negligible inside the row-norm; the gradient through `rowNorm`
  effectively projects `α C_j V^T` onto the perpendicular complement of
  `W_0` and divides by `‖W_0‖`, killing all per-anchor adaptation.

After this fix a typical Parkinsons run goes from
`anchor_dispersion ≈ 0.0` (collapsed onto PLS) to `anchor_dispersion ≈ 1.15`
(genuinely per-anchor) and `eff_rank` from 1.0 to ~2.8.

### 4.2  Normalise the smoothness penalty

`laplacian_smoothness` was originally `Σ_{j,j'} K_{jj'} ‖C_j − C_{j'}‖²`.
For `T = 650` anchors on Wine Quality the value reached `~10³` while
`gp_log_marginal` lived in `O(10)`, so even `λ_s = 0.1` made smoothness
dominate the entire loss and crushed `C_j` to zero.

The fix is to divide by `Σ_{j,j'} K_{jj'}`
(`normalize_by_graph_weight=True`, default ON). This makes the value
`O(1)` regardless of `T` and lets `λ_s ∈ [10^{-4}, 10^{-2}]` be the
useful range across all dataset sizes.

### 4.3  Add a perturbation scale `α`

Even with rows normalised, `C_j V^T` after a few hundred Adam steps has
a frobenius norm of `~1`; the same as `W_0`. To allow C to actually rotate
W more than 45°, parameterise `W_j = rowNorm(W_0 + α · C_j V^T)`. In
practice `α ∈ {1, 3}` is enough; `α = 10` makes optimisation noisier and
hurts on Parkinsons.

### 4.4  Track best-W across outer iterations

The outer EM loss is **not** monotone — re-running JumpGP-CEM with a
freshly updated `W` can flip a borderline neighbourhood from inlier to
outlier and produce a transient regression. With `track_best_W=True` we
keep the `W` from the outer iteration with the lowest M-step loss; in
the Parkinsons grid the best outer was 3, 4, or 5 depending on `(α, λ_s)`,
not always the last one.

## 5. Recommended defaults (validated on Parkinsons, Q ≈ 5, n=35)

```python
CemEmConfig(
    n_outer=5,
    n_m_steps=80,
    lr=0.01,
    lambda_gate=0.1,
    lambda_smooth=0.01,        # graph-normalised
    lambda_scale=0.0,          # row_normalize_W replaces this term
    normalize_smoothness=True,
    track_best_W=True,
    grad_clip=10.0,
)
CoeffProjection(
    coeff_scale=1.0,           # α=1; α=3 is also fine, α=10 is worse
    row_normalize=True,        # row-normalise W (and pre-normalise W_0_init)
    init_C_scale=0.0,          # start from pure PLS, let optimiser move C
)
# CLI:  --n 35 --m1 2 --n_outer 5 --n_m_steps 80 --lr 0.01
#       --lambda_gate 0.1 --lambda_smooth 0.01 --lambda_scale 0
#       --coeff_scale 1 --row_normalize_W 1 --normalize_smoothness 1
#       --track_best_W 1
```

## 6. Result summary

On Parkinsons (128 anchors, single seed, n=35):

| method | RMSE | CRPS | Wall (s) | cov95 | w95 |
|---|---|---|---|---|---|
| `pls_only` | 7.141 | 3.940 | 16.5 | 0.703 | 13.71 |
| `lmjgp_predict_vi` | 4.493 | 1.728 | 70.5 | 0.922 | 10.62 |
| `cem_em_coeff` (best) | **3.580** | **0.722** | 39.9 | 0.711 | 1.42 |

A 50% RMSE / 82% CRPS reduction over PLS-only and a 20% RMSE / 58% CRPS
reduction over the existing variational LMJGP. Wall time is lower than the
VI baseline because we avoid MC over `q(V)` at predict time.

For the multi-seed cross-method study (XGBoost-bootstrap / DKL / CEM-EM on
all three UCI datasets) see the entry in
[`experiments_log.md`](experiments_log.md).

## 7. Where the code lives

- `djgp/projections/cem_em_projection.py` — model + EM loop + diagnostics.
- `djgp/projections/lowrank_factorization.py` — `build_V_basis` and
  `compute_pls_W0` used to initialise `W_0` and the basis `V`.
- `djgp/projections/projected_jumpgp.py` — `run_projected_jumpgp` driver
  used at evaluation time to take learned `{W_j}` and produce per-anchor
  `(μ, σ)` via JumpGP-CEM.
- `djgp/projections/__init__.py` — public exports.
- `experiments/uci/compare_cem_em_projection.py` — UCI runner.
- `experiments/synthetic/compare_cem_em_projection_synth.py` — synthetic
  runner (with oracle and `subspace_alignment_to_truth`).
- `utils1.py::jumpgp_ld_wrapper` — wrapper around the vendored JumpGP-LD
  CEM that exposes `(w, ms, logtheta, r)` to the M-step.

## 8. Inductive supervised extension

`experiments/uci/compare_inductive_projection.py` adds a cleaner inductive
comparison path:

- Train CEM-EM coefficients on labeled training anchors `D_anchor`, with local
  neighborhoods drawn from `D_base`.
- Add an optional leave-one-anchor predictive NLL through
  `CemEmConfig.lambda_anchor_pred`; this term uses frozen CEM inliers and kernel
  parameters, but predicts the anchor label without putting it in the local fit.
- Return the learned `C` and `W0` from `train_projection_cem_em`, then use
  `djgp.projections.inductive.gp_condition_coefficients` to induce and sample
  `C_*` at true test anchors.
- Aggregate projected JumpGP predictions over sampled `W_*` with
  `run_projected_jumpgp_ensemble`.

This should be described as empirical-Bayes coefficient-field interpolation,
not a full Bayesian posterior, because `C_A` is still a MAP estimate. The
runner defaults to standardizing `y` and uses `anchor_pred_var_floor=0.05` in
standardized-y space to avoid overconfident CEM predictive variances dominating
the supervised anchor loss.

## 9. Things still on the TODO list (low-priority but documented)

- Sigma-clip / median-fallback in `run_projected_jumpgp` to defend against
  the rare anchor where local GP fitting blows up `σ` to inf (saw this at
  `α=3, λ_smooth=0.01` on Parkinsons — RMSE was fine but CRPS / w95
  exploded due to one anchor).
- Optional `eval_callback(W, outer)` in `train_projection_cem_em` is wired
  but not yet used; if you want true validation-driven early stopping you
  can pass a function that re-runs JumpGP on a held-out anchor slice.
- Coverage @ 95% is consistently a bit underconfident on Parkinsons.
  Consider conformal post-hoc inflation if calibration matters more than
  sharpness for downstream use.
