# Projection Method Mathematics and Implementation Audit

This document records the math object, optimization target, prediction
pipeline, implementation path, third-party packages, and likely reviewer
challenge points for the repaired projection methods and baselines used in the
recent UCI / synthetic experiments.

It is deliberately implementation-facing: when the implemented method is an
empirical-Bayes surrogate rather than a full Bayesian posterior, the document
says so.

## Notation

- Training data: `D_train = {(x_i, y_i)}_{i=1}^N`, `x_i in R^D`.
- Test anchors: `x_*^(j)`, `j = 1,...,T`.
- Supervised projection-training anchors: `a in A`, with labels `(x_a,y_a)`.
- Local neighborhood for an anchor `a`:

  ```text
  N(a) = {(x_i,y_i)}_{i=1}^n
  ```

  For strict supervised variants, `(x_a,y_a)` is not included in `N(a)`.
- Learned projected dimension: `Q`.
- Projection at anchor `j`: `W_j in R^{Q x D}`.
- Projected coordinate: `z = W_j x`.
- Low-rank coefficient projection form:

  ```text
  W_j = rowNorm(W0 + alpha C_j V^T),
  V in R^{D x Rv}, C_j in R^{Q x Rv}.
  ```

  `rowNorm` normalizes each row of `W_j` to unit norm when enabled.
- Local JumpGP / projected JumpGP returns per-anchor Gaussian predictive
  approximations `(mu_j, sigma_j^2)`.
- Mixture aggregation over projection samples:

  ```text
  mu_bar_j = (1/M) sum_m mu_j^(m)

  sigma_bar_j^2 =
      (1/M) sum_m [ sigma_j^(m)^2 + (mu_j^(m) - mu_bar_j)^2 ].
  ```

## External Packages and Vendored Code

### XGBoost

Implementation:

- `djgp/baselines/xgboost_bootstrap.py`
- `xgboost.XGBRegressor`

This is a standard external package baseline. The implementation trains
`B = n_bootstrap` boosted-tree regressors on bootstrap resamples.

Predictive distribution:

```text
f_b(x) = prediction from bootstrap model b
mu(x) = mean_b f_b(x)
sigma_epi^2(x) = Var_b[f_b(x)]
sigma_a^2 = mean_i (y_i - mean_b f_b(x_i))^2
sigma^2(x) = sigma_epi^2(x) + sigma_a^2
```

Challenge point:

- This is not a calibrated Bayesian posterior. It is a practical bootstrap
  Gaussian approximation with a homoscedastic residual plug-in. It is a strong
  point-prediction baseline, but uncertainty comparisons should be described
  carefully.

### DKL-SVGP

Implementation:

- `djgp/baselines/dkl.py`
- `gpytorch.models.ApproximateGP`
- `gpytorch.variational.VariationalStrategy`
- `gpytorch.variational.CholeskyVariationalDistribution`
- `gpytorch.likelihoods.GaussianLikelihood`
- `gpytorch.mlls.VariationalELBO`
- `gpytorch.kernels.ScaleKernel(RBFKernel(ard_num_dims=d_latent))`

Model:

```text
h_theta(x) = NN feature map in R^{d_latent}
f(h) ~ sparse variational GP
y | f ~ N(f(h_theta(x)), sigma_n^2)
```

The neural feature map is:

```text
Linear(D, d_hidden) -> ReLU -> Linear(d_hidden, d_latent)
```

The code standardizes `y` internally:

```text
y_z = (y - mean_train_y) / std_train_y,
```

trains the DKL model on `y_z`, and maps predictive mean/std back to original
`y` scale before RMSE / CRPS / coverage.

Challenge point:

- This is a legitimate GPyTorch SVGP baseline. It is still sensitive to feature
  standardization, inducing-point initialization, NN size, and learning rate.
  For paper claims, call it a strong but preprocessing-sensitive GP baseline.

### PCA / PLS Basis Construction

Implementation:

- `djgp/projections/lowrank_factorization.py`
- `sklearn.decomposition.PCA`
- `sklearn.cross_decomposition.PLSRegression`
- `sklearn.preprocessing.StandardScaler`

PLS directions are learned on standardized `X` and then converted back to raw-X
scale:

```text
X_std = StandardScaler().fit_transform(X)
PLSRegression(n_components=K, scale=False).fit(X_std, y)
V_raw = diag(1 / scale_X) V_std
```

PCA directions are similar:

```text
X_std = StandardScaler().fit_transform(X)
V_raw = diag(1 / scale_X) PCA_components^T
```

The combined basis is:

```text
V_tilde = [V_PLS, V_PCA, V_random]
V = qr(V_tilde)
```

Challenge point:

- PLS uses labels. That is fine for supervised methods and supervised
  initialization, but the paper must state that `W0` / `V` are training-label
  derived empirical-Bayes components.

## Local JumpGP Prediction Module

Implementation:

- `jumpgp` facade
- vendored `JumpGaussianProcess/`
- `djgp/projections/projected_jumpgp.py`
- `djgp/projections/inductive.py::run_projected_jumpgp_ensemble`

For a fixed projection `W_j`, neighborhood points are projected:

```text
Z_N = X_N W_j^T, z_* = x_* W_j^T.
```

Then the vendored JumpGP CEM implementation is called to obtain a local
predictive mean and variance. In the projection-ensemble methods, this local
predictor is treated as a plug-in module:

```text
W_j^(m) -> JumpGP-CEM -> (mu_j^(m), sigma_j^(m)^2).
```

Challenge point:

- This is not a standard PyPI package; it is vendored research code. It should
  be described as the paper's JumpGP implementation via the local `jumpgp`
  facade, not as an external authoritative library.

## Original Transductive LMJGP

Implementation:

- `djgp/variational.py`
- `djgp/variational.py::qW_from_qV`
- `djgp/variational.py::compute_ELBO`
- `djgp/variational.py::train_vi`
- `djgp/variational.py::predict_vi`
- UCI wrapper: `experiments/uci/compare_uci_baselines.py::_run_djgp_block`

### Projection GP Prior

For each output row `q` and input dimension `d`, the projection field has a GP
prior represented by inducing variables:

```text
V_{:,q,d} = W_{q,d}(Z),   Z in R^{m2 x D}.
```

Variational posterior:

```text
q(V_{:,q,d}) = N(mu_{V,q,d}, diag(sigma_{V,q,d}^2)).
```

The code uses independent diagonal variational covariance per `(q,d)`.

Given an anchor `x_j`, GP conditioning induces:

```text
q(W_j) = integral p(W_j | V) q(V) dV.
```

For row `q`, with

```text
K_ZZ^q = var_w exp(-||Z-Z'||^2 / (2 ell_q^2))
K_xZ^q = var_w exp(-||x_j-Z||^2 / (2 ell_q^2))
A_j^q = K_xZ^q (K_ZZ^q)^{-1},
```

the implemented mean is:

```text
mu_W[j,q,d] = A_j^q mu_V[:,q,d].
```

The implemented marginal variance is:

```text
Var(W_{j,q,d}) =
    var_w - A_j^q K_Zx^q
    + sum_k [(K_ZZ^q)^{-1} K_Zx^q]_k^2 sigma_V[k,q,d]^2.
```

`qW_from_qV` returns a diagonal covariance matrix over input dimensions:

```text
cov_W[j,q,:,:] = diag(Var(W_{j,q,1}),...,Var(W_{j,q,D})).
```

### ELBO

For each test anchor `j`, a local region is built from the training-neighbor
set `N(x_*^(j))`. This is why the original method is transductive: test inputs
are used as projection anchors during optimization, though test labels are not.

Local latent variables:

- `u_j`: inducing values for local GP response.
- `omega_j`: gate/logistic parameters.
- `U_j`: outlier / jump mixture parameter.

The code optimizes:

```text
ELBO =
  sum_j data_term_j
  - KL(q(V) || p(V))
  - sum_j KL(q(u_j) || p(u_j)).
```

The data term uses analytic expected RBF-kernel quantities under `q(W_j)`:

```text
E_q(W_j)[K_fu], E_q(W_j)[K_uf K_fu],
```

and Gauss-Hermite quadrature for the expected log-sigmoid gate:

```text
E_q(W_j)[log sigmoid(omega_j^T [1, W_j x])]
E_q(W_j)[log(1 - sigmoid(omega_j^T [1, W_j x]))].
```

In code, the per-neighbor mixture contribution is:

```text
T1 = log N(y | GP local inlier predictive term)
     + E_q[log sigmoid(gate)]

T2 = log U_j + E_q[log(1 - sigmoid(gate))]

region_elbo_j = sum_i logsumexp(T1_{ji}, T2_{ji}).
```

### Prediction

At prediction time, `predict_vi` samples projection matrices from `q(W_*)`,
row-normalizes sampled rows, runs local JumpGP prediction for each sample, then
moment-matches the mixture.

Challenge points:

- It is transductive because `X_test` defines anchors during training.
- `q(W)` is diagonal by input dimension in the current implementation.
- ELBO improvement does not imply learned directions are non-trivial; the KL can
  keep `q(W)` close to an isotropic prior if directions are weakly identified.

## LMJGP with PLS-KL Tricks

Implementation:

- `djgp/variational_tricks.py`
- `experiments/synthetic/compare_lmjgp_tricks_synth.py`

This is not a new probabilistic model. It is the original LMJGP ELBO with
initialization and optimization modifications.

The objective implemented in `compute_objective_with_tricks` is:

```text
loss =
  - [ data_term/T - beta_W KL(q(V)||p(V))/T - KL_u/T ]
  + lambda_row R_row
  + lambda_aux R_aux.
```

For `pls_kl`, the settings are:

```text
mu_V_mode = "pls"
beta_W = KL warmup schedule from beta_init to beta_final
lambda_row = 0
lambda_aux = 0
```

The PLS initialization sets the inducing mean so that the induced projection
mean roughly matches a supervised PLS direction field. The KL warmup delays the
full prior pull:

```text
beta_W(step) = linear warmup over warmup_steps.
```

Other trick variants, when enabled, add:

```text
R_row = mean_jq (E_q ||W_{j,q,:}||^2 - target)^2
```

or an auxiliary correlation loss between projected distances and response
differences:

```text
R_aux = -Corr(d_proj^2, d_y^2).
```

Challenge points:

- `lmjgp_pls_kl` is an optimization/initialization ablation, not a separate
  Bayesian model.
- PLS uses training labels. This is valid as supervised initialization, but
  must be acknowledged.
- It can numerically blow up in some runs; the synthetic train=1000 LH run had
  one exploded seed, recorded separately.

## Strict Supervised / Inductive LMJGP

Implementation:

- `experiments/uci/compare_inductive_projection.py`
- `_train_lmjgp_supervised_predictive`
- `_qv_anchor_predictive_nll`
- `_anchor_gp_predictive_nll_batch`
- `_run_lmjgp_supervised`

### Split

The supervised method separates:

```text
D_base    : neighborhood pool
D_anchor  : labeled projection-training anchors
D_test    : final test points
```

For each labeled anchor `a in D_anchor`, the neighborhood `N(a)` is built only
from `D_base`. The anchor label `y_a` is never inserted as a region observation.

### Objective

This method trains `q(V)` by a predictive VI objective:

```text
L_supervised =
  (1/|A|) sum_{a in A} E_{q(W_a)}
      [ - log p_GP(y_a | x_a, N(a), W_a) ]
  + lambda_KL KL(q(V)||p(V)) / |A|.
```

Equivalently, maximizing:

```text
sum_a E_q(W_a)[log p(y_a | x_a, N(a), W_a)]
  - beta KL(q(V)||p(V)).
```

The local predictive density is an RBF GP plug-in on projected coordinates:

```text
Z_N = X_N W_a^T,
z_a = x_a W_a^T,
K_NN = sigma_f^2 exp(-||z_i-z_l||^2/(2 ell^2)) + sigma_n^2 I,
k_aN = sigma_f^2 exp(-||z_a-z_i||^2/(2 ell^2)).
```

After centering by the local-neighbor mean:

```text
mu_a = mean(y_N) + k_aN K_NN^{-1}(y_N - mean(y_N))
var_a = sigma_f^2 + sigma_n^2 - k_aN K_NN^{-1} k_Na
```

and the predictive NLL is:

```text
0.5 [ log(2 pi var_a) + (y_a - mu_a)^2 / var_a ].
```

The trainable local-GP hyperparameters are shared:

```text
log_lengthscale in R^Q,
log_signal_var scalar,
log_noise_var scalar.
```

### Reparameterized `W_a`

The implementation explicitly samples `W_a` with a differentiable Cholesky
path:

```text
q(W_a) = N(mu_a^W, Sigma_a^W)
L_a = chol(Sigma_a^W + jitter I)
W_a^(m) = mu_a^W + L_a eps_m.
```

In code:

```text
L_W = torch.linalg.cholesky(cov_W + 1e-8 I)
W_samples = mu_W + einsum(L_W, eps)
W_samples = rowNormalize(W_samples)
```

This was added specifically to avoid a detached / non-differentiable predictive
term. The runner can perform a predictive-only gradient check and records:

```text
has_qV_gradient,
mu_V_grad_norm,
sigma_V_grad_norm,
Z_grad_norm.
```

### Prediction

After training on labeled anchors, test points are used only at prediction
time. `q(W_*)` is induced by GP conditioning through `q(V)`, and `predict_vi`
uses test neighborhoods plus projection samples to produce mixture predictions.

Challenge points:

- This is not the original DJGP full joint ELBO. It is a supervised predictive
  VI objective.
- The local predictive density is a differentiable local-GP surrogate during
  training; final prediction still uses the JumpGP refit module.
- It is more inductive than the original transductive LMJGP because `X_test`
  is not used to train `q(V)`.

## CEM-EM Projection MAP

Implementation:

- `djgp/projections/cem_em_projection.py`
- `CoeffProjection`
- `train_projection_cem_em`
- `experiments/uci/compare_inductive_projection.py::_run_cem_em_transductive`
- `experiments/uci/compare_inductive_projection.py::_run_cem_em_supervised`

### Parameterization

```text
W_j = rowNorm(W0 + alpha C_j V^T).
```

Here:

- `W0` is initialized from PLS and is learnable in these experiments.
- `V` is a fixed QR-orthonormalized basis from PLS/PCA/random directions.
- `C_j` is an anchor-specific coefficient matrix.

### Generalized EM

The method alternates:

1. C-step: with `W_j` frozen, call JumpGP CEM on projected neighborhood data
   and cache inlier indicators, gate weights, and GP hyperparameters.
2. M-step: with cached JumpGP quantities frozen, update `W0` and `C_j` by
   differentiating a surrogate loss.

The implemented M-step loss has the form:

```text
loss =
  sum_j [
      - log p_GP(y_{I_j} | Z_{I_j}; cached theta_j)
      + lambda_gate BCE(gate_j(Z_j), r_j)
      + lambda_anchor_pred NLL(y_j | x_j, N(j), W_j)
  ]
  + lambda_smooth sum_{j,j'} S_jj' ||C_j - C_j'||_F^2
  + lambda_scale row_norm_penalty.
```

For transductive CEM-EM:

```text
X_fit = X_test, y_fit = None,
lambda_anchor_pred = 0.
```

Thus test labels are not used, but test inputs are optimization anchors.

For supervised CEM-EM:

```text
X_fit = X_anchor, y_fit = y_anchor,
neighborhoods from D_base,
lambda_anchor_pred > 0.
```

The supervised version uses anchor labels through the leave-one-anchor
predictive term but excludes the anchor point from its neighborhood.

Challenge points:

- The CEM-EM MAP method is not a Bayesian posterior. It is a deterministic
  generalized-EM / empirical-risk projection learner aligned with the JumpGP
  CEM prediction module.
- The C-step relies on the vendored JumpGP CEM solver, and the M-step optimizes
  a differentiable frozen-CEM surrogate rather than the exact original CEM
  objective.

## CEM-EM GP-Conditional Ensemble

Implementation:

- `djgp/projections/inductive.py::gp_condition_coefficients`
- `sample_induced_coefficients`
- `induced_W_from_coefficients`
- `run_projected_jumpgp_ensemble`

After CEM-EM obtains MAP coefficients `C_A`, each coefficient dimension is
treated as a noisy empirical observation of a GP field:

```text
c_p(x) ~ GP(0, k)
C_A[:,p] = c_p(X_A) + empirical noise.
```

With:

```text
K_AA = k(X_A, X_A) + rho I,
K_*A = k(X_*, X_A),
```

the implemented conditional mean is:

```text
m_* = K_*A K_AA^{-1} C_A.
```

The simple GP-conditional version uses the same scalar marginal conditional
variance for all coefficient dimensions:

```text
s_*^2 = k(x_*,x_*) - K_*A K_AA^{-1} K_A*.
```

Samples:

```text
C_*^(m) = m_* + s_* eps_m,
W_*^(m) = rowNorm(W0 + alpha C_*^(m) V^T).
```

Then each `W_*^(m)` is passed to projected JumpGP and mixture-aggregated.

Challenge point:

- This is empirical-Bayes interpolation of MAP coefficients. It should not be
  called a full posterior over `C_A` because uncertainty in the learned MAP
  coefficients is not represented.

## CEM-EM Diagonal Laplace Ensemble

Implementation:

- `djgp/projections/laplace_cem.py`
- `diagonal_cem_data_hessian`
- `gp_condition_coefficients_laplace_diag`
- `sample_induced_coefficients_diag`

This starts from MAP coefficients `C_hat_A` and computes a diagonal / blockwise
Laplace approximation around the frozen-CEM surrogate.

For a coefficient dimension `p`, the intended local approximation is:

```text
q(C_{A,p}) approx N(C_hat_{A,p}, Sigma_{A,p})
```

with diagonal anchor covariance:

```text
Var(C_{a,p}) =
  1 / ( H_data[a,p] + [K_AA^{-1}]_{aa} ).
```

The test conditional adds two uncertainty sources:

```text
base_gp_var(x_*) =
  k(x_*,x_*) - K_*A K_AA^{-1} K_A*

extra_laplace_var(x_*,p) =
  B_* Var(C_{A,p}) B_*^T,
  B_* = K_*A K_AA^{-1}.
```

So:

```text
Var(C_{*,p}) = base_gp_var + extra_laplace_var.
```

Challenge points:

- This is diagonal Laplace, not full Laplace. It ignores off-diagonal posterior
  covariance between anchors and coefficient dimensions.
- Hessian failures are caught and recorded; skipped/failed curvature estimates
  use conservative fallbacks.

## CEM-EM Full-Covariance VI Ensemble

Implementation:

- `djgp/projections/vi_cem.py`
- `train_cem_coefficient_vi`
- `gp_condition_coefficients_vi_fullcov`
- `sample_induced_coefficients_fullcov`
- Main selected synthetic method:
  `cem_em_transductive_vi_fullcov_ensemble`

### Variational Posterior over Coefficients

Let `P = Q Rv`. Flatten each anchor coefficient:

```text
C_A in R^{A x Q x Rv}
C_flat in R^{A x P}.
```

For each coefficient dimension `p`, the implementation uses a full covariance
over anchors:

```text
q(c_{A,p}) = N(m_p, S_p),
S_p = L_p L_p^T.
```

The Cholesky factor `L_p` is parameterized by unconstrained raw lower-triangular
entries with a softplus diagonal:

```text
diag(L_p) = softplus(raw_diag) + floor.
```

GP prior:

```text
p(c_{A,p}) = N(0, K_AA).
```

KL:

```text
KL(q || p) =
  0.5 sum_p [
      tr(K_AA^{-1} S_p)
      + m_p^T K_AA^{-1} m_p
      - A
      + log |K_AA|
      - log |S_p|
  ].
```

### Objective

The VI objective is:

```text
loss =
  (1/A) [
      E_{q(C_A)} frozen_CEM_loss(C_A)
      + beta KL(q(C_A)||p(C_A))
  ].
```

`frozen_CEM_loss` is the same differentiable surrogate used by the CEM-EM
M-step:

```text
gate BCE + negative GP inlier log marginal
+ optional anchor predictive NLL.
```

The expectation is estimated by Monte Carlo reparameterization:

```text
c_{A,p}^{(m)} = m_p + L_p eps_p^{(m)}.
```

### Test Conditioning

For query/test anchors:

```text
B = K_*A K_AA^{-1}
m_*p = B m_p
S_*p =
    K_** - K_*A K_AA^{-1} K_A*
    + B S_p B^T.
```

The first term is ordinary GP interpolation uncertainty. The second term
propagates variational uncertainty in the learned anchor coefficients.

Samples:

```text
c_*p^(m) ~ N(m_*p, S_*p)
C_*^(m) -> W_*^(m) = rowNorm(W0 + alpha C_*^(m) V^T)
```

and final prediction uses projected JumpGP ensemble aggregation.

Challenge points:

- This is the cleanest CEM posterior-style variant implemented so far, but the
  likelihood is still the frozen-CEM surrogate, not the exact original
  non-differentiable JumpGP CEM objective.
- It is full covariance across anchors for each coefficient dimension, but it
  assumes independence across coefficient dimensions `p`.
- In transductive mode, `X_test` is still used as the anchor set for fitting
  `C_A`; test labels are not used.

## Low-Rank Residual LMJGP

Implementation:

- `djgp/variational_lowrank.py`
- `train_vi_lowrank_residual`
- `predict_vi_lowrank_residual`
- `experiments/uci/run_repaired_projection_variants.py`

### Model

This replaces the full mean-zero projection GP:

```text
W(x) in R^{Q x D}
```

with:

```text
W(x) = W0 + s C(x) V^T,
```

where:

- `W0 in R^{Q x D}` is fixed during this VI.
- `V in R^{D x R}` is fixed and orthonormal.
- `s` is `residual_scale`.
- `C(x) in R^{Q x R}` is random.

Each coefficient field has an independent GP:

```text
c_{q,r}(x) ~ GP(0,k_q).
```

Inducing variables:

```text
U_{:,q,r} = c_{q,r}(Z)
q(U_{:,q,r}) = N(mu_U[:,q,r], diag(sigma_U[:,q,r]^2)).
```

For an anchor `x_j`, GP conditioning gives:

```text
q(C_j) = N(mu_C_j, Sigma_C_j).
```

Because `W_j = W0 + s C_j V^T` is linear in `C_j`:

```text
mu_W_j = W0 + s mu_C_j V^T
Cov(W_{j,q,:}) = s^2 V Cov(C_{j,q,:}) V^T.
```

The original LMJGP expected-kernel formulas still apply because `W_j delta` is
Gaussian.

### ELBO

The low-rank residual ELBO is the same form as original LMJGP, replacing
`KL(q(V)||p(V))` by `KL(q(U)||p(U))` for the coefficient GPs:

```text
ELBO_lowrank =
  data_term
  - KL(q(U)||p(U))
  - sum_j KL(q(u_j)||p(u_j)).
```

### Variants

In `run_repaired_projection_variants.py`, implemented `W0,V` choices include:

```text
lmjgp_lowrank_pls_pca:
  W0 = PLS W0
  V = QR([PLS, PCA, random])

lmjgp_lowrank_cem_mean_pls_pca:
  W0 = mean_j W_j^CEM
  V = QR([PLS, PCA, random])

lmjgp_lowrank_cem_mean_cem_svd:
  W0 = mean_j W_j^CEM
  V = top right singular vectors of centered CEM rows,
      QR-completed by fallback V if needed.
```

### Prediction

At test anchors:

```text
C_*^(m) ~ q(C_*)
W_*^(m) = rowNorm(W0 + s C_*^(m) V^T)
```

then projected JumpGP ensemble aggregation.

Challenge points:

- `W0` and `V` are fixed empirical-Bayes components. If `W0` comes from CEM,
  the low-rank residual method inherits the transductive/supervised status of
  that CEM fit.
- This is lower-dimensional and easier to optimize than full LMJGP, but it
  restricts all projection variation to `span(V)`.

## Residual-Ridge Variants

Implementation:

- `experiments/uci/compare_parkinsons_grouped_split.py`
- `experiments/uci/compare_smalltrain_multiseed_grouped.py`
- `experiments/uci/try_lmjgp_remedies.py`

Model:

```text
y = g_ridge(x) + r(x).
```

Train ridge on the training set, then run LMJGP / supervised LMJGP on residuals:

```text
r_i = y_i - g_ridge(x_i).
```

Prediction:

```text
mu_y(x_*) = g_ridge(x_*) + mu_r(x_*)
sigma_y(x_*) = sigma_r(x_*).
```

Challenge point:

- This is a pragmatic global-mean correction, not part of the original DJGP
  model. It is best presented as a failure-mode remedy / ablation, not as the
  central method unless formalized.

## Method Status Summary

| Method | Inductive? | Uses test X during fitting? | Uses test y? | Posterior claim |
|---|---:|---:|---:|---|
| `xgb_bootstrap` | yes | no | no | bootstrap Gaussian approximation |
| `dkl_svgp` | yes | no | no | standard SVGP variational posterior via GPyTorch |
| `jgp_pca` | yes | no | no | deterministic PCA + local JumpGP |
| `lmjgp_transductive` | no | yes | no | variational posterior under original LMJGP ELBO |
| `lmjgp_pls_kl` | no | yes | no | same LMJGP posterior, PLS init + KL warmup |
| `lmjgp_supervised_reparam` | yes | no | no | supervised predictive VI over projection GP |
| `cem_em_transductive_map_or_mean` | no | yes | no | MAP / generalized EM |
| `cem_em_supervised_map_or_mean` | yes | no | no | supervised MAP / generalized EM |
| `cem_em_*_gp_cond_ensemble` | depends on anchor mode | depends | no | empirical-Bayes GP interpolation of MAP coefficients |
| `cem_em_*_laplace_ensemble` | depends on anchor mode | depends | no | diagonal Laplace empirical-Bayes approximation |
| `cem_em_*_vi_fullcov_ensemble` | depends on anchor mode | depends | no | full-anchor-covariance VI over coefficient values, independent over coefficient dimensions |
| `lmjgp_lowrank_*` | depends on anchor/W0 source | depends | no | coefficient-GP VI around fixed `W0,V` |

## Recommended Paper Wording

Use precise wording:

- For original LMJGP:
  "transductive projection-anchor inference; test labels are never used."
- For strict supervised LMJGP:
  "supervised predictive variational training of the projection field on
  labeled training anchors; test inputs are used only at prediction time."
- For CEM full-cov VI:
  "a variational empirical-Bayes posterior over coefficient values under a
  differentiable frozen-CEM surrogate."
- For CEM GP-conditioning only:
  "empirical-Bayes interpolation of MAP coefficients; not a full posterior."
- For low-rank residual LMJGP:
  "fixed empirical-Bayes projection mean plus a low-rank coefficient GP
  residual field."

Avoid claiming:

- that CEM-EM MAP is Bayesian;
- that GP-conditional MAP interpolation is a full posterior;
- that supervised LMJGP optimizes the original DJGP full joint ELBO;
- that the vendored JumpGP module is an external standard library.

