# Additive Global GP + Local Residual Self-CEM with Uncertainty Propagation

## 1. Motivation

Current self-CEM improves uncertainty calibration, but its posterior mean is often close to the mean-W / raw-local prediction. A simple deterministic residual correction can improve RMSE, but if it is added after self-CEM as a plug-in, its uncertainty is poorly integrated.

The goal is to define a probabilistic global-local model where:

1. A global GP explains smooth, broad-scale mean structure.
2. A local self-CEM / JumpGP component explains discontinuities, local nonstationarity, and boundary effects.
3. The posterior uncertainty of the global GP is propagated into the local self-CEM likelihood during training.
4. The final predictive mean and variance are produced by a single additive probabilistic model, not by an ad hoc correction.

---

## 2. Notation

Let the training data be

\[
\mathcal D = \{(x_i,y_i)\}_{i=1}^N,
\qquad
x_i\in\mathbb R^D,
\qquad
y_i\in\mathbb R.
\]

Let \(x_*\) be a test point.

For each anchor \(a\), define a local neighborhood

\[
N_a = \{i_1,\dots,i_k\},
\qquad
|N_a|=k.
\]

Let

\[
X_a = 
\begin{bmatrix}
x_{i_1}^\top \\
\vdots \\
x_{i_k}^\top
\end{bmatrix}
\in\mathbb R^{k\times D},
\qquad
y_a =
\begin{bmatrix}
y_{i_1}\\
\vdots\\
y_{i_k}
\end{bmatrix}
\in\mathbb R^k.
\]

For a test anchor \(x_*\), its local training neighborhood is denoted

\[
N_*=\{i_1,\dots,i_k\},
\qquad
X_* \in \mathbb R^{k\times D},
\qquad
y_*^{\mathrm{loc}}\in\mathbb R^k.
\]

The local CEM model has a binary latent indicator for each neighbor:

\[
\gamma_{ai}\in\{0,1\},
\]

where

\[
\gamma_{ai}=1
\]

means point \(i\in N_a\) is an inlier for the local GP component, and

\[
\gamma_{ai}=0
\]

means point \(i\in N_a\) belongs to the local background / outlier component.

Let the inlier set for anchor \(a\) be

\[
I_a=\{i\in N_a:\gamma_{ai}=1\}.
\]

---

## 3. Additive global-local model

We model the response as

\[
y_i = g(x_i) + h_a(x_i) + \epsilon_i,
\qquad
i\in N_a.
\]

Here:

- \(g(x)\) is a global smooth latent function.
- \(h_a(x)\) is an anchor-specific local residual function.
- \(\epsilon_i\) is observation noise.

The global component is

\[
g \sim \mathcal{GP}(0,k_g).
\]

For each anchor \(a\), the local residual inlier component is

\[
h_a \sim \mathcal{GP}(m_a,k_{h,a}),
\]

where \(m_a\) is a local constant mean and \(k_{h,a}\) is the local kernel.

The observation model for inliers is

\[
y_i \mid g_i,h_{ai},\gamma_{ai}=1
\sim
\mathcal N(g_i+h_{ai},\sigma_a^2).
\]

The background model for outliers is

\[
y_i \mid g_i,\gamma_{ai}=0
\sim
\mathcal N(g_i+b_a,\tau_a^2),
\]

where \(b_a\) is a local background mean and \(\tau_a^2\) is a broad background variance.

In the simplest version, one may set

\[
b_a = 0,
\]

after subtracting the global GP posterior mean, but the more general form keeps \(b_a\) explicit.

---

## 4. Approximate posterior for the global GP

We maintain a posterior approximation

\[
q(g).
\]

For any local neighborhood \(X_a\), define the posterior mean and covariance

\[
\mu_g(X_a)
=
\mathbb E_{q(g)}[g(X_a)]
\in\mathbb R^k,
\]

\[
\Sigma_g(X_a,X_a)
=
\operatorname{Cov}_{q(g)}[g(X_a)]
\in\mathbb R^{k\times k}.
\]

For a single point \(x_i\),

\[
\mu_{g,i} = \mu_g(x_i),
\qquad
v_{g,i} = \Sigma_g(x_i,x_i).
\]

The key modeling choice is that \(\Sigma_g\) is not only used at prediction time. It is also included inside the local self-CEM likelihood.

---

## 5. Local residual likelihood after integrating out the global GP

For a fixed anchor \(a\), define the global-residualized response vector

\[
r_a = y_a - \mu_g(X_a).
\]

This is not treated as noise-free. Since \(g(X_a)\) is uncertain under \(q(g)\), the residual covariance must include \(\Sigma_g(X_a,X_a)\).

For an inlier set \(I_a\), write

\[
X_{a,I}=X_a[I_a],
\qquad
y_{a,I}=y_a[I_a],
\qquad
r_{a,I}=r_a[I_a].
\]

Let

\[
\mu_{g,a,I} = \mu_g(X_{a,I}),
\]

\[
\Sigma_{g,a,I}
=
\Sigma_g(X_{a,I},X_{a,I}).
\]

Let the local residual GP covariance be

\[
K_{h,a,I}
=
k_{h,a}(X_{a,I},X_{a,I}).
\]

Then the inlier marginal likelihood after integrating over the global GP uncertainty is

\[
y_{a,I}
\mid
\gamma_{a,I}=1
\sim
\mathcal N
\left(
\mu_{g,a,I}+m_a\mathbf 1,
\;
K_{h,a,I}+\Sigma_{g,a,I}+\sigma_a^2 I
\right).
\]

Equivalently, in residualized form,

\[
r_{a,I}
\mid
\gamma_{a,I}=1
\sim
\mathcal N
\left(
m_a\mathbf 1,
\;
K_{h,a,I}+\Sigma_{g,a,I}+\sigma_a^2 I
\right).
\]

Define the effective inlier covariance

\[
C_{a,I}
=
K_{h,a,I}
+
\Sigma_{g,a,I}
+
\sigma_a^2 I.
\]

Then the inlier log marginal likelihood is

\[
\log p_{\mathrm{in}}(y_{a,I})
=
-\frac12
(r_{a,I}-m_a\mathbf 1)^\top
C_{a,I}^{-1}
(r_{a,I}-m_a\mathbf 1)
-\frac12\log |C_{a,I}|
-\frac{|I_a|}{2}\log(2\pi).
\]

This is the central local likelihood used by self-CEM.

---

## 6. Diagonal and block global-variance variants

There are two practical ways to inject global GP uncertainty into the local likelihood.

### 6.1 Diagonal version

Use only marginal global GP variances:

\[
\Sigma_{g,a,I}
\approx
\operatorname{diag}
\left(
v_g(x_i): i\in I_a
\right).
\]

Then

\[
C_{a,I}
=
K_{h,a,I}
+
\operatorname{diag}(v_g(X_{a,I}))
+
\sigma_a^2 I.
\]

Equivalently, the local observation noise becomes heteroskedastic:

\[
\sigma_{a,i,\mathrm{eff}}^2
=
\sigma_a^2+v_g(x_i).
\]

This version is easiest to implement.

### 6.2 Block covariance version

Use the full posterior covariance block:

\[
\Sigma_{g,a,I}
=
\Sigma_g(X_{a,I},X_{a,I}).
\]

Then

\[
C_{a,I}
=
K_{h,a,I}
+
\Sigma_g(X_{a,I},X_{a,I})
+
\sigma_a^2 I.
\]

This is theoretically cleaner because uncertainty in the global trend is correlated across nearby points.

---

## 7. Local background likelihood with global uncertainty

For an outlier point \(i\in N_a\), define

\[
r_{ai}=y_i-\mu_g(x_i).
\]

The background likelihood is

\[
y_i\mid \gamma_{ai}=0
\sim
\mathcal N
\left(
\mu_g(x_i)+b_a,
\;
\tau_a^2+v_g(x_i)
\right).
\]

Equivalently,

\[
r_{ai}\mid \gamma_{ai}=0
\sim
\mathcal N
\left(
b_a,
\;
\tau_a^2+v_g(x_i)
\right).
\]

Thus

\[
\log p_{\mathrm{out}}(y_i)
=
-\frac12
\frac{(r_{ai}-b_a)^2}{\tau_a^2+v_g(x_i)}
-\frac12\log(\tau_a^2+v_g(x_i))
-\frac12\log(2\pi).
\]

This means global GP uncertainty affects both the inlier and outlier likelihoods.

---

## 8. Self-CEM responsibility update

Let the local gate probability be

\[
\pi_{ai}
=
p(\gamma_{ai}=1\mid x_i,x_a).
\]

For example, using a logistic gate,

\[
\pi_{ai}
=
\sigma(s_{ai}),
\]

where

\[
s_{ai}
=
\alpha_a
+
\beta_a^\top \phi_a(x_i).
\]

Here \(\phi_a(x_i)\) may include local projected coordinates, raw coordinates, distance-to-anchor features, or uncertain-W features.

The posterior responsibility is

\[
q(\gamma_{ai}=1)
=
\rho_{ai}.
\]

For a soft self-CEM update,

\[
\rho_{ai}
=
\frac{
\pi_{ai}
p_{\mathrm{in},ai}
}{
\pi_{ai}p_{\mathrm{in},ai}
+
(1-\pi_{ai})p_{\mathrm{out},ai}
}.
\]

Equivalently,

\[
\operatorname{logit}(\rho_{ai})
=
\log\frac{\pi_{ai}}{1-\pi_{ai}}
+
\log p_{\mathrm{in},ai}
-
\log p_{\mathrm{out},ai}.
\]

For hard CEM,

\[
\gamma_{ai}^{\mathrm{new}}
=
\mathbf 1
\left[
\operatorname{logit}(\rho_{ai})>0
\right].
\]

The subtlety is \(p_{\mathrm{in},ai}\). A full GP inlier likelihood is joint over the inlier set, not pointwise. A practical CEM implementation uses either:

1. joint inlier set likelihood for the current selected set, or
2. leave-one-out conditional likelihoods.

For leave-one-out updates, define

\[
I_{-i}=I_a\setminus\{i\}.
\]

Let the effective covariance over \(I_a\) be

\[
C_{a,I_a}
=
K_{h,a,I_a}
+
\Sigma_{g,a,I_a}
+
\sigma_a^2 I.
\]

Partition this covariance as

\[
C_{a,I_a}
=
\begin{bmatrix}
C_{-i,-i} & c_{-i,i}\\
c_{i,-i} & c_{ii}
\end{bmatrix}.
\]

Then the conditional inlier predictive distribution for point \(i\) given other inliers is

\[
r_i
\mid
r_{I_{-i}},\gamma_i=1
\sim
\mathcal N
\left(
\mu_{i\mid -i},
s^2_{i\mid -i}
\right),
\]

where

\[
\mu_{i\mid -i}
=
m_a
+
c_{i,-i}
C_{-i,-i}^{-1}
(r_{I_{-i}}-m_a\mathbf 1),
\]

\[
s^2_{i\mid -i}
=
c_{ii}
-
c_{i,-i}
C_{-i,-i}^{-1}
c_{-i,i}.
\]

Then

\[
p_{\mathrm{in},ai}
=
\mathcal N
\left(
r_i;
\mu_{i\mid -i},
s^2_{i\mid -i}
\right).
\]

The outlier likelihood remains

\[
p_{\mathrm{out},ai}
=
\mathcal N
\left(
r_i;
b_a,
\tau_a^2+v_g(x_i)
\right).
\]

This gives a responsibility update that propagates global GP uncertainty into the local CEM label update.

---

## 9. Local self-CEM M-step

Given responsibilities \(\rho_{ai}\) or hard labels \(\gamma_{ai}\), update the local parameters

\[
\theta_a
=
\{m_a,\sigma_a^2,\theta_{h,a},\alpha_a,\beta_a,b_a,\tau_a^2\}.
\]

The local objective is

\[
\mathcal L_a(\theta_a)
=
\mathcal L_{a,\mathrm{in}}
+
\mathcal L_{a,\mathrm{out}}
+
\mathcal L_{a,\mathrm{gate}}
-
\mathcal R_a.
\]

### 9.1 Inlier term

For hard labels,

\[
\mathcal L_{a,\mathrm{in}}
=
\log
\mathcal N
\left(
r_{a,I_a};
m_a\mathbf 1,
K_{h,a,I_a}
+
\Sigma_{g,a,I_a}
+
\sigma_a^2 I
\right).
\]

For soft labels, one simple approximation is to use weights

\[
w_{ai}=\rho_{ai}
\]

and define a diagonal weighting matrix

\[
W_a=\operatorname{diag}(w_{ai}:i\in N_a).
\]

A weighted GP likelihood can be approximated by inflating noise:

\[
\sigma_{a,i,\mathrm{soft}}^2
=
\frac{\sigma_a^2+v_g(x_i)}{\max(w_{ai},\epsilon_w)}.
\]

Then

\[
C_{a,\mathrm{soft}}
=
K_{h,a}(X_a,X_a)
+
\operatorname{diag}
\left(
\sigma_{a,i,\mathrm{soft}}^2
\right).
\]

The soft inlier objective is

\[
\mathcal L_{a,\mathrm{in}}^{\mathrm{soft}}
=
\log
\mathcal N
\left(
r_a;
m_a\mathbf 1,
C_{a,\mathrm{soft}}
\right).
\]

The block-covariance version replaces the diagonal \(v_g\) by the block \(\Sigma_g(X_a,X_a)\), while still applying responsibility weighting to the local noise term.

### 9.2 Outlier term

For hard labels,

\[
\mathcal L_{a,\mathrm{out}}
=
\sum_{i\in N_a\setminus I_a}
\log
\mathcal N
\left(
r_{ai};
b_a,
\tau_a^2+v_g(x_i)
\right).
\]

For soft labels,

\[
\mathcal L_{a,\mathrm{out}}
=
\sum_{i\in N_a}
(1-\rho_{ai})
\log
\mathcal N
\left(
r_{ai};
b_a,
\tau_a^2+v_g(x_i)
\right).
\]

### 9.3 Gate term

The gate likelihood is

\[
\mathcal L_{a,\mathrm{gate}}
=
\sum_{i\in N_a}
\left[
\rho_{ai}\log\pi_{ai}
+
(1-\rho_{ai})\log(1-\pi_{ai})
\right].
\]

For hard labels, replace \(\rho_{ai}\) by \(\gamma_{ai}\).

### 9.4 Regularization

A standard regularizer is

\[
\mathcal R_a
=
\lambda_{\beta}\|\beta_a\|_2^2
+
\lambda_m m_a^2
+
\lambda_{\theta}\|\theta_{h,a}\|_2^2.
\]

If using uncertain-W local kernels, include the KL or qR regularizer separately.

---

## 10. Local residual kernel with uncertain W

If the local kernel depends on a projection matrix \(W_a\), the deterministic version is

\[
k_{h,a}(x_i,x_j)
=
s_a^2
\exp
\left(
-\frac12
\|W_a(x_i-x_j)\|_{\Lambda_a^{-1}}^2
\right).
\]

If \(W_a\) is uncertain under \(q(W_a)\), use the moment-matched kernel

\[
\bar k_{h,a}(x_i,x_j)
=
\mathbb E_{q(W_a)}
\left[
s_a^2
\exp
\left(
-\frac12
\|W_a(x_i-x_j)\|_{\Lambda_a^{-1}}^2
\right)
\right].
\]

Let

\[
\delta_{ij}=x_i-x_j.
\]

If

\[
W_a\delta_{ij}
\sim
\mathcal N(\mu_{ij},S_{ij}),
\]

then

\[
\bar k_{h,a}(x_i,x_j)
=
s_a^2
\left|
I+\Lambda_a^{-1}S_{ij}
\right|^{-1/2}
\exp
\left(
-\frac12
\mu_{ij}^\top
(\Lambda_a+S_{ij})^{-1}
\mu_{ij}
\right).
\]

Then the effective inlier covariance is

\[
C_{a,I}
=
\bar K_{h,a,I}
+
\Sigma_{g,a,I}
+
\sigma_a^2 I.
\]

This makes global uncertainty and projection uncertainty enter the same local GP covariance, but from different sources.

---

## 11. Global GP update given local self-CEM states

To integrate the global GP into the optimizer, update \(q(g)\) using pseudo-observations produced by the local residual model.

Suppose each training point \(i\) is covered by a set of anchors

\[
A(i)=\{a:i\in N_a\}.
\]

For each pair \((a,i)\), local self-CEM gives a posterior residual mean

\[
\mu_{h,ai}
=
\mathbb E[h_a(x_i)\mid \text{local state }a],
\]

and residual variance

\[
v_{h,ai}
=
\operatorname{Var}[h_a(x_i)\mid \text{local state }a].
\]

It also gives inlier responsibility

\[
\rho_{ai}.
\]

A local prediction for the residual contribution at \(x_i\) is

\[
\hat h_{ai}
=
\rho_{ai}\mu_{h,ai}
+
(1-\rho_{ai})b_a.
\]

Its uncertainty can be approximated by mixture variance:

\[
\hat v_{h,ai}
=
\rho_{ai}
\left[
v_{h,ai}
+
(\mu_{h,ai}-\hat h_{ai})^2
\right]
+
(1-\rho_{ai})
\left[
\tau_a^2
+
(b_a-\hat h_{ai})^2
\right].
\]

Aggregate across anchors using weights \(\omega_{ai}\), for example normalized inverse-distance weights:

\[
\omega_{ai}\ge 0,
\qquad
\sum_{a\in A(i)}\omega_{ai}=1.
\]

Then

\[
\hat h_i
=
\sum_{a\in A(i)}
\omega_{ai}\hat h_{ai}.
\]

Assuming anchor-local residual estimates are conditionally independent, a simple variance aggregation is

\[
\hat v_{h,i}
=
\sum_{a\in A(i)}
\omega_{ai}^2 \hat v_{h,ai}.
\]

Define the pseudo-target for the global GP as

\[
\tilde y_i
=
y_i-\hat h_i.
\]

The pseudo-observation noise is

\[
\tilde\sigma_i^2
=
\sigma_\epsilon^2+\hat v_{h,i}.
\]

Then the global GP update uses

\[
\tilde y_i
\sim
\mathcal N(g(x_i),\tilde\sigma_i^2).
\]

The global GP objective is

\[
\mathcal L_g
=
\sum_{i=1}^N
\mathbb E_{q(g_i)}
\left[
\log
\mathcal N
\left(
\tilde y_i;
g_i,
\tilde\sigma_i^2
\right)
\right]
-
\mathrm{KL}(q(g)\Vert p(g)).
\]

For a Gaussian posterior marginal

\[
q(g_i)=\mathcal N(\mu_{g,i},v_{g,i}),
\]

the expected log likelihood is

\[
\mathbb E_{q(g_i)}
\left[
\log
\mathcal N
\left(
\tilde y_i;
g_i,
\tilde\sigma_i^2
\right)
\right]
=
-\frac12
\frac{
(\tilde y_i-\mu_{g,i})^2+v_{g,i}
}{
\tilde\sigma_i^2
}
-\frac12\log \tilde\sigma_i^2
-\frac12\log(2\pi).
\]

Thus

\[
\mathcal L_g
=
-\frac12
\sum_{i=1}^N
\left[
\frac{
(\tilde y_i-\mu_{g,i})^2+v_{g,i}
}{
\tilde\sigma_i^2
}
+
\log \tilde\sigma_i^2
+
\log(2\pi)
\right]
-
\mathrm{KL}(q(g)\Vert p(g)).
\]

This lets the global GP fit be updated inside the same probabilistic framework.

---

## 12. Joint approximate ELBO

The full additive model objective can be written as

\[
\mathcal L
=
\sum_a
\mathbb E_{q(g)q(h_a)q(\gamma_a)}
\left[
\log p(y_{N_a}\mid g_{N_a},h_a,\gamma_a)
\right]
-
\mathrm{KL}(q(g)\Vert p(g))
-
\sum_a
\mathrm{KL}(q(h_a)\Vert p(h_a)).
\]

The local likelihood factor is

\[
p(y_{N_a}\mid g_{N_a},h_a,\gamma_a)
=
\prod_{i\in N_a}
\left[
p_{\mathrm{in}}(y_i\mid g_i,h_{ai})^{\gamma_{ai}}
p_{\mathrm{out}}(y_i\mid g_i)^{1-\gamma_{ai}}
\right].
\]

Because local GP inlier likelihoods are correlated across inliers, the inlier part is more accurately represented as

\[
p(y_{a,I_a}\mid g_{a,I_a},\gamma_{a,I_a}=1)
=
\mathcal N
\left(
y_{a,I_a};
g_{a,I_a}+m_a\mathbf 1,
K_{h,a,I_a}+\sigma_a^2I
\right).
\]

Taking expectation over \(q(g)\) yields

\[
\mathbb E_{q(g)}
\left[
\log
\mathcal N
\left(
y_{a,I_a};
g_{a,I_a}+m_a\mathbf 1,
K_{h,a,I_a}+\sigma_a^2I
\right)
\right].
\]

A tractable approximation is obtained by replacing this expected log likelihood with the marginal likelihood after integrating \(g\):

\[
\log
\mathcal N
\left(
y_{a,I_a};
\mu_g(X_{a,I_a})+m_a\mathbf 1,
K_{h,a,I_a}
+
\Sigma_g(X_{a,I_a},X_{a,I_a})
+
\sigma_a^2I
\right).
\]

This is exactly the local likelihood used above.

---

## 13. Coordinate ascent training algorithm

### Step 0: Initialization

Fit an initial global GP \(q^{(0)}(g)\) using either:

1. a smooth sparse GP on \((X,y)\), or
2. an OOB / K-fold GP to avoid overfitting the training residuals.

For each local anchor \(a\), initialize:

\[
\rho_{ai}^{(0)}
\]

using existing CEM labels or raw self-CEM labels.

Initialize local parameters:

\[
\theta_a^{(0)}
=
\{m_a,\sigma_a^2,\theta_{h,a},\alpha_a,\beta_a,b_a,\tau_a^2\}.
\]

### Step 1: Local self-CEM update given global GP

At iteration \(t\), compute

\[
\mu_g^{(t)}(X_a),
\qquad
\Sigma_g^{(t)}(X_a,X_a).
\]

For each anchor \(a\), compute residuals

\[
r_a^{(t)}
=
y_a-\mu_g^{(t)}(X_a).
\]

Update responsibilities using

\[
\operatorname{logit}(\rho_{ai}^{(t+1)})
=
\log\frac{\pi_{ai}^{(t)}}{1-\pi_{ai}^{(t)}}
+
\log p_{\mathrm{in},ai}^{(t)}
-
\log p_{\mathrm{out},ai}^{(t)},
\]

where both \(p_{\mathrm{in},ai}^{(t)}\) and \(p_{\mathrm{out},ai}^{(t)}\) include global GP variance.

Then update local parameters by maximizing

\[
\theta_a^{(t+1)}
=
\arg\max_{\theta_a}
\mathcal L_a(\theta_a;q^{(t)}(g),\rho_a^{(t+1)}).
\]

### Step 2: Global GP update given local self-CEM states

For each training point \(i\), compute the local residual estimate

\[
\hat h_i^{(t+1)}
\]

and local residual uncertainty

\[
\hat v_{h,i}^{(t+1)}.
\]

Construct

\[
\tilde y_i^{(t+1)}
=
y_i-\hat h_i^{(t+1)}.
\]

Set

\[
\tilde\sigma_i^{2,(t+1)}
=
\sigma_\epsilon^2+\hat v_{h,i}^{(t+1)}.
\]

Update the global GP posterior by maximizing

\[
q^{(t+1)}(g)
=
\arg\max_{q(g)}
\left[
\sum_i
\mathbb E_{q(g_i)}
\log
\mathcal N
\left(
\tilde y_i^{(t+1)};
g_i,
\tilde\sigma_i^{2,(t+1)}
\right)
-
\mathrm{KL}(q(g)\Vert p(g))
\right].
\]

### Step 3: Repeat

Repeat Step 1 and Step 2 for \(T\) outer cycles.

---

## 14. Test-time prediction

For a test point \(x_*\), compute the global GP posterior

\[
q(g_*)=
\mathcal N(\mu_{g,*},v_{g,*}).
\]

Choose a local neighborhood \(N_*\) using one of:

1. raw-X KNN,
2. projected-Z KNN,
3. raw candidate pool plus W/Z reranking,
4. validation-gated raw vs projected neighborhood.

Run local self-CEM on \(N_*\) using residualized training responses

\[
r_{*,N}
=
y_{N_*}
-
\mu_g(X_{N_*}),
\]

and include

\[
\Sigma_g(X_{N_*},X_{N_*})
\]

or its diagonal inside the local covariance.

The local residual predictive distribution is

\[
q(h_*\mid x_*,N_*)
=
\mathcal N(\mu_{h,*},v_{h,*}).
\]

The additive predictive distribution is

\[
q(y_*\mid x_*,\mathcal D)
=
\mathcal N
\left(
\mu_{g,*}+\mu_{h,*},
\;
v_{g,*}+v_{h,*}+2c_{gh,*}+\sigma_\epsilon^2
\right).
\]

Under the mean-field approximation,

\[
c_{gh,*}=0,
\]

so

\[
\mu_{y,*}
=
\mu_{g,*}+\mu_{h,*},
\]

\[
v_{y,*}
=
v_{g,*}+v_{h,*}+\sigma_\epsilon^2.
\]

If \(\sigma_\epsilon^2\) is already included in \(v_{h,*}\), then use

\[
v_{y,*}
=
v_{g,*}+v_{h,*}.
\]

The implementation must avoid counting \(\sigma_\epsilon^2\) twice.

---

## 15. Sampled-W ensemble extension

If local self-CEM uses uncertain \(W\), one can produce a stronger mean path by sampling deterministic projections.

Draw

\[
W_*^{(s)}\sim q(W_*),
\qquad
s=1,\dots,S.
\]

For each sample \(s\):

1. Set \(\operatorname{Var}(W_*^{(s)})=0\).
2. Select or rerank neighbors using \(W_*^{(s)}\).
3. Run deterministic-W residual self-CEM.
4. Obtain

\[
\mu_{h,*}^{(s)},
\qquad
v_{h,*}^{(s)}.
\]

The sample-specific full predictive mean is

\[
\mu_{y,*}^{(s)}
=
\mu_{g,*}
+
\mu_{h,*}^{(s)}.
\]

The sample-specific full predictive variance is

\[
v_{y,*}^{(s)}
=
v_{g,*}
+
v_{h,*}^{(s)}.
\]

Moment-match the mixture:

\[
\bar\mu_{y,*}
=
\frac1S
\sum_{s=1}^S
\mu_{y,*}^{(s)}.
\]

\[
\bar v_{y,*}
=
\frac1S
\sum_{s=1}^S
\left[
v_{y,*}^{(s)}
+
\left(\mu_{y,*}^{(s)}\right)^2
\right]
-
\bar\mu_{y,*}^2.
\]

Equivalently,

\[
\bar v_{y,*}
=
\underbrace{
\frac1S\sum_{s=1}^S v_{y,*}^{(s)}
}_{\text{within-sample uncertainty}}
+
\underbrace{
\frac1S\sum_{s=1}^S
\left(
\mu_{y,*}^{(s)}-\bar\mu_{y,*}
\right)^2
}_{\text{between-sample uncertainty}}.
\]

This extension allows sampled \(W\) to affect the posterior mean through neighborhood selection, CEM labels, and local refitting.

---

## 16. Avoiding double counting

A naive two-stage procedure can double count uncertainty:

1. Fit a full GP to \(y\).
2. Compute residuals.
3. Fit self-CEM to residuals.
4. Add GP variance and self-CEM variance.

This is problematic because the first GP may already explain local jumps as noise or posterior variance.

The additive model avoids this in three ways.

### 16.1 Interpret global GP as a trend component

The global GP is not the full predictor for \(y\). It is the latent smooth component \(g\).

The local component \(h_a\) explains residual structure.

Thus

\[
\operatorname{Var}(y_*)
=
\operatorname{Var}(g_*)
+
\operatorname{Var}(h_*)
+
\operatorname{Var}(\epsilon_*),
\]

not because two independent predictors are being added, but because the generative model is additive.

### 16.2 Propagate global uncertainty during local training

The local self-CEM likelihood uses

\[
K_{h,a}
+
\Sigma_g
+
\sigma_a^2I.
\]

Therefore local variance is learned conditional on global uncertainty.

### 16.3 Update global GP with local uncertainty

The global GP update uses

\[
\tilde y_i=y_i-\hat h_i
\]

with noise

\[
\tilde\sigma_i^2
=
\sigma_\epsilon^2+\hat v_{h,i}.
\]

Thus the global GP does not overfit regions where the local residual model is uncertain.

---

## 17. Recommended implementation variants

### Variant A: `globalgp_resid_selfcem_diagvar`

Use:

\[
C_{a,I}
=
K_{h,a,I}
+
\operatorname{diag}(v_g(X_{a,I}))
+
\sigma_a^2I.
\]

Final prediction:

\[
\mu_y=\mu_g+\mu_h,
\qquad
v_y=v_g+v_h.
\]

This is the minimal uncertainty-aware version.

### Variant B: `globalgp_resid_selfcem_blockvar`

Use:

\[
C_{a,I}
=
K_{h,a,I}
+
\Sigma_g(X_{a,I},X_{a,I})
+
\sigma_a^2I.
\]

This is the cleaner local-likelihood version.

### Variant C: `oob_globalgp_resid_selfcem_diagvar`

Compute \(\mu_g(x_i)\) and \(v_g(x_i)\) for training points using K-fold or OOB prediction.

This reduces in-sample residual shrinkage.

### Variant D: `alt_globalgp_selfcem`

Alternate:

\[
q(g)
\rightarrow
\{\rho_a,\theta_a\}_{a=1}^A
\rightarrow
q(g)
\rightarrow
\cdots
\]

This is the most coherent version.

### Variant E: `alt_globalgp_selfcem_ensW`

Add sampled deterministic-W residual self-CEM ensemble at test time.

This is the most likely version to improve RMSE, because sampled \(W\) can change neighborhoods and local CEM states.

---

## 18. Diagnostics to report

For each variant, report:

### Prediction metrics

\[
\mathrm{RMSE},
\qquad
\mathrm{CRPS},
\qquad
\mathrm{Cov90},
\qquad
\mathrm{Width90}.
\]

### Mean decomposition

\[
\mu_y(x_*)
=
\mu_g(x_*)+\mu_h(x_*).
\]

Report:

\[
\operatorname{RMSE}(\mu_g),
\qquad
\operatorname{RMSE}(\mu_h),
\qquad
\operatorname{RMSE}(\mu_g+\mu_h).
\]

### Variance decomposition

\[
v_y(x_*)
=
v_g(x_*)+v_h(x_*).
\]

Report averages:

\[
\mathbb E[v_g],
\qquad
\mathbb E[v_h],
\qquad
\mathbb E[v_y].
\]

For sampled-W ensemble, also report:

\[
v_{\mathrm{within}}
=
\frac1S\sum_s v_y^{(s)},
\]

\[
v_{\mathrm{between}}
=
\frac1S\sum_s
(\mu_y^{(s)}-\bar\mu_y)^2.
\]

### Local CEM diagnostics

Report:

\[
\frac1{|N_a|}
\sum_{i\in N_a}\rho_{ai},
\]

label-change rate,

\[
\frac1{|N_a|}
\sum_{i\in N_a}
\mathbf 1[
\gamma_{ai}^{\mathrm{new}}\neq \gamma_{ai}^{\mathrm{old}}
],
\]

and local noise floor hit rate:

\[
\frac1A
\sum_a
\mathbf 1[
\sigma_a^2 = \sigma_{\mathrm{floor}}^2
].
\]

### Global-local interaction diagnostics

Report:

\[
\frac{
\mathbb E_i[v_g(x_i)]
}{
\mathbb E_i[v_h(x_i)]
},
\]

and the residual variance reduction:

\[
\operatorname{Var}(y)
-
\operatorname{Var}(y-\mu_g).
\]

Also report whether the final model is improving RMSE because of:

1. global mean correction,
2. local residual self-CEM changes,
3. sampled-W neighborhood/refit changes,
4. variance-only calibration.

---

## 19. Expected behavior

If the method works, it should satisfy:

\[
\mathrm{RMSE}(\mu_g+\mu_h)
<
\mathrm{RMSE}(\mu_h^{\mathrm{raw\ selfCEM}}),
\]

while maintaining

\[
\mathrm{Cov90}\approx 0.75\text{--}0.85
\]

and avoiding excessive width inflation:

\[
\mathrm{Width90}_{\mathrm{additive}}
\not\gg
\mathrm{Width90}_{\mathrm{raw\ selfCEM}}.
\]

A bad failure mode is:

\[
\mathrm{RMSE}\downarrow
\quad
\text{but}
\quad
\mathrm{Cov90}\downarrow
\]

which means the global GP behaved like a deterministic plug-in correction.

Another bad failure mode is:

\[
\mathrm{Cov90}\uparrow
\quad
\text{but}
\quad
\mathrm{Width90}\uparrow\uparrow
\]

which means global and local uncertainty were double counted.

The desired regime is:

\[
\mathrm{RMSE}\downarrow,
\qquad
\mathrm{CRPS}\downarrow,
\qquad
\mathrm{Cov90}\text{ stable or improved},
\qquad
\mathrm{Width90}\text{ moderate}.
\]

---

## 20. Summary

The proposed method is not:

\[
\text{fit GP}
\rightarrow
\text{fit residual self-CEM}
\rightarrow
\text{add predictions}.
\]

Instead, it is an additive latent model:

\[
y(x)=g(x)+h_a(x)+\epsilon.
\]

The global GP posterior uncertainty enters the local self-CEM likelihood through

\[
K_{h,a}
+
\Sigma_g
+
\sigma_a^2I.
\]

The local self-CEM posterior uncertainty enters the global GP update through heteroskedastic pseudo-observation noise:

\[
\tilde\sigma_i^2
=
\sigma_\epsilon^2+\hat v_{h,i}.
\]

The final predictive distribution is

\[
\mu_y
=
\mu_g+\mu_h,
\]

\[
v_y
=
v_g+v_h
\]

under a mean-field global-local approximation.

This gives a clean probabilistic path for improving RMSE through a global smooth mean component while preserving self-CEM's uncertainty-calibration advantage.