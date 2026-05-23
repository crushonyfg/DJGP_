# DMGP + Local Residual Self-CEM/JGP: A Global-Local Additive Metric GP

## 0. Purpose

This document describes a probabilistic method that combines:

1. a global learned-metric GP, similar in spirit to Deep Mahalanobis GP (DMGP), for sample-efficient smooth trend learning;
2. a local residual JumpGP / self-CEM component for boundary sharpening and local discontinuities;
3. uncertainty propagation from the global GP into local self-CEM training;
4. optional feedback from local residual uncertainty back into the global GP update.

The target use case is a setting where a global model is data-efficient but too smooth near jumps, while a local JumpGP/self-CEM model can sharpen boundaries but has limited sample efficiency because each anchor only sees a local neighborhood.

The core model is

$$
y(x)=g(z_\theta(x))+h_a(z_\theta(x))+\epsilon,
$$

where $z_\theta(x)=W_\theta(x)x$ is a learned global metric representation, $g$ is a global smooth GP, and $h_a$ is an anchor-specific local residual JumpGP/self-CEM function.

The intended interpretation is not “fit a GP, then plug in a residual correction.” Instead, it is an additive global-local latent function model with a normalized composite variational objective.

---

## 1. High-level motivation

A standard global GP or DMGP has access to all training data and can learn a sample-efficient smooth trend. However, if the target function has jumps or regime boundaries, a smooth global GP tends to average across the boundary. Near a discontinuity, the global posterior mean can be biased:

$$
\mu_g(x)\approx \text{smooth interpolation across both sides of the boundary}.
$$

A local JumpGP/self-CEM model can separate local inliers from outliers and produce a sharper boundary-aware prediction. However, it is fitted from only a local neighborhood, so it can have poor sample efficiency and unstable mean estimates.

The proposed model combines both:

$$
\text{global metric GP} \quad + \quad \text{local residual JumpGP/self-CEM}.
$$

The global model captures broad trend using all data. The local model captures residual jump structure that remains after the global trend.

This is a GP/JGP analogue of residual learning. The difference from an ordinary deterministic residual correction is that the residual is not treated as noise-free. The global GP posterior uncertainty is propagated into the local self-CEM likelihood.

---

## 2. Data and notation

Training data:

$$
\mathcal D=\{(x_i,y_i)\}_{i=1}^N,
\qquad
x_i\in\mathbb R^D,
\qquad
y_i\in\mathbb R.
$$

The learned metric representation is

$$
z_i=z_\theta(x_i)=W_\theta(x_i)x_i\in\mathbb R^Q.
$$

For a test point,

$$
z_*=z_\theta(x_*).
$$

Let $\mathcal A$ be the set of local anchors. For anchor $a$, define a local neighborhood

$$
N_a=\{i_1,\dots,i_k\},
\qquad
|N_a|=k.
$$

The corresponding local representation and response arrays are

$$
Z_a=
\begin{bmatrix}
z_{i_1}^\top\\
\vdots\\
z_{i_k}^\top
\end{bmatrix}
\in\mathbb R^{k\times Q},
\qquad
y_a=
\begin{bmatrix}
y_{i_1}\\
\vdots\\
y_{i_k}
\end{bmatrix}
\in\mathbb R^k.
$$

For each local point $i\in N_a$, introduce a JumpGP/self-CEM indicator

$$
\gamma_{ai}\in\{0,1\}.
$$

The interpretation is:

$$
\gamma_{ai}=1
\quad \Longleftrightarrow \quad
i \text{ is an inlier for local residual GP } h_a,
$$

$$
\gamma_{ai}=0
\quad \Longleftrightarrow \quad
i \text{ belongs to the local background/outlier component}.
$$

Define

$$
I_a=\{i\in N_a:\gamma_{ai}=1\},
$$

$$
O_a=N_a\setminus I_a.
$$

---

## 3. Shared metric representation

The model uses a shared learned representation

$$
z_\theta(x)=W_\theta(x)x.
$$

The matrix-valued function

$$
W_\theta(x)\in\mathbb R^{Q\times D}
$$

may be parameterized by a neural network, sparse GP, low-rank residual model, or another structured map.

A typical low-rank residual parameterization is

$$
W_\theta(x)=W_0+C_\theta(x)V^\top,
$$

where

$$
W_0\in\mathbb R^{Q\times D},
\qquad
C_\theta(x)\in\mathbb R^{Q\times r},
\qquad
V\in\mathbb R^{D\times r},
\qquad
r\ll D.
$$

The representation $z_\theta$ is shared by the global GP and the local residual self-CEM/JGP head.

A plain DMGP-style model would use

$$
y_i=g(z_\theta(x_i))+\epsilon_i.
$$

The proposed model instead uses

$$
y_i=g(z_\theta(x_i))+h_a(z_\theta(x_i))+\epsilon_i,
$$

so it can be viewed as DMGP plus a local residual JumpGP/self-CEM layer.

---

## 4. Generative model

### 4.1 Global trend GP

The global trend is

$$
g\sim\mathcal{GP}(0,k_g).
$$

The global kernel is defined in learned metric space:

$$
k_g(x,x')=k_g(z_\theta(x),z_\theta(x')).
$$

For example, using an RBF kernel,

$$
k_g(z,z')=
s_g^2
\exp\left(
-\frac12
(z-z')^\top\Lambda_g^{-1}(z-z')
\right),
$$

where

$$
\Lambda_g=\operatorname{diag}(\ell_{g,1}^2,\dots,\ell_{g,Q}^2).
$$

The global GP should be constrained to represent smooth trend rather than sharp boundary behavior. Common constraints are:

$$
\ell_g \text{ large},
\qquad
M_g \text{ moderate/small inducing count},
\qquad
\sigma_g^2 \text{ not too small}.
$$

### 4.2 Local residual JumpGP/self-CEM

For each anchor $a$, define a local residual function

$$
h_a\sim\mathcal{GP}(m_a,k_{h,a}).
$$

The local residual kernel also acts in metric space:

$$
k_{h,a}(x_i,x_j)=k_{h,a}(z_i,z_j).
$$

For example,

$$
k_{h,a}(z_i,z_j)=
s_{h,a}^2
\exp\left(
-\frac12
(z_i-z_j)^\top\Lambda_{h,a}^{-1}(z_i-z_j)
\right).
$$

The local residual component should be more flexible and more local than the global GP. A useful prior separation is

$$
\ell_g \gg \ell_{h,a}.
$$

For example, one may enforce

$$
\ell_g \ge c\,\ell_{h,a},
\qquad
c>1.
$$

### 4.3 Inlier observation model

For $i\in N_a$ with $\gamma_{ai}=1$,

$$
y_i=g(z_i)+h_a(z_i)+\epsilon_{ai},
$$

where

$$
\epsilon_{ai}\sim\mathcal N(0,\sigma_a^2).
$$

Thus

$$
y_i\mid g,h_a,\gamma_{ai}=1
\sim
\mathcal N(g(z_i)+h_a(z_i),\sigma_a^2).
$$

### 4.4 Outlier/background observation model

For $i\in N_a$ with $\gamma_{ai}=0$, use a local background residual around the global trend:

$$
y_i=g(z_i)+b_a+\eta_{ai},
$$

where

$$
\eta_{ai}\sim\mathcal N(0,\tau_a^2).
$$

Thus

$$
y_i\mid g,\gamma_{ai}=0
\sim
\mathcal N(g(z_i)+b_a,\tau_a^2).
$$

The background variance $\tau_a^2$ should generally be broad enough to avoid forcing all points into the inlier GP.

### 4.5 Gate model

The inlier probability is

$$
\pi_{ai}=p(\gamma_{ai}=1\mid z_i,z_a).
$$

A logistic gate is

$$
\pi_{ai}=\sigma(s_{ai}),
$$

$$
s_{ai}=\alpha_a+\beta_a^\top\phi(z_i,z_a).
$$

Possible gate features include:

$$
\phi(z_i,z_a)=
\begin{bmatrix}
1\\
z_i-z_a\\
\|z_i-z_a\|^2\\
\text{raw distance features}\\
\text{response-initialization features}
\end{bmatrix}.
$$

The gate should be regularized because it is local and can overfit.

---

## 5. Global GP posterior approximation

Use a sparse variational GP for the global trend.

Let inducing locations in metric space be

$$
U_g=\{u_m\}_{m=1}^{M_g}.
$$

Let

$$
u_g=g(U_g).
$$

Prior:

$$
p(u_g)=\mathcal N(0,K_{UU}).
$$

Variational posterior:

$$
q(u_g)=\mathcal N(m_u,S_u).
$$

For any set of metric inputs $Z$, the posterior marginal is

$$
q(g(Z))=\mathcal N(\mu_g(Z),\Sigma_g(Z,Z)).
$$

The standard sparse GP expressions are:

$$
\mu_g(Z)=K_{ZU}K_{UU}^{-1}m_u,
$$

$$
\Sigma_g(Z,Z)=
K_{ZZ}
+
K_{ZU}K_{UU}^{-1}(S_u-K_{UU})K_{UU}^{-1}K_{UZ}.
$$

For a single point $z_i$,

$$
\mu_{g,i}=\mu_g(z_i),
\qquad
v_{g,i}=\Sigma_g(z_i,z_i).
$$

---

## 6. Local likelihood after integrating global GP uncertainty

A key part of the method is that the local self-CEM likelihood does not treat the global trend estimate as deterministic.

For anchor $a$, define

$$
\mu_{g,a}=\mu_g(Z_a),
\qquad
\Sigma_{g,a}=\Sigma_g(Z_a,Z_a).
$$

The residualized response is

$$
r_a=y_a-\mu_{g,a}.
$$

This residual is still uncertain because $g$ is uncertain. Therefore, when fitting the local residual GP, $\Sigma_{g,a}$ must be included in the local covariance.

For an inlier set $I_a$, define

$$
Z_{a,I}=Z_a[I_a],
\qquad
y_{a,I}=y_a[I_a],
\qquad
r_{a,I}=r_a[I_a].
$$

The global posterior block is

$$
\Sigma_{g,a,I}=\Sigma_g(Z_{a,I},Z_{a,I}).
$$

The local residual covariance is

$$
K_{h,a,I}=k_{h,a}(Z_{a,I},Z_{a,I}).
$$

The inlier likelihood after integrating over $q(g)$ is

$$
y_{a,I}
\mid
\gamma_{a,I}=1
\sim
\mathcal N
\left(
\mu_g(Z_{a,I})+m_a\mathbf 1,
\;
K_{h,a,I}
+
\Sigma_g(Z_{a,I},Z_{a,I})
+
\sigma_a^2I
\right).
$$

Equivalently,

$$
r_{a,I}
\mid
\gamma_{a,I}=1
\sim
\mathcal N
\left(
m_a\mathbf 1,
\;
K_{h,a,I}
+
\Sigma_{g,a,I}
+
\sigma_a^2I
\right).
$$

Define the effective covariance

$$
C_{a,I}=K_{h,a,I}+\Sigma_{g,a,I}+\sigma_a^2I.
$$

Then

$$
\log p_{\mathrm{in}}(y_{a,I})
=
-\frac12
(r_{a,I}-m_a\mathbf 1)^\top C_{a,I}^{-1}(r_{a,I}-m_a\mathbf 1)
-\frac12\log|C_{a,I}|
-\frac{|I_a|}{2}\log(2\pi).
$$

This is the central equation of the method.

---

## 7. Diagonal and block global-variance variants

### 7.1 Diagonal version

Use only marginal global variances:

$$
\Sigma_{g,a,I}
\approx
\operatorname{diag}
\left(
v_g(z_i): i\in I_a
\right).
$$

Then

$$
C_{a,I}=K_{h,a,I}+\operatorname{diag}(v_g(Z_{a,I}))+\sigma_a^2I.
$$

This is equivalent to a heteroskedastic local noise term:

$$
\sigma_{a,i,\mathrm{eff}}^2=\sigma_a^2+v_g(z_i).
$$

This variant is simplest and should be implemented first.

### 7.2 Block version

Use the full local global-posterior block:

$$
C_{a,I}=K_{h,a,I}+\Sigma_g(Z_{a,I},Z_{a,I})+\sigma_a^2I.
$$

This is theoretically cleaner because global trend uncertainty is correlated across nearby points.

---

## 8. Outlier likelihood with global uncertainty

For $i\in O_a$, define

$$
r_{ai}=y_i-\mu_g(z_i).
$$

After integrating global uncertainty, the outlier likelihood is

$$
y_i\mid \gamma_{ai}=0
\sim
\mathcal N
\left(
\mu_g(z_i)+b_a,
\;
v_g(z_i)+\tau_a^2
\right).
$$

Equivalently,

$$
r_{ai}\mid \gamma_{ai}=0
\sim
\mathcal N
\left(
b_a,
\;
v_g(z_i)+\tau_a^2
\right).
$$

Therefore,

$$
\log p_{\mathrm{out}}(y_i)
=
-\frac12
\frac{(r_{ai}-b_a)^2}{v_g(z_i)+\tau_a^2}
-\frac12\log(v_g(z_i)+\tau_a^2)
-\frac12\log(2\pi).
$$

This makes global GP uncertainty influence both inlier and outlier likelihoods.

---

## 9. Local CEM/self-CEM responsibility update

For soft self-CEM, use responsibilities

$$
\rho_{ai}=q(\gamma_{ai}=1).
$$

The exact GP inlier likelihood is joint over an inlier set, not pointwise. For pointwise responsibility updates, use a leave-one-out conditional inlier likelihood.

Let $I_a$ be the current inlier set and suppose $i\in I_a$. Let

$$
I_{-i}=I_a\setminus\{i\}.
$$

The effective covariance over $I_a$ is

$$
C_{a,I_a}=K_{h,a,I_a}+\Sigma_{g,a,I_a}+\sigma_a^2I.
$$

Partition it as

$$
C_{a,I_a}
=
\begin{bmatrix}
C_{-i,-i} & c_{-i,i}\\
c_{i,-i} & c_{ii}
\end{bmatrix}.
$$

Then

$$
r_i\mid r_{I_{-i}},\gamma_{ai}=1
\sim
\mathcal N(\mu_{i\mid -i},s_{i\mid -i}^2),
$$

where

$$
\mu_{i\mid -i}
=
m_a+c_{i,-i}C_{-i,-i}^{-1}(r_{I_{-i}}-m_a\mathbf 1),
$$

$$
s_{i\mid -i}^2
=
c_{ii}-c_{i,-i}C_{-i,-i}^{-1}c_{-i,i}.
$$

Thus the pointwise inlier score is

$$
p_{\mathrm{in},ai}=\mathcal N(r_i;\mu_{i\mid -i},s_{i\mid -i}^2).
$$

The pointwise outlier score is

$$
p_{\mathrm{out},ai}=\mathcal N(r_i;b_a,v_g(z_i)+\tau_a^2).
$$

The responsibility update is

$$
\rho_{ai}
=
\frac{\pi_{ai}p_{\mathrm{in},ai}}
{\pi_{ai}p_{\mathrm{in},ai}+(1-\pi_{ai})p_{\mathrm{out},ai}}.
$$

Equivalently,

$$
\operatorname{logit}(\rho_{ai})
=
\log\frac{\pi_{ai}}{1-\pi_{ai}}
+
\log p_{\mathrm{in},ai}
-
\log p_{\mathrm{out},ai}.
$$

For hard self-CEM,

$$
\gamma_{ai}=\mathbf 1[\rho_{ai}>1/2].
$$

A practical implementation can use soft responsibilities for early iterations and hard labels later.

---

## 10. Local self-CEM M-step

Given labels or responsibilities, update local parameters

$$
\psi_a=\{m_a,s_{h,a}^2,\Lambda_{h,a},\sigma_a^2,b_a,\tau_a^2\},
$$

and gate parameters

$$
\phi_a=\{\alpha_a,\beta_a\}.
$$

### 10.1 Hard-label local objective

For hard labels,

$$
\mathcal L_{a,\mathrm{in}}
=
\log
\mathcal N
\left(
y_{a,I_a};
\mu_g(Z_{a,I_a})+m_a\mathbf 1,
K_{h,a,I_a}+\Sigma_{g,a,I_a}+\sigma_a^2I
\right).
$$

The outlier term is

$$
\mathcal L_{a,\mathrm{out}}
=
\sum_{i\in O_a}
\log
\mathcal N
\left(
y_i;
\mu_g(z_i)+b_a,
v_g(z_i)+\tau_a^2
\right).
$$

The gate term is

$$
\mathcal L_{a,\mathrm{gate}}
=
\sum_{i\in N_a}
\left[
\gamma_{ai}\log\pi_{ai}
+
(1-\gamma_{ai})\log(1-\pi_{ai})
\right].
$$

The local objective is

$$
\mathcal L_a=
\mathcal L_{a,\mathrm{in}}
+
\mathcal L_{a,\mathrm{out}}
+
\mathcal L_{a,\mathrm{gate}}
-
\Omega_a.
$$

### 10.2 Soft-label approximation

For soft responsibilities, a simple approximation is weighted local likelihood.

Let

$$
w_{ai}=\rho_{ai}.
$$

One practical approximation is noise inflation:

$$
\sigma_{a,i,\mathrm{soft}}^2
=
\frac{\sigma_a^2+v_g(z_i)}{\max(w_{ai},\epsilon_w)}.
$$

Then

$$
C_{a,\mathrm{soft}}=K_{h,a}(Z_a,Z_a)+\operatorname{diag}(\sigma_{a,i,\mathrm{soft}}^2).
$$

The soft inlier objective is

$$
\mathcal L_{a,\mathrm{in}}^{\mathrm{soft}}
=
\log
\mathcal N
\left(
r_a;
m_a\mathbf 1,
C_{a,\mathrm{soft}}
\right).
$$

The soft outlier term is

$$
\mathcal L_{a,\mathrm{out}}^{\mathrm{soft}}
=
\sum_{i\in N_a}
(1-\rho_{ai})
\log
\mathcal N
\left(
r_i;
b_a,
v_g(z_i)+\tau_a^2
\right).
$$

The soft gate term is

$$
\mathcal L_{a,\mathrm{gate}}^{\mathrm{soft}}
=
\sum_{i\in N_a}
\left[
\rho_{ai}\log\pi_{ai}
+
(1-\rho_{ai})\log(1-\pi_{ai})
\right].
$$

---

## 11. Normalized composite variational objective

Because local neighborhoods overlap, the local likelihoods cannot be treated as independent exact likelihood factors without correction. The safe formulation is a normalized composite variational objective.

Let $c_i$ be the number of local neighborhoods containing point $i$:

$$
c_i=\sum_{a\in\mathcal A}\mathbf 1[i\in N_a].
$$

Define an anchor weight

$$
\omega_a
=
\frac1{|N_a|}
\sum_{i\in N_a}
\frac1{\max(c_i,1)}.
$$

If neighborhoods are uniform and coverage is nearly uniform, use the approximation

$$
\omega_a=\frac{N}{|\mathcal A|k}.
$$

The objective is

$$
\boxed{
\mathcal J
=
\sum_{a\in\mathcal A}
\omega_a\mathcal L_a
-
\beta_g\operatorname{KL}(q(u_g)\Vert p(u_g))
-
\Omega_\theta
-
\sum_a\Omega_a.
}
$$

Here

$$
\mathcal L_a=
\mathcal L_{a,\mathrm{in}}
+
\mathcal L_{a,\mathrm{out}}
+
\mathcal L_{a,\mathrm{gate}}.
$$

The representation regularizer is

$$
\Omega_\theta
=
\lambda_{\theta}\|\theta\|^2
+
\lambda_{\mathrm{prox}}\|\theta-\theta_0\|^2
+
\lambda_{\mathrm{smooth}}\mathbb E_x\|\nabla_x z_\theta(x)\|_F^2,
$$

where $\theta_0$ is the pretrained global metric parameter.

The local regularizer can include

$$
\Omega_a
=
\lambda_\beta\|\beta_a\|^2
+
\lambda_m m_a^2
+
\lambda_b b_a^2
+
\lambda_s s_{h,a}^2
+
\Omega_{\mathrm{sep},a}.
$$

A lengthscale separation penalty is

$$
\Omega_{\mathrm{sep},a}
=
\lambda_{\mathrm{sep}}
\left[
\max
\left(
0,
\log\ell_{h,a}-\log\ell_g+\log c
\right)
\right]^2.
$$

This encourages

$$
\ell_g\ge c\ell_{h,a}.
$$

The parameter $\beta_g$ controls the global GP KL strength.

This objective should be described as a normalized composite variational objective, not as an exact full-data ELBO.

---

## 12. Why not use DMGP loss plus self-CEM loss directly?

The tempting objective

$$
\mathcal L_{\mathrm{bad}}
=
\mathcal L_{\mathrm{DMGP}}(y;g,z_\theta)
+
\lambda
\sum_a
\mathcal L_{\mathrm{selfCEM},a}(y;g,h_a,z_\theta)
$$

is not clean because the global GP and local CEM both directly fit the same $y$ as separate predictors. This invites a double-counting criticism.

The proposed objective avoids this by using a single additive local likelihood:

$$
y=g+h_a+\epsilon.
$$

The global GP appears inside the local likelihood through

$$
\mu_g
\quad\text{and}\quad
\Sigma_g,
$$

rather than as a separate full-data predictor that is later corrected.

---

## 13. Updating the global GP using local residual uncertainty

The composite objective above can be optimized jointly, but a stable coordinate version is preferred.

After local self-CEM has been fitted, each point $i$ may be covered by multiple anchors

$$
A(i)=\{a:i\in N_a\}.
$$

Each anchor gives a local residual estimate

$$
\hat h_{ai},
$$

and residual uncertainty

$$
\hat v_{h,ai}.
$$

For anchor $a$, if the inlier responsibility is $\rho_{ai}$, define the local mixture mean

$$
\hat h_{ai}
=
\rho_{ai}\mu_{h,ai}
+
(1-\rho_{ai})b_a.
$$

Define the local mixture variance

$$
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
$$

Aggregate across anchors using coverage weights

$$
\rho_{ai}^{\mathrm{cover}}\ge 0,
\qquad
\sum_{a\in A(i)}\rho_{ai}^{\mathrm{cover}}=1.
$$

For example, use inverse-distance weights in metric space.

The aggregated residual mean is

$$
\hat h_i
=
\sum_{a\in A(i)}
\rho_{ai}^{\mathrm{cover}}\hat h_{ai}.
$$

Assuming independent local residual estimates, the aggregated variance is

$$
\hat v_{h,i}
=
\sum_{a\in A(i)}
(\rho_{ai}^{\mathrm{cover}})^2
\hat v_{h,ai}.
$$

Construct a pseudo target for the global GP:

$$
\tilde y_i=y_i-\hat h_i.
$$

Its heteroskedastic noise is

$$
\tilde\sigma_i^2=\sigma_\epsilon^2+\hat v_{h,i}.
$$

The global GP update objective is

$$
\mathcal L_g
=
\sum_{i=1}^N
\mathbb E_{q(g_i)}
\log
\mathcal N
\left(
\tilde y_i;
g_i,
\tilde\sigma_i^2
\right)
-
\beta_g\operatorname{KL}(q(u_g)\Vert p(u_g)).
$$

For

$$
q(g_i)=\mathcal N(\mu_{g,i},v_{g,i}),
$$

the expected log likelihood is

$$
\mathbb E_{q(g_i)}
\log
\mathcal N
\left(
\tilde y_i;
g_i,
\tilde\sigma_i^2
\right)
=
-\frac12
\frac{
(\tilde y_i-\mu_{g,i})^2+v_{g,i}
}{
\tilde\sigma_i^2
}
-\frac12\log \tilde\sigma_i^2
-\frac12\log(2\pi).
$$

Thus

$$
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
\log\tilde\sigma_i^2
+
\log(2\pi)
\right]
-
\beta_g\operatorname{KL}(q(u_g)\Vert p(u_g)).
$$

This gives a natural way to update the global GP inside the global-local model: local residual uncertainty controls how strongly each pseudo target influences the global trend.

---

## 14. Training strategy

A fully joint update from scratch is not recommended. The parameter groups are too entangled:

$$
\theta: \text{shared metric / W layer},
$$

$$
q(u_g): \text{global GP posterior},
$$

$$
\psi_a: \text{local residual GP and noise parameters},
$$

$$
\phi_a: \text{local gate parameters},
$$

$$
\gamma_a: \text{local CEM labels/responsibilities}.
$$

The recommended training procedure is staged coordinate optimization.

---

## 15. Stage 0: pretrain global metric GP

Train a smooth global metric GP:

$$
z_\theta(x)=W_\theta(x)x,
$$

$$
g\sim\mathcal{GP}(0,k_g).
$$

Use a standard sparse GP ELBO:

$$
\mathcal L_{\mathrm{pre}}
=
\sum_i
\mathbb E_{q(g_i)}
\log
\mathcal N(y_i;g_i,\sigma_{\mathrm{pre}}^2)
-
\operatorname{KL}(q(u_g)\Vert p(u_g))
-
\Omega_\theta.
$$

The goal is not to let this model solve every boundary. The global GP should be a trend learner. Use one or more of:

$$
\ell_g\ge \ell_{g,\min},
$$

$$
\sigma_{\mathrm{pre}}^2\ge \sigma_{\min}^2,
$$

$$
M_g \text{ moderate},
$$

$$
\lambda_\theta \text{ nontrivial},
$$

$$
\lambda_{\mathrm{smooth}}\mathbb E_x\|\nabla_x z_\theta(x)\|_F^2.
$$

The output of this stage is

$$
\theta_0,
\qquad
q_0(g).
$$

---

## 16. Stage 1: freeze global model and fit local residual self-CEM

Freeze

$$
\theta=\theta_0,
\qquad
q(g)=q_0(g).
$$

For each anchor $a$, compute

$$
\mu_g(Z_a),
\qquad
\Sigma_g(Z_a,Z_a).
$$

Then fit local residual self-CEM using

$$
y_{a,I}
\sim
\mathcal N
\left(
\mu_g(Z_{a,I})+m_a\mathbf 1,
K_{h,a,I}+\Sigma_g(Z_{a,I},Z_{a,I})+\sigma_a^2I
\right).
$$

The local CEM loop is:

1. initialize $\gamma_a$ using existing CEM, raw self-CEM, or response-based initialization;
2. update local hyperparameters $\psi_a$;
3. update gate parameters $\phi_a$;
4. update labels or soft responsibilities $\rho_a$;
5. repeat 1--2 weak refresh cycles.

A noise floor should be retained:

$$
\sigma_a^2\ge \sigma_{\mathrm{floor}}^2.
$$

This protects uncertainty calibration.

---

## 17. Stage 2: update global GP using local residual uncertainty

After local residual self-CEM is fitted, compute

$$
\hat h_i,
\qquad
\hat v_{h,i}
$$

for each training point.

Then define

$$
\tilde y_i=y_i-\hat h_i,
$$

$$
\tilde\sigma_i^2=\sigma_\epsilon^2+\hat v_{h,i}.
$$

Update the global GP posterior by maximizing

$$
\mathcal L_g
=
\sum_i
\mathbb E_{q(g_i)}
\log
\mathcal N
\left(
\tilde y_i;
g_i,
\tilde\sigma_i^2
\right)
-
\beta_g\operatorname{KL}(q(u_g)\Vert p(u_g)).
$$

This can update only $q(u_g)$, or also global kernel hyperparameters. Initially, do not update $W_\theta$ in this stage.

---

## 18. Stage 3: alternate local and global updates

Repeat:

$$
q(g)
\rightarrow
\{\gamma_a,\psi_a,\phi_a\}_{a\in\mathcal A}
\rightarrow
q(g)
\rightarrow
\cdots
$$

for a small number of outer cycles, for example 2--4.

This coordinate ascent interpretation is more stable and easier to debug than fully joint optimization.

---

## 19. Stage 4: optional slow fine-tuning of the W layer

Only after the frozen-$W$ version works should $W_\theta$ be fine-tuned.

Use the objective

$$
\mathcal J_\theta
=
\lambda_g\mathcal L_g
+
\lambda_{\mathrm{loc}}
\sum_a\omega_a\mathcal L_a
-
\lambda_{\mathrm{prox}}\|\theta-\theta_0\|^2
-
\Omega_\theta.
$$

Start with

$$
\lambda_g=1,
$$

$$
\lambda_{\mathrm{loc}}\in[0.05,0.2].
$$

Use local-loss warmup:

$$
\lambda_{\mathrm{loc}}(t)
=
\min
\left(
\lambda_{\max},
\frac{t}{T_{\mathrm{warm}}}\lambda_{\max}
\right).
$$

Use a smaller learning rate for $\theta$:

$$
\eta_\theta \ll \eta_\psi,\eta_\phi.
$$

A practical rule is

$$
\eta_\theta=0.1\eta_\psi.
$$

Use a proximal penalty:

$$
\lambda_{\mathrm{prox}}\|\theta-\theta_0\|^2.
$$

This prevents the local residual objective from destroying the global metric representation.

---

## 20. Gradient balancing for W fine-tuning

The local objective can dominate the representation gradients. Track

$$
r_\theta
=
\frac{
\|\nabla_\theta \mathcal L_{\mathrm{loc}}\|
}{
\|\nabla_\theta \mathcal L_g\|+\epsilon
}.
$$

Enforce

$$
r_\theta\le r_{\max}.
$$

If

$$
r_\theta>r_{\max},
$$

rescale

$$
\nabla_\theta \mathcal L_{\mathrm{loc}}
\leftarrow
\frac{r_{\max}}{r_\theta}
\nabla_\theta \mathcal L_{\mathrm{loc}}.
$$

Use

$$
r_{\max}=0.2
$$

early and optionally increase to

$$
r_{\max}=0.5
$$

later.

This keeps $W_\theta$ primarily a global metric representation, with only controlled local adaptation.

---

## 21. Test-time prediction

For a test point $x_*$, compute

$$
z_*=z_\theta(x_*).
$$

The global GP posterior is

$$
q(g_*)=\mathcal N(\mu_{g,*},v_{g,*}).
$$

Choose a local neighborhood $N_*$. Options:

1. raw-X KNN;
2. learned-Z KNN;
3. raw candidate pool plus learned-Z reranking;
4. validation-gated raw vs learned-Z retrieval.

The safest option is often raw candidate pool plus learned-Z reranking:

$$
C_*=\text{top-}M\text{ raw-X neighbors}.
$$

Then select $k$ neighbors from $C_*$ using

$$
d_{\mathrm{hybrid}}(i,*)
=
(1-\lambda)d_{\mathrm{raw}}(x_i,x_*)
+
\lambda d_z(z_i,z_*).
$$

Use small $\lambda$ initially. This protects raw locality while allowing the learned metric to refine neighbor choice.

Run local residual self-CEM on $N_*$ using global-residualized targets:

$$
r_{*,N}=y_{N_*}-\mu_g(Z_{N_*}),
$$

and local covariance

$$
K_{h,*}+\Sigma_g(Z_{N_*},Z_{N_*})+\sigma_*^2I.
$$

The local residual predictive distribution is

$$
q(h_*)=\mathcal N(\mu_{h,*},v_{h,*}).
$$

The full predictive distribution is

$$
q(y_*)=\mathcal N(\mu_{y,*},v_{y,*}),
$$

with

$$
\mu_{y,*}=\mu_{g,*}+\mu_{h,*}.
$$

Under a mean-field global-local approximation,

$$
v_{y,*}=v_{g,*}+v_{h,*}.
$$

If observation noise is not already included in $v_{h,*}$, use

$$
v_{y,*}=v_{g,*}+v_{h,*}+\sigma_\epsilon^2.
$$

The implementation must avoid adding the same observation noise twice.

---

## 22. Sampled-W local residual ensemble

If the local self-CEM head has uncertain $W$, a sampled deterministic-W ensemble can improve RMSE because each sample can change the local neighborhood, labels, hyperparameters, and posterior mean.

Draw

$$
W_*^{(s)}\sim q(W_*),
\qquad
s=1,\dots,S.
$$

For each sample $s$:

1. set $\operatorname{Var}(W_*^{(s)})=0$;
2. rerank or select neighbors using $W_*^{(s)}$;
3. run deterministic-W residual self-CEM;
4. obtain

$$
\mu_{h,*}^{(s)},
\qquad
v_{h,*}^{(s)}.
$$

The sample-specific full predictive mean is

$$
\mu_{y,*}^{(s)}=\mu_{g,*}+\mu_{h,*}^{(s)}.
$$

The sample-specific full predictive variance is

$$
v_{y,*}^{(s)}=v_{g,*}+v_{h,*}^{(s)}.
$$

Moment-match the mixture:

$$
\bar\mu_{y,*}
=
\frac1S
\sum_{s=1}^S
\mu_{y,*}^{(s)}.
$$

$$
\bar v_{y,*}
=
\frac1S
\sum_{s=1}^S
\left[
v_{y,*}^{(s)}+(\mu_{y,*}^{(s)})^2
\right]
-
\bar\mu_{y,*}^2.
$$

Equivalently,

$$
\bar v_{y,*}
=
\underbrace{
\frac1S\sum_s v_{y,*}^{(s)}
}_{\text{within-sample uncertainty}}
+
\underbrace{
\frac1S\sum_s
(\mu_{y,*}^{(s)}-\bar\mu_{y,*})^2
}_{\text{between-sample uncertainty}}.
$$

This extension is the closest analogue of LMJGP's sampled-W refit head, but with a global trend component and residual self-CEM calibration.

---

## 23. Identifiability and constraints

The decomposition

$$
y=g+h
$$

is not identifiable without restrictions, because for any function $\delta$,

$$
g+h=(g+\delta)+(h-\delta).
$$

The model is defensible only if global and local components have different inductive biases.

Use the following restrictions.

### 23.1 Smooth global trend

Constrain the global GP:

$$
\ell_g \text{ large},
$$

$$
M_g \text{ moderate},
$$

$$
\sigma_g^2 \text{ not too small},
$$

$$
\Omega_\theta \text{ nontrivial}.
$$

### 23.2 Local residual shrinkage

Constrain local residuals:

$$
\lambda_s\sum_a s_{h,a}^2,
$$

$$
\sigma_a^2\ge\sigma_{\mathrm{floor}}^2,
$$

$$
\ell_{h,a}\le \ell_g/c.
$$

### 23.3 CEM gating

The local residual component is active only through local inlier structure:

$$
\gamma_{ai}=1.
$$

The gate must be regularized:

$$
\lambda_\beta\|\beta_a\|^2.
$$

### 23.4 Composite-likelihood normalization

Use $\omega_a$ to avoid overcounting points covered by many anchors.

---

## 24. Main review-risk points and defenses

### Challenge 1: “This is not an exact likelihood.”

Defense: It is a normalized composite variational likelihood. Local JumpGP neighborhoods overlap, so an exact factorized likelihood is not claimed.

### Challenge 2: “The global GP and local GP double-count uncertainty.”

Defense: The local likelihood includes global GP posterior covariance:

$$
K_h+\Sigma_g+\sigma^2I.
$$

The local model is trained under global uncertainty, not after subtracting a deterministic GP mean.

### Challenge 3: “The decomposition is not identifiable.”

Defense: The model uses separated priors and constraints:

$$
\ell_g\gg \ell_h,
$$

global trend smoothness, local CEM gating, local residual shrinkage, and coverage-normalized composite likelihood.

### Challenge 4: “This is just DMGP.”

Defense: Plain DMGP is

$$
y=g(z_\theta(x))+\epsilon.
$$

This model is

$$
y=g(z_\theta(x))+h_a(z_\theta(x))+\epsilon.
$$

The local residual JumpGP/self-CEM component explicitly sharpens boundary residuals and propagates global uncertainty.

### Challenge 5: “The local loss can destroy the learned metric.”

Defense: Start with frozen $W_\theta$. Fine-tune only with proximal regularization and gradient balancing.

---

## 25. Recommended ablation ladder

### A. `globalgp_resid_selfcem_diagvar`

Use frozen global metric GP.

Local covariance:

$$
C_{a,I}=K_{h,a,I}+\operatorname{diag}(v_g(Z_{a,I}))+\sigma_a^2I.
$$

This is the simplest uncertainty-aware residual self-CEM variant.

### B. `globalgp_resid_selfcem_blockvar`

Use frozen global metric GP.

Local covariance:

$$
C_{a,I}=K_{h,a,I}+\Sigma_g(Z_{a,I},Z_{a,I})+\sigma_a^2I.
$$

This is the cleaner theory version.

### C. `alt_globalgp_selfcem`

Alternate:

$$
q(g)
\rightarrow
\{\gamma_a,\psi_a,\phi_a\}
\rightarrow
q(g)
\rightarrow
\cdots.
$$

Use local residual uncertainty as heteroskedastic noise for global GP updates.

### D. `alt_globalgp_selfcem_finetuneW`

Same as C, but slowly fine-tune $W_\theta$ with proximal penalty and gradient balancing.

### E. `alt_globalgp_selfcem_ensW`

Same as C or D, with sampled deterministic-W residual self-CEM ensemble at test time.

---

## 26. Diagnostics to report

### 26.1 Standard prediction metrics

Report:

$$
\mathrm{RMSE},
\qquad
\mathrm{CRPS},
\qquad
\mathrm{Cov90},
\qquad
\mathrm{Width90}.
$$

### 26.2 Mean decomposition

Report:

$$
\mu_y(x_*)=\mu_g(x_*)+\mu_h(x_*).
$$

Metrics:

$$
\mathrm{RMSE}(\mu_g),
\qquad
\mathrm{RMSE}(\mu_h),
\qquad
\mathrm{RMSE}(\mu_g+\mu_h).
$$

### 26.3 Variance decomposition

Report:

$$
v_y(x_*)=v_g(x_*)+v_h(x_*).
$$

Averages:

$$
\mathbb E[v_g],
\qquad
\mathbb E[v_h],
\qquad
\mathbb E[v_y].
$$

For sampled-W ensemble:

$$
v_{\mathrm{within}}=\frac1S\sum_s v_y^{(s)},
$$

$$
v_{\mathrm{between}}=\frac1S\sum_s(\mu_y^{(s)}-\bar\mu_y)^2.
$$

### 26.4 Local CEM diagnostics

Inlier fraction:

$$
\frac1{|N_a|}\sum_{i\in N_a}\gamma_{ai}.
$$

Soft inlier fraction:

$$
\frac1{|N_a|}\sum_{i\in N_a}\rho_{ai}.
$$

Label-change rate:

$$
\frac1{|N_a|}
\sum_{i\in N_a}
\mathbf 1[
\gamma_{ai}^{\mathrm{new}}\neq\gamma_{ai}^{\mathrm{old}}
].
$$

Noise-floor hit rate:

$$
\frac1{|\mathcal A|}
\sum_a
\mathbf 1[\sigma_a^2=\sigma_{\mathrm{floor}}^2].
$$

### 26.5 Representation diagnostics

Track:

$$
\|W_\theta-W_{\theta_0}\|,
$$

$$
\mathbb E_x\|W_\theta(x)\|_F,
$$

$$
\mathbb E_x\|\nabla_x z_\theta(x)\|_F.
$$

If using raw-pool plus $z$-rerank, track raw overlap:

$$
\mathrm{overlap}
=
\frac{
|\mathrm{KNN}_{\mathrm{raw}}(x_*)\cap \mathrm{KNN}_{\mathrm{hybrid}}(x_*)|
}{k}.
$$

### 26.6 Global-local interaction diagnostics

Report:

$$
\frac{\mathbb E[v_g]}{\mathbb E[v_h]},
$$

$$
\operatorname{Var}(y)-\operatorname{Var}(y-\mu_g),
$$

and

$$
\operatorname{Var}(y-\mu_g)-\operatorname{Var}(y-\mu_g-\mu_h).
$$

These show how much variance is explained by the global trend and local residual.

---

## 27. Expected successful behavior

A successful method should satisfy:

$$
\mathrm{RMSE}(\mu_g+\mu_h)
<
\mathrm{RMSE}(\mu_h^{\mathrm{raw\ selfCEM}}),
$$

while preserving calibrated uncertainty:

$$
\mathrm{Cov90}\approx 0.75\text{--}0.85.
$$

It should avoid excessive width inflation:

$$
\mathrm{Width90}_{\mathrm{global+local}}
\not\gg
\mathrm{Width90}_{\mathrm{raw\ selfCEM}}.
$$

A good outcome is:

$$
\mathrm{RMSE}\downarrow,
\qquad
\mathrm{CRPS}\downarrow,
\qquad
\mathrm{Cov90}\text{ stable or improved},
\qquad
\mathrm{Width90}\text{ moderate}.
$$

A bad deterministic-plugin failure mode is:

$$
\mathrm{RMSE}\downarrow
\quad\text{but}\quad
\mathrm{Cov90}\downarrow.
$$

A double-counting failure mode is:

$$
\mathrm{Cov90}\uparrow
\quad\text{but}\quad
\mathrm{Width90}\uparrow\uparrow.
$$

---

## 28. Minimal implementation checklist

### Inputs

- $X_{\mathrm{train}},y_{\mathrm{train}},X_{\mathrm{test}}$
- learned metric dimension $Q$
- global inducing count $M_g$
- local neighbor count $k$
- anchor set $\mathcal A$
- local self-CEM parameters
- noise floor $\sigma_{\mathrm{floor}}^2$

### Outputs

For each test point:

$$
\mu_g,
\quad
v_g,
\quad
\mu_h,
\quad
v_h,
\quad
\mu_y,
\quad
v_y.
$$

Also save diagnostics:

- local inlier fraction;
- label-change rate;
- raw-neighbor overlap;
- global/local variance decomposition;
- sampled-W within/between variance if applicable.

### Core local covariance

The core implementation line is:

$$
C_{a,I}=K_{h,a,I}+\Sigma_{g,a,I}+\sigma_a^2I.
$$

For the diagonal version:

$$
C_{a,I}=K_{h,a,I}+\operatorname{diag}(v_g(Z_{a,I}))+\sigma_a^2I.
$$

### Core prediction

$$
\mu_y=\mu_g+\mu_h,
$$

$$
v_y=v_g+v_h.
$$

---

## 29. Summary

The proposed method is a global-local additive probabilistic model:

$$
y(x)=g(z_\theta(x))+h_a(z_\theta(x))+\epsilon.
$$

The global GP provides sample-efficient smooth trend learning in a learned metric space. The local residual self-CEM/JGP provides boundary sharpening and local discontinuity modeling.

The defining local likelihood is

$$
y_{a,I}
\sim
\mathcal N
\left(
\mu_g(Z_{a,I})+m_a\mathbf 1,
K_{h,a,I}
+
\Sigma_g(Z_{a,I},Z_{a,I})
+
\sigma_a^2I
\right).
$$

This equation makes the method more than a residual plugin: the global GP posterior uncertainty is propagated into local self-CEM training.

The objective is a normalized composite variational objective:

$$
\mathcal J
=
\sum_a\omega_a\mathcal L_a
-
\beta_g\operatorname{KL}(q(u_g)\Vert p(u_g))
-
\Omega_\theta
-
\sum_a\Omega_a.
$$

The recommended training procedure is:

$$
\text{pretrain global metric GP}
\rightarrow
\text{fit residual self-CEM with global uncertainty}
\rightarrow
\text{update global GP using local residual uncertainty}
\rightarrow
\text{optionally fine-tune }W_\theta.
$$

The method can be interpreted as:

$$
\boxed{
\text{DMGP trend head}
+
\text{local residual JumpGP/self-CEM boundary-sharpening head}.
}
$$
