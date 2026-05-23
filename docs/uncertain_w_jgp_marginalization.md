# Uncertain-W JGP：在 JGP-CEM/VEM 中闭式或近似闭式传播投影不确定性

## 0. 目的

当前 DJGP/LMJGP 的 prediction 通常采用 Monte Carlo 方式传播投影不确定性：

$$
W^{(s)} \sim q(W),\quad s=1,\dots,S,
$$

每个 sampled projection 下重新投影 local neighborhood、重新 fit JGP-CEM/VEM，然后用 total variance aggregation 得到最终预测均值和方差。

这个做法统计上直接，但计算很慢。本文档讨论一个替代方案：直接修改 JGP-CEM/VEM，使其接受投影矩阵的 posterior mean 和 posterior covariance：

$$
q(W_a)=\mathcal N(M_a,\Sigma_a),
$$

并将 $$W_a$$ 的不确定性通过 closed-form 或 approximation 传入 JGP 的 kernel、gate 和最终 predictive variance 中。

目标是构造一个 no-MC 或 low-MC 的近似：

$$
p(y_\ast\mid D)
=
\int p_{\text{JGP}}(y_\ast\mid W_a,D)q(W_a\mid D)dW_a,
$$

但避免对 $$W_a$$ 采样后反复 fit JGP。

---

## 1. 基本设定

考虑一个 test anchor 或 local anchor：

$$
x_\ast \in \mathbb R^D.
$$

其 local neighborhood 为：

$$
D_n=\{(x_i,y_i):i=1,\dots,n\}.
$$

local projection matrix 为：

$$
W_a\in \mathbb R^{Q\times D}.
$$

给定 $$W_a$$，projected inputs 为：

$$
z_i=W_ax_i,
$$

$$
z_\ast=W_ax_\ast.
$$

JGP-CEM 引入 latent inlier indicator：

$$
v_i\in\{0,1\}.
$$

其中：

$$
v_i=1
$$

表示 local point $$i$$ 与 anchor $$x_\ast$$ 属于同一个 regime；

$$
v_i=0
$$

表示它属于其他 regime，即 outlier。

给定 inlier set：

$$
I=\{i:v_i=1\},
$$

JGP prediction 使用 inlier points 上的 GP posterior：

$$
y_I\mid W_a,v,\theta
\sim
\mathcal N
\left(
\mu\mathbf 1,
K_I(W_a)+\sigma^2 I
\right).
$$

其中：

$$
K_I(W_a)[i,l]=k(W_ax_i,W_ax_l),\quad i,l\in I.
$$

如果使用 squared exponential kernel：

$$
k(z_i,z_l)
=
\sigma_f^2
\exp\left(
-\frac{1}{2\ell^2}\|z_i-z_l\|^2
\right).
$$

现在假设我们不知道 $$W_a$$ 的确切值，而是有 posterior：

$$
q(W_a)=\mathcal N(M_a,\Sigma_a).
$$

注意这里的 Gaussian 是对 $$\operatorname{vec}(W_a)$$ 而言：

$$
\operatorname{vec}(W_a)\sim
\mathcal N
\left(
\operatorname{vec}(M_a),
\Sigma_a
\right).
$$

---

## 2. 精确 marginalization 为什么困难

理想上我们想要：

$$
p(y_\ast\mid D)
=
\int p_{\text{JGP}}(y_\ast\mid W_a,D)q(W_a)dW_a.
$$

但是精确计算很难，原因有三点。

第一，CEM 的 inlier set 依赖 $$W_a$$：

$$
I(W_a)=\{i:v_i(W_a)=1\}.
$$

这是一个离散、非连续的函数。

第二，GP posterior 中有矩阵逆：

$$
(K_I(W_a)+\sigma^2 I)^{-1}.
$$

通常：

$$
\mathbb E_W
\left[
K(W)^{-1}
\right]
\ne
\left(\mathbb E_W[K(W)]\right)^{-1}.
$$

第三，CEM / JGP 中的 local hyperparameters，例如 gate parameter $$\nu$$、noise $$\sigma^2$$、kernel scale 等，通常也可能随 $$W_a$$ 的变化而重新估计：

$$
\hat\theta(W_a),\quad \hat\nu(W_a).
$$

因此，精确积分：

$$
\int p_{\text{JGP}}(y_\ast\mid W_a,D,\hat\theta(W_a),\hat\nu(W_a))q(W_a)dW_a
$$

基本不可闭式。

本文采用一个 moment-matched approximation：不对整个 JGP posterior 精确积分，而是将 $$W_a$$ 的 posterior uncertainty 先闭式传入 kernel 和 gate，再在这个 uncertain-kernel JGP 上做 CEM/VEM 和 prediction。

---

## 3. Uncertain-W kernel 的推导

### 3.1 投影差分的分布

对任意 pair $$(i,l)$$，定义原空间差分：

$$
\delta_{il}=x_i-x_l\in\mathbb R^D.
$$

投影后差分为：

$$
\Delta z_{il}=W_a\delta_{il}\in\mathbb R^Q.
$$

因为 $$W_a$$ 是 Gaussian，且 $$\Delta z_{il}$$ 是 $$W_a$$ 的线性函数，所以：

$$
\Delta z_{il}\sim
\mathcal N(\mu_{\Delta,il},S_{\Delta,il}).
$$

其中：

$$
\mu_{\Delta,il}=M_a\delta_{il}.
$$

为了写协方差，令：

$$
w=\operatorname{vec}(W_a)\in\mathbb R^{QD}.
$$

构造线性矩阵：

$$
B_{il}=I_Q\otimes \delta_{il}^\top.
$$

则：

$$
\Delta z_{il}=B_{il}w.
$$

因此：

$$
S_{\Delta,il}=B_{il}\Sigma_aB_{il}^\top.
$$

如果 posterior covariance 是 row-independent，即每一行 $$W_{a,q:}$$ 独立，且：

$$
W_{a,q:}\sim \mathcal N(M_{a,q:},\Sigma_{a,q}),
$$

那么：

$$
S_{\Delta,il}[q,q]
=
\delta_{il}^\top\Sigma_{a,q}\delta_{il}.
$$

如果不同行之间独立，则 $$S_{\Delta,il}$$ 是 diagonal；若行之间相关，则它是 full $$Q\times Q$$ matrix。

---

### 3.2 Squared exponential kernel 的期望

给定 $$W_a$$ 时：

$$
k(W_ax_i,W_ax_l)
=
\sigma_f^2
\exp\left(
-\frac{1}{2\ell^2}
\|W_a(x_i-x_l)\|^2
\right).
$$

因为：

$$
\Delta z_{il}=W_a\delta_{il}
\sim
\mathcal N(\mu_{\Delta,il},S_{\Delta,il}),
$$

我们需要计算：

$$
\bar k(i,l)
=
\mathbb E_{q(W_a)}
\left[
\sigma_f^2
\exp\left(
-\frac{1}{2\ell^2}\|\Delta z_{il}\|^2
\right)
\right].
$$

这是 Gaussian quadratic exponential expectation。一般公式为：若：

$$
x\sim\mathcal N(\mu,S),
$$

则：

$$
\mathbb E
\left[
\exp\left(-\frac{1}{2}x^\top A x\right)
\right]
=
|I+SA|^{-1/2}
\exp\left(
-\frac{1}{2}\mu^\top A(I+SA)^{-1}\mu
\right),
$$

其中 $$A$$ 是 positive semidefinite。

在 SE kernel 中：

$$
A=\ell^{-2}I_Q.
$$

因此：

$$
\bar k(i,l)
=
\sigma_f^2
\left|I_Q+\frac{1}{\ell^2}S_{\Delta,il}\right|^{-1/2}
\exp\left(
-\frac{1}{2}
\mu_{\Delta,il}^\top
(\ell^2I_Q+S_{\Delta,il})^{-1}
\mu_{\Delta,il}
\right).
$$

这就是 uncertain-W expected kernel。

注意：如果 $$S_{\Delta,il}=0$$，即没有 projection uncertainty，则：

$$
\bar k(i,l)
=
\sigma_f^2
\exp\left(
-\frac{1}{2\ell^2}\|M_a(x_i-x_l)\|^2
\right),
$$

退化回 deterministic mean-W kernel。

---

### 3.3 直观解释

Uncertain-W kernel 有两种效应。

第一，mean projection 决定 expected projected distance：

$$
\mu_{\Delta,il}=M_a(x_i-x_l).
$$

第二，projection uncertainty 通过 $$S_{\Delta,il}$$ 进入 kernel。

若 $$S_{\Delta,il}$$ 很大，说明 pair $$(i,l)$$ 的 projected distance 对 $$W_a$$ 很不确定，则 kernel 会被额外 shrink：

$$
\left|I_Q+\ell^{-2}S_{\Delta,il}\right|^{-1/2}
$$

会变小。

这表示：如果两个点的相似性高度依赖不确定的 projection direction，则我们不应过度相信它们高度相关。

---

## 4. 在 JGP-CEM prediction 中使用 uncertain-W kernel

假设 CEM 已经得到 inlier set：

$$
I=\{i:v_i=1\}.
$$

用 uncertain-W kernel 构造：

$$
\bar K_I[i,l]=\bar k(i,l),\quad i,l\in I.
$$

test-train covariance 为：

$$
\bar k_{\ast I}[i]=\bar k(\ast,i),\quad i\in I,
$$

其中：

$$
\delta_{\ast i}=x_\ast-x_i,
$$

$$
\mu_{\Delta,\ast i}=M_a(x_\ast-x_i),
$$

$$
S_{\Delta,\ast i}=B_{\ast i}\Sigma_aB_{\ast i}^\top.
$$

然后使用标准 GP posterior form：

$$
\mu_\ast
=
\hat\mu
+
\bar k_{\ast I}^\top
(\bar K_I+\hat\sigma^2I)^{-1}
(y_I-\hat\mu\mathbf 1).
$$

$$
s_{\ast,\text{UW}}^2
=
\bar k_{\ast\ast}
-
\bar k_{\ast I}^\top
(\bar K_I+\hat\sigma^2I)^{-1}
\bar k_{\ast I}.
$$

对 SE kernel：

$$
\bar k_{\ast\ast}=\sigma_f^2.
$$

如果预测的是 noisy observation，可以再加 observation noise：

$$
s_{y_\ast,\text{UW}}^2=s_{\ast,\text{UW}}^2+\hat\sigma^2.
$$

---

## 5. 这个 prediction variance 是否已经完全包含 W uncertainty？

不完全。

上面的 uncertain-W kernel 是一个 moment-matched GP approximation：

$$
K(W_a)
\quad\Rightarrow\quad
\bar K=\mathbb E[K(W_a)].
$$

然后用 $$\bar K$$ 做 GP prediction。

但精确的 law of total variance 是：

$$
\operatorname{Var}(y_\ast\mid D)
=
\mathbb E_{q(W_a)}
\left[
\operatorname{Var}(y_\ast\mid D,W_a)
\right]
+
\operatorname{Var}_{q(W_a)}
\left[
\mathbb E(y_\ast\mid D,W_a)
\right].
$$

Uncertain-W kernel 部分主要把 $$W_a$$ 的 uncertainty 传播到 covariance matrix 中，但它不等于精确计算：

$$
\mathbb E_W
\left[
k_W^\top(K_W+\sigma^2I)^{-1}y
\right].
$$

因此，若希望更保守地传播 $$W_a$$ uncertainty，可以额外加入 posterior mean sensitivity correction。

---

## 6. Delta-method variance correction

令 deterministic JGP/GP predictive mean 为：

$$
\mu_\ast(W_a).
$$

在 posterior mean $$M_a$$ 处一阶线性化：

$$
\mu_\ast(W_a)
\approx
\mu_\ast(M_a)
+

abla_w\mu_\ast(M_a)^\top
(w-m_w),
$$

其中：

$$
w=\operatorname{vec}(W_a),
$$

$$
m_w=\operatorname{vec}(M_a).
$$

于是：

$$
\operatorname{Var}_{q(W_a)}[\mu_\ast(W_a)]
\approx

abla_w\mu_\ast(M_a)^\top
\Sigma_a
\nabla_w\mu_\ast(M_a).
$$

最终 variance 可以写成：

$$
s_{\ast,\text{final}}^2
=
s_{\ast,\text{UW}}^2
+
\lambda_\Delta
\nabla_w\mu_\ast(M_a)^\top
\Sigma_a
\nabla_w\mu_\ast(M_a).
$$

其中 $$\lambda_\Delta$$ 可设为 1。若该 correction 过大或过小，也可以在 validation 上调节，但它本质上仍是 Bayesian linear uncertainty propagation，而不是 conformal calibration。

实际实现中，$$\nabla_w\mu_\ast(M_a)$$ 可以用自动微分得到。

---

## 7. Kernel-vector covariance correction

如果希望避免对 full predictive mean 自动微分，可以使用 kernel-vector covariance 近似。

定义：

$$
\alpha
=
(\bar K_I+\sigma^2I)^{-1}
(y_I-\mu\mathbf 1).
$$

预测均值近似为：

$$
\mu_\ast(W_a)\approx \mu + k_{\ast I}(W_a)^\top\alpha.
$$

于是：

$$
\operatorname{Var}_{q(W_a)}[\mu_\ast(W_a)]
\approx
\alpha^\top
\operatorname{Cov}_{q(W_a)}(k_{\ast I}(W_a))
\alpha.
$$

其中：

$$
\operatorname{Cov}_{q(W_a)}(k_{\ast I})[i,l]
=
\mathbb E[k_{\ast i}(W_a)k_{\ast l}(W_a)]
-
\bar k_{\ast i}\bar k_{\ast l}.
$$

对 SE kernel，第一项也可以闭式计算。因为：

$$
k_{\ast i}(W_a)k_{\ast l}(W_a)
=
\sigma_f^4
\exp
\left(
-rac{1}{2\ell^2}
\left[
\|W_a(x_\ast-x_i)\|^2
+
\|W_a(x_\ast-x_l)\|^2
\right]
\right).
$$

令：

$$
\delta_i=x_\ast-x_i,
$$

$$
\delta_l=x_\ast-x_l.
$$

定义联合向量：

$$
r=
\begin{bmatrix}
W_a\delta_i\\
W_a\delta_l
\end{bmatrix}
\in\mathbb R^{2Q}.
$$

因为 $$W_a$$ Gaussian，$$r$$ 也是 Gaussian：

$$
r\sim \mathcal N(\mu_r,S_r).
$$

其中：

$$
\mu_r=
\begin{bmatrix}
M_a\delta_i\\
M_a\delta_l
\end{bmatrix}.
$$

如果：

$$
B_i=I_Q\otimes \delta_i^\top,
$$

$$
B_l=I_Q\otimes \delta_l^\top,
$$

则：

$$
B_r=
\begin{bmatrix}
B_i\\
B_l
\end{bmatrix},
$$

$$
S_r=B_r\Sigma_aB_r^\top.
$$

令：

$$
A_r=\frac{1}{\ell^2}
\begin{bmatrix}
I_Q & 0\\
0 & I_Q
\end{bmatrix}.
$$

则：

$$
\mathbb E[k_{\ast i}(W_a)k_{\ast l}(W_a)]
=
\sigma_f^4
|I_{2Q}+S_rA_r|^{-1/2}
\exp
\left(
-rac{1}{2}
\mu_r^\top
A_r(I_{2Q}+S_rA_r)^{-1}
\mu_r
\right).
$$

这个 correction 比 delta method 更闭式，但实现更复杂，因为需要为 test point 的 inlier set 构造 $$|I|\times |I|$$ 的 kernel-vector covariance matrix。

---

## 8. Gate uncertainty 的传播

JGP 的 gate 通常是：

$$
p(v_i=1\mid W_a,\nu)
=
\sigma(\xi_i),
$$

其中：

$$
\xi_i=\nu^\top
\begin{bmatrix}
1\\
W_a(x_i-x_\ast)
\end{bmatrix}.
$$

令：

$$
\delta_{i\ast}=x_i-x_\ast.
$$

因为：

$$
W_a\delta_{i\ast}
\sim
\mathcal N(\mu_{\Delta,i\ast},S_{\Delta,i\ast}),
$$

所以 gate logit：

$$
\xi_i\sim \mathcal N(m_{\xi i},s_{\xi i}^2).
$$

其中：

$$
m_{\xi i}
=
\nu_0+
u_z^\top M_a\delta_{i\ast},
$$

$$
s_{\xi i}^2
=
\nu_z^\top S_{\Delta,i\ast}\nu_z.
$$

需要计算：

$$
\bar\pi_i
=
\mathbb E[\sigma(\xi_i)].
$$

logistic-normal expectation 没有精确初等闭式。一个常用近似为：

$$
\bar\pi_i
\approx
\sigma
\left(
\frac{m_{\xi i}}
{\sqrt{1+\pi s_{\xi i}^2/8}}
\right).
$$

或者使用 Gauss-Hermite quadrature：

$$
\bar\pi_i
=
\int \sigma(\xi)\mathcal N(\xi;m_{\xi i},s_{\xi i}^2)d\xi.
$$

令 $$\xi=m_{\xi i}+\sqrt{2}s_{\xi i}t$$，则：

$$
\bar\pi_i
\approx
\frac{1}{\sqrt\pi}
\sum_{r=1}^{H}
\omega_r
\sigma(m_{\xi i}+\sqrt{2}s_{\xi i}t_r),
$$

其中 $$t_r,\omega_r$$ 是 Gauss-Hermite nodes 和 weights。

---

## 9. 修改 CEM update

原始 hard CEM 中，给定当前 $$f_i$$ 或 GP posterior fitted value，会比较：

$$
p(v_i=1\mid \cdot)
$$

与：

$$
p(v_i=0\mid \cdot).
$$

在 uncertain-W CEM 中，可以使用：

$$
\log A_i
=
\log \bar\pi_i
+
\log p_{\text{in}}(y_i),
$$

$$
\log B_i
=
\log(1-\bar\pi_i)
+
\log p_0(y_i).
$$

其中 $$p_0(y_i)$$ 是 outlier likelihood，可以是：

$$
p_0(y_i)=1/u,
$$

或 broad background：

$$
p_0(y_i)=\mathcal N(y_i;m_0,s_0^2).
$$

inlier likelihood 可以近似为：

$$
p_{\text{in}}(y_i)
\approx
\mathcal N(y_i;m_{f,i},s_{f,i}^2+\sigma^2),
$$

其中 $$m_{f,i},s_{f,i}^2$$ 来自 uncertain-W kernel GP posterior。

然后 hard update：

$$
v_i
=
\mathbf 1(\log A_i>\log B_i).
$$

为了稳定，不建议一开始就完全 hard。可以使用 soft responsibility：

$$
\rho_i
=
\frac{A_i}{A_i+B_i}.
$$

然后如果需要 CEM hard labels，再设：

$$
v_i=\mathbf 1(\rho_i>0.5).
$$

---

## 10. Uncertain-W VEM 版本

VEM 版本比 CEM 更自然，因为它不需要过早 harden labels。

令：

$$
q(v_i=1)=\rho_i.
$$

使用 log-odds update：

$$
\log\frac{\rho_i}{1-
\rho_i}
=
\mathbb E_{q(W_a)}
\left[
\log p(v_i=1\mid W_a,\nu)
-
\log p(v_i=0\mid W_a,\nu)
\right]
+
\mathbb E_{q(f_i)}
\left[
\log p(y_i\mid f_i,v_i=1)
\right]
-
\log p_0(y_i).
$$

Gate 部分可用：

$$
\mathbb E[\log \sigma(\xi_i)]
$$

和：

$$
\mathbb E[\log(1-\sigma(\xi_i))]
$$

这两个 expectation 可用 Gauss-Hermite quadrature 计算。

Likelihood 部分：

$$
\mathbb E_{q(f_i)}
[
\log p(y_i\mid f_i,v_i=1)
]
=
-rac{1}{2}\log(2\pi\sigma^2)
-rac{(y_i-m_{f,i})^2+s_{f,i}^2}{2\sigma^2}.
$$

其中 $$q(f)$$ 由 uncertain-W kernel GP posterior 给出。

这个 VEM 版本能同时保留 label uncertainty 和 projection uncertainty，因此理论上比 hard CEM 更适合 UQ。

---

## 11. 如果 W 来自 q(R)，如何得到 q(W_a)

若 projection field 由 inducing variables $$R$$ 给出：

$$
W_{qd}(x)=k_w(x,U)K_{UU}^{-1}R_{:qd}+\epsilon_{qd}(x),
$$

其中：

$$
\epsilon_{qd}(x)\mid R
\sim
\mathcal N(0,c_0(x)).
$$

如果：

$$
q(R_{:qd})=\mathcal N(\mu_{qd},\Sigma_{qd}),
$$

则 marginal：

$$
q(W_{qd}(x_a))=\mathcal N(m_{a,qd},v_{a,qd}).
$$

其中：

$$
a_a^\top=k_w(x_a,U)K_{UU}^{-1}.
$$

mean：

$$
m_{a,qd}=a_a^\top\mu_{qd}.
$$

variance：

$$
v_{a,qd}=c_0(x_a)+a_a^\top\Sigma_{qd}a_a.
$$

这里：

$$
c_0(x_a)=k_w(x_a,x_a)-k_w(x_a,U)K_{UU}^{-1}k_w(U,x_a).
$$

如果使用 deterministic conditional mean approximation，则设置：

$$
c_0(x_a)=0.
$$

但如果要完整传播 $$W\mid R$$ uncertainty，应保留 $$c_0(x_a)$$。

把所有 $$(q,d)$$ 拼起来，即得到：

$$
q(W_a)=\mathcal N(M_a,\Sigma_a).
$$

这个 $$\Sigma_a$$ 包含两部分：

第一，来自 $$q(R)$$ 的 posterior uncertainty：

$$
a_a^\top\Sigma_{qd}a_a.
$$

第二，来自 sparse GP conditional residual 的 uncertainty：

$$
c_0(x_a).
$$

如果只 sample $$R$$ 后取 $$E[W\mid R]$$，就会漏掉第二部分。

---

## 12. 两层 uncertainty 的 no-MC 近似

完整 decomposition 是：

$$
q(W_a)
=
\int p(W_a\mid R)q(R)dR.
$$

如果二者都是 Gaussian-linear，那么 marginal 仍然是 Gaussian：

$$
q(W_a)=\mathcal N(M_a,\Sigma_a).
$$

所以 no-MC uncertain-W JGP 可以直接用 marginal $$q(W_a)$$，不需要显式 sample $$R$$。

这相当于把：

$$
R\sim q(R),\quad W_a\sim p(W_a\mid R)
$$

两层不确定性合并成：

$$
W_a\sim q(W_a).
$$

然后用前面的 expected kernel：

$$
\bar k(i,l)=\mathbb E_{q(W_a)}[k(W_ax_i,W_ax_l)].
$$

这就是完全 no-MC 的核心版本。

---

## 13. Anchor-conditioned vs pointwise W(x)

### 13.1 Anchor-conditioned version

主推版本：

$$
z_i^{(a)}=W(x_a)x_i.
$$

这里同一个 local neighborhood 共用同一个 projection matrix。此时只需要 $$q(W(x_a))$$。

pair kernel expectation 使用：

$$
\Delta z_{il}=W(x_a)(x_i-x_l).
$$

这和原 DJGP 的 local constant projection 一致，工程也最简单。

### 13.2 Pointwise version

如果使用：

$$
z_i=W(x_i)x_i,
$$

则 pair difference 是：

$$
\Delta z_{il}=W(x_i)x_i-W(x_l)x_l.
$$

这仍然是 Gaussian，因为 $$W(x_i)$$ 和 $$W(x_l)$$ 在 GP posterior 下联合 Gaussian。

但此时需要 joint posterior：

$$
q(W(x_i),W(x_l)).
$$

这会涉及 cross-covariance：

$$
\operatorname{Cov}(W_{qd}(x_i),W_{qd}(x_l)).
$$

虽然仍然可闭式，但实现复杂度明显更高。

如果目标是修正 DJGP，而不是变成 DMGP embedding + JGP，建议优先使用 anchor-conditioned version。

---

## 14. 推荐算法：Uncertain-W JGP-CEM

输入：

$$
D_n=\{(x_i,y_i)\}_{i=1}^n,
\quad x_\ast,
\quad q(W_a)=\mathcal N(M_a,\Sigma_a).
$$

步骤：

1. 使用 $$M_a$$ 初始化 projected inputs：

   $$
   z_i=M_ax_i.
   $$

2. 初始化 JGP-CEM labels $$v_i$$。

3. 重复直到收敛或固定步数：

   a. 根据当前 inlier set，用 uncertain-W kernel 构造 $$\bar K_I$$。

   b. 计算 GP posterior fitted mean 和 variance：

   $$
   m_{f,i},s_{f,i}^2.
   $$

   c. 对每个点计算 uncertain gate probability：

   $$
   \bar\pi_i\approx
   \mathbb E_{q(W_a)}[\sigma(\nu^\top[1,W_a(x_i-x_\ast)])].
   $$

   d. 更新：

   $$
   \rho_i
   =
   \frac{
   \bar\pi_i\mathcal N(y_i;m_{f,i},s_{f,i}^2+\sigma^2)
   }{
   \bar\pi_i\mathcal N(y_i;m_{f,i},s_{f,i}^2+\sigma^2)
   +(1-\bar\pi_i)p_0(y_i)
   }.
   $$

   e. CEM hard version：

   $$
   v_i=\mathbf 1(\rho_i>0.5).
   $$

4. 最终 inlier set $$I$$ 下，用 uncertain-W kernel 做 prediction。

5. prediction variance 可选加入 delta correction：

   $$
   s_{\ast,\text{final}}^2
   =
   s_{\ast,\text{UW}}^2
   +
   \nabla_w\mu_\ast^\top\Sigma_a\nabla_w\mu_\ast.
   $$

---

## 15. 推荐算法：Uncertain-W JGP-VEM

输入同上。

初始化 responsibilities：

$$
\rho_i\in(0,1).
$$

迭代：

1. 用 uncertain-W kernel 构造 full local covariance：

$$
\bar K=(\bar k(i,l))_{i,l=1}^n.
$$

2. 用 soft inlier weights 构造 approximate likelihood precision。可使用：

$$
\Gamma=\operatorname{diag}(\rho_1,\dots,\rho_n).
$$

3. 更新 q(f)：

$$
S_f
=
(\bar K^{-1}+\sigma^{-2}\Gamma)^{-1},
$$

$$
m_f
=
S_f
(\bar K^{-1}\mu\mathbf 1+\sigma^{-2}\Gamma y).
$$

4. 更新每个 $$\rho_i$$：

$$
\log\frac{\rho_i}{1-\rho_i}
=
G_i
+
L_i
-
\log p_0(y_i),
$$

其中 gate contribution：

$$
G_i
=
\mathbb E
[
\log\sigma(\xi_i)
]
-
\mathbb E
[
\log(1-\sigma(\xi_i))
],
$$

可用 Gauss-Hermite。

likelihood contribution：

$$
L_i
=
-rac{1}{2}\log(2\pi\sigma^2)
-
\frac{(y_i-m_{f,i})^2+S_{f,ii}}{2\sigma^2}.
$$

5. 重复 1 到 4。

Prediction 使用最终 $$\rho_i$$ 或 $$\rho_i>0.5$$ 的 inlier set，也可以使用 weighted GP prediction。

---

## 16. 与 MC sampling 的关系

MC 方法计算：

$$
W^{(s)}\sim q(W_a),
$$

$$
\mu_\ast^{(s)},s_\ast^{2(s)}
=\text{JGP}(W^{(s)}).
$$

然后：

$$
\mu_\ast=\frac{1}{S}\sum_s\mu_\ast^{(s)},
$$

$$
s_\ast^2
=
\frac{1}{S}\sum_s
\left[
s_\ast^{2(s)}+(
\mu_\ast^{(s)}-\mu_\ast)^2
\right].
$$

Uncertain-W kernel 方法相当于用 analytic moment matching 替代大量 MC samples。

优点：

1. 无需每个 $$W$$ sample 重新 fit JGP；
2. 速度更快；
3. 可以直接接入 CEM/VEM；
4. 显式使用 $$W$$ posterior variance。

缺点：

1. 无法精确处理 CEM hard-label 随 $$W$$ 变化的离散跳变；
2. 近似了 $$\mathbb E[K^{-1}]$$；
3. 如果 $$q(W)$$ 很宽，moment-matched kernel 可能过度平滑；
4. 对 non-SE kernel 不一定有简单闭式。

---

## 17. 实现建议

### 17.1 第一版最小实现

实现函数：

$$
\texttt{expected\_se\_kernel\_under\_gaussian\_W}(X_1,X_2,M_W,\Sigma_W,\ell,\sigma_f)
$$

返回：

$$
\bar K.
$$

先支持 row-independent diagonal covariance：

$$
\Sigma_{a,q}=\operatorname{diag}(v_{q1},\dots,v_{qD}).
$$

此时：

$$
S_{\Delta,il}[q,q]
=
\sum_{d=1}^{D}v_{qd}\delta_{il,d}^2.
$$

所以每个 pair 的 $$S_{\Delta,il}$$ 是 diagonal，计算 determinant 和 inverse 很便宜：

$$
\left|I_Q+\ell^{-2}S\right|
=
\prod_{q=1}^{Q}
(1+S_{qq}/\ell^2).
$$

$$
\mu^\top(\ell^2I+S)^{-1}\mu
=
\sum_q
\frac{\mu_q^2}{\ell^2+S_{qq}}.
$$

于是：

$$
\bar k(i,l)
=
\sigma_f^2
\prod_q
\left(1+\frac{S_{qq}}{\ell^2}\right)^{-1/2}
\exp\left(
-rac{1}{2}
\sum_q
\frac{\mu_q^2}{\ell^2+S_{qq}}
\right).
$$

这非常容易实现。

### 17.2 Gate expectation

第一版使用 logistic-normal approximation：

$$
\bar\pi_i
\approx
\sigma
\left(
\frac{m_{\xi i}}{\sqrt{1+\pi s_{\xi i}^2/8}}
\right).
$$

如果数值不稳，再切换 Gauss-Hermite。

### 17.3 Prediction variance

第一版使用：

$$
s_{\ast,\text{UW}}^2.
$$

第二版加入 delta correction：

$$
\nabla_w\mu_\ast^\top\Sigma_W\nabla_w\mu_\ast.
$$

---

## 18. 推荐实验

对每个 dataset 比较：

1. mean-W JGP：只用 $$M_W$$；
2. MC-W JGP：sample $$W$$；
3. Uncertain-W kernel JGP；
4. Uncertain-W kernel JGP + delta correction；
5. Uncertain-W VEM-JGP。

关键指标：

- RMSE；
- CRPS；
- Cov90；
- Width90；
- runtime。

预期结果：

如果 uncertain-W kernel 有效，它应当：

1. 比 mean-W JGP coverage 更好；
2. 比 MC-W JGP 快；
3. CRPS 接近或优于 MC-W；
4. 在 LH 这种高不确定边界问题上显著改善 under-coverage。

---

## 19. 总结

可以直接修改 JGP-CEM/VEM，让它接受：

$$
q(W_a)=\mathcal N(M_a,\Sigma_a),
$$

而不是只接受 deterministic $$W_a$$。

核心数学工具是：

$$
\bar k(i,l)
=
\mathbb E_{q(W_a)}[k(W_ax_i,W_ax_l)].
$$

对 SE kernel，该 expectation 有闭式：

$$
\bar k(i,l)
=
\sigma_f^2
\left|I_Q+\frac{1}{\ell^2}S_{\Delta,il}\right|^{-1/2}
\exp\left(
-\frac{1}{2}
\mu_{\Delta,il}^\top
(\ell^2I_Q+S_{\Delta,il})^{-1}
\mu_{\Delta,il}
\right).
$$

Gate uncertainty 通过 logistic-normal approximation 或 Gauss-Hermite 传播：

$$
\bar\pi_i
=
\mathbb E[\sigma(\xi_i)].
$$

Prediction variance 可以使用：

$$
s_{\ast,\text{UW}}^2
$$

并可加入 delta-method correction：

$$
\nabla_w\mu_\ast^\top\Sigma_W\nabla_w\mu_\ast.
$$

这条路线比纯 MC sampling 更快，比只用 mean-W 更有统计意义，也比 post-hoc calibration 更符合 GP/JGP 的 uncertainty propagation 逻辑。
