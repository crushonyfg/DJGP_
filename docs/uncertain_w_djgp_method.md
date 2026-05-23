pg# Uncertain-W DJGP-EM：用闭式 moment propagation 替代 MC-sampling 的 DJGP-EM 方案

## 1. 核心问题

原始 DJGP / LMJGP 中，projection matrix 通常写成局部形式：

$$
W_j \in \mathbb R^{Q\times D},
$$

第 $$j$$ 个 test anchor 的 local training point 被投影为：

$$
z_i^{(j)} = W_j x_i^{(j)}.
$$

然后在 projected local data 上 fit JumpGP。

如果我们已经有了 projection posterior：

$$
q(W_j\mid D),
$$

那么严格的 predictive distribution 应该是：

$$
p(y_\ast\mid D)
=
\int p_{\text{JGP}}(y_\ast\mid W_j,D)
q(W_j\mid D)dW_j.
$$

原始 DJGP prediction 通常用 Monte Carlo 做这个积分：

1. sample 多个 $$W_j^{(s)}\sim q(W_j)$$；
2. 每个 $$W_j^{(s)}$$ 下投影 local data；
3. 每个 sample 下重新 fit local JGP；
4. 对 predictive means / variances 做 ensemble。

这个方法统计上合理，但计算很慢，而且 prediction noise 比较大。

本方案想解决的问题是：

$$
\boxed{
\text{能不能不 sample }W_j，\text{而是直接把 }q(W_j)\text{ 的 mean/var 闭式传入 JGP？}
}
$$

答案是：可以做一个近似闭式版本。它不是 exact marginalization，但可以作为一个 moment-matched EM / VEM surrogate。

---

## 2. 为什么 exact marginalization 不可行

给定 $$W_j$$，JGP-CEM 的 prediction 是标准 GP posterior。假设 CEM 得到 inlier set：

$$
I_j(W_j)=\{i:v_{ij}(W_j)=1\}.
$$

预测均值为：

$$
\mu_\ast(W_j)
=
\hat m_j
+
k_{\ast I}(W_j)^\top
\left(K_I(W_j)+\hat\sigma_j^2 I\right)^{-1}
(y_I-\hat m_j\mathbf 1).
$$

预测方差为：

$$
s_\ast^2(W_j)
=
k_{\ast\ast}
-
k_{\ast I}(W_j)^\top
\left(K_I(W_j)+\hat\sigma_j^2 I\right)^{-1}
k_{\ast I}(W_j).
$$

严格 marginalization 要算：

$$
\mathbb E_{q(W_j)}[\mu_\ast(W_j)]
$$

和：

$$
\mathbb E_{q(W_j)}[s_\ast^2(W_j)]
+
\operatorname{Var}_{q(W_j)}[\mu_\ast(W_j)].
$$

困难在于：

1. CEM inlier set $$I_j(W_j)$$ 随 $$W_j$$ 离散变化；
2. kernel inverse $$K(W_j)^{-1}$$ 对 $$W_j$$ 非线性；
3. local hyperparameters $$\theta_j(W_j)$$ 和 gate parameters $$\nu_j(W_j)$$ 可能也随 $$W_j$$ 重新优化；
4. 因此

$$
\mathbb E_W\left[k_W^\top(K_W+\sigma^2I)^{-1}y\right]
\neq
\bar k^\top(\bar K+\sigma^2I)^{-1}y.
$$

所以 exact closed-form marginalization 基本不可行。

但我们可以做一个更 tractable 的 approximation：

$$
\boxed{
K(W_j)\text{ replaced by }\bar K_j=\mathbb E_{q(W_j)}[K(W_j)]
}
$$

也就是把随机 projection layer 的不确定性先传进 kernel，再用这个 expected kernel 做 JGP-CEM / VEM。

---

## 3. 基本设定

对某个 anchor $$j$$，假设 projection posterior 是 Gaussian：

$$
q(W_j)=\mathcal N(M_j,\Sigma_j).
$$

其中：

$$
M_j=\mathbb E[W_j].
$$

如果使用 row-independent covariance，可以写成：

$$
W_{j,q:}\sim \mathcal N(M_{j,q:},\Sigma_{j,q}),
\quad q=1,\dots,Q.
$$

这里 $$W_{j,q:}$$ 是 $$W_j$$ 的第 $$q$$ 行。

对于任意两个 local points $$x_i,x_l$$，定义：

$$
\delta_{il}=x_i-x_l.
$$

随机 projected difference 为：

$$
\Delta z_{il}=W_j\delta_{il}.
$$

由于 $$W_j$$ 是 Gaussian，$$\Delta z_{il}$$ 也是 Gaussian：

$$
\Delta z_{il}\sim \mathcal N(\mu_{\Delta,il},S_{\Delta,il}).
$$

其中：

$$
\mu_{\Delta,il}=M_j\delta_{il}.
$$

如果 rows independent，则：

$$
S_{\Delta,il}[q,q]
=
\delta_{il}^\top \Sigma_{j,q}\delta_{il}.
$$

如果不同 rows 之间也相关，则可以用 vectorization 表示。

令：

$$
w_j=\operatorname{vec}(W_j).
$$

则存在矩阵 $$B_{il}$$，使得：

$$
\Delta z_{il}=B_{il}w_j.
$$

于是：

$$
\mu_{\Delta,il}=B_{il}\mu_w,
$$

$$
S_{\Delta,il}=B_{il}\Sigma_w B_{il}^\top.
$$

---

## 4. Squared Exponential kernel 下的 closed-form expected kernel

假设 latent GP kernel 是 squared exponential：

$$
k(z_i,z_l)
=
\sigma_f^2
\exp
\left(
-\frac{1}{2\ell^2}
\|z_i-z_l\|^2
\right).
$$

在 projected space 中：

$$
z_i-z_l=W_j(x_i-x_l)=W_j\delta_{il}.
$$

因此：

$$
k_W(i,l)
=
\sigma_f^2
\exp
\left(
-\frac{1}{2\ell^2}
\|W_j\delta_{il}\|^2
\right).
$$

我们想计算：

$$
\bar k(i,l)
=
\mathbb E_{q(W_j)}[k_W(i,l)].
$$

因为：

$$
\Delta z_{il}=W_j\delta_{il}
\sim
\mathcal N(\mu_{\Delta,il},S_{\Delta,il}),
$$

所以：

$$
\bar k(i,l)
=
\sigma_f^2
\mathbb E_{\Delta z}
\left[
\exp
\left(
-\frac{1}{2\ell^2}
\Delta z^\top\Delta z
\right)
\right].
$$

使用 Gaussian quadratic exponential identity：

若：

$$
x\sim \mathcal N(\mu,S),
$$

则：

$$
\mathbb E
\left[
\exp\left(-\frac12 x^\top A x\right)
\right]
=
|I+SA|^{-1/2}
\exp
\left(
-\frac12
\mu^\top A(I+SA)^{-1}\mu
\right),
$$

其中 $$A$$ 是 positive semidefinite matrix。

令：

$$
A=\ell^{-2}I_Q.
$$

得到：

$$
\boxed{
\bar k(i,l)
=
\sigma_f^2
\left|I_Q+\frac{1}{\ell^2}S_{\Delta,il}\right|^{-1/2}
\exp
\left(
-\frac12
\mu_{\Delta,il}^\top
(\ell^2I_Q+S_{\Delta,il})^{-1}
\mu_{\Delta,il}
\right)
}
$$

这个公式把 $$W_j$$ 的 posterior mean 和 variance 都传入了 kernel。

如果 $$S_{\Delta,il}=0$$，则退化为 deterministic kernel：

$$
\bar k(i,l)
=
\sigma_f^2
\exp
\left(
-\frac{1}{2\ell^2}
\|M_j(x_i-x_l)\|^2
\right).
$$

---

## 5. Uncertain-W kernel 的直观含义

公式中有两部分：

### 5.1 Mean projection distance

$$
\mu_{\Delta,il}=M_j(x_i-x_l).
$$

它表示 posterior mean projection 下两个点的距离。

### 5.2 Projection uncertainty term

$$
S_{\Delta,il}
$$

表示这个 pair 的 projected difference 因 $$W_j$$ posterior uncertainty 产生的方差。

如果 $$S_{\Delta,il}$$ 很大，则：

$$
\left|I_Q+\frac{1}{\ell^2}S_{\Delta,il}\right|^{-1/2}
$$

会变小，off-diagonal covariance 通常会下降。

这表示：当 projection 不确定时，模型降低两个点之间的相关性，从而增加 predictive uncertainty。

这比只用 $$M_j$$ 做 projection 更合理。

---

## 6. 用 uncertain-W kernel 改写 JGP-CEM prediction

假设 CEM 已经得到 inlier set：

$$
I_j=\{i:v_{ij}=1\}.
$$

使用 uncertain-W kernel 构造：

$$
\bar K_{I_jI_j}[i,l]=\bar k(i,l),
\quad i,l\in I_j.
$$

test-train covariance 为：

$$
\bar k_{\ast I_j}[i]
=
\bar k(\ast,i).
$$

其中：

$$
\delta_{\ast i}=x_\ast-x_i.
$$

同样使用上面的 expected kernel formula。

预测均值：

$$
\boxed{
\mu_\ast
=
\hat m_j
+
\bar k_{\ast I_j}^\top
\left(
\bar K_{I_jI_j}+\hat\sigma_j^2I
\right)^{-1}
(y_{I_j}-\hat m_j\mathbf 1)
}
$$

预测方差：

$$
\boxed{
s_\ast^2
=
\bar k_{\ast\ast}
-
\bar k_{\ast I_j}^\top
\left(
\bar K_{I_jI_j}+\hat\sigma_j^2I
\right)^{-1}
\bar k_{\ast I_j}
}
$$

对 SE kernel：

$$
\bar k_{\ast\ast}=\sigma_f^2.
$$

如果预测 noisy observation $$y_\ast$$，再加 observation noise：

$$
s_{y,\ast}^2=s_\ast^2+\hat\sigma_j^2.
$$

---

## 7. 这个 variance 是否已经完整包含 W uncertainty？

不完全。

用 $$\bar K=\mathbb E[K(W)]$$ 的 GP posterior 是一个 moment-matched approximation。它把 $$W$$ uncertainty 传入 prior covariance，但没有精确计算：

$$
\operatorname{Var}_{q(W)}[\mu_\ast(W)].
$$

严格的 total variance 是：

$$
\operatorname{Var}(y_\ast\mid D)
=
\mathbb E_{q(W)}
[
\operatorname{Var}(y_\ast\mid D,W)
]
+
\operatorname{Var}_{q(W)}
[
\mathbb E(y_\ast\mid D,W)
].
$$

Uncertain-W kernel 近似主要覆盖第一部分和部分 covariance shrinkage，但第二部分未必完全覆盖。

因此可以加一个 delta-method correction。

---

## 8. Delta-method correction for mean uncertainty

设 deterministic-W JGP predictive mean 为：

$$
\mu_\ast(W).
$$

在 posterior mean $$M_j$$ 处一阶 Taylor 展开：

$$
\mu_\ast(W)
\approx
\mu_\ast(M_j)
+
g_\ast^\top
\operatorname{vec}(W-M_j),
$$

其中：

$$
g_\ast
=
\nabla_{\operatorname{vec}(W)}
\mu_\ast(W)
\bigg|_{W=M_j}.
$$

于是：

$$
\operatorname{Var}_{q(W)}[\mu_\ast(W)]
\approx
 g_\ast^\top \Sigma_w g_\ast.
$$

最终 prediction variance 可以写成：

$$
\boxed{
s_{\ast,\text{final}}^2
=
s_{\ast,\text{UW-kernel}}^2
+
\lambda_\Delta
 g_\ast^\top\Sigma_w g_\ast
}
$$

其中 $$\lambda_\Delta$$ 默认取 1。

这不是 conformal，也不是后处理 calibration，而是 Bayesian first-order uncertainty propagation。

---

## 9. Kernel-vector covariance correction

另一个更闭式但实现更复杂的 correction 是考虑：

$$
\operatorname{Var}_W[k_{\ast I}(W)].
$$

令：

$$
\alpha
=
(\bar K_{I I}+\sigma^2I)^{-1}(y_I-m\mathbf 1).
$$

近似：

$$
\mu_\ast(W)
\approx
m+k_{\ast I}(W)^\top\alpha.
$$

因此：

$$
\operatorname{Var}_W[\mu_\ast(W)]
\approx
\alpha^\top
\operatorname{Cov}_W(k_{\ast I})
\alpha.
$$

其中：

$$
\operatorname{Cov}_W(k_{\ast I})[i,l]
=
\mathbb E_W[k(Wx_\ast,Wx_i)k(Wx_\ast,Wx_l)]
-
\bar k_{\ast i}\bar k_{\ast l}.
$$

对 SE kernel，第一项也可以通过 Gaussian quadratic exponential identity 计算。

因为：

$$
k(Wx_\ast,Wx_i)k(Wx_\ast,Wx_l)
=
\sigma_f^4
\exp
\left(
-\frac{1}{2\ell^2}
\|W\delta_{\ast i}\|^2
-\frac{1}{2\ell^2}
\|W\delta_{\ast l}\|^2
\right).
$$

令：

$$
w=\operatorname{vec}(W).
$$

则 exponent 是一个 quadratic form：

$$
-\frac12 w^\top A_{il}w.
$$

如果 $$w\sim\mathcal N(\mu_w,\Sigma_w)$$，则：

$$
\mathbb E
\left[
\exp\left(-\frac12w^\top A_{il}w\right)
\right]
=
|I+\Sigma_w A_{il}|^{-1/2}
\exp
\left(
-\frac12
\mu_w^\top A_{il}(I+\Sigma_w A_{il})^{-1}\mu_w
\right).
$$

所以可以闭式计算：

$$
\mathbb E_W[k_{\ast i}(W)k_{\ast l}(W)].
$$

但该 correction 工程量更大。第一版建议先用 delta-method correction。

---

## 10. Gate uncertainty

JGP-CEM / VEM 还需要 gate probability。

原始 gate logit：

$$
\xi_i
=
\nu^\top
\begin{bmatrix}
1\\
W_j(x_i-x_\ast)
\end{bmatrix}.
$$

因为 $$W_j$$ 是 Gaussian，所以 $$\xi_i$$ 是一维 Gaussian：

$$
\xi_i\sim\mathcal N(m_{\xi i},s_{\xi i}^2).
$$

其中：

$$
m_{\xi i}
=
\nu_0+
u_z^\top M_j(x_i-x_\ast).
$$

$$
s_{\xi i}^2
=
\nu_z^\top S_{\Delta,i\ast}\nu_z.
$$

需要：

$$
\bar\pi_i
=
\mathbb E_{q(W_j)}[\sigma(\xi_i)].
$$

logistic-normal expectation 没有简单精确闭式，但有常用 approximation：

$$
\boxed{
\bar\pi_i
\approx
\sigma
\left(
\frac{m_{\xi i}}
{\sqrt{1+\pi s_{\xi i}^2/8}}
\right)
}
$$

也可以用 Gauss-Hermite quadrature：

$$
\bar\pi_i
=
\int \sigma(\xi)\mathcal N(\xi;m_{\xi i},s_{\xi i}^2)d\xi.
$$

由于这是一维积分，Gauss-Hermite 很便宜，不需要 sample $$W$$。

---

## 11. Uncertain-W CEM update

CEM update 中比较 inlier 和 outlier posterior score。

使用 uncertain gate：

$$
\bar\pi_i.
$$

使用 uncertain-W GP 得到 fitted latent function 的 mean/variance：

$$
m_{f,i},
\quad
s_{f,i}^2.
$$

则 inlier score：

$$
S_{1,i}
=
\log \bar\pi_i
+
\log
\mathcal N
(y_i;m_{f,i},s_{f,i}^2+\sigma^2).
$$

outlier score：

$$
S_{0,i}
=
\log(1-\bar\pi_i)+\log p_0(y_i).
$$

若使用 uniform outlier likelihood：

$$
p_0(y_i)=\frac1u.
$$

若使用 broad background likelihood：

$$
p_0(y_i)=\mathcal N(y_i;m_0,s_0^2).
$$

CEM hard update：

$$
\boxed{
v_i
=
\mathbf 1(S_{1,i}>S_{0,i})
}
$$

这就是把 $$W$$ 的 mean/var 传入 CEM step。

---

## 12. Uncertain-W VEM update

更自然的是 soft VEM。

令：

$$
\rho_i=q(v_i=1).
$$

更新：

$$
\boxed{
\rho_i
=
\frac{\exp(S_{1,i})}
{\exp(S_{1,i})+\exp(S_{0,i})}
}
$$

即：

$$
\log\frac{\rho_i}{1-
ho_i}
=S_{1,i}-S_{0,i}.
$$

这比 CEM 更适合 uncertainty propagation，因为 regime uncertainty 不会被立即 hard-threshold 掉。

---

## 13. Uncertain-W EM training objective

如果我们想做 EM / VEM-EM 训练 $$q(R)$$，可以构造 surrogate objective。

对每个 local anchor $$j$$，用 $$q(R)$$ 诱导 $$q(W_j)$$，再构造 uncertain-W kernel：

$$
\bar K_j(q(R)).
$$

CEM hard-label objective：

$$
\mathcal Q_j(q(R))
=
\log
\mathcal N
\left(
 y_{I_j};
 \mu_j\mathbf 1,
 \bar K_{j,I_jI_j}+\sigma_j^2I
\right)
+
\sum_{i\in I_j}\log\bar\pi_{ij}
+
\sum_{i\notin I_j}\log(1-\bar\pi_{ij})
+
\sum_{i\notin I_j}\log p_0(y_i).
$$

整体目标：

$$
\boxed{
\mathcal Q(q(R))
=
\sum_j\mathcal Q_j(q(R))
-
\beta KL(q(R)\|p(R))
}
$$

对 VEM soft-label 版本：

$$
\mathcal Q_j(q(R))
=
\log
\mathcal N
\left(
 y;
 \mu_j\mathbf 1,
 \bar K_j+\sigma_j^2D_{\rho}^{-1}
\right)
+
\sum_i
\rho_i\log\bar\pi_i
+
(1-\rho_i)\log(1-\bar\pi_i)
+
\sum_i(1-\rho_i)\log p_0(y_i)
+
H(\rho).
$$

这里：

$$
D_\rho=\operatorname{diag}(\rho_1,\dots,\rho_n).
$$

也可以用更稳定的 lower-bound form，而不是直接用 $$D_\rho^{-1}$$，避免 $$\rho_i$$ 太小时数值不稳。

这个 objective 对 $$q(R)$$ 的 mean 和 variance 可微。

---

## 14. 为什么这个比原 DJGP closed-form VI 的梯度链路更好

原 DJGP closed-form VI 中，$$q(R)$$ 的 signal 大致是：

$$
q(R)
\rightarrow
q(W_j)
\rightarrow
\Psi_1,\Psi_2
\rightarrow
q(r_j)
\rightarrow
S_1,S_2
\rightarrow
\logsumexp.
$$

其中 $$\Psi_1,\Psi_2$$ 是 local inducing kernel expectations。

Uncertain-W EM 中，signal 是：

$$
q(R)
\rightarrow
q(W_j)
\rightarrow
\bar K_j=\mathbb E[K(W_j)]
\rightarrow
\log\mathcal N(y;0,\bar K_j+\sigma^2I).
$$

这条链路更短。

而且 variance path 更直接，因为：

$$
\bar k(i,l)
$$

显式依赖：

$$
S_{\Delta,il}=\operatorname{Var}_{q(W_j)}[W_j(x_i-x_l)].
$$

所以 data likelihood 会直接对 $$\operatorname{Var}(W_j)$$、进一步对 $$\operatorname{Var}(R)$$ 产生梯度。

这比原来的 $$\Psi_1,\Psi_2$$ 形式更容易让 $$R_{\log std}$$ 收到有效信号。

---

## 15. 与原 DJGP ELBO 的关系

需要注意：

$$
\log\mathcal N(y;0,\mathbb E[K(W)]+\sigma^2I)
$$

不是：

$$
\mathbb E_W[
\log\mathcal N(y;0,K(W)+\sigma^2I)
].
$$

所以 uncertain-W EM 不是原 DJGP ELBO 的严格等价形式。

更准确地说，它定义了一个 moment-matched surrogate model：

原始 stochastic projection GP mixture：

$$
p(f)=\int GP(0,K(W))q(W)dW.
$$

用一个 moment-matched GP 近似：

$$
f\approx GP(0,\bar K),
\quad
\bar K=\mathbb E[K(W)].
$$

因此它是：

$$
\boxed{
\text{moment-matched marginalized-W surrogate EM}
}
$$

不是 exact EM。

但这个近似比 mean-W JGP 更合理，也比 full MC 更快。

---

## 16. 对全局 R 的 propagation

如果 $$W_j$$ 来自 global inducing values $$R$$，则：

$$
q(R)=\mathcal N(\mu_R,\Sigma_R).
$$

对 anchor $$j$$：

$$
W_j=W(x_j^\ast;R).
$$

若使用 sparse GP conditional mean：

$$
W_{j,qd}
=
a_{j,q}^\top R_{:qd},
$$

其中：

$$
a_{j,q}^\top
=
k_w(x_j^\ast,U)(K_{UU}^{(q)})^{-1}.
$$

则：

$$
\mathbb E[W_{j,qd}]
=
a_{j,q}^\top\mu_{qd}.
$$

$$
\operatorname{Var}(W_{j,qd})
=
a_{j,q}^\top\Sigma_{qd}a_{j,q}
+
\sigma^2_{\text{cond},j,q}.
$$

其中 $$\sigma^2_{\text{cond},j,q}$$ 是 sparse GP conditional residual variance。如果使用 deterministic training conditional，则可以先忽略它；但为了完整 UQ，应该加上。

于是 $$q(R)$$ 的 mean/variance 会自然传到 $$q(W_j)$$ 的 mean/variance，再传到 uncertain-W kernel。

---

## 17. 是否需要 sample R

如果只使用 moment-matched $$q(W_j)$$，那么 prediction 可以完全不 sample $$R$$。

流程是：

$$
q(R)
\Rightarrow
q(W_j)
\Rightarrow
\bar K_j
\Rightarrow
\mu_\ast,s_\ast^2.
$$

这是一种 fully deterministic approximate marginalization。

但也可以做 hybrid：

1. training 时不 sample，用 uncertain-W kernel；
2. prediction 时 sample 少量 $$R$$ 作为额外 check；
3. 或者只对 high-uncertainty points sample。

如果计算压力大，第一版建议不 sample。

---

## 18. 对 pointwise W(x) 的扩展

如果使用 pointwise embedding：

$$
z_i=W(x_i)x_i,
$$

则 pair difference 为：

$$
\Delta z_{il}
=
W(x_i)x_i-W(x_l)x_l.
$$

因为 $$W(x_i)$$ 和 $$W(x_l)$$ 来自同一个 GP posterior，它们联合 Gaussian，所以：

$$
\Delta z_{il}
\sim
\mathcal N(\mu_{\Delta,il},S_{\Delta,il}).
$$

然后仍然使用 same expected SE kernel formula：

$$
\bar k(i,l)
=
\sigma_f^2
\left|I_Q+\frac{1}{\ell^2}S_{\Delta,il}\right|^{-1/2}
\exp
\left(
-\frac12
\mu_{\Delta,il}^\top
(\ell^2I_Q+S_{\Delta,il})^{-1}
\mu_{\Delta,il}
\right).
$$

但工程上要计算 $$W(x_i)$$ 与 $$W(x_l)$$ 的 joint covariance，更复杂。

如果目标是修正 DJGP，而不是变成 DMGP + JGP，推荐先做 anchor-conditioned version。

---

## 19. 推荐实现顺序

### 19.1 第一版：Uncertain-W JGP-CEM prediction only

目标：先验证 no-MC prediction 是否工作。

输入已有：

$$
q(W_j)=\mathcal N(M_j,\Sigma_j).
$$

实现：

1. 用 $$M_j$$ 跑原 JGP-CEM，得到 inlier set。
2. 用 $$M_j,\Sigma_j$$ 构造 uncertain-W kernel $$\bar K$$。
3. 用 $$\bar K$$ 替代 deterministic kernel 做 GP posterior prediction。
4. 可选加 delta-method variance correction。

这一步不改训练，只改 prediction。

---

### 19.2 第二版：Uncertain-W JGP-CEM training

在 CEM-EM 里加入 uncertain-W kernel。

C-step：

$$
q(R)\Rightarrow q(W_j)\Rightarrow \bar K_j,\bar\pi_j.
$$

用 uncertain-W CEM 更新 $$v_j$$。

M-step：

固定 $$v_j$$，最大化：

$$
\sum_j\mathcal Q_j(q(R))-\beta KL(q(R)\|p(R)).
$$

---

### 19.3 第三版：Uncertain-W VEM

用 soft responsibilities：

$$
\rho_{ij}=q(v_{ij}=1).
$$

用 uncertain-W kernel 构造 approximate $$q(f_j)$$，更新 $$\rho_j$$，再更新 $$q(R)$$。

这是理论上最漂亮的版本，但实现稍复杂。

---

## 20. 主要风险

### 20.1 Variance 可能被用来降低 correlation

因为 $$S_{\Delta,il}$$ 增大会降低 off-diagonal covariance，模型可能用 projection variance 来解释复杂数据，而不是学 mean projection。

应对：

1. KL 控制 $$q(R)$$；
2. 先 MAP-R，再打开 variance；
3. 给 $$\sigma_R$$ 上限；
4. monitor data gradient / KL gradient。

---

### 20.2 CEM labels 仍可能太 hard

Uncertain-W kernel 不自动解决 hard CEM 的问题。

建议：

1. 使用 VEM soft responsibilities；
2. 或 CEM labels 做 floor：

$$
\rho=\epsilon+(1-2\epsilon)v.
$$

---

### 20.3 Moment matching 可能低估 uncertainty

因为：

$$
\log\mathcal N(y;0,\mathbb E[K(W)]+\sigma^2I)
$$

不是 exact marginal likelihood。

建议加：

1. delta-method mean correction；
2. kernel-vector covariance correction；
3. 或少量 MC sanity check。

---

## 21. 推荐命名

可以叫：

$$
\boxed{\text{Uncertain-W DJGP-EM}}
$$

或：

$$
\boxed{\text{Marginalized-W JGP}}
$$

或：

$$
\boxed{\text{Moment-Matched DJGP-EM}}
$$

如果要强调它不是 exact marginalization，我建议：

$$
\boxed{\text{Moment-Matched Uncertain-W DJGP}}
$$

---

## 22. 一句话总结

这个方法把原来基于 MC 的：

$$
W\sim q(W),\quad \text{fit JGP under each }W
$$

替换成闭式近似：

$$
q(W)\Rightarrow \bar K=\mathbb E[K(W)]\Rightarrow \text{fit one uncertain-W JGP}.
$$

它不是 exact DJGP marginal likelihood，但它给了一个可微、快速、能传播 $$W$$ posterior variance 的 EM/VEM surrogate。

如果实现成功，它很可能是“原 DJGP 的 EM 版修正”里最自然的一条路线。

