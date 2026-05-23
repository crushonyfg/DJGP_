# Pairwise-Regularized Amortized DJGP：简化版方法说明

## 0. 核心观点

这个方法的目标不是另起一个“NN 降维 + JGP”的模型，而是尽量保留 DJGP 的结构，同时修正原始 DJGP 中 $$W$$ 学习信号弱的问题。

原始 DJGP 的核心是：每个 local neighborhood 使用一个 local projection matrix，然后在 projected space 中 fit JGP。这里保留这个设定，但把每个 anchor 独立的 $$W_j$$ 改成由一个平滑的全局 projection field 生成：

$$
W_j = W(x_j^\ast).
$$

第 $$j$$ 个 local neighborhood 中所有点仍然共用同一个 local projection：

$$
z_i^{(j)} = W(x_j^\ast)x_i, \qquad z_\ast^{(j)} = W(x_j^\ast)x_j^\ast.
$$

这和 pointwise embedding

$$
z_i = W(x_i)x_i
$$

不同。后者更像 DMGP embedding + JGP head，离原 DJGP 更远。本文档推荐主线使用 anchor-conditioned 版本。

方法的核心修改是：在 CEM/VEM-EM 的 M-step 中，不只最大化 local JGP likelihood，而是额外加入一个由 JGP latent inlier/outlier labels 诱导出的 pairwise metric likelihood。这个 pairwise 项直接训练 projected geometry，让同 regime 的点更近、跨 regime 的点更远。

最终目标是：

$$
\text{local DJGP/JGP structure}
+
\text{smooth shared } W(x)
+
\text{pairwise regime-separation signal}
+
\text{projection uncertainty propagation}.
$$

---

## 1. Projection field prior

对每个 projection entry 放一个 GP prior：

$$
W_{qd}(x) \sim GP(0,k_w(x,x')),
$$

其中 $$q=1,\dots,Q$$，$$d=1,\dots,D$$。

选 inducing inputs：

$$
U=\{u_m\}_{m=1}^{M}.
$$

inducing values 为：

$$
R_{mqd}=W_{qd}(u_m).
$$

对固定 $$(q,d)$$，记：

$$
R_{:qd}=(R_{1qd},\dots,R_{Mqd})^\top.
$$

prior 为：

$$
p(R_{:qd}) = \mathcal N(0,K_{UU}^{(q)}).
$$

给定 $$R$$，任意位置 $$x$$ 的 conditional mean 为：

$$
\mu_{W,qd}(x\mid R)
=
k_w(x,U)
\left(K_{UU}^{(q)}\right)^{-1}
R_{:qd}.
$$

因此：

$$
\mu_W(x\mid R)=\left[\mu_{W,qd}(x\mid R)\right]_{q,d}.
$$

原始 DJGP 里 $$q(W_j)$$ 也是由 $$p(W\mid R)q(R)$$ 诱导出来的；这里保留这一思想，只是把 projection 从 transductive anchor variables 改成 shared field $$W(x)$$。

---

## 2. Variational posterior over R

使用 variational posterior：

$$
q(R)=\prod_{q=1}^{Q}\prod_{d=1}^{D}q(R_{:qd}),
$$

$$
q(R_{:qd})=\mathcal N(\mu_{qd},\Sigma_{qd}).
$$

最简单实现是 diagonal covariance：

$$
\Sigma_{qd}=\operatorname{diag}(\sigma_{1qd}^2,\dots,\sigma_{Mqd}^2).
$$

训练建议分两阶段：

1. **MAP-R 阶段**：固定 $$\Sigma_{qd}$$ 很小，只训练 $$\mu_{qd}$$。
2. **Posterior-R 阶段**：再打开 $$\Sigma_{qd}$$，学习 projection uncertainty。

---

## 3. W|R 的 conditional variance

不能只使用 $$\mu_W(x\mid R)$$。给定 $$R$$ 后，$$W(x)$$ 本身仍有 GP conditional variance：

$$
\operatorname{Var}(W_{qd}(x)\mid R)
=
k_w(x,x)
-
k_w(x,U)K_{UU}^{-1}k_w(U,x).
$$

对 anchor $$a$$，令：

$$
W_a = W(c_a).
$$

给定 $$R$$：

$$
\operatorname{vec}(W_a)\mid R
\sim
\mathcal N(m_a(R),\Sigma_{0,a}).
$$

其中：

$$
m_a(R)=\operatorname{vec}(\mu_W(c_a\mid R)),
$$

$$
\Sigma_{0,a}=\operatorname{Var}(\operatorname{vec}(W(c_a))\mid R).
$$

如果不同 $$(q,d)$$ 的 projection GP 独立，$$\Sigma_{0,a}$$ 通常是 diagonal 或 block-diagonal。

这部分是重要的：

$$
q(R) \text{ 表示 projection function 的 global uncertainty},
$$

而

$$
W\mid R \text{ 表示 sparse GP conditional residual uncertainty}.
$$

预测时应该传播两者。

---

## 4. Local JGP-CEM / VEM

训练 anchors 为：

$$
\mathcal A=\{c_a\}_{a=1}^{A}.
$$

每个 anchor 有 local neighborhood：

$$
N_a=\{i_1,\dots,i_n\}.
$$

对 anchor $$a$$，local projection 是：

$$
W_a = W(c_a).
$$

local projected input 为：

$$
z_i^{(a)}=W_a x_i,\qquad i\in N_a.
$$

JGP latent variable：

$$
v_{ai}\in\{0,1\}.
$$

其中 $$v_{ai}=1$$ 表示 point $$i$$ 与 anchor $$a$$ 属于 same regime；$$v_{ai}=0$$ 表示 outlier / other regime。

### Gate model

建议用 centered projected coordinates：

$$
p(v_{ai}=1\mid W_a,\nu_a)
=
\sigma\left(
\nu_a^\top
\begin{bmatrix}
1\\
W_a(x_i-c_a)
\end{bmatrix}
\right).
$$

这样比直接用 $$W_a x_i$$ 更符合 local boundary approximation。

### Inlier GP likelihood

令：

$$
I_a=\{i\in N_a:v_{ai}=1\}.
$$

对 inlier points：

$$
p(y_{I_a}\mid W_a,v_a,\theta_a)
=
\mathcal N
\left(
y_{I_a};
\mu_a\mathbf 1,
K_a(W_a)+\sigma_a^2I
\right),
$$

其中：

$$
K_a(W_a)[i,l]
=
k_f(W_a x_i,W_a x_l),
\qquad i,l\in I_a.
$$

### Outlier likelihood

原 JGP 使用 uniform dummy likelihood。对单点 prediction 这合理，但对 global $$W(x)$$ learning 会导致长期 outlier points 的 response 不进入学习。

推荐实现中先用 broad background：

$$
p_0(y_i)=\mathcal N(y_i;\bar y,c_y^2s_y^2),
$$

其中 $$c_y=3\sim 5$$。

hard CEM likelihood 可以写成：

$$
\ell_{\text{JGP},a}(W_a)
=
\log \mathcal N
\left(
y_{I_a};\mu_a\mathbf 1,K_a(W_a)+\sigma_a^2I
\right)
+
\sum_{i\in N_a}\log p(v_{ai}\mid W_a,\nu_a)
+
\sum_{i:v_{ai}=0}\log p_0(y_i).
$$

---

## 5. Pairwise regime likelihood

CEM/VEM 得到 local regime labels 后，构造 pairwise same-regime probability。

hard CEM：

$$
s_{ail}=\mathbf 1(v_{ai}=v_{al}).
$$

soft VEM：

$$
\rho_{ai}=q(v_{ai}=1),
$$

$$
s_{ail}=\rho_{ai}\rho_{al}+(1-\rho_{ai})(1-\rho_{al}).
$$

为了避免过硬，做 gamma floor：

$$
\tilde\rho_{ai}=\epsilon+(1-2\epsilon)\rho_{ai},
$$

例如 $$\epsilon=0.03$$。然后用 $$\tilde\rho$$ 构造 $$s_{ail}$$。

定义 projected distance：

$$
d_{ail}(W_a)=\|W_a(x_i-x_l)\|^2.
$$

same-regime pair 希望距离小，cross-regime pair 希望距离大。

使用 logistic margin likelihood：

$$
\log p(s_{ail}\mid W_a)
=
s_{ail}\log\sigma(m-d_{ail}(W_a))
+
(1-s_{ail})\log\sigma(d_{ail}(W_a)-m).
$$

pairwise term：

$$
\ell_{\text{pair},a}(W_a)
=
\sum_{i<l,\ i,l\in N_a}
\omega_{ail}
\left[
 s_{ail}\log\sigma(m-d_{ail}(W_a))
 +(1-s_{ail})\log\sigma(d_{ail}(W_a)-m)
\right].
$$

权重可以用 response difference：

$$
\omega_{ail}=
\operatorname{clip}\left(\frac{|y_i-y_l|}{s_y},0,w_{\max}\right).
$$

也可以加入 entropy：

$$
\omega_{ail}
=
\operatorname{clip}\left(\frac{|y_i-y_l|}{s_y},0,w_{\max}\right)
+
\eta_H\{H(\rho_{ai})+H(\rho_{al})\}.
$$

---

## 6. Training objective

使用 generalized EM / composite likelihood objective：

$$
\mathcal L
=
\tau_{\text{jgp}}
\sum_{a\in\mathcal A}
\mathbb E_{q(R)p(W_a\mid R)}
\left[
\ell_{\text{JGP},a}(W_a)
\right]
+
\lambda_{\text{pair}}
\sum_{a\in\mathcal A}
\mathbb E_{q(R)p(W_a\mid R)}
\left[
\ell_{\text{pair},a}(W_a)
\right]
-
\beta KL(q(R)\|p(R)).
$$

这不是普通 pure generative ELBO，而是 pairwise-regularized composite ELBO。它比原始 DJGP ELBO更适合学习 $$W$$，因为 pairwise term 给 $$W$$ 一条短梯度路径：

$$
R \to W_a \to d_{ail} \to \ell_{\text{pair}}.
$$

建议训练 schedule：

### Stage 1: pairwise warm start

$$
\tau_{\text{jgp}}=0,
\qquad
\lambda_{\text{pair}}>0,
\qquad
\beta \text{ small}.
$$

### Stage 2: joint fine-tuning

$$
\tau_{\text{jgp}}>0,
\qquad
\lambda_{\text{pair}}>0,
\qquad
\beta \uparrow 1.
$$

不要在 Stage 2 完全关闭 pairwise term。否则 CEM/JGP likelihood 可能把 pairwise 学到的 geometry 洗掉。

---

## 7. 如何传播 W 的不确定性

不建议把 calibration / conformal 当作主方法。主方法应该直接通过 projection posterior 传播 uncertainty。

有两种可行方式。

---

### 7.1 方式 A：两层 Monte Carlo，最忠实

预测时采样：

$$
R^{(s)}\sim q(R),\qquad s=1,\dots,S_R.
$$

然后对每个 $$R^{(s)}$$，再采样：

$$
W_a^{(s,t)}\sim p(W_a\mid R^{(s)}),
\qquad t=1,\dots,S_W.
$$

对 test point $$x_\ast$$，anchor-conditioned projection 为：

$$
W_\ast^{(s,t)}=W(x_\ast;R^{(s)},\xi^{(s,t)}).
$$

然后：

$$
z_i^{(s,t)}=W_\ast^{(s,t)}x_i,
\qquad
z_\ast^{(s,t)}=W_\ast^{(s,t)}x_\ast.
$$

在每个 sampled latent space 中 fit local JGP，得到：

$$
\mu_\ast^{(s,t)},
\qquad
\sigma_\ast^{2(s,t)}.
$$

最终：

$$
\mu_\ast=
\frac{1}{S_RS_W}
\sum_{s=1}^{S_R}
\sum_{t=1}^{S_W}
\mu_\ast^{(s,t)}.
$$

$$
\sigma_\ast^2
=
\frac{1}{S_RS_W}
\sum_{s=1}^{S_R}
\sum_{t=1}^{S_W}
\left[
\sigma_\ast^{2(s,t)}
+
(\mu_\ast^{(s,t)}-\mu_\ast)^2
\right].
$$

这直接对应 law of total variance：

$$
\operatorname{Var}(y_\ast)
=
\mathbb E_{R,W\mid R}[\operatorname{Var}(y_\ast\mid R,W)]
+
\operatorname{Var}_{R,W\mid R}(\mathbb E[y_\ast\mid R,W]).
$$

这个方法最干净，但计算最贵。

---

### 7.2 方式 B：闭式 uncertain-W kernel，推荐实现

给定 $$R$$，如果只使用 $$W\mid R$$ 的 conditional Gaussian uncertainty，可以把它闭式传播到 kernel 里。

令：

$$
\Delta x_{il}=x_i-x_l.
$$

给定 $$R$$：

$$
\Delta z_{il}=W_a\Delta x_{il}.
$$

由于 $$W_a\mid R$$ 是 Gaussian，$$\Delta z_{il}\mid R$$ 也是 Gaussian：

$$
\Delta z_{il}\mid R
\sim
\mathcal N(\mu_{\Delta,il},S_{\Delta,il}).
$$

其中：

$$
\mu_{\Delta,il}
=
\mu_W(c_a\mid R)\Delta x_{il}.
$$

若 $$\operatorname{vec}(W_a)\mid R\sim\mathcal N(m_a(R),\Sigma_{0,a})$$，则：

$$
S_{\Delta,il}
=
A_{il}\Sigma_{0,a}A_{il}^\top,
$$

其中：

$$
A_{il}=I_Q\otimes (\Delta x_{il})^\top.
$$

对 squared exponential kernel：

$$
k_f(z_i,z_l)
=
\sigma_f^2
\exp\left(-\frac{1}{2\ell^2}\|z_i-z_l\|^2\right),
$$

有闭式 expectation：

$$
\bar k_R(i,l)
=
\mathbb E_{W_a\mid R}
[k_f(W_a x_i,W_a x_l)].
$$

结果为：

$$
\bar k_R(i,l)
=
\sigma_f^2
\left|I_Q+\frac{1}{\ell^2}S_{\Delta,il}\right|^{-1/2}
\exp\left(
-rac{1}{2}
\mu_{\Delta,il}^\top
(\ell^2 I_Q+S_{\Delta,il})^{-1}
\mu_{\Delta,il}
\right).
$$

如果使用 anisotropic lengthscale matrix $$\Lambda$$：

$$
k_f(z_i,z_l)
=
\sigma_f^2
\exp\left(-\frac{1}{2}(z_i-z_l)^\top\Lambda^{-1}(z_i-z_l)\right),
$$

则：

$$
\bar k_R(i,l)
=
\sigma_f^2
\left|I_Q+\Lambda^{-1}S_{\Delta,il}\right|^{-1/2}
\exp\left(
-rac{1}{2}
\mu_{\Delta,il}^\top
(\Lambda+S_{\Delta,il})^{-1}
\mu_{\Delta,il}
\right).
$$

然后在 local JGP 中用 uncertain-W kernel matrix：

$$
\bar K_R[i,l]=\bar k_R(i,l).
$$

测试点和训练点之间：

$$
\Delta x_{i\ast}=x_i-x_\ast,
$$

同样得到：

$$
\bar k_{R,\ast}[i]
=
\mathbb E_{W_a\mid R}
[k_f(W_a x_i,W_a x_\ast)].
$$

self kernel：

$$
\bar k_{R,\ast\ast}=\sigma_f^2.
$$

给定 $$R$$ 的 local GP predictive mean 和 variance 近似为：

$$
\mu_\ast(R)
=
\mu_a+
\bar k_{R,\ast}^\top
(\bar K_R+
\sigma_a^2I)^{-1}
(y_I-\mu_a\mathbf 1).
$$

$$
\sigma_\ast^2(R)
=
\bar k_{R,\ast\ast}
-
\bar k_{R,\ast}^\top
(\bar K_R+
\sigma_a^2I)^{-1}
\bar k_{R,\ast}.
$$

这个近似把 $$W\mid R$$ 的 variance 直接传进 local kernel。直观上，当 $$W\mid R$$ 不确定时，$$S_{\Delta,il}$$ 变大，kernel correlation 会被修正，预测 variance 通常会更保守。

然后只需要对 $$R\sim q(R)$$ 做外层 Monte Carlo：

$$
R^{(s)}\sim q(R).
$$

每个 $$R^{(s)}$$ 下用 uncertain-W kernel 得到：

$$
\mu_\ast^{(s)},
\qquad
\sigma_\ast^{2(s)}.
$$

最终聚合：

$$
\mu_\ast=
\frac{1}{S_R}\sum_s\mu_\ast^{(s)}.
$$

$$
\sigma_\ast^2
=
\frac{1}{S_R}\sum_s
\left[
\sigma_\ast^{2(s)}
+
(\mu_\ast^{(s)}-\mu_\ast)^2
\right].
$$

这样同时传播了：

1. $$W\mid R$$ 的 conditional GP uncertainty，进入 $$\bar K_R$$；
2. $$R\sim q(R)$$ 的 global posterior uncertainty，通过外层 MC 聚合。

这是比 post-hoc calibration 更符合 GP spirit 的 UQ 方案。

---

## 8. Gate uncertainty under W|R

gate logit 为：

$$
\xi_{ai}=\nu_a^\top
\begin{bmatrix}
1\\
W_a(x_i-c_a)
\end{bmatrix}.
$$

给定 $$R$$，由于 $$W_a\mid R$$ Gaussian：

$$
\xi_{ai}\mid R \sim \mathcal N(m_{\xi,ai},s_{\xi,ai}^2).
$$

其中：

$$
m_{\xi,ai}
=
\nu_{a0}
+
\nu_{a,1:Q}^\top
\mu_W(c_a\mid R)(x_i-c_a).
$$

$$
s_{\xi,ai}^2
=
B_{ai}\Sigma_{0,a}B_{ai}^\top,
$$

其中：

$$
B_{ai}
=
\nu_{a,1:Q}^\top
\otimes
(x_i-c_a)^\top.
$$

需要计算：

$$
\mathbb E_{W_a\mid R}[\log \sigma(\xi_{ai})]
$$

和：

$$
\mathbb E_{W_a\mid R}[\log(1-\sigma(\xi_{ai}))].
$$

这没有简单 elementary closed form，但可以用低阶 Gauss-Hermite quadrature。若只需要 inlier probability，可以用 logistic-normal approximation：

$$
\mathbb E[\sigma(\xi)]
\approx
\sigma\left(
\frac{m_\xi}{\sqrt{1+\pi s_\xi^2/8}}
\right).
$$

这会让 gate 在 $$W\mid R$$ 不确定时变软，避免过度 confident CEM labels。

---

## 9. 为什么不用 calibration / conformal 作为核心

Calibration 或 conformal 可以作为最终报告的附加 sanity check，但不应该是主方法核心。否则确实会产生问题：如果任何模型都能后处理校准，那就很难说明 GP/JGP 的 UQ 有独特价值。

本方法的核心 UQ 应该来自：

$$
q(R)\quad\text{和}\quad p(W\mid R).
$$

也就是：

$$
\text{projection field uncertainty}
\to
\text{kernel uncertainty}
\to
\text{JGP predictive uncertainty}.
$$

推荐主结果优先报告不做 conformal 的版本。如果 coverage 仍偏低，再讨论 calibration 作为工程补救，而不是方法核心。

---

## 10. 最推荐的最终简化算法

### Training

1. 选 anchors $$\mathcal A$$。
2. 对每个 anchor 固定 raw-X neighborhood $$N_a$$。
3. 初始化 $$q(R)$$，推荐 PCA/PLS init。
4. C-step：用 $$\mathbb E[R]$$ 和 $$\mu_W(c_a\mid \mathbb E[R])$$ 投影 local data，跑 JGP-CEM/VEM，得到 $$\rho_{ai}$$。
5. 构造 pairwise labels $$s_{ail}$$。
6. M-step：最大化

$$
\mathcal L
=
\tau_{\text{jgp}}
\sum_a
\mathbb E_{q(R)p(W_a\mid R)}
[\ell_{\text{JGP},a}]
+
\lambda_{\text{pair}}
\sum_a
\mathbb E_{q(R)p(W_a\mid R)}
[\ell_{\text{pair},a}]
-
\beta KL(q(R)\|p(R)).
$$

7. 训练前期使用 pairwise warm start，后期 joint fine-tuning。

### Prediction

对每个 test point：

1. sample $$R^{(s)}\sim q(R)$$。
2. 对每个 $$R^{(s)}$$，用 uncertain-W kernel 闭式整合 $$W\mid R^{(s)}$$。
3. 在该 kernel 下 fit local JGP / local GP on inliers。
4. 得到 $$\mu_\ast^{(s)},\sigma_\ast^{2(s)}$$。
5. 用 total variance 聚合。

---

## 11. 最重要的 ablation

必须比较：

1. 原 LMJGP。
2. 原 LMJGP learned $$q(W_j)$$ vs prior $$q(W_j)$$。
3. Pairwise-only。
4. Pairwise warm start + JGP fine-tune。
5. Joint pairwise + JGP。
6. 只用 $$R$$ samples，不整合 $$W\mid R$$。
7. 整合 $$W\mid R$$ uncertain kernel。
8. anchor-conditioned $$W(x_\ast)$$ vs pointwise $$W(x_i)$$。

关键判断：

如果加入 $$W\mid R$$ uncertain kernel 后 Cov90 明显上升且 CRPS 不崩，说明 UQ 传播是有效的。

如果 joint pairwise + JGP 优于 pairwise-only，说明 JGP likelihood fine-tuning 有用。

如果 anchor-conditioned 版本优于 pointwise transfer，说明该方法确实更适合 DJGP 而不是 DMGP-style embedding。

