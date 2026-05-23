# LMJGP / DJGP Original ELBO 与 Direct Analytic Prediction：诊断、已尝试 remedy、暂时结论与后续路线

更新时间：2026-05-19

本文档总结当前围绕 LMJGP / DJGP 的 original analytic ELBO、direct VI analytic prediction head、tempered responsibility、two-timescale W-learning 等问题的讨论和实验结论。

---

## 0. 当前问题的总览

我们最初希望 original analytic ELBO 同时完成两件事：

1. 学到有用的 projection posterior：

\[
q(R) \rightarrow q(W_j),
\]

使 sampled-\(W\) + local JGP-CEM prediction 更好。

2. 直接通过 analytic VI head 做预测：

\[
\hat y_j = m_j + \zeta_j,
\qquad
\zeta_j = \Psi_1 K^{-1}\mu_u.
\]

但实验显示，original ELBO 的 direct analytic path 存在多层 shortcut / bad basin：

\[
q(R) \rightarrow q(W_j) \rightarrow \Psi_1,\Psi_2 \rightarrow \Phi^T r \rightarrow \mu_u \rightarrow \zeta \rightarrow \rho \rightarrow \text{prediction}.
\]

其中任何一环都可能断掉。当前逐步诊断后的核心结论是：

\[
\boxed{
\text{direct analytic path 可以被修到“学得动”，但不一定提升 test mean；}
}
\]

\[
\boxed{
\text{真正 paper-relevant 的目标应回到：学到对 sampled-W+CEM 有用的 }q(W).
}
}
\]

---

## 1. 模型与关键数学对象

### 1.1 Local residual form

令

\[
r = y - m_j,
\qquad
\Phi = \Psi_1 K^{-1},
\qquad
\zeta = \Phi\mu_u.
\]

其中：

- \(m_j\)：local mean；
- \(r\)：local residual；
- \(\Psi_1=\mathbb E_{q(W_j)}[K_{fu}]\)；
- \(\Psi_2=\mathbb E_{q(W_j)}[K_{uf}K_{fu}]\)；
- \(\mu_u\)：local inducing residual GP mean；
- \(\zeta\)：direct analytic residual GP prediction on local training points。

在 fixed \(W,\Psi_1,\Psi_2,m_j,\sigma_n\) 下，local residual GP 的 quadratic objective 给出近似 stationary point：

\[
\mu_u^*
=
(G+\sigma_n^2K^{-1})^{-1}\Phi^T r,
\]

其中

\[
G = \sum_i K^{-1}\Psi_{2,i}K^{-1}.
\]

因此：

\[
\zeta_* = \Phi \mu_u^*.
\]

这个式子是后续所有 shrinkage 诊断的核心。

---

### 1.2 三个 z 指标

#### `z_learn/r`

\[
\frac{\|\zeta_{learn}\|}{\|r\|}
= 
\frac{\|\Phi\mu_u\|}{\|y-m_j\|}.
\]

表示当前 learned \(\mu_u\) 产生的 residual GP mean 占 residual 的比例。

#### `z_star/r`

\[
\frac{\|\zeta_*\|}{\|r\|}
= 
\frac{\|\Phi(G+\sigma_n^2K^{-1})^{-1}\Phi^T r\|}{\|r\|}.
\]

表示在当前 ELBO quadratic 下，若只优化 \(\mu_u\)，理论上能得到的 residual GP signal。

#### `z_ridge/r`

用 ridge regression 在 \(\Phi\) 上拟合 residual 的 upper diagnostic：

\[
b_{ridge}=\arg\min_b \|r-\Phi b\|^2+\lambda\|b\|^2,
\qquad
\zeta_{ridge}=\Phi b_{ridge}.
\]

如果 `z_ridge/r` 高而 `z_star/r` 低，说明 \(\Phi\) 有信号，但 ELBO 的 \(G+\sigma_n^2K^{-1}\) 把 signal 压住。

---

### 1.3 Mixture responsibility 与 \(r_{mix}\)

Mixture 下每个点的 posterior inlier responsibility 为：

\[
\rho_i
=
\frac{\exp(S_{1,i})}{\exp(S_{1,i})+\exp(S_{2,i})}
=
\sigma(S_{1,i}-S_{2,i}).
\]

平均 responsibility：

\[
r_{mix}=\bar\rho=\frac1n\sum_i \rho_i.
\]

注意：

- `z_star/r` 里的 \(r\) 是 residual \(y-m_j\)；
- `r_mix` 里的 \(r\) 是 responsibility 的简写；
- 二者不是同一个对象。

---

## 2. 第一层问题：scale、local mean 与 noise escape

### 2.1 Raw response scale 问题

原始 y scale 很大时，inlier quadratic：

\[
Q_i\approx \frac{(y_i-\mu_i)^2}{2\sigma_n^2}
\]

会非常大，导致 early stage：

\[
S_1 \ll S_2,
\qquad
\rho_i\approx0.
\]

这解释了最初 gate collapse 和巨大 negative ELBO。

已修复：

- `--standardize_y`；
- 显式 local mean \(m_j\)；
- residual form：\(r=y-m_j\)。

结论：scale 和 local mean 是必要修复，但不是最终瓶颈。

---

### 2.2 \(\sigma_n\) 是什么

\(\sigma_n\) 是 observation noise 的标准差，不是 variance。进入 likelihood 和 shrinkage 的是：

\[
\sigma_n^2.
\]

在 local inlier likelihood 中：

\[
y_i \mid f_i, v_i=1 \sim N(f_i,\sigma_n^2).
\]

它有两个作用：

1. 控制 residual quadratic 的惩罚强度：

\[
Q_i\propto \frac1{\sigma_n^2}.
\]

2. 控制 \(\mu_u^*\) denominator 中的 KL/prior shrinkage：

\[
\mu_u^*
=
(G+\sigma_n^2K^{-1})^{-1}\Phi^T r.
\]

\(\sigma_n\) 大时，\(\sigma_n^2K^{-1}\) 变大，\(\mu_u^*\) 被压小。

---

### 2.3 Shrinkage 分解诊断

诊断形式：

\[
z_*(\beta_G,\beta_K)
=
\Phi(\beta_GG+\beta_K\sigma_n^2K^{-1})^{-1}\Phi^Tr.
\]

主要结果：

- baseline D3 at step 300：

\[
z_*/r\approx0.0045,
\qquad
z_{learn}/r\approx0.0047,
\qquad
\sigma_n\approx1.29.
\]

- 降低 \(\beta_K\) 明显提高 `z_star/r`；
- 降低 \(\beta_G\) 也有帮助，但在 \(\beta_K=1\) 时单独降 \(G\) 作用有限；
- `z_ridge/r` 明显高于 `z_star/r`。

结论：

\[
\boxed{
\sigma_n^2K^{-1}\text{ 是一级锁；}G/\Psi_2\text{ 是二级锁。}
}
\]

---

### 2.4 固定 \(\sigma_n\) 诊断

固定 \(\sigma_n\) 后结果：

| 变体 | \(\sigma_n\) | z_star/r | z_learn/r | r_mix |
|---|---:|---:|---:|---:|
| snfix0.2 | 0.20 | 0.090 | 0.038 | 0.18 |
| snfix0.3 | 0.30 | 0.070 | 0.036 | 0.25 |
| snfix0.5 | 0.50 | 0.043 | 0.027 | 0.34 |
| snmax0.8 | 0.79 | 0.018 | 0.013 | — |

结论：

\[
\boxed{
固定较小 \sigma_n \text{ 能解开 pure-inlier shrinkage。}
}
\]

但副作用是：\(\sigma_n\) 越小，inlier branch 越严格，模型更容易把 high-residual 点推给 outlier。

---

## 3. 第二层问题：mixture responsibility selective filtering

### 3.1 Batch \(\rho\)-floor 的失败

尝试：

\[
\bar\rho\ge\rho_{min}.
\]

实现为：

\[
-\lambda_\rho\operatorname{softplus}(\rho_{min}-\bar\rho)^2.
\]

结果：

| 变体 | z_learn/r | z*pure/r | z*ρ/r | \(\bar\rho\) |
|---|---:|---:|---:|---:|
| snfix0.3_rhofloor0.3 | 0.0046 | 0.054 | 0.002 | 0.71 |
| lam5 | 0.0046 | 0.054 | 0.002 | 0.71 |
| refresh0.3 | 0.0029 | 0.054 | 0.002 | 0.72 |

解释：

- floor 成功提高了平均 inlier 数量；
- 但 mixture-consistent star 仍接近 0；
- high `z*pure/r` 是错误口径，不能代表 mixture 下真实学习空间。

---

### 3.2 为什么高 \(\bar\rho\) 不等于高 GP signal

Mixture-consistent star 是：

\[
\mu_u^{*,\rho}
=
(G_\rho+\sigma_n^2K^{-1})^{-1}\Phi^TD_\rho r,
\]

其中：

\[
G_\rho=\sum_i\rho_iK^{-1}\Psi_{2,i}K^{-1}.
\]

核心 RHS 是：

\[
\Phi^TD_\rho r.
\]

如果 \(\rho_i\) 选择性给 low-residual 点高权重，给 high-residual 点低权重，则：

\[
\|\Phi^TD_\rho r\|\ll \bar\rho\|\Phi^Tr\|.
\]

实验证据：

| 指标 | mixture+floor |
|---|---:|
| \(B_\rho/(\bar\rho B_{pure})\) | 0.027 |
| corr(\(\rho,|r|\)) | -0.93 |
| weighted r² / unweighted r² | 0.003 |
| cos(\(\Phi^TD_\rho r, \Phi^Tr\)) | -0.15 |

结论：

\[
\boxed{
\rho\text{ 在做选择性删信号：high residual 点被系统性降权。}
}
\]

这是一种 objective shortcut：与其让 local GP 解释 high residual，不如把 high residual 点移出 inlier branch。

---

## 4. 第三层 remedy：tempered / softened responsibility

### 4.1 Pure warmup

先用 pure-inlier 训练，不让 mixture 早期自选择 \(\rho\)：

\[
\tilde\rho_i=1.
\]

30×120 结果：

| 变体 | z_learn/r | z*ρ/r | Br/Bp | RMSE |
|---|---:|---:|---:|---:|
| uniform_warm | 0.022 | 0.014 | 0.15 | 1.85 |
| rhofloor | 0.005 | 0.002 | 0.02 | 2.34 |

结论：pure warmup 能明显缓解 selective filtering，但 abrupt release 到 full mixture 后仍会塌。

---

### 4.2 Softened \(\rho\) blend curriculum

核心设计：

\[
\tilde\rho_i(t)
=(1-\alpha_t)\rho_i^0+\alpha_t\rho_i^{ELBO}.
\]

最有效设置：

\[
\rho_i^0=1,
\qquad
\alpha_t:0\rightarrow0.5.
\]

30×120 结果：

| 变体 | z_learn/r | z*tilde/r | z*rho/r | Br/Bp | RMSE |
|---|---:|---:|---:|---:|---:|
| warm80 → α=0.5 | 0.030 | 0.047 | 0.014 | 0.49 | 1.79 |
| warm80 → α=1.0 | 0.021 | 0.005 | 0.005 | 0.05 | 1.84 |
| pointblend γ=0.3 | 0.014 | 0.022 | 0.007 | 0.29 | 1.77 |
| rhofloor | 0.005 | 0.002 | 0.002 | 0.02 | 2.34 |

结论：

\[
\boxed{
full collapsed mixture endpoint \alpha=1 \text{ 会重新删掉 residual GP signal。}
}
\]

\[
\boxed{
\alpha_{end}\approx0.5 \text{ 是当前最稳的 tempered responsibility 设置。}
}
\]

---

### 4.3 200×300 full 结果

| 变体 | z_learn/r | z*tilde/r | Br/Bp | SSEg | RMSE |
|---|---:|---:|---:|---:|---:|
| pure | 0.186 | — | — | 0.12 | 1.75 |
| warm120→0.5 | 0.152 | 0.188 | 0.52 | 0.10 | 1.79 |
| warm120→0.7 | 0.120 | 0.138 | 0.32 | 0.07 | 1.82 |
| warm120→0.5 + refresh | 0.205 | 0.207 | 0.52 | 0.14 | 1.79 |

结论：

\[
\boxed{
\sigma_n\text{ fixed} + pure warmup + \tilde\rho\text{ blend to }0.5 + \mu_u^*\text{ refresh}
}
\]

能使 direct residual GP path 真正学起来。

但 RMSE 没有明显优于 pure/local mean baseline，说明：

\[
\boxed{
训练侧 residual fit 改善 \not\Rightarrow test mean RMSE 改善。
}
\]

---

## 5. 第四层问题：train residual fit 与 test-anchor correction 不对齐

### 5.1 Test-anchor alignment 诊断

定义 local mean baseline test error：

\[
e_0 = y_* - m_j.
\]

direct analytic correction 为 \(\zeta_*\)。

MSE 改变量：

\[
(e_0-\zeta_*)^2-e_0^2
=
\zeta_*^2-2e_0\zeta_*.
\]

所以 \(\zeta_*\) 变大只有在：

\[
e_0\zeta_* > \frac12\zeta_*^2
\]

时才会改善 test RMSE。

诊断结果：

| 指标 | full + refresh | freeze_W |
|---|---:|---:|
| train_SSEg | 0.18 | 0.07 |
| test_SSEg | 0.015 | 0.012 |
| cos(\(e_0,\zeta\)) | 0.17 | 0.15 |
| W_drift | 1.06 | 0.10 |
| RMSE_raw | 765 | 748 |
| RMSE@W0 | 740 | 748 |
| RMSE_cem | 755 | 687 |

结论：

\[
\boxed{
\zeta \text{ 在 training residual 上学起来了，但对 test anchor 的方向弱对齐。}
}
\]

---

### 5.2 W drift 伤 direct head 和 CEM

full joint training：

\[
W_{drift}\approx1.06.
\]

这时 direct RMSE 和 CEM RMSE 都较差。

freeze-W：

\[
W_{drift}\approx0.10,
\]

direct RMSE 恶化较小，CEM RMSE 显著更好。

说明：

\[
\boxed{
当前 analytic ELBO 的 W-gradient 与 downstream CEM/test prediction 不完全对齐。}
}
\]

不是 W 完全不能学，而是自由 joint update 容易把 \(W\) 从较好的 \(W_0\) 拉偏。

---

## 6. 第五层 remedy：two-timescale / envelope-style W learning

### 6.1 为什么需要 two-timescale

我们真正想优化的是 profile objective：

\[
F(W)=\max_\theta \mathcal L(W,\theta),
\]

其中 \(\theta\) 包括：

\[
\theta=\{\mu_u,\Sigma_u,m_j,\rho,\sigma_n,gate,Z,\sigma_k,\ldots\}.
\]

但 joint Adam 实际做的是：

\[
W_{t+1}=W_t+\eta_W\nabla_W\mathcal L(W_t,\theta_t),
\]

其中 \(\theta_t\neq\theta^*(W_t)\)。

如果 \(W\) 更新太快，local GP / \(\rho\) / \(m_j\) 还没适应当前 \(W\)，则 \(W\)-gradient 可能是 stale / lagged 的。

---

### 6.2 已实现 two-timescale training loop

每个 outer step：

1. Fast inner：只更新 \(\mu_u,\Sigma_u\)，默认 5 次；
2. Refresh：

\[
\mu_u\leftarrow(1-\alpha)\mu_u+\alpha\mu_u^*(\tilde\rho);
\]

3. Slow W：只更新 \(\mu_B,\sigma_B\)，学习率：

\[
lr_W=lr\times lr_{W,scale};
\]

4. 可选 refresh after W；
5. Slow other：每隔若干步更新 \(m_j\), gate, inducing Z, lengthscales, \(\sigma_k\) 等。

---

### 6.3 Smoke 结果

| 变体 | W_drift | z_learn/r | test_SSEg | RMSE_raw | RMSE_cem |
|---|---:|---:|---:|---:|---:|
| slowW5x | 0.00 | 0.06 | 0.006 | 752 | 641 |
| slowW5x_trust | 0.58 | 0.19 | 0.015 | 762 | 680 |
| joint refresh | 1.06 | 0.25 | 0.015 | 765 | 755 |

解读：

- joint training：W drift 大，CEM 变差；
- slowW5x：几乎冻结 W，但 CEM 最好；
- trust 允许 moderate drift，优于 joint 但不如 freeze-like slowW；
- 当前最好结果主要来自 \(W\approx W_0\) + fast local adaptation。

暂时结论：

\[
\boxed{
当前 analytic ELBO 的 W update 必须非常慢、受限，或者通过 line search/acceptance 控制。}
}
\]

---

## 7. 当前最可能的问题链条

从目前所有诊断看，问题可分层理解：

### Layer 1：raw scale / missing local mean

已基本修复。

### Layer 2：noise escape / shrinkage

\[
\sigma_n\uparrow \Rightarrow \sigma_n^2K^{-1}\uparrow \Rightarrow \zeta_*\downarrow.
\]

fixed/capped \(\sigma_n\) 有效。

### Layer 3：mixture responsibility selective filtering

\[
\rho_i\downarrow \text{ for high } |r_i|,
\quad
\Phi^TD_\rho r \approx0.
\]

tempered \(\tilde\rho\) 有效。

### Layer 4：training residual fit 不等于 test anchor improvement

\[
train\_SSEg \uparrow
\not\Rightarrow
 test\_SSEg \uparrow.
\]

direct head 的 \(\zeta_*\) 与 test residual 弱对齐。

### Layer 5：W drift / proxy mismatch

\[
\nabla_W\mathcal L_{analytic}
\not\parallel
\nabla_W \text{CEM/test prediction quality}.
\]

slowW/freezeW 对 CEM 更好。

---

## 8. 已尝试 tricks 与暂时结论表

| Trick | 目的 | 结果 | 当前判断 |
|---|---|---|---|
| standardize_y | 修 scale | 有效 | 必须保留 |
| explicit \(m_j\) | 修 local mean | 有效但不充分 | 必须保留，且需防 drift |
| \(s=x^T\Sigma_Wx\) penalty | 降低 E[K] smoothing | 降 \(s\)，RMSE 不动 | 非主线 |
| \(\sigma_V\) anneal/clamp | 集中 q(W) | variance 降，RMSE 不动 | 非主线 |
| gate stabilization | 防 gate collapse | 单独无效 | gate 是后果 |
| fixed gate slope | 减少 W/ν non-identifiability | gate variation 改，ζ不动 | 等 local signal 修好后再看 |
| D3 structured posterior | 恢复 W geometry | std(W) 有变化，但 ζ弱 | 有帮助但不够 |
| shrinkage β table | 分解 G/K | 定位 \(\sigma_n^2K^{-1}\) | 关键诊断 |
| snfix/snmax | 堵 noise escape | z* 大幅上升 | 有效 diagnostic/remedy |
| batch rho floor | 堵 outlier escape | \(\bar\rho\) 高但 signal 低 | 不够，quota gaming |
| rho anchor penalty | 限制 rho shape | 暂时无效 | 低优先级 |
| pure warmup | 防 early selective filtering | 有效 | 高优先级 |
| soft rho blend to 0.5 | tempered responsibility | 显著有效 | 当前主 remedy |
| alpha -> 1 | 恢复 full mixture | 重新塌 | 不建议 |
| mu-star refresh | 让 μu 跟上 star | 有效 | 推荐保留 |
| freeze-W | 检查 W drift | direct/CEM 更好 | 强诊断 |
| slowW/two-timescale | 减少 lagged W gradient | CEM 最好在 near-freeze | 主线之一 |
| trust W | 控制 drift | 比 joint 好但不如 freeze-like | 需要 sweep |
| sampled-W+CEM | 主 prediction 路线 | 比 direct head 好 | 应作为最终验证 |

---

## 9. 下一步建议

### 9.1 不再优先追更高 `z_learn/r`

当前已经能达到：

\[
z_{learn}/r\approx0.15\sim0.20.
\]

但 RMSE 不一定改善。继续推高 \(\zeta\) 可能只是更强地拟合 local training residual。

新的目标应改为：

\[
\boxed{
\text{test-anchor alignment} + \text{CEM downstream performance}.
}
}
\]

---

### 9.2 优先实验 1：W drift grid / trust region

固定当前 best training scaffold：

\[
\sigma_n=0.3,
\quad
pure\ warmup,
\quad
\tilde\rho\to0.5,
\quad
\mu_u^*\ refresh,
\quad
two-timescale.
\]

扫：

\[
lr_{W,scale}\in\{0,0.02,0.05,0.1,0.2\},
\]

\[
\lambda_W\in\{0,5,20\}.
\]

目标：找 CEM RMSE 对 W_drift 的曲线。

判读：

- 最佳在 W_drift≈0：当前 ELBO 不应学 W mean；
- 小 drift 最好：trust-region W learning 有价值；
- 大 drift 最好：说明之前只是 timescale 太慢。

---

### 9.3 优先实验 2：W interpolation diagnostic

设：

\[
W(t)=W_0+t(W_{learned}-W_0),
\qquad
 t\in[0,1].
\]

评估：

\[
\mathcal L_{ELBO}(W(t)),
\quad
RMSE_{direct}(W(t)),
\quad
RMSE_{CEM}(W(t)).
\]

如果 \(t\uparrow\) 时 ELBO 上升而 CEM RMSE 变差，则明确说明：

\[
\boxed{
analytic ELBO 的 W-objective 与 downstream CEM prediction mismatch。
}
}
\]

---

### 9.4 优先实验 3：W line search / profile step

每个 outer step：

1. 固定 W，inner 更新 local 参数 10–20 步；
2. 计算 W 梯度方向 \(d_W\)；
3. 尝试：

\[
W(\tau)=W+\tau d_W,
\quad
\tau\in\{0,0.25,0.5,1,2,4\}\eta_W;
\]

4. 对每个 candidate 做 \(\mu_u^*_{\tilde\rho}\) refresh；
5. 用 cheap ELBO+trust 或 local heldout/CEM proxy 选 \(\tau\)。

记录：

\[
P(\tau=0),
\quad
E[\tau],
\quad
RMSE_{CEM},
\quad
W_{drift}.
\]

这能判断：是否存在沿当前 W-gradient 的有益非零 step。

---

### 9.5 优先实验 4：local heldout / cross-fit objective

当前 train_SSEg 与 test_SSEg 分离。建议把 local neighborhood 分成：

\[
D_{fit},D_{val}.
\]

训练 local 参数用 \(D_{fit}\)，选择 W / early stop 用 \(D_{val}\)。

目标：避免 W 和 \(\zeta\) 只提高 local training residual fit。

---

### 9.6 优先实验 5：sampled-W+CEM 作为主评估

因为 manuscript 主 prediction 是：

\[
W^{(m)}\sim q(W_j),
\quad
\text{fit local JGP-CEM on } W^{(m)}X,
\quad
\text{MC aggregate}.
\]

所以所有 training remedy 最终要看：

\[
RMSE_{CEM},
\quad
CRPS_{CEM}.
\]

direct analytic head 可以作为 diagnostic，不应作为主 prediction 结论。

---

## 10. 当前建议的主 candidate

短期主 candidate：

\[
\boxed{
\sigma_n=0.3\ fixed
+ pure\ warmup
+ \tilde\rho\to0.5
+ \mu_u^*_{\tilde\rho}\ refresh
+ two\text{-}timescale
+ W\ trust/line\ search.
}
}
\]

更具体：

1. \(\sigma_n=0.3\) 或 cap/prior around 0.3；
2. 前 120 step pure-inlier：\(\tilde\rho=1\)；
3. 120–300/600 step：\(\alpha:0\to0.5\)；
4. 周期性 refresh：

\[
\mu_u\leftarrow(1-\eta)\mu_u+\eta\mu_u^*_{\tilde\rho};
\]

5. W update 使用 slow/two-timescale；
6. 尝试 small nonzero W learning with trust or line search；
7. final prediction 使用 sampled-W+CEM。

---

## 11. Paper-level implication

当前 evidence 支持如下叙事：

1. original analytic ELBO 不是单纯实现 bug；
2. 它作为 composite variational pseudo-objective 有多个 shortcut；
3. direct analytic VI prediction head 容易被 noise、responsibility、W-drift 和 test-alignment 问题影响；
4. sampled-W + local JGP-CEM prediction 更符合方法本意；
5. direct analytic head 更适合作为 diagnostic / ablation；
6. 若要强化 VI training，应采用 tempered responsibility、noise control、\(\mu_u\) coordinate refresh、two-timescale W learning、trust/line-search 等机制。

可在 manuscript 中表达为：

> The variational objective is primarily used to infer a locally coherent posterior over projection matrices. Because the collapsed analytic prediction head can be sensitive to responsibility self-selection and projection-drift shortcuts, final prediction is performed by sampling projections from the learned posterior and refitting local JumpGPs, which better aligns with the downstream piecewise-continuous surrogate task.

---

## 12. 一句话总结

当前我们已经确认：

\[
\boxed{
\text{原始 ELBO direct head 的失败不是“没学”，而是“学的目标和 test/CEM 不完全对齐”。}
}
\]

已经修复或定位：

- raw scale / local mean；
- \(\sigma_n\) escape；
- \(G+\sigma_n^2K^{-1}\) shrinkage；
- mixture responsibility selective filtering；
- learned-star gap；
- train/test residual correction mismatch；
- W drift 与 downstream CEM mismatch。

当前最重要的后续方向：

\[
\boxed{
\text{不要再只追 }z/r；
\text{要验证并优化 }W\text{ 对 sampled-W+CEM prediction 的实际价值。}
}
\]
