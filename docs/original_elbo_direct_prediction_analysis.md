# Original ELBO + Direct Analytic Prediction：问题诊断、已尝试方法与后续路线总结

## 0. 背景与目标

当前讨论对象是 LMJGP / DJGP 中的 **original analytic ELBO** 与 **direct VI analytic prediction head**。

核心路径是：

\[
q(R) \rightarrow q(W_j) \rightarrow \Psi_1,\Psi_2 \rightarrow \zeta_j=\Psi_1K^{-1}\mu_u \rightarrow p_{\mathrm{in}}(y\mid x) \rightarrow S_1/S_2 \rightarrow \rho.
\]

其中：

- \(q(R)\)：global inducing projection posterior；
- \(q(W_j)\)：由 \(q(R)\) 和 GP conditional induced 出来的 local projection posterior；
- \(\Psi_1=\mathbb E_{q(W_j)}[K_{fu}]\)，\(\Psi_2=\mathbb E_{q(W_j)}[K_{uf}K_{fu}]\)；
- \(\zeta_i = [\Psi_1 K^{-1}\mu_u]_i\)：local inlier GP residual mean；
- \(S_1,S_2\)：inlier / outlier 两个分支的 log-score；
- \(\rho=\operatorname{softmax}(S_1,S_2)\)：数据后验 inlier responsibility。

目标是理解为什么：

1. ELBO 可以明显改善；
2. direct VI prediction 的 RMSE 却几乎不动；
3. gate / inlier GP / projection posterior 哪一层断了；
4. 未来如何让 original ELBO 真正给 \(W\) 和 local GP 提供有效监督信号。

---

## 1. 当前已确认的主要问题

### 1.1 Raw response scale 导致 inlier likelihood 爆炸

原始 LH 数据中 \(y\) 量级约为 \(O(10^2\sim10^3)\)，而早期 local GP mean \(\mu_f\) 和噪声 \(\sigma_n\) 通常是 \(O(1)\)。因此 inlier quadratic：

\[
Q_i \approx \frac{(y_i-\mu_i)^2}{2\sigma_n^2}
\]

会自然到 \(10^6\) 量级。例如：

\[
y\approx 800,\quad \mu\approx 1,\quad \sigma_n\approx 0.5
\]

则：

\[
Q \approx \frac{800^2}{2\cdot 0.5^2}\approx 1.28\times 10^6.
\]

这解释了 smoke 中的：

\[
\Delta T = T_{\mathrm{in}}-T_{\mathrm{out}}\approx -1.4\times 10^6.
\]

这不是神秘的 mixture 现象，而是 likelihood 尺度失衡。

诊断结果：

- 加 `--standardize_y` 后，ELBO 从 \(-10^7\) 量级变为 \(O(10^2)\)；
- 说明 response scale 是第一层硬问题；
- 但标准化后 pure inlier 仍基本等于 local mean oracle，说明还有更深层问题。

---

### 1.2 缺少或未充分处理显式 local mean \(m_j\)

如果模型写成：

\[
f_j(z)=m_j+\sigma_{k,j}g_j(z),\qquad g_j\sim GP(0,C),
\]

那么 inlier quadratic 应该写成 residual form：

\[
Q_{j,i}
=
\frac{
(y_i^{(j)}-m_j)^2
-2(y_i^{(j)}-m_j)\zeta_{j,i}
+V_{1,j,i}+B_{j,i}
}{2\sigma_{n,j}^2}.
\]

如果 ELBO 中没有显式 \(m_j\)，除非代码已经对 \(y\) 做了 local centering，否则模型相当于让 GP residual 去拟合一个几百量级的 local plane。这会导致 inlier likelihood 初期极差。

已做诊断：

- `--local_center_y`：每个 region 用 \(y-\bar y_{N_j}\) 训练 residual，预测时加回邻域均值；
- 结果：oracle / pure inlier / fixedW 都约等于 local mean oracle，RMSE raw 约 737；
- 说明 local mean 是必要修复，但 residual GP 仍然没学到明显额外结构。

---

### 1.3 Gate collapse 是后果，不是唯一根因

原始观察：

\[
\pi\rightarrow 0.001,\qquad r\approx 0,
\]

最初看起来像 gate collapse。但后续 forced gate / pure inlier 诊断显示：

\[
\mathrm{RMSE}(\pi=1)\approx \mathrm{RMSE}(\text{mixture}),
\]

而且：

\[
\operatorname{std}(\mu_f)\approx 0.
\]

因此不是 “好的 GP 被 gate 挡住”，而是 **inlier GP path 自己也没有学出非平凡预测**。

数学上，responsibility 为：

\[
r_i
=
\frac{\exp(T_{\mathrm{in},i})}
{\exp(T_{\mathrm{in},i})+\exp(T_{\mathrm{out},i})}.
\]

若 early stage 中：

\[
T_{\mathrm{in}}\ll T_{\mathrm{out}},
\]

则：

\[
r_i\approx 0.
\]

GP 路径梯度被 \(r_i\) 乘掉，形成自锁：

\[
\text{bad inlier}\Rightarrow r\approx0\Rightarrow \text{GP no gradient}\Rightarrow \text{bad inlier}.
\]

但当前更深层的问题是：即使关掉 gate，inlier GP mean path 仍然很弱。

---

### 1.4 Analytic \(E_q[K]\) 会把 kernel feature 平滑，但不是“必然无效”

RBF kernel expectation 下，若：

\[
w_k\sim N(\mu_k,\Sigma_k),\qquad z_{ik}=w_k^Tx_i,
\]

则：

\[
z_{ik}\sim N(m_{ik},s_{ik}),
\quad
m_{ik}=\mu_k^Tx_i,
\quad
s_{ik}=x_i^T\Sigma_kx_i.
\]

对应的 expected kernel 大致是：

\[
\Psi_{1,i\ell}
=
\prod_{k=1}^Q
\frac{1}{\sqrt{1+s_{ik}}}
\exp\left(
-
\frac{(m_{ik}-\tilde z_{\ell k})^2}
{2(1+s_{ik})}
\right).
\]

这说明：

1. \(s_{ik}\) 大时，distance contrast 被 \(1+s_{ik}\) 平滑；
2. 每个 coordinate 有 shrinkage factor \((1+s_{ik})^{-1/2}\)；
3. 多维 \(Q\) 下 shrinkage 会相乘；
4. 若 \(\mu_W\approx0\) 且局部 \(s_i\) 相近，则 \(\Psi_1(i,:)\) 对不同 \(i\) 近似相同。

若：

\[
\Psi_1 \approx \mathbf 1 c^T,
\]

则：

\[
\zeta=
\Psi_1K^{-1}\mu_u
\approx
\mathbf 1(c^TK^{-1}\mu_u),
\]

即 local GP 只能表达常数。

但是这个现象不意味着 analytic marginalization 本身必然失败。理想情况下，若 \(q(W_j)\) 集中在有意义的 local classifier / metric 附近，并且 \(s_i\) 小，则 \(E_q[K]\) 仍然应该有结构。当前失败说明：**ELBO/parameterization/optimization 没有把 posterior 推到这个好 basin。**

---

### 1.5 \(\Phi^Tr\) 是 y 传到 local GP estimate 的关键瓶颈

令：

\[
r=y-m_j,
\qquad
\Phi=\Psi_1K^{-1},
\qquad
\zeta=\Phi\mu_u.
\]

在 pure-inlier quadratic 中，对 \(\mu_u\) 的 stationary point 近似满足：

\[
\mu_u^*
=
(G+\sigma_n^2K^{-1})^{-1}\Phi^Tr,
\]

其中：

\[
G=\sum_i K^{-1}\Psi_{2,i}K^{-1}.
\]

因此，\(y\) 传到 \(\mu_u\) 的直接通道是：

\[
\boxed{\Phi^Tr}.
\]

如果 \(\Phi^Tr\approx0\)，则：

\[
\mu_u^*\approx0,
\qquad
\zeta\approx0.
\]

即使 \(\Psi_1\) 有 row variation，只要该 variation 与 residual \(r\) 不对齐，local GP 仍然不会学到。

---

### 1.6 Learned gate slope \(\nu\) 与 \(W\) 存在 non-identifiability

Gate logit：

\[
h_i=\nu_0+\nu^TWx_i.
\]

真正的原始空间 gate normal 是：

\[
a=W^T\nu.
\]

因此很多 \((W,\nu)\) 组合给出同一个 gate boundary。例如对可逆变换 \(A\)：

\[
W'=AW,
\qquad
\nu'=A^{-T}\nu,
\]

则：

\[
(W')^T\nu'=W^T\nu.
\]

这会导致 gate 信息被 \(\nu\) 吸收，而不一定传给 \(W\)。同时 GP kernel 主要看 \(W^TW\)，还有旋转/scale等不识别自由度。

因此，固定 gate slope：

\[
\nu_{1:Q}=e_1
\quad\text{or}\quad
\nu_{1:Q}=\mathbf 1/Q
\]

只学习 intercept \(\nu_0\)，是一个合理的 identifiability remedy。

---

## 2. 已尝试方法与结果总结

### 2.1 Remedy A：\(s=x^T\Sigma_Wx\) penalty

目的：直接压小 analytic kernel expectation 中的 smoothing factor。

形式：

\[
\mathcal L\leftarrow \mathcal L-\lambda_s\operatorname{mean}\log(1+s).
\]

结果：

- \(s\) 从约 8 降到约 3.8；
- \(\sigma_W\) 下降；
- 但 RMSE 不动；
- gate 仍坍缩到 \(\pi\approx0.001\)。

解读：A 确实控制了一部分 posterior variance，但没解决 inlier GP / gate 的更深问题。

---

### 2.2 Remedy B：\(\sigma_V\) annealing / clamp

目的：训练前期让 \(q(W)\) 更集中，避免 \(E_q[K]\) 早期变平。

结果：

- \(s\) 可降到约 1.5–2；
- \(\sigma_W\) 降到约 0.25–0.30；
- RMSE 仍不动；
- gate 仍 \(\pi\to0.001\)。

解读：即使 posterior variance 已明显变小，inlier GP 仍然没有学出非平凡预测。说明问题不只是 \(s\) 太大。

---

### 2.3 Remedy C：Gate stabilization

尝试变体：

| 变体 | 改动 |
|---|---|
| C1 | \(\omega_0=2\) 初始化 |
| C2 | \(\omega_0\) lower clamp |
| C3 | \(\pi\)-floor prior / inlier-rate prior |
| C4 | force inlier warmup，即训练前期关掉 outlier branch |
| C5 | `U_logit` upper clamp |

诊断结果：

- \(\pi\) 可以暂时变高，但 \(r\) 仍接近 0；
- \(\Delta T\approx -10^6\)；
- \(\operatorname{std}(\mu_f)\approx0\)；
- \(\mathrm{RMSE}(\pi=1)\approx \mathrm{RMSE}(\text{mixture})\)。

解读：Gate stabilization 不能单独解决，因为 inlier GP path 自己没学动。Gate collapse 是后果之一。

---

### 2.4 Scale / local mean diagnostic

新增：

- `--standardize_y`
- `--local_center_y`
- `oracle_local_mean`
- `pure_inlier`
- `fixedW_inlier`
- `fixedW_predict`

关键结果：

| 设置 | 观察 | 解读 |
|---|---|---|
| `standardize_y` | ELBO 从 \(-10^7\) 到 \(O(10^2)\) | raw scale 是硬问题 |
| `pure_inlier` | 仍等于 oracle local mean | 去掉 gate 后 GP residual 仍不贡献 |
| `fixedW_inlier` | 有少量 \(std(\mu_f)\)，但远小于 \(y\) residual | 几何未完全死，但 amplitude / feature-to-output 路径弱 |
| `E_q[K]` | \(std(m_W)\approx0\) | analytic expectation 更容易抹平 feature |

---

### 2.5 显式 \(m_j\) 的 inlier quadratic

已实现：

\[
Q_{j,i}
=
\frac{
(y_i-m_j)^2
-2(y_i-m_j)\zeta_i
+V_i+B_i
}{2\sigma_n^2}.
\]

作用：修正 local mean / scale mismatch。

结果：仍接近 local mean oracle，说明 \(m_j\) 修正必要但不充分。

---

### 2.6 Remedy D：结构化残差 projection posterior

目标：避免 full entrywise \(q(R)\) 把 \(W\) 拉回零均值 prior，同时限制无意义自由度。

变体：

| 变体 | 投影结构 | 其它 |
|---|---|---|
| C0_mj | 原 full \(q(R)\) | 显式 \(m_j\) |
| D1 | \(W=W_0+\Delta W\) full residual | KL 对 residual |
| D2 | \(W=W_0+BU^T\)，low-rank residual | \(r=3\) |
| D3 | D2 + inducing \(C\leftarrow W_0X\) | local inducing 更贴近 projected cloud |
| D4 | D3 + MC \(\Psi_1\) | sampled kernel statistic |

Smoke 结果：

| 变体 | 结果 |
|---|---|
| C0_mj | \(std(m_W)\approx0\) |
| D2/D3/D4 | \(std(m_W)\approx0.18\)，说明几何有 variation |
| D3/D4 | \(std(\zeta)\) 提升，但 \(\zeta/y_{res}\) 仍很小 |
| D4≈D3 | 短程下 MC \(\Psi\) 未明显优于 analytic |

解读：Remedy D 成功修了 projection geometry 的一部分，但 \(\Psi_1\to K^{-1}\mu_u\to\zeta\) 这一段仍弱。

---

### 2.7 \(\Phi^Tr\) 对齐诊断与 auxiliary loss

新增诊断：

- centered signal：

\[
\frac{\|\Phi_c^Tr_c\|}{\|\Phi_c\|_F\|r_c\|}
\]

- ridge \(R^2\)：当前 \(\Phi\) 最优线性组合最多解释多少 centered residual；
- variance temperature test：\(\Sigma_W^{(\tau)}=\tau\Sigma_W\)，\(\tau\in\{0,0.25,0.5,1\}\)。

D3 smoke 结果：

| 指标 | 观察 |
|---|---|
| \(R^2@\tau=0\) | 约 0.09 |
| \(R^2@\tau=1\) | 约 0.11 |
| ridge \(z/r\) | 约 0.30 |
| learned \(\zeta/r\) | 约 0.086 |

解读：

1. 当前 \(\Phi\) 不是完全无信号；
2. 方差 / analytic smoothing 不是 D3 当前主瓶颈；
3. learned \(q(u)\) 没有充分利用 feature signal。

Auxiliary loss：

- \(L_{cov}\)：最大化 normalized \(\|\Phi_c^Tr_c\|^2\)；
- \(L_{ridge}\)：最大化 ridge feature fit。

观察：

- `D3_cov1` 能提高 \(R^2\)，但 loss 过强，ELBO 爆炸；
- `D3_ridge0.1` 作用较弱；
- 需要更细权重 sweep 和 clipping。

---

## 3. 当前最合理的解释

目前不是简单的 “analytic marginalization 必然让 \(W\) trivial”。更准确是：

\[
\boxed{
\text{当前 ELBO 存在多个 cheap bad basins，而 good basin 没有被稳定找到。}
}
\]

这些 cheap basins 包括：

1. local mean \(m_j\) 解释大部分 signal；
2. gate/outlier 解释所有点；
3. \(q(W)\) 被 KL 拉回无信息 prior；
4. \(\Psi_1\) 有 variation 但与 residual 不对齐；
5. \(\Phi\) 有 signal，但 \(q(u)\)/\(\Psi_2\)/\(\sigma_n\)/KL 把 signal 压住。

当前 D3 + alignment 诊断已经显示：

\[
\Phi^Tr\neq 0,
\]

但：

\[
\zeta_{learned} \ll \zeta_{ridge}.
\]

因此下一阶段重点应从 “让 \(W\) 有 variation” 转向：

\[
\boxed{
\text{让 }q(u),\sigma_k,\sigma_n,\Psi_2\text{ 把已有 feature signal 转成 }\zeta.
}
\]

---

## 4. 后续改进方向与诊断

### 4.1 计算 ELBO-consistent \(\mu_u^*\)

当前 ridge 上界未包含真实 ELBO 的 \(\Psi_2\)、\(K^{-1}\)、\(\sigma_n\) 惩罚。应计算：

\[
\mu_u^*
=
(G+\sigma_n^2K^{-1})^{-1}\Phi^Tr,
\]

并比较：

\[
\zeta_{learned}=\Phi\mu_u,
\]

\[
\zeta_{elbo-star}=\Phi\mu_u^*,
\]

\[
\zeta_{ridge}=\Phi b_{ridge}.
\]

判读：

| 结果 | 含义 |
|---|---|
| learned \(\ll\) elbo-star \(\approx\) ridge | \(q(u)\) 优化没跟上 |
| learned \(\approx\) elbo-star \(\ll\) ridge | ELBO 二阶项 / noise / KL 压住 signal |
| ridge 本身低 | \(\Phi\) feature 不对齐 |

这是下一步最关键诊断。

---

### 4.2 \(q(u)\) refresh / closed-form coordinate update

若 \(\mu_u\) 没吃满 signal，可以尝试：

1. 用 \(\mu_u^*\) 初始化；
2. 每 20–50 步做 partial refresh：

\[
\mu_u\leftarrow (1-\alpha)\mu_u+\alpha\mu_u^*.
\]

3. 降低 \(KL(q(u)\|p(u))\) 权重 warmup；
4. 对 \(\Sigma_u\) 使用更稳定参数化或先固定。

诊断：

- \(\|\mu_u\|\)、\(\|K^{-1}\mu_u\|\)；
- \(KL_u\)；
- \(\sigma_n\)、\(\sigma_k\)；
- \(\zeta_{learned}/\zeta_{elbo-star}\)。

---

### 4.3 更稳的 alignment auxiliary

当前 `cov1` 太强。建议：

- sweep \(\lambda_{cov}\in\{0.01,0.03,0.1,0.3\}\)；
- 对 \(L_{cov}\) 做 clip；
- 只让 alignment loss 影响 projection 参数 \(B,U\)，不要影响 \(m_j,\sigma_k,\sigma_n,\mu_u\)；
- target 用 `r_align=(y-mean(y)).detach()`，避免 \(m_j\) 作弊。

成功指标：

\[
\zeta/r \ge 0.05\sim0.1,
\]

且 ELBO 不爆炸。

---

### 4.4 Fixed gate slope：吸收 \(\nu\) 到 \(W\)

动机：

\[
h_i=\nu_0+\nu^TWx_i,
\quad
\text{gate normal } a=W^T\nu.
\]

\(W\) 与 \(\nu\) 非 identifiable。可试：

| 变体 | 形式 |
|---|---|
| G1 | \(\nu_{1:Q}=\mathbf 1/Q\)，只学 \(\nu_0\) |
| G2 | \(\nu_{1:Q}=e_1\)，只学 \(\nu_0\) |
| G3 | \(e_1+\) learnable temperature \(\tau_g\) |

期望效果：

- gate signal 必须进入 \(W\)，不能被 \(\nu\) 吸收；
- \(std(h)\)、\(std(\pi)\)、gate-to-W gradient 上升；
- 若有效，\(R^2\)、\(zeta/r\) 也可能上升。

诊断：

- \(std(\mu_\xi)\)、\(var_\xi\)、\(\pi\)、\(r\)、\(\Delta T\)；
- \(\|\nabla_W S_{gate}\|\)；
- \(\|W\|\)、row norm、scale coupling；
- \(R^2\)、\(z_{ridge}/r\)、\(z_{learned}/r\)。

风险：fixed slope 把 gate scale 与 GP kernel scale 绑在 \(W\) 上，可能需要 row norm control 或 gate temperature。

---

### 4.5 Better inducing placement and conditioning

D3 显示 `C <- W0X` 有帮助。后续可尝试：

- local inducing \(\tilde z_j\) 用 kmeans\((W_0X_j)\)；
- random subset of \(W_0X_j\)；
- learnable \(\tilde z\) with small LR；
- monitor inducing distance：

\[
\min_\ell\|W_0x_i-\tilde z_\ell\|.
\]

诊断：

- \(\Psi_1\) row-std；
- \(\Psi_1\) effective rank；
- condition number of \(K_{uu}\)；
- \(zeta/r\)。

---

### 4.6 MC-\(\Psi\) / sampled-W objective

D4 短程未明显优于 D3，但仍值得更长训练。

若最终发现：

\[
R^2(0)\gg R^2(1),
\]

或 analytic \(\Psi\) effective rank 低，则说明 \(E_q[K]\) smoothing 仍是瓶颈。

可尝试：

\[
\Psi_1\approx \frac1S\sum_s K(W^{(s)}X,\tilde Z),
\]

或直接训练 sampled-W / mean-W local GP/JGP objective。

这也与 manuscript 当前 prediction 路线一致：sample \(W\)，每个 sampled projection 上 fit local JumpGP-CEM，再 MC aggregate。

---

### 4.7 Composite pseudo-ELBO 的处理

当前 local neighborhoods 可能重叠，因此目标更准确应称为：

\[
\boxed{\text{composite variational pseudo-ELBO}}
\]

而不是严格 global likelihood ELBO。

后续可尝试：

- composite weights：

\[
w_{ij}=\frac{1}{\#\{j':i\in N_{j'}\}};
\]

- sample-splitting / cross-fitting diagnostics；
- 只用 selected anchors 训练 \(q(R)\)；
- 比较 weighted vs unweighted objective 对 \(W\) 学习的影响。

但当前单 region pure-inlier 也弱，因此 composite issue 不是第一优先级。

---

## 5. 推荐短期实验路线

### Step 1：D3 上加 ELBO-star 诊断

输出：

- \(z_{learned}/r\)；
- \(z_{elbo-star}/r\)；
- \(z_{ridge}/r\)；
- \(\sigma_n,\sigma_k,KL_u,\|K^{-1}\mu_u\|\)。

目的：确定瓶颈是 \(q(u)\) 优化、ELBO 二阶项，还是 feature 本身。

---

### Step 2：D3 + mild alignment sweep

尝试：

\[
\lambda_{cov}\in\{0.01,0.03,0.1,0.3\}.
\]

加 clip，并只更新 projection 参数。

成功指标：

\[
z_{learned}/r\ge 0.1
\]

且 ELBO 不爆炸。

---

### Step 3：Fixed \(\nu\) variants

跑：

- G1：\(\nu=\mathbf 1/Q\)；
- G2：\(\nu=e_1\)；
- G3：\(\nu=e_1\) + temperature。

与 D3 / D3+alignment 联用。

看 fixed \(\nu\) 是否增加：

- gate-to-W gradient；
- \(R^2\)；
- \(zeta/r\)。

---

### Step 4：q(u) refresh / closed-form warmup

若 \(z_{elbo-star}\) 明显大于 learned \(\zeta\)，实现 \(\mu_u^*\) 初始化或周期性 partial update。

---

### Step 5：与 sampled-W + local JGP-CEM 对比

如果 structured direct VI 仍无法超过 local mean oracle，但 sampled-W+CEM 能明显超过，则 manuscript 主路线应明确：

- VI 用于学习 projection posterior；
- prediction 用 sampled-W + local JGP-CEM；
- direct analytic head 作为 diagnostic / ablation，而非主 prediction head。

---

## 6. 需要持续记录的诊断指标

每个 experiment 建议记录：

### Projection / kernel feature

- \(std(m_W)\)
- \(\Psi_1\) row-std
- \(\Psi_1\) effective rank
- \(R^2_{\tau=0},R^2_{\tau=1}\)
- centered signal \(\|\Phi_c^Tr_c\|/(\|\Phi_c\|\|r_c\|)\)
- variance temperature test \(\tau=0,0.25,0.5,1\)

### Local GP path

- \(z_{learned}/r\)
- \(z_{elbo-star}/r\)
- \(z_{ridge}/r\)
- \(\|\mu_u\|\)
- \(\|K^{-1}\mu_u\|\)
- \(KL_u\)
- \(\sigma_k,\sigma_n\)

### Gate path

- \(std(h)\)
- \(\mu_\xi,\sigma_\xi^2\)
- \(\pi\) mean/std
- \(r\) mean/std
- \(\Delta T\)
- gate-to-W gradient norm

### Prediction

- RMSE_work
- RMSE_raw
- RMSE(\(\pi=1\))
- oracle local mean RMSE
- sampled-W+CEM RMSE

---

## 7. Paper-level implication

目前证据支持以下叙事：

1. Original analytic ELBO 的形式不一定错误；
2. 但它是 local composite variational pseudo-objective，存在 local mean / outlier / KL / non-identifiability 等 cheap bad basins；
3. Direct analytic prediction head 不应作为主方法卖点；
4. 主方法应强调：

\[
q(R)\text{ learns a projection posterior, then prediction samples }W_j\text{ and runs local JumpGP-CEM.}
\]

5. Direct analytic head 可以作为 diagnostic / ablation，用来解释为什么 naive VI prediction underperforms；
6. Structured residual posterior、explicit local mean、fixed gate slope、feature-alignment loss 可以作为 remedy / future work / robustness variants。

---

## 8. 一句话总结

当前 original ELBO + direct analytic prediction 的失败不是单一 bug，而是多层链路断裂：

\[
\text{raw scale} \rightarrow m_j \rightarrow E_q[K] \rightarrow \Phi^Tr \rightarrow q(u) \rightarrow gate.
\]

已经确认：

- scale 和 local mean 必须修；
- gate collapse 是后果；
- structured \(W_0+BU^T\) 能恢复部分 geometry；
- \(\Phi\) 已经有一定 residual signal；
- 现在最大问题是 learned \(q(u)\) / ELBO 二阶项 / noise-scale 没有把 feature signal 充分转化成 \(\zeta\)。

下一步最关键的诊断是：

\[
\boxed{
\zeta_{learned}\quad vs\quad \zeta_{elbo-star}\quad vs\quad \zeta_{ridge}.
}
\]

这会决定后续应该优化 \(q(u)\)，调整 ELBO scale/penalty，还是继续增强 \(W\)-feature alignment。
