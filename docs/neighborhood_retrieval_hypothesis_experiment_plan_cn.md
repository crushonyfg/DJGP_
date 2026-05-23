# W-aware Neighborhood Retrieval：假想、方法与实验设置

## 1. 背景

当前 uncertain-W self-CEM / qR alternating CEM 已经显示出一个比较稳定的现象：

1. raw-neighborhood self-CEM head 在 LH 和 UCI 上能改善 CRPS / coverage；
2. 直接学习一个 spatial projection field $$x\mapsto W(x)$$ 不一定稳定优于 shuffled / no-qR control；
3. 只有当 $$W$$ 参与 neighbor retrieval 时，learned $$W(x)$$ 才开始显示出空间绑定信号；
4. pure W-KNN 又太激进，容易破坏 raw locality，导致 RMSE 变差；
5. hybrid raw + W retrieval 在 seed0 上有效，但 5 seeds 不稳定。

这说明当前瓶颈可能不是 uncertain-W kernel 本身，而是：

$$
W(x) \text{ 还没有被放在一个足够可辨识、且足够稳定的位置。}
$$

JGP 的 prediction 很依赖 neighborhood。如果 neighborhood 固定在 raw-X 上，$$W$$ 只能影响 local kernel / gate / uncertainty，JGP-CEM 可以通过 labels、noise、lengthscale、gate 重新适配，因此 $$W$$ 的空间绑定不重要。

如果让 $$W$$ 完全决定 neighborhood，模型又会丢掉 raw locality，选进过远或不稳定的点，local GP mean 会受损。

因此，更合理的设想是：

$$
\boxed{
W(x) \text{ 不应完全替代 raw neighborhood，而应作为 supervised local retrieval correction。}
}
$$

---

## 2. 核心假想

### 2.1 当前 fixed-neighborhood objective 的问题

当前很多 qR / uncertain-W CEM 训练目标可以抽象为：

$$
\mathcal Q(qR; N_a)
=
\sum_a
\log p_{\text{JGP}}
\left(
 y_{N_a}\mid X_{N_a}, q(W_a)
\right)
-
\beta KL(qR\|pR),
$$

其中 $$N_a$$ 是给定的 local neighborhood。

如果 $$N_a$$ 固定，那么这个目标无法直接比较：

$$
N_a^{old}
\quad \text{vs} \quad
N_a^{new}(W).
$$

也就是说，模型优化的是“在这个 neighborhood 上如何 fit 得好”，而不是“哪个 neighborhood 更适合预测 anchor”。

因此，$$W$$ 即使能把 inliers 拉近、outliers 推远，也未必能得到稳定训练信号，因为 neighborhood selection 的好坏没有被直接监督。

---

### 2.2 需要 supervised anchor predictive objective

更合理的目标是：对 training anchors，固定预测目标 $$y_a$$，然后让不同 retrieval rule 竞争，看哪个 neighborhood 能更好预测 anchor。

对 anchor $$a$$，定义 candidate pool：

$$
C_a=\operatorname{KNN}_{raw}(x_a,K_{pool}).
$$

在这个固定 pool 内，用某个 distance rule 选最终 neighbors：

$$
N_a(W)=\operatorname{TopK}_{i\in C_a}[-d_W(a,i)].
$$

然后优化 leave-anchor-out predictive objective：

$$
\mathcal J(W)
=
\sum_{a\in\mathcal A}
\log p_{\text{JGP}}
\left(
 y_a
 \mid
 \{(x_i,y_i):i\in N_a(W), i\neq a\}
\right).
$$

这个目标更可比，因为每个 anchor 的预测目标 $$y_a$$ 是固定的。

因此它能回答：

$$
\text{这个 } W \text{ 选出来的 neighborhood 是否真的更能预测 anchor？}
$$

---

## 3. 三类 retrieval distance

### 3.1 Raw retrieval

基线距离：

$$
d_{raw}(a,i)
=
\|x_i-x_a\|^2.
$$

选：

$$
N_a^{raw}=\operatorname{TopK}_{i\in C_a}[-d_{raw}(a,i)].
$$

优点：稳定、保留 local approximation。

缺点：不能利用 learned metric；在 jump boundary 附近，raw KNN 可能混入 outliers。

---

### 3.2 Anchor-conditioned W retrieval

这是更贴近 DJGP 的版本。

对 anchor $$a$$ 使用 local projection：

$$
W_a=W(x_a).
$$

定义：

$$
d_{anc}(a,i)
=
\mathbb E_{q(W_a)}
\left[
\|W_a(x_i-x_a)\|^2
\right].
$$

若：

$$
q(W_a)=\mathcal N(M_a,\Sigma_a),
$$

则：

$$
d_{anc}(a,i)
=
\|M_a(x_i-x_a)\|^2
+
\operatorname{tr}(S_{a,i}),
$$

其中：

$$
S_{a,i}
=
\operatorname{Var}_{q(W_a)}[W_a(x_i-x_a)].
$$

优点：保留原 DJGP 的 local-constant projection 假设：

$$
W(x)\approx W(x_a)
\quad \text{near anchor } x_a.
$$

缺点：如果 candidate pool 太大，$$W(x_a)$$ 对远点可能不合适。

---

### 3.3 Pointwise global-Z retrieval

这是更接近 DMGP-style embedding 的版本。

定义：

$$
z_i=W(x_i)x_i,
\qquad
z_a=W(x_a)x_a.
$$

retrieval distance：

$$
d_{pw}(a,i)
=
\mathbb E_q
\left[
\|W(x_i)x_i-W(x_a)x_a\|^2
\right].
$$

然后在全局 mapped dataset 或 fixed candidate pool 内选 KNN。

优点：更像一个全局 learned latent space，可能适合 smooth global geometry。

缺点：偏离 DJGP 的 local constant assumption；已有实验中 pointwise W-KNN 和 DMGP embedding + JGP 并不稳定。

---

### 3.4 Hybrid raw + W retrieval

定义：

$$
d_{hyb}(a,i)
=
(1-\lambda)d_{raw}(a,i)
+
\lambda d_W(a,i).
$$

其中 $$d_W$$ 可以是 anchor-conditioned 或 pointwise distance。

$$\lambda$$ 控制 learned W 对 raw neighborhood 的重排强度。

- $$\lambda=0$$：完全 raw retrieval；
- $$\lambda=1$$：pure W-KNN；
- 中间值：raw locality + W correction。

经验上，pure W-KNN 太激进。更合理的是让 W 替换少量 raw neighbors。

建议记录：

$$
\operatorname{overlap}
=
\frac{|N_a^{hyb}\cap N_a^{raw}|}{K}.
$$

与其只解释 $$\lambda$$，更推荐报告 overlap。因为不同数据集上 raw/W distance scale 不同，同一个 $$\lambda$$ 的含义可能不同。

---

## 4. 为什么 selected-neighborhood marginal likelihood 不可比

不建议直接优化：

$$
\log p(y_{N_a(W)}\mid W).
$$

原因是不同 $$W$$ 会选择不同的 $$N_a(W)$$。这个目标可能偏向选择一组内部非常平滑、很好拟合的点，而不是能预测 anchor 的点。

例如，模型可能选到一批 response 很接近的点，让 local marginal likelihood 很高，但这些点未必靠近 anchor，也未必能预测 $$y_a$$。

因此更合理的是：

$$
\log p(y_a\mid D_{N_a(W)}).
$$

即每个 anchor 是一个固定 supervised prediction task。

---

## 5. Supervised anchor predictive objective

### 5.1 Hard top-k version

对 training anchors $$\mathcal A$$，定义：

$$
N_a(W)=\operatorname{TopK}_{i\in C_a}[-d_W(a,i)].
$$

目标：

$$
\mathcal J_{pred}(W)
=
\sum_{a\in\mathcal A}
\log p_{\text{uncertain-JGP}}
\left(
 y_a
 \mid
 D_{N_a(W)\setminus \{a\}}
\right).
$$

这能直接比较不同 W-induced neighborhoods。

缺点：TopK 不可微。可以用 EM-style refresh：每隔若干步用当前 $$W$$ 刷新 neighborhoods。

---

### 5.2 Ranking surrogate

为了给 W 连续梯度，可以加 anchor-centric ranking loss。

给定当前 CEM labels，对 anchor $$a$$：

$$
P_a=\{i:v_{ai}=1\},
\qquad
O_a=\{i:v_{ai}=0\}.
$$

希望 inlier 比 outlier 更近：

$$
d_W(a,p)<d_W(a,o),
\quad
p\in P_a,
\quad o\in O_a.
$$

定义 ranking objective：

$$
\mathcal L_{rank}
=
\sum_a
\sum_{p\in P_a}
\sum_{o\in O_a}
\log \sigma
\left(
\frac{d_W(a,o)-d_W(a,p)}{\tau}
\right).
$$

这个比普通 pairwise loss 更适合 JGP retrieval，因为它不要求所有 same-regime pairs 都彼此很近，只要求：

$$
\text{相对 anchor 而言，inliers 排在 outliers 前面。}
$$

---

### 5.3 Combined objective

qR M-step 可以写成：

$$
\mathcal Q(qR)
=
\tau_{pred}
\sum_a
\log p_{\text{uncertain-JGP}}
\left(
 y_a\mid D_{N_a}
\right)
+
\lambda_{rank}\mathcal L_{rank}
-
\beta KL(qR\|pR).
$$

其中：

- $$N_a$$ 可由 current W/hybrid retrieval refresh；
- $$\mathcal L_{rank}$$ 给 W 一个 retrieval-aware signal；
- KL 控制 W field 的 smoothness 和 complexity。

---

## 6. Soft retrieval 版本

TopK 离散且不平滑，可以考虑 soft retrieval。

对 candidate pool $$C_a$$：

$$
\pi_{ai}
=
\frac{
\exp(-d_{ai}/\tau)
}{
\sum_{l\in C_a}\exp(-d_{al}/\tau)
}.
$$

可以定义 soft anchor predictive objective 或 soft ranking。

一种简单 proxy：让 CEM inliers 权重大、outliers 权重低：

$$
\mathcal L_{soft-ret}
=
\sum_a
\sum_{i\in C_a}
\left[
\rho_{ai}\log \pi_{ai}
+
(1-\rho_{ai})\log(1-\pi_{ai})
\right].
$$

其中 $$\rho_{ai}=q(v_{ai}=1)$$ 或 smoothed hard labels。

实际 prediction 时仍可用 hard top-k。

---

## 7. Gated hybrid retrieval

5-seed 结果显示：hybrid retrieval 在部分 seed 上有效，但不是稳定默认。更稳的方案是：默认 raw retrieval，只在 training-anchor validation 认为 hybrid 更好时启用。

### 7.1 Rule-based gate

启用 hybrid if：

$$
\operatorname{overlap}(N_{hyb},N_{raw})\ge o_{min}
$$

且：

$$
\widehat{\operatorname{NLL}}_{hyb}
<
\widehat{\operatorname{NLL}}_{raw}-\tau.
$$

其中 NLL 用 training anchors 的 leave-anchor-out predictive NLL 估计。

推荐：

$$
o_{min}=0.85.
$$

### 7.2 Learned gate

为每个 anchor 构造特征：

1. raw/W neighbor overlap；
2. local y-range；
3. CEM entropy；
4. W distance margin；
5. selected inlier fraction；
6. local noise estimate；
7. gate norm。

训练一个简单 classifier / regressor，预测 hybrid 是否优于 raw。

test 时根据这个 gate 决定使用 raw or hybrid。

---

## 8. 实验设置建议

### 8.1 Stage 1：LH seed0 retrieval mechanism screen

固定数据：

$$
\text{LH},\quad \text{train}=1000,\quad \text{test}=200,\quad \text{seed}=0.
$$

固定 head：

$$
\texttt{self-CEM2 + nstd0.1 + kcov0.1}.
$$

比较：

| Variant | Distance | Training signal |
|---|---|---|
| raw | raw KNN | current uncertain-W self-CEM |
| anchor W-KNN | $$d_{anc}$$ | ranking + qR |
| pointwise W-KNN | $$d_{pw}$$ | ranking + qR |
| hybrid anchor | raw + $$d_{anc}$$ | ranking + qR |
| hybrid pointwise | raw + $$d_{pw}$$ | ranking + qR |
| gated hybrid | raw or hybrid | anchor predictive gate |

### 8.2 Hybrid λ sweep

For anchor hybrid：

$$
\lambda\in\{0.1,0.15,0.2,0.25,0.3,0.4,0.5\}.
$$

但不要只看 λ。必须记录：

$$
\text{raw overlap},
\quad
\text{selected inlier fraction},
\quad
\text{label change rate},
\quad
\text{noise floor fraction}.
$$

### 8.3 Candidate pool and neighborhood size

比较：

$$
K_{pool}\in\{100,200,500\},
$$

$$
K_{neighbor}\in\{35,50,100\}.
$$

预期：

- pool 太小：W 没有足够空间修正 neighbors；
- pool 太大：W 容易选入非局部点；
- neighbor 太大：JGP local approximation 可能失效；
- neighbor 太小：hyperparams / CEM labels 不稳。

---

## 9. Controls

每个 promising retrieval variant 必须跑 controls：

1. learned W / learned qR；
2. prior W / prior qR；
3. shuffled W / shuffled qR；
4. raw-only retrieval；
5. no-qR raw self-CEM。

判断标准：

若：

$$
\text{learned} < \text{shuffled}
$$

且：

$$
\text{learned} < \text{prior},
$$

说明 spatial binding 起作用。

若 learned 与 shuffled/no-qR 接近，则说明主要收益来自 head 或 raw component，而不是 learned W field。

---

## 10. Metrics and diagnostics

必须记录：

1. RMSE；
2. CRPS；
3. Cov90；
4. Width90；
5. raw overlap；
6. selected inlier fraction；
7. label change rate；
8. inlier rate；
9. local noise mean；
10. fraction of anchors at noise floor；
11. lengthscale mean；
12. gate norm；
13. W-distance margin；
14. qR KL；
15. R_mu data/KL ratio；
16. R_log_std data/KL ratio；
17. per-test CRPS delta against raw baseline。

尤其重要的是 per-test delta：

$$
\Delta_i
=
\operatorname{CRPS}_{hyb,i}
-
\operatorname{CRPS}_{raw,i}.
$$

如果 improvement 来自少数 points，需要谨慎；若大多数 points 小幅改善，则更稳。

---

## 11. 解释规则

### 11.1 Pure W-KNN 好

若 pure W-KNN 好，说明 learned W 已经足够定义 local geometry。

这种情况目前不太符合已有结果。

### 11.2 Hybrid 好，pure W-KNN 差

这是当前最可能的成功形态。

解释：

$$
W(x) \text{ 能修正 raw neighborhood，但不能替代 raw locality。}
$$

### 11.3 Hybrid seed0 好但 5-seed 不稳

说明 W retrieval 有 seed-specific signal，但需要 gating / validation selection。

### 11.4 learned 不赢 shuffled

说明 W distribution 有用，但 spatial assignment 不可靠。

### 11.5 learned 赢 shuffled/prior

说明 learned spatial W binding 真的被 retrieval 使用了。

---

## 12. 推荐下一步最小实验

如果让 Codex 继续做，建议按以下顺序：

### Experiment A：hybrid stability diagnostics

使用已有 5-seed hybrid结果，按 seed/test point分析：

- 哪些 seed hybrid 改善；
- 哪些 test points 改善；
- improvement 与 overlap / inlier fraction / y-range 的关系。

### Experiment B：gated hybrid seed0

在 LH seed0 上实现：

1. raw retrieval；
2. hybrid anchor λ=0.25 / 0.4；
3. rule-based gated hybrid；
4. oracle gated hybrid based on anchor predictive NLL。

oracle gated hybrid 是上界，用于判断 gating 是否有潜力。

### Experiment C：rank-based hybrid

用 rank mixing 代替 raw distance mixing：

$$
d_{hyb-rank}
=
(1-\lambda)\operatorname{rank}_{raw}
+
\lambda\operatorname{rank}_{W}.
$$

这个可能比 distance mixing 更稳定，因为 λ 的含义不受尺度影响。

### Experiment D：soft retrieval

实现 softmax weights：

$$
\pi_{ai}\propto \exp(-d_{ai}/\tau).
$$

先用 soft retrieval loss 训练 W，再 hard top-k prediction。

### Experiment E：5 seeds only after gated/rank variant wins seed0

不要直接继续扩大所有 hybrid λ。只有 gated/rank variant 在 seed0 稳定赢 raw/no-qR，再跑 5 seeds。

---

## 13. 当前方法定位

若 gated hybrid 成功，可以将方法定位为：

$$
\boxed{
\text{Uncertain-W CEM with supervised W-aware neighborhood refinement}
}
$$

核心贡献：

1. uncertain-W expected kernel 传播 projection uncertainty；
2. self-CEM weak refresh 改善 local probabilistic prediction；
3. supervised anchor predictive / ranking objective 让 W 控制 neighbor retrieval；
4. hybrid/gated retrieval 保留 raw locality，避免 pure W-KNN 破坏 local GP。

如果 gated hybrid 不成功，则应把 W retrieval 降级为 diagnostic，主方法仍然是：

$$
\boxed{
\text{raw-neighborhood uncertain-W self-CEM prediction head}
}
$$

