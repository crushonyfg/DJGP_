# 关于 DJGP / LMJGP 中 W-learning 的阶段性总结与后续 ablation 计划

本文档总结最近几轮关于 projection matrix / projection field $$W$$ 的讨论，重点围绕：为什么原始 $$q(W)$$ / $$q(R)$$ 学不强、uncertain-W CEM 的当前定位、low-rank residual prior、ARD / group penalty、以及如何设计后续 ablation。

---

## 1. 当前核心判断

目前实验结果支持一个比较稳定的判断：

$$
\text{当前原始 LMJGP / DJGP VI 的 } q(W) \text{ 没有学到强的 spatially-bound projection field。}
$$

也就是说，当前 learned $$q(W_j)$$ 或 learned $$q(R)$$ 的提升，很多时候不像是来自一个真正有效的映射：

$$
x_j^\ast \mapsto q(W_j),
$$

而更像来自：

1. local JumpGP-CEM prediction head 本身很强；
2. sampled / prior-like projection ensemble 提供了一些随机投影平均效果；
3. uncertain-W CEM head、noise floor、weak CEM refresh 改善了概率预测；
4. 但 learned projection field 本身在 shuffled / no-qR controls 下还没有表现出足够强的空间绑定作用。

因此，后续方法应分成两条线：

第一，继续把 uncertain-W CEM 作为一个 improved probabilistic prediction head。

第二，如果要证明 learned $$W$$ 本身有价值，需要限制 $$W$$ 的 prior space，并加入更合理的 structure，例如 low-rank residual、ARD / group penalty、basis selection、或者更强的 projection-field controls。

---

## 2. 为什么原始 W-learning 信号弱

原始 DJGP / LMJGP 的 projection posterior 通常通过 sparse GP inducing variables 表示：

$$
q(R) \rightarrow q(W_j).
$$

但原始 ELBO 中，$$W_j$$ 对 likelihood 的影响经常经过较长链路：

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
\log\sum\exp.
$$

这个链路有几个问题：

1. $$W$$ 的几何信息被 local inducing statistics 压缩；
2. gate / responsibility 很容易变硬；
3. local mean、noise、amplitude、gate 等 nuisance parameters 会吸收 response signal；
4. final prediction 时每个 projected sample 又重新 fit JumpGP-CEM，导致 final performance 不一定依赖 learned $$W$$；
5. KL 会把 $$q(R)$$ 拉回 prior，如果 data gradient 不足，就会出现 posterior collapse 或 prior-like posterior。

因此，单纯让原始 VI 学 $$q(W)$$，很容易出现：

$$
q(W) \approx p(W),
$$

或者：

$$
\text{learned } q(W) \approx \text{prior/shuffled/global controls}.
$$

---

## 3. Uncertain-W CEM 当前结论

最近实现的 uncertain-W CEM 证明了一件重要的事情：

$$
\text{可以在不损害 JumpGP-CEM 点预测的情况下，把 } W \text{ 的不确定性传进预测方差。}
$$

核心是把 deterministic kernel：

$$
k(Wx_i,Wx_l)
$$

替换成 uncertain-W expected kernel：

$$
\bar k(i,l)
=
\mathbb E_{q(W)}[k(Wx_i,Wx_l)].
$$

对于 squared exponential kernel，如果：

$$
\Delta z_{il}=W(x_i-x_l)\sim \mathcal N(\mu_{\Delta,il},S_{\Delta,il}),
$$

则：

$$
\bar k(i,l)
=
\sigma_f^2
\left|I_Q+\frac{1}{\ell^2}S_{\Delta,il}\right|^{-1/2}
\exp\left(
-\frac12
\mu_{\Delta,il}^\top
(\ell^2 I_Q+S_{\Delta,il})^{-1}
\mu_{\Delta,il}
\right).
$$

当前最有效的 prediction-head 版本是：

$$
\text{mean-W CEM init}
\rightarrow
\text{uncertain-W hyperopt/gate}
\rightarrow
\text{weak self-CEM refresh}
\rightarrow
\text{qR M-step}
\rightarrow
\text{periodic weak refresh}.
$$

5-seed 结果显示，在 LH 上主候选：

```text
qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1
```

相对 original CEM 有稳定提升：

$$
\text{CRPS: } 272.4 \rightarrow 250.8,
$$

$$
\text{Cov90: } 0.362 \rightarrow 0.781.
$$

但 controls 显示 learned qR、shuffled qR、no-qR head 很接近。因此当前应保守表述为：

$$
\boxed{
\text{uncertain-W CEM 是一个 improved probabilistic prediction head，}
\text{但尚未充分证明 learned projection field 是主要增益来源。}
}
$$

---

## 4. 为什么考虑 low-rank residual prior

当前 full GP prior over every entry of $$W(x)$$ 空间太大：

$$
W(x)\in\mathbb R^{Q\times D}.
$$

若每个 $$W_{qd}(x)$$ 都有一个 GP，inducing variables 数量是：

$$
M\times Q\times D.
$$

在 local CEM/JGP 信号弱的情况下，这个空间太自由，容易导致：

1. learned qR 有梯度，但不形成稳定 projection field；
2. learned / shuffled / no-qR controls 接近；
3. posterior variance 或 random projection effect 贡献大于 mean projection；
4. spatial assignment 不明显。

因此建议改成 centered low-rank residual prior：

$$
\boxed{
W(x)=W_0+C(x)V^\top
}
$$

其中：

$$
W_0\in\mathbb R^{Q\times D}
$$

是稳定 base projection，例如 PLS、pairwise-warm-start、或其他 supervised projection；

$$
C(x)\in\mathbb R^{Q\times r},
$$

是平滑变化的低维 coefficient field；

$$
V\in\mathbb R^{D\times r},
$$

是 residual directions；

并且：

$$
r\ll D.
$$

这样 residual 部分：

$$
\Delta W(x)=C(x)V^\top
$$

只能在少数方向上变化，从而显著减少自由度。

---

## 5. Low-rank residual 的 uncertain-W kernel 仍然可用

如果对 $$C(x)$$ 放 GP prior：

$$
C_{qr}(x)\sim GP(0,k_c),
$$

则给定 posterior 后：

$$
C(x) \text{ 是 Gaussian},
$$

而：

$$
W(x)=W_0+C(x)V^\top
$$

是 $$C(x)$$ 的线性变换，所以 $$W(x)$$ 仍然是 Gaussian。

对 anchor-conditioned 版本：

$$
W_a=W(c_a)=W_0+C(c_a)V^\top.
$$

对 pair：

$$
\delta_{il}=x_i-x_l.
$$

投影差：

$$
\Delta z_{il}=W_a\delta_{il}
=W_0\delta_{il}+C(c_a)V^\top\delta_{il}.
$$

令：

$$
b_{il}=V^\top\delta_{il}\in\mathbb R^r.
$$

对第 $$q$$ 个 latent coordinate：

$$
\Delta z_{il,q}
=(W_0\delta_{il})_q+C_q(c_a)b_{il}.
$$

若：

$$
C_q(c_a)\sim \mathcal N(m_{aq},S_{aq}),
$$

则：

$$
\mu_{\Delta,q}
=(W_0\delta_{il})_q+m_{aq}^\top b_{il},
$$

$$
S_{\Delta,qq}
=b_{il}^\top S_{aq}b_{il}.
$$

因此可以直接放进现有 uncertain-W expected kernel 公式。这个改法不会破坏现有 uncertain-W CEM 框架。

---

## 6. Low-rank residual 本身不等于 feature selection

需要注意：

$$
W(x)=W_0+C(x)V^\top
$$

本身只是限制 residual variation 的 rank。它不自动忽略噪声 feature。

例如 synthetic expansion：

$$
x=(z,z^2,\eta),
$$

其中 $$\eta$$ 是纯噪声或不相关维度。我们希望 projection metric：

$$
\|W(x_i-x_l)\|^2
$$

主要依赖 $$z,z^2$$，而不依赖 $$\eta$$。

要实现这种 selection，需要加 feature-wise shrinkage，例如 ARD 或 residual group penalty。

---

## 7. 推荐的 residual group penalty

最推荐先对 residual 部分加 feature-column group penalty，而不是直接对整个 $$W$$ 加强 ARD。

定义：

$$
\Delta W(c_a)=C(c_a)V^\top.
$$

对每个原始 feature column $$d$$，定义 residual usage：

$$
G_d
=
\sqrt{
\frac{1}{A}
\sum_{a=1}^{A}
\|\Delta W(c_a)_{:d}\|_2^2
+\epsilon
}.
$$

加 penalty：

$$
\boxed{
\mathcal R_{\text{res-col}}
=
\lambda_\Delta
\sum_{d=1}^{D}G_d
}
$$

其作用是：如果 residual 在第 $$d$$ 个原始 feature 上没有必要，就把整列 residual 压小。

这样比直接对整个 $$W$$ 做 sparsity 更安全，因为 base projection $$W_0$$ 的信息不会被删掉；正则只限制局部 residual 不要乱用噪声维度。

---

## 8. ARD prior 的风险与使用方式

可以引入 feature scale：

$$
W_{:d}(x)=\gamma_d \tilde W_{:d}(x),
\quad
\gamma_d\ge 0.
$$

给 $$\gamma_d$$ 一个 shrinkage prior，例如：

$$
p(\gamma_d)\propto \exp(-\lambda \gamma_d),
$$

对应 penalty：

$$
\lambda\sum_d \gamma_d.
$$

如果第 $$d$$ 维是噪声，likelihood gain 小，正则会压低 $$\gamma_d$$。

但 ARD / sparsity prior 有风险：如果惩罚过强，会把有用但弱的 direction 压掉。

因此建议：

1. 使用 weak ARD，而不是 hard selection；
2. 只对 residual 加 ARD，不直接对整个 $$W$$ 加强惩罚；
3. 使用 annealing：先让模型找方向，再逐渐加正则；
4. 用 validation / controls 选强度。

推荐 annealing：

$$
\lambda_{\text{ARD}}(t)=0
\quad \text{for early steps},
$$

然后逐渐增加到：

$$
\lambda_{\max}.
$$

推荐初始网格：

$$
\lambda_\Delta\in\{10^{-4},10^{-3},10^{-2}\}.
$$

如果非高噪声 setting 上 RMSE/CRPS 明显下降，说明正则太强。

---

## 9. 如何避免 V 选得不好

固定：

$$
W(x)=W_0+C(x)V^\top
$$

时，如果 $$V$$ 的 span 不包含真正有用的 residual direction，则模型无法补救。因此需要让 $$V$$ 有足够安全的覆盖。

### 9.1 Overcomplete candidate basis

不要只用一个来源选 $$V$$。建议拼接多个 basis：

$$
V=[V_{\text{PLS}},V_{\text{PCA}},V_{\text{pairwise}},V_{\text{random}}].
$$

其中：

- $$V_{\text{PLS}}$$ 捕捉 response 相关方向；
- $$V_{\text{PCA}}$$ 捕捉输入主变化方向；
- $$V_{\text{pairwise}}$$ 捕捉 boundary / same-cross regime 方向；
- 少量 random directions 作为兜底。

总 rank 可以稍大：

$$
r=4,8,16.
$$

再用 shrinkage 控制有效 rank。

### 9.2 Basis scale selection

可以写成：

$$
W(x)=W_0+C(x)\operatorname{diag}(s)V^\top,
$$

其中：

$$
s_r\ge 0.
$$

对 $$s$$ 加弱稀疏 penalty：

$$
\lambda_s\sum_r s_r.
$$

这样 $$V$$ 可以包含多种候选方向，最终由 $$s$$ 启用或关闭某些 basis。

### 9.3 可学习 V，但要有正交约束

如果固定 $$V$$ 仍然受限，可以让 $$V$$ 可学习：

$$
W(x)=W_0+C(x)V^\top.
$$

但必须加约束：

$$
V^\top V=I.
$$

否则有 scale confounding：

$$
C(x)V^\top=(aC(x))(V/a)^\top.
$$

实现上可以每若干步对 $$V$$ 做 QR orthonormalization，或者用 Stiefel manifold retraction。

建议训练顺序：

1. 固定 $$W_0,V$$，只学 $$C(x)$$；
2. 如果效果受限，再解冻 $$V$$；
3. 解冻 $$V$$ 时使用小 learning rate；
4. 保持 $$V^\top V=I$$。

### 9.4 Full-rank escape hatch

可以保留一个小的 full residual：

$$
W(x)=W_0+C(x)V^\top+\epsilon B(x),
$$

其中 $$B(x)\in\mathbb R^{Q\times D}$$ 是 full residual，但给强 penalty：

$$
\lambda_B\|B(x)\|^2,
\quad
\lambda_B\gg 1.
$$

这样如果 $$V$$ 漏掉关键方向，模型仍有一条代价较高的逃生通道。

---

## 10. 不建议做 hard row-normalization

在 uncertain-W Gaussian 路线里，不建议直接做：

$$
W(x)\leftarrow \frac{W(x)}{\|W_q(x)\|}.
$$

因为 row normalization 是非线性的，会破坏 $$W(x)$$ 的 Gaussian 性，使 uncertain-W expected kernel 的闭式不再严格成立。

更稳的方式是：

1. 令 $$W_0$$ row-normalized；
2. 保持 residual scale 小；
3. 对 posterior mean 加 soft orthogonality penalty：

$$
\lambda_{\text{orth}}
\sum_a
\left\|
W(c_a)W(c_a)^\top-I_Q
\right\|_F^2.
$$

或者只对 $$\mathbb E[W(c_a)]$$ 加这个 penalty。

---

## 11. 建议的 low-rank residual 训练目标

在当前 uncertain-W CEM qR M-step 上，把 full $$W(x)$$ 换成：

$$
W(x)=W_0+C(x)V^\top.
$$

M-step objective：

$$
\mathcal Q(q(C))
=
\mathcal Q_{\text{uncertain-JGP}}(q(C))
-
\beta KL(q(C)\|p(C))
-
\lambda_\Delta \mathcal R_{\text{res-col}}
-
\lambda_s\|s\|_1
-
\lambda_{\text{orth}}\mathcal R_{\text{orth}}.
$$

其中：

$$
\mathcal Q_{\text{uncertain-JGP}}
$$

是目前已经实现的 uncertain-W CEM hard-label GP marginal likelihood + GH gate terms。

第一版建议只用：

$$
\mathcal Q(q(C))
=
\mathcal Q_{\text{uncertain-JGP}}(q(C))
-
\beta KL(q(C)\|p(C))
-
\lambda_\Delta \mathcal R_{\text{res-col}}.
$$

不要一开始同时学 $$V$$ 和太多正则项。

---

## 12. 推荐 ablation 计划

### 12.1 先做 seed0，验证结构是否有效

保持当前最强 prediction head：

```text
qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1
```

新增 low-rank variants：

| variant | W0 | V | r | penalty | 目的 |
|---|---|---|---:|---|---|
| lowrank_pls_r2 | PLS | PCA/PLS residual | 2 | none | 测低秩是否足够 |
| lowrank_pls_r4 | PLS | PCA/PLS residual | 4 | none | 测 rank 增大 |
| lowrank_pair_r2 | pairwise warm | PCA/PLS | 2 | none | 测 supervised W0 |
| lowrank_pair_r4 | pairwise warm | PCA/PLS | 4 | none | 测 supervised W0 + 大 rank |
| lowrank_pair_r4_ard1e-3 | pairwise warm | overcomplete | 4 | residual group | 测噪声选择 |
| lowrank_pair_r8_scale | pairwise warm | overcomplete | 8 | basis scale | 避免 V 太窄 |

Controls：

| control | 目的 |
|---|---|
| no-qR head | 判断 head-only gain |
| prior lowrank qR | 判断 learned posterior 是否有用 |
| shuffled lowrank qR | 判断 spatial binding |
| full WGP qR | 判断 low-rank 是否比 full 更稳 |
| original CEM | baseline |

### 12.2 判断标准

如果 low-rank learned 明显优于 no-qR / prior / shuffled，说明 projection field 终于有 spatial binding signal。

如果 low-rank 和 no-qR 仍接近，说明主要收益仍然来自 uncertain-W CEM head，而不是 learned W。

如果 low-rank 比 full WGP 更好，说明 full $$W(x)$$ prior space 过大是瓶颈之一。

如果 low-rank 加 residual group penalty 在高噪声 expansion 上更好，而在 L2/LH 不掉，说明 ARD / feature selection 有价值。

---

## 13. Feature relevance diagnostics

为了判断 low-rank / ARD 是否真的在做 selection，建议输出每个 feature 的 relevance：

$$
R_d
=
\frac{1}{A}
\sum_{a=1}^{A}
\mathbb E_q
\left[
\|W(c_a)_{:d}\|_2^2
\right].
$$

也可以只看 residual relevance：

$$
R_d^{\Delta}
=
\frac{1}{A}
\sum_{a=1}^{A}
\mathbb E_q
\left[
\|\Delta W(c_a)_{:d}\|_2^2
\right].
$$

对于 synthetic：

$$
x=(z,z^2,\eta),
$$

可以定义 signal energy：

$$
\text{signal energy}
=
\frac{
\sum_{d\in\text{signal}}R_d
}{
\sum_{d=1}^{D}R_d
}.
$$

如果方法有效，应看到：

$$
\text{signal energy}\uparrow,
\quad
R_d \text{ for noise dims } \downarrow.
$$

同时要比较 learned / prior / shuffled，避免把 prior geometry 误认为 learning。

---

## 14. V-span diagnostic

为了判断 $$V$$ 是否太局限，可以计算 full-W objective 对 $$W$$ 的梯度 $$G_W$$，再分解到 $$V$$ span 内外。

投影到 $$V$$ span：

$$
G_{\parallel}=G_WVV^\top.
$$

垂直部分：

$$
G_{\perp}=G_W-G_{\parallel}.
$$

诊断指标：

$$
\text{outside ratio}
=
\frac{\|G_{\perp}\|_F}{\|G_W\|_F+\epsilon}.
$$

如果 outside ratio 很大，说明当前 $$V$$ 漏掉了很多有用方向，low-rank residual 太局限。

如果 outside ratio 小，说明当前 $$V$$ span 足够覆盖主要 learning signal。

---

## 15. 推荐给 Codex 的实现顺序

### Step 1: 实现 low-rank residual moments

新增参数化：

$$
W(x)=W_0+C(x)V^\top.
$$

复用 sparse GP q(R) moment 逻辑，但 R 现在对应 $$C(x)$$，而不是 full $$W(x)$$。

输出 anchor-conditioned moments：

$$
\mathbb E[W(c_a)],
\quad
\operatorname{Var}(W(c_a)).
$$

并确保 expected SE kernel 可以直接用这些 moments。

### Step 2: 固定 W0,V，只学 C(x)

先做 MAP-C 或 small posterior std。

不要先学 V。

### Step 3: 加 residual group penalty

实现：

$$
\mathcal R_{\text{res-col}}
=
\sum_{d=1}^{D}
\sqrt{
\frac{1}{A}
\sum_a
\|\Delta W(c_a)_{:d}\|_2^2+\epsilon
}.
$$

注意这里 $$\epsilon$$ 是小数值稳定项。

### Step 4: 跑 seed0 ablation

L2/LH + structured noise synthetic。

先不要扩 5 seeds。

### Step 5: 加 V overcomplete / basis scale

实现：

$$
W(x)=W_0+C(x)\operatorname{diag}(s)V^\top.
$$

固定 overcomplete $$V$$，学习 $$s$$，加弱 penalty。

### Step 6: 如果 low-rank 有收益，再扩 5 seeds

只扩最有希望的 1–2 个 variants。

---

## 16. 当前不建议继续做的方向

### 16.1 不建议继续调 simultaneous pairwise auxiliary

实验显示：

- 小 pairwise 权重几乎无影响；
- 大 pairwise 权重损伤 CRPS；
- pair/data gradient 上来后并没有提升 final prediction。

因此当前 hard-label pairwise target 和 uncertain-JGP objective 不一致，不适合作为主 M-step regularizer。

pairwise 可以保留为：

1. warm start；
2. basis construction；
3. diagnostic；
4. 非主 objective。

### 16.2 不建议直接 full pointwise NN W(x) 作为主线

类似：

$$
z_i=W(x_i)x_i
$$

的 deterministic NN embedding + JGP head 虽然在某些 seed 上有信号，但偏离 DJGP 较远，而且 UQ 需要额外处理。

主线应优先保留 anchor-conditioned local projection：

$$
W_j=W(x_j^\ast),
\quad
z_i^{(j)}=W_jx_i.
$$

---

## 17. 最推荐的下一步主线

当前建议主线为：

$$
\boxed{
\text{Uncertain-W CEM head}
+
\text{Low-rank residual projection prior}
+
\text{residual group penalty}
}
$$

具体模型：

$$
W_j=W(x_j^\ast)=W_0+C(x_j^\ast)V^\top.
$$

训练：

$$
\mathcal Q(q(C))
=
\mathcal Q_{\text{uncertain-JGP}}(q(C))
-
\beta KL(q(C)\|p(C))
-
\lambda_\Delta \mathcal R_{\text{res-col}}.
$$

prediction：

1. 用 current $$q(C)$$ 得到 $$q(W_j)$$；
2. 用 uncertain-W expected kernel；
3. 用 weak self-CEM refresh；
4. 用 noise floor 防止 overconfidence；
5. 用 small kcov correction 传播 projection uncertainty。

当前默认 head：

```text
refresh10 self-CEM2 nstd0.1 kcov0.1
```

推荐先测：

```text
lowrank_pair_r4_resgroup1e-3_refresh10_self_cem2_nstd0.1_kcov0.1
```

以及：

```text
lowrank_pls_r4_resgroup1e-3_refresh10_self_cem2_nstd0.1_kcov0.1
```

---

## 18. 论文表述建议

如果 low-rank residual 成功，可以把故事写成：

原始 DJGP 中 full local projection posterior 过自由，且 local JGP decoder 很强，导致 learned $$W$$ 不稳定。我们提出 centered low-rank residual projection prior：

$$
W(x)=W_0+C(x)V^\top,
$$

用 empirical supervised projection $$W_0$$ 作为中心，只允许平滑低秩 residual 变化。配合 uncertain-W kernel，projection uncertainty 可以闭式传播到 JGP prediction；配合 residual group penalty，可以在高噪声 expansion 中抑制无关 features。

如果 low-rank residual 没有明显超过 controls，则应保守写成：

uncertain-W CEM improves probabilistic local prediction, but learned projection field remains difficult to identify under the current JGP objective. Low-rank residual priors and ARD-style shrinkage are promising but require stronger evidence.

---

## 19. 简短 checklist

后续 ablation 应至少输出：

- RMSE / CRPS / Cov90 / Width90；
- learned vs prior vs shuffled vs no-qR；
- R_mu / R_log_std data/KL ratio；
- label change rate；
- inlier rate；
- local noise / lengthscale / signal variance；
- feature relevance $$R_d$$；
- residual feature relevance $$R_d^\Delta$$；
- signal energy for structured-noise synthetic；
- V outside-gradient ratio。

这些指标能判断：

1. 是否真的学到了 $$W$$；
2. 是否只是 prediction head 改善；
3. 是否有 feature selection；
4. 是否因为正则过强丢失信息；
5. $$V$$ 是否太局限。
