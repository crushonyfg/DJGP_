# LMJGP 与 CEM 的改进汇总

更新时间：2026-05-23

本文档系统记录当前 repo 在 **原始 LMJGP/DJGP** 与 **JumpGP-CEM 局部头** 两条线上的全部主要改进。
目标是给一份 "回到这个项目时只需读这一篇" 的索引：每一项写明
**动机 → 数学/算法做了什么 → 代码位置 → 推荐 doc**，便于追溯。

涉及的目录：

- `djgp/variational*.py` — 原始 LMJGP/DJGP 的 ELBO 与 prediction。
- `djgp/projections/` — 所有围绕 projection (W) 与 CEM 的改进模块。
- `djgp/baselines/global_sparse_gp.py` — global Mahalanobis-GP 后端。
- `djgp/gate_fixed_slope.py`, `djgp/structured_projection.py`,
  `djgp/gp_feature_alignment.py`, `djgp/elbo_shrinkage.py`, `djgp/jumpgp_bridge.py`
  — LMJGP 的辅助层。
- `experiments/synthetic/compare_*.py` — 对应的 runner。
- `docs/*.md` — 每条改进对应的设计文档。

---

## 0. 名词与基线

| 简写 | 含义 | 关键代码 |
|---|---|---|
| **LMJGP / DJGP** | paper 中 variational projection-GP (`q(R) → q(W_j) → Ψ_1, Ψ_2 → q(u_j) → ELBO`) | `djgp/variational.py`, `DJGP_test.py` |
| **JGP-CEM (vendored)** | 传统局部 JumpGP-LD CEM，每个 anchor 一次 hard CEM | `JumpGaussianProcess/JumpGP_LD.py`, `utils1.jumpgp_ld_wrapper` |
| **W** | local projection matrix `Q × D`，对应 paper 中的 `W_j`（"M-step"）和全局 `R`（"E-step over inducing"） |
| **CEM head** | 用 (W,Z) 后的 local JumpGP-CEM 给 `(μ, σ)` |
| **Self-CEM** | repo 自实现的可微 CEM 驱动，替换 vendored CEM | `djgp/projections/uncertain_w_jgp.py::run_uncertain_w_self_cem` |

整体目标演化：

```
原始 LMJGP (ELBO + MC over q(V) at predict)
        │
        ├── 修复 ELBO shortcut / shrinkage / W drift                                   (§1)
        ├── 把 W 学法换掉                                                              (§2 CEM-EM, §3 uncertain-W EM, §4 MC-CEM REINFORCE)
        ├── 把 CEM head 换成自研 self-CEM，能传入 q(W) mean/var                        (§5)
        ├── 加 global-trend GP 做 additive global+local residual self-CEM             (§6)
        └── 加诊断/可视化以判断 W 是否真的学到了对 CEM 有用的方向                      (§7)
```

---

## 1. 原始 LMJGP ELBO 的诊断与就地修复

参考文档：[`lmjgp_elbo_diagnostics_summary.md`](lmjgp_elbo_diagnostics_summary.md)、
[`w_learning_ablation_plan_cn.md`](w_learning_ablation_plan_cn.md)、
[`projection_methods_math.md`](projection_methods_math.md)、
[`original_elbo_direct_prediction_analysis.md`](original_elbo_direct_prediction_analysis.md)。

### 1.1 链路与已知问题

原 ELBO 的 W signal 走：

\[
q(R) \to q(W_j) \to \Psi_1,\Psi_2 \to q(u_j) \to S_1,S_2 \to \log\sum\exp.
\]

经过逐层诊断，五层 bug:

| 层 | 现象 | 已修复 |
|---|---|---|
| 1. raw `y` scale | gate collapse，巨大负 ELBO | `--standardize_y` + explicit local mean `m_j` + residual `r=y-m_j` |
| 2. shrinkage 锁 | `σ_n^2 K^{-1}` 主锁、`G/Ψ_2` 次锁压住 `μ_u^*` | fixed/capped `σ_n` (`snfix0.3`)，`elbo_shrinkage.py` 的 β 分解诊断 |
| 3. mixture ρ 选择性过滤 | `ρ_i` 把 high-residual 点系统性降权，`Φ^T D_ρ r ≈ 0` | pure warmup + tempered `ρ_i^0 → 0.5` blend curriculum (`α_end=0.5`)；`μ_u^*` refresh |
| 4. train residual fit ≠ test improvement | `train_SSEg ↑` 但 `cos(e_0, ζ_*) ≈ 0.15` | "不再追更高 z_learn/r" — 目标改为 test-anchor alignment + CEM RMSE |
| 5. W drift | joint Adam W drift ~1.06，破坏 CEM | freeze-W / slow-W / two-timescale，trust-region |

### 1.2 通用 trick 与开关

- **KL warmup**：`djgp/variational_tricks.py::KLWarmupSchedule`
- **mu_V supervised init**：PLS/SIR 初始化 `q(R)` 均值，`init_mu_V_supervised`
- **q(W) target calibration**：`calibrate_mu_V_to_target_mu_W` 保证初始 `||W||` 在合理量级
- **lengthscale 中位数启发**：`init_lengthscales_median_heuristic`
- **row-norm penalty / row-norm-square penalty**：`row_norm_penalty_mu_W`,
  `row_norm_penalty_E_norm_sq`
- **辅助 metric correlation loss**：`auxiliary_metric_corr_loss`, `auxiliary_metric_corr_loss_A`
- **训练编排器**：`compute_objective_with_tricks`, `train_vi_with_tricks`

### 1.3 MC ELBO（取代 closed-form `Ψ_1, Ψ_2`）

参考文档：[`refactor_plan.md`](refactor_plan.md)、`docs/projection_methods_math.md` §"Monte Carlo ELBO"。

代码：`djgp/variational_mc.py`
- `qW_from_qR` / `sample_W` / `kl_qR`
- `compute_mc_elbo` — 每步 reparam 抽 `W`，对 local data 算 `log p(y|W)`
- `train_mc_vi` / `predict_mc_vi` / `predict_mc_vi_direct`

收益：跳过 `Ψ_1, Ψ_2`，让 `q(R)` 的 W gradient 直接来自 likelihood；信号链路更短。
代价：方差更大、wall time 更高，需要 reparam 与 grad clip。

### 1.4 Structured projection

代码：`djgp/structured_projection.py`、`docs/projection_methods_math.md`。

| 函数 | 作用 |
|---|---|
| `qW_lowrank_residual` | `W(x) = W_0 + C_θ(x) V^T`，`V` 固定，`C` 学习；正则化 `q(W)` 的 free DOF |
| `qW_full_centered` | 给 `q(W)` 减去 mean，方便加 prior |
| `s_penalty_log_lowrank` | 控制 `E[K]` smoothing 强度（`s = x^T Σ_W x`） |
| `propagate_sigma_B_to_X` | `Σ_W → Σ_X` 传播 |
| `clamp_projection` | 对 `σ_V` 加上限，防 q(W) 方差爆炸 |
| `sample_W_from_proj` | 一致采样 W |
| `projection_w_drift_metrics` | drift 诊断 |

### 1.5 Fixed-slope gate

代码：`djgp/gate_fixed_slope.py`。

减少 `W ↔ ν`（投影 × gate 斜率）的非可辨性：
- `learned` (legacy)
- `fixed_ones`：`v = 1/Q`
- `fixed_e1`：`v = e_1`
- `fixed_e1_temp`：`v = e_1`，但温度 `τ_g = softplus(log_tau_gate)` 可学

### 1.6 GP feature alignment 与 shrinkage 诊断

代码：`djgp/gp_feature_alignment.py`、`djgp/elbo_shrinkage.py`、
`docs/lmjgp_elbo_diagnostics_summary.md`。

诊断量（每个都对应了 §1.1 的一类 bug）：

| 量 | 含义 |
|---|---|
| `z_learn / r` | learned `μ_u` 给出的 residual GP signal 占 `r` 的比例 |
| `z_star / r` | 当前 ELBO quadratic 下，单优化 `μ_u` 理论可达的上界 |
| `z_ridge / r` | ridge regression upper bound（去除 ELBO 限制） |
| `B_ρ / (ρ̄ B_pure)` | ρ-mixture 与 pure-inlier 信号比 |
| `corr(ρ, |r|)` | ρ 是否选择性删 high-residual 点 |
| `β_G, β_K` 表 | 把 `(βG·G + βK·σ_n² K^{-1})^{-1}` 的两块解锁分开看 |
| `σ_n` sweep | 显示 noise escape |

辅助工具：
- `refresh_mu_u_toward_star` — `μ_u ← (1-α)μ_u + α·μ_u^*(ρ̃)`
- `mixture_elbo_star_metrics` — mixture-consistent star
- `mixture_rho_signal_diagnostics`、`ridge_r2`、`signal_strength`

### 1.7 Two-timescale / trust-region W learning

参考 §1.1 layer 5。training loop:
1. fast inner：`μ_u, Σ_u` 5 步
2. refresh：`μ_u ← (1-α)μ_u + α μ_u^*(ρ̃)`
3. slow W：`lr_W = lr × lr_W_scale`
4. 可选 W refresh
5. slow other：`m_j, gate, Z, lengthscale, σ_k` 每若干步一次

结论：**当前 ELBO 的 W gradient 与 CEM downstream 不对齐**；freeze-W / slow-W 反而 CEM RMSE 最好。这是后续 §3 §4 的动机。

---

## 2. CEM-aligned EM Projection（M-step 与 CEM 同口径）

参考文档：[`cem_em_projection.md`](cem_em_projection.md)；代码：
`djgp/projections/cem_em_projection.py`、`djgp/projections/projected_jumpgp.py`、
`djgp/projections/lowrank_factorization.py`。

### 2.1 模型

\[
W_j = \text{rowNorm}(W_0 + \alpha \cdot C_j V^T), \qquad
W_0 \in \mathbb R^{Q\times D},\ C_j \in \mathbb R^{Q\times R_v},\ V \in \mathbb R^{D\times R_v}.
\]

`V = I_D` 时退化为 unconstrained per-anchor W（`cem_em_free` 上界）。

### 2.2 算法

- **C-step**：对每个 anchor 跑 `jumpgp_ld_wrapper(..., mode="CEM")`，缓存
  `(r_j, w_j, log\theta_j, m_{s,j})`。
- **M-step**：在 inlier 子集上最大化

  \[
  \sum_j \log p_\mathrm{GP}(y_{I_j}; \log\theta_j, m_{s,j})
  + \lambda_g \sum_j -\mathrm{BCE}(\sigma(w_j \cdot Z_j + m_{s,j}), r_j)
  - \lambda_s \sum_{j,j'} K_{jj'} \|C_j - C_{j'}\|_F^2
  - \lambda_r \cdots.
  \]

  对 `(W_0, \{C_j\})` 做 Adam，对 cached CEM internals 视为常数。

### 2.3 关键 bug fix（不要回滚）

| Fix | 现象 → 解决 |
|---|---|
| **Row-normalize W & W_0_init** | raw-X PLS 行范数 ~10^4 会把 `α·C_jV^T` 淹没；`rowNorm` 把 `C` 投影到 `W_0` 法平面 → `anchor_dispersion ≈ 0` ⇒ pre-norm `W_0_init` |
| **Smoothness 归一化** | `Σ K_{jj'}\|C_j-C_{j'}\|^2` 对大 `T` 量级 10^3，压死 GP marginal ⇒ 除以 `Σ K_{jj'}` |
| **Perturbation scale α** | `α∈{1,3}` 足够；`α=10` 噪声大 |
| **Track best-W** | M-step loss 非单调（CEM 重 refit 会翻转 borderline 点） → 保留 best |

### 2.4 与 LMJGP-VI 的比较

UCI Parkinsons (128 anchors, n=35, seed×5)：

| Method | RMSE | CRPS |
|---|---:|---:|
| `pls_only` | 7.14 | 3.94 |
| `lmjgp_predict_vi` (paper baseline) | 4.49 | 1.73 |
| `cem_em_coeff` (best) | **3.58** | **0.72** |

50% RMSE / 82% CRPS 改善。

### 2.5 不同 C-posterior 后端

- `vi_cem.py` — full-covariance VI Gaussian over `C_j`
- `laplace_cem.py` — diagonal Hessian Laplace approximation
- `inductive.py` — empirical-Bayes coefficient-field interpolation（`gp_condition_coefficients`），
  用于 inductive evaluation
- `lowrank_factorization.py` — `build_V_basis`（PLS + PCA + random orth）与 `compute_pls_W0`

---

## 3. Uncertain-W JGP/CEM（把 q(W) 闭式传入 kernel/gate）

参考文档：[`uncertain_w_jgp_marginalization.md`](uncertain_w_jgp_marginalization.md)（推导）、
[`uncertain_w_djgp_method.md`](uncertain_w_djgp_method.md)（中文方案）。
代码：`djgp/projections/uncertain_w_jgp.py`（~4600 行，整个项目最大模块）。

### 3.1 关键封闭式：expected SE kernel

\[
\bar k(i,l) = \sigma_f^2
\left|I_Q + \tfrac{1}{\ell^2} S_{\Delta,il}\right|^{-1/2}
\exp\!\left(-\tfrac12 \mu_{\Delta,il}^\top (\ell^2 I_Q + S_{\Delta,il})^{-1}\mu_{\Delta,il}\right).
\]

实现：
- `expected_se_kernel_anchor` / `expected_se_kernel_anchor_fullcov`
- `expected_se_kernel_pointwise`
- `kernel_vector_covariance_anchor` / `kernel_vector_covariance_pointwise` —
  for delta-method second-order correction
- `gate_moments_anchor` / `gate_moments_pointwise` + `gh_logsigmoid_expectations`
  — uncertain-W gate（logistic-normal 用 Gauss-Hermite quadrature）

### 3.2 Uncertain-W CEM / VEM

- `run_uncertain_w_self_cem`：把 `W_mu_anchor, W_var_anchor` 传入；不再需要 sample W
- `run_uncertain_w_self_vem`：soft `ρ_i = σ(S_1 - S_0)`；带温度、`responsibility_floor`、`damping`
- `uncertain_w_cem_update_labels` / `uncertain_w_vem_update_responsibilities`
- `uncertain_w_cem_predict_from_labels` / `uncertain_w_jgp_vem_predict`
- `weighted_gp_train_posterior` — 软标签下的 weighted GP posterior

### 3.3 闭式 q(R) M-step（不 MC）

`train_qr_uncertain_w_cem_mstep` (`UncertainWQRMstepConfig`)
- 固定 C-step labels / logθ / gate
- M-step 直接对 `q(R) → q(W) → E[K(W)]` 反传 GP marginal log-likelihood
- 支持：anchor pairwise / rank loss、anchor predictive NLL（leave-one-out
  inductive supervised loss）、residual group penalty、basis scale 学习、mean
  parameterization 选项、SVD log-scale clamp、orthogonality penalty。

### 3.4 Sparse-GP projection field

代码：`djgp/projections/global_wgp_cem.py::SparseGPProjectionField`。

\[
W_{q,d}(x) = k(x, U)\, K(U,U)^{-1}\, R_{:,q,d}.
\]

- 共享 inducing 上 `q(R) = \mathcal N(R_\mu, \text{diag}(R_{\log std}^2))`
- DTC conditional：`sparse_gp_uncertain_w_moments_for_mode` 同时支持 `mode="anchor"`
  和 `mode="pointwise"`，给出 `W_mu, W_var`（含 conditional residual variance）
- `sparse_gp_lowrank_residual_w_moments_for_mode` — `W = W_0 + V·R` 的低秩残差版本

### 3.5 MC-CEM REINFORCE / supervised anchor（route B）

代码：`train_qr_mc_cem_mstep` + `MCCEMQRMstepConfig`；
runner：`experiments/synthetic/compare_mc_cem_qr_lh.py`,
`compare_l2_reinforce_advantage.py`。

| 选项 | 含义 |
|---|---|
| `objective="stopgrad_reparam"` | route A：reparam 抽 W，detached 上 refit CEM，再对 frozen-CEM logpdf 反传 |
| `objective="score_function"` | route B：REINFORCE，`reward * ∇log q(W)` |
| `reward_mode="supervised_anchor"` | reward = anchor predictive logpdf；anchor 不进 local fit |
| `reward_advantage` | `center` / `center_std` / `rank` / `rank_norm`（per-anchor 跨 W sample 中心化） |
| `anchor_weight` | `uniform` / `spread_std` / `spread_range` / `top_fraction` / `spread_threshold`，可选 curriculum |
| `anchor_weight_curriculum_steps` | 早期按 spread 加权，后期 blend 回 uniform |
| `log_sample_rewards` | 记录 per-step 每个 W sample × anchor 的 reward、`ΔR vs W_0`、top-anchor indices |
| `restore_state` | `best_loss` / `final` / `best_eval` |

诊断脚本：
- `experiments/synthetic/l2_reinforce_advantage_specs.py`：7 个推荐 variant
- `experiments/synthetic/l2_reinforce_diagnostics.py`：top-anchor overlap、rank
  stability、ΔR vs W₀、W drift
- `experiments/synthetic/compare_l2_reinforce_advantage.py`：完整 metrics.csv

> 25+8 step、7 variant 实验结论（见对话历史与 `l2_reinforce_advantage_cmp_25x8/metrics.csv`）：per-anchor centering 实现稳定，但 hard-anchor weighting 在当前 L2 setting 下 **未压低 CEM RMSE**；部分 variant W drift 变大同时指标不变 / 略差。下一步应加 ranking-stability filter 过滤 noisy hard anchors。

### 3.6 与 paper ELBO 的关系

`log N(y; 0, E[K(W)] + σ²I)` 不是 `E_W[log N(y; 0, K(W) + σ²I)]`；
是 moment-matched marginalized-W surrogate，不是 exact ELBO。
但它的好处是：

- 链路更短（绕开 `Ψ_1, Ψ_2 → μ_u`），data gradient 直接打到 `Var(R)`
- 不需要 MC 即可把 W posterior variance 传入 predictive variance
- 自然适合做 self-CEM / self-VEM 的 fixed-label M-step

---

## 4. Global-Z supervised JGP / self-CEM 交替

参考文档：[`global_z_supervised_jgp_plan_cn.md`](global_z_supervised_jgp_plan_cn.md)、
[`neighborhood_retrieval_hypothesis_experiment_plan_cn.md`](neighborhood_retrieval_hypothesis_experiment_plan_cn.md)。

代码：`experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py`,
`compare_projected_dataset_selfcem_l2_lh.py`。

核心想法：

\[
z = W_\theta(x)\,x,\quad N_a(\theta) = \text{KNN}(z_a, \{z_i\}, K).
\]

- E-step：detached `Z_train` 上重新选 KNN
- M-step：固定 KNN，用 supervised anchor predictive NLL 反传 `θ`
- 不在 raw pool 内 reranking；让 `W` 真正控制全局 neighborhood geometry

variants：`static`, `lowrank` (`W_0 + C(x)V^T`), `mlp`，加 `shuffled / random`
对照检查 spatial binding 是否真学到。

---

## 5. CEM head 本身的改进（self-CEM 系列）

### 5.1 总览

| 模块 | 角色 |
|---|---|
| vendored `jumpgp_ld_wrapper` | 旧 baseline；不可微 |
| `run_uncertain_w_self_cem` (`SelfCEMConfig`) | 自研可微 CEM 驱动 |
| `run_uncertain_w_self_vem` (`SelfVEMConfig`) | 软 responsibility 版本 |
| `_final_sampled_self_cem_predict` (in `compare_mc_cem_qr_lh.py`) | 采样 W 的 ensemble CEM 预测头 |
| `global_residual_selfcem.py` | global GP + 局部 residual self-CEM 的additive 头 (§6) |

### 5.2 Self-CEM 的具体改进

代码：`djgp/projections/uncertain_w_jgp.py`（hyperopt / gate / labels 三块）。

1. **可学习 kernel hyperparameters**（`fit_uncertain_kernel_hyperparams_from_labels`,
   `FixedLabelHyperoptConfig`）
   - 学 `lengthscale, signal_var, noise_var`；带 box constraints
     (`min_noise_var, max_noise_var, min/max_signal_var, min/max_lengthscale`)
   - `batched=True` 时 anchors 并行优化，相对 per-anchor loop 大幅加速
2. **可学习 gate**（`fit_uncertain_gate_from_labels`, `FixedLabelGateConfig`）
   - `linear` 或 `quadratic` basis；带 L2 / `max_norm` 正则；Gauss-Hermite
     integration 处理 uncertain-W gate
   - `center_linear_gate`（`djgp/jumpgp_bridge.py`）把 `w_0 + w_slope^T z`
     转成 anchor-relative `b + w_slope^T (z - z_anchor)`
3. **Outlier model**（`outlier_mode`）
   - `background_normal`：广 Normal `N(y; mean, (σ × outlier_sigma_mult × background_scale_mult)^2)`
   - `constant`：可选 `learn_outlier_constant` 学带 bounded log-prob
   - `min_inliers`, `max_inlier_frac` 安全防护
4. **可选的 W 不确定性传入** — 见 §3，`W_mu_anchor`, `W_var_anchor`,
   `W_cov_anchor` 全协方差；`mode="anchor"` 或 `"pointwise"`
5. **`refit_after_update`** — 每次 label 更新后 refit hyperparams 与 gate
6. **soft responsibility VEM** (`SelfVEMConfig`) — 温度、floor、damping，防 hard
   collapse；soft `ρ_i` 可直接被 §3.3 的 q(R) M-step 用
7. **CEM cache reuse** — `cem_em_projection` 把 `(r_j, w_j, logθ_j, m_{s,j})`
   缓存复用，让 C-step 和 M-step 共享状态

### 5.3 Sampled-W CEM ensemble at predict time

代码：`experiments/synthetic/compare_mc_cem_qr_lh.py::_final_sampled_self_cem_predict`,
`_sample_anchor_w`。

流程：
1. 训练后从 `q(R) → q(W_anchor)` 给出每个 test anchor 的 W posterior
2. 抽 `n_samples` 个 `W^{(s)}`，对每个独立 refit local CEM
3. 把 `n_samples × T` 个 `(μ, σ²)` 做 ensemble：

   \[
   \bar\mu = \tfrac{1}{S}\sum_s \mu^{(s)},\quad
   \bar\sigma^2 = \tfrac{1}{S}\sum_s \sigma^{(s)2} + \mathrm{Var}_s(\mu^{(s)}).
   \]

这是当前最稳的 final prediction head（优于 analytic direct head，见 §1.7）。

---

## 6. DMGP + 局部 residual self-CEM（additive global+local）

参考文档：[`dmgp_local_residual_selfcem.md`](dmgp_local_residual_selfcem.md)、
[`residual_ablation.md`](residual_ablation.md)。

代码：`djgp/projections/global_residual_selfcem.py`、
`djgp/baselines/global_sparse_gp.py`、
`experiments/synthetic/compare_global_residual_selfcem_lh.py`,
`compare_global_wgp_cem_l2_lh.py`,
`compare_dmgp_uncertain_w_cem_head.py`。

### 6.1 模型

\[
y(x) = g(z_\theta(x)) + h_a(z_\theta(x)) + \epsilon,\quad
z_\theta(x) = W_\theta(x)\,x.
\]

- `g`：global sparse SVGP（learned Mahalanobis 特征） — `GlobalGPModel`
- `h_a`：anchor-specific local residual JumpGP / self-CEM
- 关键：local CEM **不把** `g` 视为 deterministic，而是把 `Σ_g(X_a, X_a)` 加入
  local covariance：

  \[
  C_{a,I} = K_{h,a,I} + \Sigma_g(X_{a,I}, X_{a,I}) + \sigma_a^2 I.
  \]

### 6.2 5 个 variant（`VARIANTS = ("diagvar", "blockvar", "oob_diagvar", "alt", "alt_ensW")`）

| Variant | 区别 |
|---|---|
| `diagvar` | `Σ_g` 用 marginal `v_g(z_i)`：等价于 heteroscedastic local noise floor `σ²_eff = σ_a² + v_g(z_i)` |
| `blockvar` | full block `Σ_g(X_a, X_a)`（含 off-diag） |
| `oob_diagvar` | 同 `diagvar`，但 `μ_g, v_g` 通过 `predict_oob_kfold` 给出，避免 train 上 double-counting |
| `alt` | 交替更新 `q(g) ↔ {ρ_a, θ_a}`；global refit 用 `y_i - \hat h_i` 与 `σ_eps² + \hat v_{h,i}` heteroscedastic pseudo-obs |
| `alt_ensW` | `alt` + sampled-W deterministic-CEM ensemble at prediction time |

### 6.3 Predict-time

加性 mean/var：

\[
\mu_y = \mu_g + \mu_h, \quad v_y = v_g + v_h.
\]

`σ_eps²` 已折进 `v_h`，不重复加。`alt_ensW` 在外层多加 sample-over-`W` 的
mixture variance。

### 6.4 配套 global GP 后端

`djgp/baselines/global_sparse_gp.py`：
- `GlobalGPModel` — gpytorch SVGP，学 lengthscale / variance / noise
- 支持 `_RawSVGP`（raw X）与 `_LatentSVGP`（feature-mapped Mahalanobis）
- `predict_oob_kfold` / `predict_oob_ensemble` — 防 train double-counting

---

## 7. 诊断与可视化

| 目的 | 工具 |
|---|---|
| ELBO shrinkage 分解 | `djgp/elbo_shrinkage.py` (`elbo_star_solve`, `beta_shrinkage_grid`, `sigma_n_shrinkage_sweep`) |
| ρ-selective filtering | `djgp/gp_feature_alignment.py` (`mixture_rho_signal_diagnostics`) |
| W drift | `djgp/structured_projection.py::projection_w_drift_metrics`、`experiments/synthetic/l2_reinforce_diagnostics.py::w_drift_frobenius_mean` |
| reward signal / top-anchor 稳定性 | `l2_reinforce_diagnostics.py` (`summarize_reward_trace`, `write_spread_histogram`) |
| anchor predictive NLL alignment | `UncertainWQRMstepConfig.anchor_pred_weight` + leave-one-out anchor logpdf |
| projection quality | `djgp/projections/metric_variants.py::projection_quality_diagnostics`, `lowrank_factorization.per_anchor_W_diagnostics` |
| 5-seed 汇总 | `docs/five_seed_results_summary.md`, `five_seed_results_total_table.{md,csv}` |
| 实验时间线 | `docs/experiments_log.md`（持续追加） |

---

## 8. 推荐入口（"现在去跑实验" 速查）

| 任务 | 入口脚本 |
|---|---|
| paper LMJGP baseline（`predict_vi`） | `DJGP_test.py`、`experiments/synthetic/compare_paper_synthetic_baselines.py` |
| CEM-EM projection (UCI) | `experiments/uci/compare_cem_em_projection.py`、`compare_inductive_projection.py` |
| Self-CEM / Uncertain-W 主线 | `experiments/synthetic/compare_uncertain_w_cem_l2_lh.py`、`compare_mc_cem_qr_lh.py` |
| Supervised REINFORCE q(R) 训练 | `experiments/synthetic/train_supervised_reinforce_l2.py` |
| REINFORCE advantage 对比 | `experiments/synthetic/compare_l2_reinforce_advantage.py` |
| Global+Residual additive | `experiments/synthetic/compare_global_residual_selfcem_lh.py`、`compare_global_wgp_cem_l2_lh.py` |
| Global-Z supervised JGP / self-CEM 交替 | `experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py`、`compare_projected_dataset_selfcem_l2_lh.py` |
| LMJGP 各 ELBO trick 对照 | `experiments/synthetic/compare_lmjgp_tricks_synth.py`、`compare_lmjgp_*_paper.py` 系列 |

---

## 9. 当前主线判断（2026-05-23 时点）

1. **paper LMJGP ELBO + direct analytic head 的 W learning 链路是弱的**：
   高 `z_learn/r` 不等价于 test improvement；test 上反而是 `W ≈ W_0` + 强 local
   CEM 头最稳。
2. **CEM-EM projection（§2）已经在 UCI 上明显胜过 `predict_vi`**：当前最强的
   "学 W" 路线之一。
3. **Uncertain-W CEM（§3）作为 prediction head 是稳的改进**：可以在不损害点
   预测的前提下把 `W` 不确定性传进 predictive variance；但 q(R) 的 EM 训练能否
   反过来明显提升 RMSE，目前仍只是 "持平 / 微弱改善"。
4. **REINFORCE supervised anchor（§3.5）**：per-anchor centering advantage 是必
   要的，但单纯 hard-anchor weighting 还没拉开 CEM RMSE。下一步要加 ranking
   stability filter 与 ΔR-vs-W₀ 控制。
5. **DMGP + residual self-CEM（§6）**：把 global GP 的 posterior variance 闭式
   传入 local CEM 是 promising 的概率/标定改进；与单纯加 deterministic residual
   不同，避免了 "残差是 noise-free" 的隐性假设。
6. **CEM head 本身**：自研 self-CEM（§5）已经覆盖 vendored CEM 的全部能力，
   并支持 uncertain-W、soft VEM、batched hyperopt、anchor-relative gate、
   constant outlier 学习等；它是后续所有 W-learning 的稳定底座。

---

## 10. 推荐阅读顺序

1. [`paper_code_map.md`](paper_code_map.md) + [`repository_map.md`](repository_map.md) — 全局结构
2. 本文件 — 改进汇总
3. [`cem_em_projection.md`](cem_em_projection.md) — CEM-aligned EM（最完整的 case study）
4. [`uncertain_w_djgp_method.md`](uncertain_w_djgp_method.md) / [`uncertain_w_jgp_marginalization.md`](uncertain_w_jgp_marginalization.md) — uncertain-W EM 推导
5. [`dmgp_local_residual_selfcem.md`](dmgp_local_residual_selfcem.md) — additive global+local 模型
6. [`lmjgp_elbo_diagnostics_summary.md`](lmjgp_elbo_diagnostics_summary.md) — 原 ELBO 的层层诊断
7. [`w_learning_ablation_plan_cn.md`](w_learning_ablation_plan_cn.md) — W learning 的现状与计划
8. [`global_z_supervised_jgp_plan_cn.md`](global_z_supervised_jgp_plan_cn.md) — 全局 Z-space 重投影方向
9. [`experiments_log.md`](experiments_log.md) — 实验时间线（最近变化）
