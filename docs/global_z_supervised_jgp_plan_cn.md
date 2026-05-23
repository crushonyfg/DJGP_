# Global-Z Supervised JGP：用 (W(x)x) 重投影整个 dataset 的 JGP 监督实验

## 2026-05-17 更新：self-CEM 交替版

最新要测试的不是旧版 surrogate-only runner，而是把 global-Z 放回当前
self-CEM 流程里：

1. 用当前 \(W_\theta(x)\) 把整个 train/test dataset 映射为
   \(z_i=W_\theta(x_i)x_i\)；
2. 在全局 \(Z\)-space 里重新选 KNN；
3. 在这些新 neighborhoods 上重新 fit uncertain-W self-CEM，得到 labels /
   hyperparameters；
4. 固定新 neighborhoods 和 CEM 划分，在 supervised anchor predictive NLL +
   regularization 下更新 \(W_\theta\)；
5. 重复上述 refresh/M-step。

对应实现和结果见
`experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py` 的
`*_selfcem*` variants，以及 `docs/experiments_log.md` 中
`2026-05-17 - global_z_selfcem_supervised_lh_seed0_20260517`。旧的
`static_supervised` / `lowrank_supervised` 是 RBF-GP surrogate-only 版本，
不能代替这次 self-CEM 交替实验。

## 1. 背景与定位

本实验是对 `docs/neighborhood_retrieval_hypothesis_experiment_plan_cn.md`
（hybrid raw+W retrieval 思路）的明确**替代方向**，不是补充。

hybrid reranking 始终把 raw locality 作为主约束，只在 raw candidate pool 内
做小幅重排：

$$
d_{\text{hyb}}=(1-\lambda) d_{\text{raw}}+\lambda d_W .
$$

这适合做 conservative correction，但**不能测试**：

$$
Z=\{z_i=W(x_i)x_i\}\text{ 是否可以成为 JGP 的新输入空间？}
$$

这条更强命题需要让 (W) 真正控制全局 neighborhood geometry，并直接用
**supervised anchor predictive objective** 评价这个 latent space 是否使 JGP
更会预测。

> 关键决策：**不再在 raw pool 内重排，而是用 (W) 重投影整个 train/test
> dataset，再在投影后的 (Z)-space 里重新选 KNN**。

---

## 2. 形式化

### 2.1 全局重投影

对所有 training points：

$$
Z_{\text{train}}=\{z_i=W_\theta(x_i)x_i\}_{i=1}^{N_{\text{train}}}.
$$

对 test point：

$$
z_\ast=W_\theta(x_\ast)x_\ast.
$$

Neighborhood 直接在 (Z)-space 选：

$$
N_\ast(\theta)=\operatorname{KNN}(z_\ast, Z_{\text{train}}, K).
$$

### 2.2 Supervised anchor predictive objective

不直接最大化 selected-neighborhood marginal likelihood（不同 (\theta) 选不同
的 (N_a(\theta))，不可比）。对每个 training anchor (a)，把它当成 pseudo-test
point，**leave-one-out** 评价 anchor predictive NLL：

$$
\mathcal A\subset\{1,\dots,N_{\text{train}}\},
\qquad
N_a(\theta)=\operatorname{KNN}(z_a,\{z_i:i\neq a\},K).
$$

可微 GP surrogate（local RBF GP）下，预测 anchor response：

$$
\log p_{\text{GP}}\bigl(y_a\mid \{(z_i,y_i):i\in N_a(\theta)\}\bigr).
$$

训练目标：

$$
\mathcal J(\theta)
=
\sum_{a\in\mathcal A}
\log p_{\text{GP}}\bigl(y_a\mid \{(z_i,y_i):i\in N_a(\theta)\}\bigr)
-
\lambda\mathcal R(\theta).
$$

注意：

- 这里的 GP 是**可微 RBF surrogate**（学习的 lengthscale / signal / noise），
  目的是给 (\theta) 一个稳定的可微 supervised signal。
- 最终 test prediction 仍用 JumpGP CEM head（不可微，但只在推理时使用），
  邻居由 (Z)-space KNN 决定。
- 选邻居本身是 hard KNN，不可微；用 alternating EM：每 (k) 步用 detached
  current (Z_{\text{train}}) 刷新一次 KNN，inner loop 内固定 (N_a(\theta))，
  通过 (Z_a(\theta), Z_{\text{nb}}(\theta)) 反传梯度。

### 2.3 W 的参数化

第一版用 deterministic (W_\theta(x))，不做 (q(R)) posterior：

- `static`: (W_\theta(x)=W_0)，单一全局 (W_0\in\mathbb R^{Q\times D})。
  这等价于学习一个全局线性 projection；和 PCA/PLS baseline 形成对比，但
  projection 是 supervised by anchor predictive NLL。
- `lowrank`: (W_\theta(x)=W_0+C_\theta(x)V^\top)。其中 (W_0\in\mathbb R^{Q\times D})
  是 PCA/PLS init 的全局 mean，(V\in\mathbb R^{D\times r}) 是 PCA/PLS residual
  basis (固定)，(C_\theta(x)\in\mathbb R^{Q\times r}) 由 MLP 输出，
  实现 input-dependent 修正。
- `mlp`: 直接 MLP 输出 (W_\theta(x)\in\mathbb R^{Q\times D})。最 flexible，
  最容易 overfit。

第一版重点跑 `lowrank` 和 `static`，因为：

- `static` 是 sanity（如果 supervised projection 都不能赢 PCA/PLS，方向就败了）；
- `lowrank` 是真正测试 (x\mapsto W(x)) 的 spatial binding。
- `mlp` 留作可选 stretch。

---

## 3. Variants

### Variant 1：deterministic global (W_\theta(x)x) + 可微 RBF GP surrogate

- E-step（每 `outer_refresh` 步刷一次）：detach (Z_{\text{train}})，对每个
  anchor 计算 (Z)-space KNN（leave-one-out, top-K）。
- M-step（`inner_steps` 步）：固定 KNN indices，反传 anchor predictive NLL
  关于 (\theta)（W 网络 + 可微 GP hyperparams）。
- 终态：刷新最后一次 (Z_{\text{train}}) 与 (Z_{\text{test}})；test prediction
  用 JumpGP CEM head 在 (Z)-space KNN 上推理。

### Variant 2：soft retrieval surrogate (optional, 后续)

对 anchor (a) 用 softmax weights：
$$
\pi_{ai}=\frac{\exp(-d_{ai}/\tau)}{\sum_l\exp(-d_{al}/\tau)},
\qquad d_{ai}=\|z_a-z_i\|^2.
$$
配合 ranking loss 用同一 (W) 网络。先不实现，留作 follow-up。

### Variant 3：uncertain-W moments (optional, 后续)

(q(W(x))=\mathcal N(M(x),S(x)))，predict 时使用 expected kernel。
第一版**不**实现，避免 posterior approximation 噪声。

---

## 4. Controls

| Control | 作用 |
|---|---|
| `raw_jgp` | 原始 X-space JGP baseline（CEM） |
| `pca_jgp` | 静态 projection baseline (X std + PCA→Q) + JGP |
| `pls_jgp` | y-aware 静态 projection baseline + JGP |
| `static_supervised` | 学习全局 (W_0)（无 input dep），supervised |
| `lowrank_supervised` | (W_0+C(x)V^\top) supervised（主实验） |
| `lowrank_shuffled` | 训练后 (W_\theta) 的输入 x 被随机置换，破坏 spatial binding |
| `lowrank_random` | (W_\theta) 不训练，random init 直接用 |
| `mlp_supervised` (optional) | full MLP W，flexible 上界 |

判断准则：

- 若 `lowrank_supervised` < `pca_jgp` 和 `pls_jgp` 和 `raw_jgp`，说明
  supervised global-Z 方向值得继续。
- 若 `lowrank_supervised` < `lowrank_shuffled`，说明 spatial binding
  (x_i\mapsto W_\theta(x_i)) 真的被学到。
- 若 `lowrank_supervised` ≈ `static_supervised`，说明 input-dependent W
  没有额外贡献，全局线性 supervised projection 就足够。
- 若 `lowrank_random` ≈ `lowrank_supervised`，说明主要收益来自 random
  projection smoothing，方向再次失败。

---

## 5. 实验设置（Stage 1：LH seed 0）

固定：

- 数据：`lh_q5_train1000`（paper LH，RFF expansion，(d=5)，(H=30)，
  N_train=1000, N_test=200）。
- Standardization：train-only X StandardScaler，y 不标准化（与
  `compare_paper_synthetic_baselines.py` 一致）。
- 学习目标 (Q=5)（与 `Q_true=5`、其它 LH baseline 一致）。
- (K_{\text{neighbor}}=35)（与 LH JGP 默认一致）。
- Anchors：默认全部 (N_{\text{train}}=1000)；可选 `kmeans_yvar` 100 anchors
  做较快版本。
- W 训练超参（默认）：`outer_refresh=10`，`inner_steps=50`，总 `epochs=200`。
- 优化器：Adam，lr=1e-2 for GP hypers, 1e-3 for W 网络。
- W 初始化：`W_0`=PLS via `_pls_directions_raw` 的 (W^\top\in\mathbb R^{Q\times D})；
  `V`=PCA-residual basis (D×r), `r=8`，与 (W_0) 正交化。
- 正则：W frobenius L2 (1e-4)，C MLP weight decay (1e-4)。
- GP surrogate：RBF kernel，learnable `log_ell` per output, `log_sf`, `log_sn`；
  per-anchor mean subtraction (`y_nb.mean()`).
- 仅 seed 0。

只有在 seed0 supervised global-Z **既赢 PCA/PLS/raw baseline，又赢 shuffled
control** 时，才扩 5 seeds 或 uncertain-W 后续版本。

---

## 6. Diagnostics

每个 variant 输出：

1. 数据级：`rmse`, `mean_crps`, `coverage_90`, `mean_width_90`, `coverage_95`,
   `mean_width_95`, `wall_sec`。
2. Z-space 几何：训练后 (\|z_i\|) 的 mean/std；Z-KNN 与 raw-X-KNN 的
   per-anchor overlap mean/std。
3. 训练动态：每 `outer_refresh` 步的 train predictive NLL，GP hypers
   (\log\ell,\log s_f,\log s_n)，refresh 后 KNN label change rate。
4. Anchor predictive：训练结束时 anchor leave-one-out RMSE / CRPS（用
   surrogate GP 与可选 JumpGP）。

---

## 7. 不做的事

- **不在 raw pool 内重排** —— 这是 hybrid 的工作，已经被本计划替代。
- **第一版不上 (q(R))** —— 避免 posterior 噪声混入这次结论。
- **不直接 marginal likelihood** —— 不可比，见 §2.2 推理。
- **不扩 seeds**，直到 seed0 supervised global-Z 已经稳定赢 PCA/PLS/raw 与
  shuffled control。

---

## 8. 推荐执行顺序

1. smoke (max_test=10, epochs=10, outer_refresh=2, inner_steps=3)，
   验证管线 + 数值稳定。
2. seed0 stage1：`raw_jgp`, `pca_jgp`, `pls_jgp`, `static_supervised`,
   `lowrank_supervised`, `lowrank_shuffled`, `lowrank_random`。
3. 若 seed0 supervised 赢，再考虑 `mlp_supervised` 与 soft retrieval。
4. 若 seed0 supervised 不赢，**不要**扩 5 seeds；该方向已被 falsify。

入口：

```text
experiments/synthetic/compare_global_z_supervised_jgp_l2_lh.py
artifacts: experiments/synthetic/global_z_supervised_jgp_lh_seed0_<YYYYMMDD>/
```
