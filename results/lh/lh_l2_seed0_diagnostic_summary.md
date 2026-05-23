# LH / L2 seed=0 诊断：z-vs-X neighborhood overlap 与 JGP-on-(z,y)

## 数据出处（provenance, 严格按 AGENTS.md "synthetic benchmark provenance" 要求登记）

- 生成器：`experiments/synthetic/compare_paper_synthetic_baselines._generate_paper_data`
  - LH：`new_highdata_gen_utils.generate_data`（caseno=0，paper L-H 高维合成数据）
  - L2：`new_highdata_gen.generate_data_phantom`（caseno=6，paper L2/phantom 图像分区）
- 扩展方法：Appendix-C RFF（`sample_rff_params` + `make_rff_features`），LH H=30，L2 H=20
- 标准化：X 用 train-only StandardScaler；z 直接来自生成器（未标准化）
- 来源文件（已存在的 npz, 由 `export_paper_synthetic_for_dmgp.py` 导出，对应同一份 seed=0 流程）：
  - LH：`experiments/synthetic/deep_mahalanobis_gp_paper_train{200,500,1000}_*seed0/lh_q5_train*_seed0.npz`
  - L2：`experiments/synthetic/deep_mahalanobis_gp_paper_train{200,500,1000}_*seed0/l2_q2_train*_seed0.npz`
- 各 split 元信息：LH d=5, Q_true=5, H=30, n=35；L2 d=2, Q_true=2, H=20, n=25

## 问题 1：z-空间 35-NN 与 X-空间 35-NN overlap

报告**test→train cross-KNN overlap**（即 JGP 在测试时实际做的检索：每个 test anchor 在 train pool 上选 KNN）。within-train 结果基本一致（差 ≤ 1）。

| Setting (seed 0) | train×test | dim(z)/dim(X) | k | mean overlap | std | min/max | mean / k |
|---|---|---|---:|---:|---:|---:|---:|
| LH train=200  | 200 / 200 | 5 / 30 | 35 | **27.09** | 2.58 | 17 / 32 | 0.774 |
| LH train=500  | 500 / 200 | 5 / 30 | 35 | **24.98** | 2.44 | 19 / 32 | 0.714 |
| LH train=1000 | 1000 / 200 | 5 / 30 | 35 | **24.62** | 2.49 | 18 / 31 | 0.703 |
| L2 train=200  | 200 / 200 | 2 / 20 | 35 | **30.92** | 2.03 | 26 / 35 | 0.883 |
| L2 train=500  | 500 / 200 | 2 / 20 | 35 | 29.07 | 2.62 | 20 / 34 | 0.830 |
| L2 train=1000 | 1000 / 200 | 2 / 20 | 35 | **32.64** | 1.36 | 28 / 35 | 0.932 |

L2 默认 paper n=25 时（同一 seed=0，同 split）：

| Setting | k | mean overlap | std | min/max | mean / k |
|---|---:|---:|---:|---:|---:|
| L2 train=200  | 25 | 22.12 | 1.79 | 17/25 | 0.885 |
| L2 train=500  | 25 | 20.68 | 2.07 | 15/25 | 0.827 |
| L2 train=1000 | 25 | 23.39 | 1.09 | 21/25 | 0.936 |

要点：

- L2（d=2 → H=20 RFF）overlap 比例非常高（83%–94%），说明 RFF 在很大程度上保留了 2D 邻域结构，X-KNN ≈ z-KNN。
- LH（d=5 → H=30 RFF）overlap 比例明显较低（70%–77%）；维度更高、几何更稀，RFF 把约 30% 邻居打乱。LH 上 X-KNN ≠ z-KNN 的差距才是 W-aware retrieval 真正可以发挥的空间。
- 随 train pool 变大，绝对 overlap 仍然显著（LH train=1000 时 35 个邻居里平均 24–25 个相同），但 fraction 随 N 上升不一致：LH 略降（pool 变密、X 的近邻噪声更明显），L2 反而升（低维生成下 X 的 KNN 更稳定）。

JSON 原始结果：`overlap_report.json`

## 问题 2：seed=0 上直接在 LH latent (z, y) fit JGP

设置（与 paper LH 一致）：

- LH train=1000, test=200, seed=0
- 输入：`z_train ∈ R^{1000×5}`（生成器原始 latent，未标准化，∈ [-0.5, 0.5]^5）
- 目标：`y_train`（原始尺度，范围约 [-841.5, 859.4]）
- JumpGP linear-boundary, mode=CEM, M=35

结果（在原始 y 尺度上）：

| 指标 | 值 |
|---|---:|
| RMSE | **929.47** |
| mean CRPS | **530.89** |
| coverage@90 | 0.560 |
| mean width@90 | 107.16 |
| coverage@95 | 0.565 |
| mean width@95 | 127.68 |
| wall time (s) | 17.10 |

参照同 split (LH seed=0 train=1000) 的 paper baselines（`paper_l2_lh_xstd_train1000_5seeds/metrics.csv`，全部在 X_standardized 上跑）：

| 方法 | RMSE | CRPS |
|---|---:|---:|
| **JGP-on-z**（本题答案） | **929.47** | **530.89** |
| jgp_pca (Q=5, n=35) | 913.19 | 530.53 |
| xgb_bootstrap | 763.53 | 449.22 |
| dkl_svgp | 658.75 | 389.73 |
| lmjgp_baseline (DJGP, learned W) | 757.81 | 385.98 |
| lmjgp_pls_kl | 738.60 | 370.93 |
| cem_em_transductive_vi_fullcov_ensemble | 808.70 | 416.21 |

观察（直接对应 `neighborhood_retrieval_hypothesis_experiment_plan_cn.md` 里的讨论）：

- 即使把 X-projection 这一步换成**真值 latent z**，JGP 仍然得 RMSE 929 / CRPS 531 —— 和 PCA 投影 + JGP（913 / 530.5）几乎完全一样的 CRPS。这意味着，在 LH 这个生成器上，**JGP 本身（35 邻域 + linear-boundary 局部 GP）才是瓶颈**，不是 X→z 投影。
- 这与 LH 实验中 learned W 投影方法（lmjgp_baseline 757/386, lmjgp_pls_kl 738/371）显著优于 jgp_pca 的事实一致：胜出的不是"找到 z"，而是变分 JGP 的 inducing-variable / soft-gating 等额外结构。
- coverage 只有 0.56（远低于 90%），说明 CEM linear-boundary 的局部 variance 严重低估了 LH 跨 region 的方差。这与 LH 用 2^(d+1)=64 个 region、quadratic boundary 的生成机制不匹配 —— linear-boundary CEM 在 35 个邻居里几乎一定会跨 2–4 个不同 region。

原始 JSON：`jgp_lh_seed0_train1000_z_M35_CEM.json`

## 日志条目建议（按 AGENTS.md "Experiment-logging policy"）

本任务是诊断性 one-off，没有持久化大批量 artifact，但建议在 `docs/experiments_log.md` 加一条短记录：

```
- 2026-05-18 LH/L2 seed=0 retrieval overlap & JGP-on-z diagnostics
  - runner: outputs/overlap_only.py, outputs/jgp_on_z.py (one-off scratch)
  - data: deep_mahalanobis_gp_paper_train{200,500,1000}_*seed0 npz exports
  - headline: LH X→z RFF overlap@k35 ≈ 0.70-0.77 fraction; L2 ≈ 0.83-0.93.
              JGP-CEM on true LH z (train=1000) hits RMSE=929.5 / CRPS=530.9,
              essentially matching jgp_pca; the bottleneck is the local
              CEM-linear JGP, not the projection.
  - status: done, supporting; see outputs/final_summary.md
```
