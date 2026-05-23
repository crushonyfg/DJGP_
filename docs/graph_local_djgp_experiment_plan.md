# Graph-Local DJGP Synthetic Benchmark 实验规划

## 0. 目标概述

本实验计划的目标是构造一个更贴近分布式系统的高维 surrogate benchmark，用来检验 DJGP/JGP 相对普通 global surrogate、普通 local surrogate、tree partition GP、GNN 和 graph-aware GP 的优势。

核心思想是：在分布式系统中，locality 往往不是由 feature-space 距离决定的，而是由外部系统结构决定的，例如网络拓扑、rack/cell 层级、service dependency graph、任务 DAG、通信权限或观测权限。因此，模型不应该默认使用 Euclidean kNN 来定义 local neighborhood，而应该允许使用一个外部 relation：

\[
\mathcal N_{\mathcal R}(i)
\]

其中 \(\mathcal R\) 可以是 graph adjacency、graph hop distance、network latency、dependency edge、routing distance 或 access constraint。

DJGP 的定位不是“别人不能使用 graph 信息”。强 baseline 当然也可以融合 graph。更合理的 claim 是：

> Graph-aware global methods can use topology, but they usually learn a single pooled response mechanism. Graph-local GP methods can use topology, but suffer in high-dimensional small-local-sample regimes. DJGP combines graph-defined local response modeling with a shared low-dimensional representation and jump/regime experts.

中文概括：

> 分布式 surrogate 的难点不是单纯“输入有 graph”，而是同时存在 topology-constrained locality、高维 noisy local state、局部异质 response、以及 congestion/failure/scheduling threshold 诱发的 jump/regime transition。DJGP 的优势应建立在这几个因素的组合上。

---

## 1. 我们的 idea 是什么

### 1.1 分布式系统中的 locality 不等于 feature-space locality

传统 local GP/JGP 通常默认：

\[
\mathcal N_i = \operatorname{kNN}(x_i)
\]

也就是邻域由 observed feature \(x_i\) 的欧氏距离或某种 feature-space distance 决定。

但是在分布式系统中，两个节点是否相关，常常不由 feature 相似性决定，而由系统结构决定。例如：

- 两台机器 feature 很像，但位于不同 rack/cell，网络上很远，负载传播关系弱；
- 两个 service 状态很不相似，但存在直接调用关系，latency 或 failure 可能强耦合；
- 两个任务的资源 profile 接近，但 DAG 中没有 parent-child 或 data dependency，互相影响很小；
- 一个节点在在线预测时只能看到 k-hop neighbor，不能读取全局状态。

因此，应该把 neighborhood 定义为：

\[
\mathcal N_G(v)=\{u:\operatorname{dist}_G(u,v)\le h\}
\]

其中 \(G=(V,E)\) 是外部图结构，\(\operatorname{dist}_G\) 是 hop distance、latency distance、routing distance 或 dependency distance。

这给 DJGP/JGP 一个更自然的应用动机：

> Local experts should be local with respect to the system relation, not necessarily local with respect to observed feature distance.

---

### 1.2 为什么需要把 neighborhood 信息和 feature 信息分开

我们希望区分两类信息：

1. **Feature information**  
   节点自身或局部观测到的状态，例如 CPU、memory、queue length、network inflow/outflow、task priority、workload features、history features。

2. **Neighborhood/relation information**  
   哪些节点、任务、服务、机器可以互相影响，或者哪些节点在部署时可被观测。这通常来自 graph、topology、DAG、rack/cell metadata、routing table、service dependency graph。

普通 global ML/GP 可以把 graph embedding 拼到 feature 里：

\[
\tilde x_v = [x_v,\phi_G(v)]
\]

然后训练一个 global model：

\[
y_v = f_{\text{global}}(\tilde x_v) + \epsilon.
\]

但这样仍然把所有 region 的 response 绑定在一个 pooled function 里。问题是，分布式系统中不同 graph region 可能有不同 response mechanism：

\[
y_v = f_{r(v)}(\cdot) + \epsilon
\]

其中 \(r(v)\) 可以对应 rack、cell、service group、DAG stage、hidden congestion regime 或 failure regime。

DJGP 的结构可以写成：

\[
z_v = W x_v,
\]

\[
m_v = \operatorname{Agg}_{u\in\mathcal N_G(v)} W x_u,
\]

\[
y_v = f_{r(v)}(z_v,m_v) + \epsilon.
\]

其中：

- \(W\)：跨 graph regions 共享的 low-dimensional representation；
- \(\mathcal N_G(v)\)：由 graph/relation 定义的 neighborhood；
- \(f_{r(v)}\)：local expert 或 local JGP；
- \(r(v)\)：region / anchor / regime / gate assignment。

所以 DJGP 的 inductive bias 是：

\[
\text{shared representation} + \text{graph-local heterogeneous response}.
\]

它不是完全 global，也不是完全 local。

---

### 1.3 为什么 jump 是自然的

分布式系统中的 response 经常不是全局平滑函数。很多机制会带来 threshold 或 regime transition：

- queue saturation：队列长度超过阈值后 latency 急剧上升；
- network congestion：链路负载超过容量后 packet delay 或 retry 增加；
- failure mode：某个 neighbor 或 service failure 后，routing/scheduling 切换；
- autoscaling threshold：负载超过阈值后触发扩容或迁移；
- priority/scheduling class switch：任务优先级不同，排队规则不同；
- DAG bottleneck：关键上游任务延迟后，下游 completion time 突变。

这些可以抽象为：

\[
y_v =
f_{r(v)}(z_v,m_v)
+
\Delta_{r(v)}\mathbf 1\{b_{r(v)}^\top m_v > \tau_{r(v)}\}
+
\epsilon_v.
\]

其中 indicator term 表示 jump。它不一定是严格不连续，也可以用 steep sigmoid 平滑近似：

\[
\Delta_{r(v)}\sigma\left(\alpha(b_{r(v)}^\top m_v-\tau_{r(v)})\right),
\quad \alpha \gg 1.
\]

因此，JGP/DJGP 的 jump/local expert 不是人为设计，而是对分布式系统中 congestion/failure/scheduling regime transition 的抽象。

---

### 1.4 我们应该避免的过度 claim

不要写：

> Global methods fail because they cannot use graph.

这不成立。GNN、graph kernel GP、global GP with graph features 都可以使用 graph。

更好的 claim 是：

> Global graph-aware methods can incorporate topology, but they usually impose a single pooled response mechanism. In high-dimensional, topology-constrained, locally heterogeneous surrogate problems, this pooling can oversmooth region-specific transitions. DJGP uses graph-defined neighborhoods to localize response modeling while sharing a representation across neighborhoods.

---

## 2. Synthetic data 如何构造

### 2.1 总体数据结构

生成一个图：

\[
G=(V,E),\quad |V|=n.
\]

每个节点 \(v\in V\) 在时间 \(t\) 有高维状态：

\[
x_{v,t}\in\mathbb R^D,\quad D\ge 30.
\]

建议初始设置：

- \(n=300\) 或 \(500\) nodes；
- \(T=20\) 或 \(50\) time steps；
- 总样本数 \(N=nT\)；
- \(D\in\{30,50,100\}\)；
- latent dimension \(q\in\{2,3,5\}\)；
- graph neighborhood radius \(h\in\{1,2\}\)。

每个样本是：

\[
(x_{v,t}, G, v, t, y_{v,t}).
\]

训练时可以给方法不同程度的 graph 信息：

- non-graph baselines：只看 \(x_{v,t}\)；
- graph-feature baselines：看 \([x_{v,t},\phi_G(v)]\)；
- graph-local methods：用 \(\mathcal N_G(v)\) 定义 local subset；
- GNN：看 \(G\)、node features 和 edge 信息；
- DJGP-graph：用 graph neighborhood 定义 anchors/local experts，同时学习 global \(W\)。

---

### 2.2 图结构生成

建议至少实现三种 graph family。

#### A. Stochastic block model: rack/cell/community

模拟 cloud rack/cell/service group。

\[
V=\bigcup_{b=1}^B V_b
\]

其中 \(B\in\{4,8,12\}\)。block 内连接概率高，block 间连接概率低：

\[
p_{\text{in}}=0.10\sim 0.20,\quad
p_{\text{out}}=0.005\sim 0.02.
\]

每个 block 可以对应一个 local response regime：

\[
r(v)=b(v).
\]

优点：容易控制 local heterogeneity 和 held-out region split。

#### B. Random geometric graph: physical/network proximity

给每个 node 一个隐含物理位置 \(p_v\in[0,1]^2\)，若距离小于半径则连接：

\[
(v,u)\in E \iff \|p_v-p_u\|_2\le \rho.
\]

重要：\(p_v\) 不应该直接等于 feature \(x_v\)。否则 feature kNN 会过强。可以让 \(x_v\) 主要来自 workload state，而 graph 来自 physical/network topology。

#### C. Fat-tree / hierarchical graph: data center topology

构造 core-aggregation-edge-rack 层级。local neighborhood 可以是同 rack、同 pod、k-hop connection。

优点：最贴近 data center story。缺点：实现稍复杂。可以作为第二阶段。

初始 1-seed screen 建议先用 SBM，因为最容易观察 local heterogeneity 和 graph-kNN vs feature-kNN 差异。

---

### 2.3 Feature 生成：latent + high-dimensional expansion

#### 默认版本：有 latent space

先生成低维真实状态：

\[
s_{v,t}\in\mathbb R^q.
\]

可以用 temporal correlated process：

\[
s_{v,t}=0.7s_{v,t-1}+0.3\mu_{r(v)}+\xi_{v,t},
\quad
\xi_{v,t}\sim\mathcal N(0,I_q).
\]

其中 \(\mu_{r(v)}\) 是 region-specific workload mean。

然后通过高维 expansion 得到 observed feature：

\[
x_{v,t}=A s_{v,t}+\eta_{v,t},
\]

\[
A\in\mathbb R^{D\times q},\quad
\eta_{v,t}\sim\mathcal N(0,\sigma_x^2 I_D).
\]

为了模拟 high-dimensional noisy observation，设置：

\[
D\in\{30,50,100\},
\quad
q\in\{2,5\},
\quad
\sigma_x\in\{0.2,0.5,1.0\}.
\]

真实 projection 可以取：

\[
W^\star = (A^\top A)^{-1}A^\top
\]

或直接随机生成 \(W^\star\in\mathbb R^{q\times D}\)，然后生成：

\[
z_{v,t}=W^\star x_{v,t}.
\]

DJGP 的任务就是从高维 \(x\) 中恢复或近似这个低维 geometry。

#### 可选版本：没有 clean latent space

为了避免 reviewer 说 synthetic 完全为 projection 方法量身定制，可以设计一个 no-clean-latent variant：

\[
x_{v,t}\sim\mathcal N(\mu_{r(v)},\Sigma_{r(v)}),
\]

输出直接依赖 sparse subset 或 nonlinear combination：

\[
y_{v,t}=f_{r(v)}(B x_{v,t},m_{v,t})+\epsilon
\]

其中 \(B\in\mathbb R^{q\times D}\) 是稀疏矩阵或 random nonlinear map。这仍然有 low effective dimension，但不是显式 linear latent recovery。

也可以设计更困难版本：

\[
z_{v,t} = \tanh(W^\star x_{v,t})
\]

这样 linear \(W\) 不再完全正确，测试 DJGP 在 mild misspecification 下是否仍有优势。

---

### 2.4 Graph neighborhood aggregation

真实 response 不只依赖节点自身状态，也依赖 graph-neighborhood summary：

\[
\mathcal N_G(v)=\{u:\operatorname{dist}_G(u,v)\le h\}.
\]

先计算 latent：

\[
z_{v,t}=W^\star x_{v,t}.
\]

然后聚合邻居：

\[
m_{v,t}
=
\sum_{u\in\mathcal N_G(v)} a_{vu} z_{u,t}.
\]

权重可以是：

\[
a_{vu}
=
\frac{\exp(-\lambda \operatorname{dist}_G(u,v))}
{\sum_{w\in\mathcal N_G(v)}
\exp(-\lambda \operatorname{dist}_G(w,v))}.
\]

默认：

\[
h=1,\quad \lambda=1.
\]

也可以包含 self-loop：

\[
v\in\mathcal N_G(v).
\]

---

### 2.5 输出函数：no-jump 版本

No-jump 版本用于测试 graph-locality 和 representation sharing，但不强调 discontinuity。

\[
y_{v,t}
=
f_{r(v)}(z_{v,t},m_{v,t})
+
\epsilon_{v,t}.
\]

可以设置：

\[
f_r(z,m)
=
\sin(a_r^\top z)
+
0.5(b_r^\top m)^2
+
c_r^\top z
+
d_r^\top m.
\]

其中 \(a_r,b_r,c_r,d_r\) 是 region-specific 参数，但共享结构相似。例如：

\[
a_r = a_0 + \delta a_r,\quad
b_r = b_0 + \delta b_r.
\]

这样不同 regions 不完全一样，但有共享 geometry。

噪声：

\[
\epsilon_{v,t}\sim\mathcal N(0,\sigma_y^2),
\quad
\sigma_y\in\{0.05,0.1,0.2\}.
\]

---

### 2.6 输出函数：jump / multi-regime 版本

Jump 版本加入 congestion/failure/scheduling threshold：

\[
y_{v,t}
=
f_{r(v)}(z_{v,t},m_{v,t})
+
\Delta_{r(v)} \mathbf 1\{b_{r(v)}^\top m_{v,t}>\tau_{r(v)}\}
+
\epsilon_{v,t}.
\]

或者 soft jump：

\[
y_{v,t}
=
f_{r(v)}(z_{v,t},m_{v,t})
+
\Delta_{r(v)}
\sigma(\alpha(b_{r(v)}^\top m_{v,t}-\tau_{r(v)}))
+
\epsilon_{v,t}.
\]

默认建议先用 hard jump，因为更容易验证 JGP/DJGP 的作用。然后可选 soft jump 做 robustness。

推荐参数：

- \(\Delta_r\in[1.0,3.0]\)；
- \(\tau_r\) 取 \(b_r^\top m\) 的 60%-80% quantile，使 jump region 不太稀有；
- \(\alpha=10\sim 30\) for soft jump；
- jump proportion 控制在 20%-40%。

---

### 2.7 Feature kNN misleading 机制

为了证明 graph neighborhood 和 feature neighborhood 不一样，需要刻意降低 feature distance 与 graph distance 的相关性。

做法：

1. graph 由 \(p_v\) 或 block membership 决定；
2. feature \(x_{v,t}\) 主要由 workload latent \(s_{v,t}\) 决定；
3. workload latent 可以在不同 graph block 之间重叠。

例如：

\[
s_{v,t}=0.7s_{v,t-1}+0.3\mu_{r(v)}+\xi_{v,t}
\]

但 \(\mu_r\) 之间不要相距太远，甚至可以让不同 block 的 \(\mu_r\) 部分重叠。这样 feature-kNN 很难恢复 graph neighborhood。

可以记录一个 diagnostic：

\[
\rho = \operatorname{corr}
\left(
\|x_i-x_j\|_2,
\operatorname{dist}_G(i,j)
\right).
\]

希望 default setting 中 \(\rho\) 较低，例如 \(0.0\sim 0.3\)。

---

### 2.8 Split 设计

至少三种 split。

#### Split 1: random node-time split

随机划分 train/test。用于基本 sanity check。

#### Split 2: future-time split

前 70% 时间训练，后 30% 时间测试：

\[
t\le T_{\text{train}} \Rightarrow \text{train},
\quad
t>T_{\text{train}} \Rightarrow \text{test}.
\]

测试 temporal extrapolation under same graph。

#### Split 3: held-out graph region split

把某些 block/community/rack 留作 test：

\[
r(v)\in \mathcal R_{\text{test}}
\Rightarrow
(v,t)\in \text{test}.
\]

这是最有价值的 split。它测试 cold-start region / new rack / new cell。可以设置少量 calibration points：

- zero-shot held-out region；
- few-shot held-out region：每个 held-out region 给 5%、10% training points。

DJGP 预期在 few-shot 下比 pure local GP 更强，因为 \(W\) 从其他 regions 共享学习。

---

## 3. 方法与 baseline 设计

### 3.1 我们的方法组

建议至少包含以下 variants：

#### A. JGP-feature

原始 feature-space kNN / anchor JGP。

\[
\mathcal N(i)=\operatorname{kNN}(x_i).
\]

目的：作为已有 JGP baseline。

#### B. JGP-graph

用 graph neighborhood 定义 local expert：

\[
\mathcal N(i)=\mathcal N_G(i).
\]

目的：证明 graph locality 本身有用。

#### C. DJGP-feature

学习 global \(W\)，但 local neighborhoods 仍用 feature kNN 或 projected feature kNN。

目的：区分 representation sharing 和 graph locality。

#### D. DJGP-graph

学习 global \(W\)，local neighborhoods 用 graph relation。

\[
z_i = W x_i,\quad
\mathcal N(i)=\mathcal N_G(i).
\]

这是主方法。

#### E. DJGP-graph-nojump

关闭 jump/local regime gate，只保留 graph-local smooth GP experts。

目的：证明 jump term 是否必要。

#### F. DJGP-graph-jump

完整版本，包含 graph-local experts + shared \(W\) + jump/regime mechanism。

目的：主 claim。

---

### 3.2 现有 baseline

保留原来已有强 baseline：

1. **XGBoost**
   - 输入：\(x\)
   - graph-aware version：\([x,\phi_G(v), \text{neighbor aggregate features}]\)

2. **DKL**
   - 输入：\(x\)
   - graph-aware version：\([x,\phi_G(v), \text{neighbor aggregate features}]\)
   - 优点：有 GP head，可做 UQ。

3. **DMGP / DGP**
   - 输入：\(x\)
   - graph-aware version：\([x,\phi_G(v), \text{neighbor aggregate features}]\)
   - 用作 global nonlinear GP baseline。

这些是已有 baseline，但不足以覆盖 graph-aware reviewer challenge。

---

### 3.3 新增合理 baseline：graph-aware global

#### A. GNN surrogate

最重要的 graph-aware neural baseline。

输入：

\[
G,\quad x_{v,t}
\]

输出：

\[
\hat y_{v,t}.
\]

候选模型：

- GCN；
- GraphSAGE；
- GAT；
- MPNN。

推荐先实现 GraphSAGE 或 GCN，避免过度复杂。

UQ 方案：

- MC dropout；
- deep ensemble；
- evidential regression；
- quantile regression；
- conformalized residual intervals。

最容易落地的是：

\[
\text{GNN ensemble} \quad K=5
\]

或者 MC dropout。1-seed screen 可以先用 MC dropout，正式多 seed 可以用 ensemble。

GNN 的存在非常重要，因为它是 reviewer 最可能提出的 baseline。

#### B. Global GP with graph features

构造 graph embedding：

\[
\phi_G(v)
\]

可选：

- Laplacian eigenvectors；
- node2vec embedding；
- degree/clustering coefficient/community id；
- block/rack one-hot；
- shortest-path summaries；
- neighbor aggregate features。

输入：

\[
\tilde x_{v,t}=[x_{v,t},\phi_G(v),\operatorname{Agg}_{u\in\mathcal N_G(v)}x_{u,t}].
\]

模型：

- sparse GP；
- DKL；
- DMGP；
- XGBoost。

这组 baseline 非常公平，因为它允许 global methods 使用 graph 信息。

#### C. Graph kernel GP

定义 kernel：

\[
k((v,x_v),(u,x_u))
=
k_x(x_v,x_u)\cdot k_G(v,u).
\]

其中：

\[
k_G(v,u)
=
\exp(-\gamma\operatorname{dist}_G(v,u))
\]

或 diffusion kernel：

\[
K_G=\exp(-\beta L_G).
\]

然后：

\[
K = K_x \odot K_G.
\]

UQ：天然 GP predictive variance。

挑战：大 \(N\) 时成本较高。可以用 sparse approximation 或先在较小 \(N\) 上跑。

#### D. Graph feature + global XGBoost

虽然 XGBoost 不天然 UQ，但作为强 point baseline 需要保留。UQ 可以用：

- quantile XGBoost；
- ensemble variance；
- conformal prediction。

建议正式报告 point metrics + conformal interval coverage。

---

### 3.4 新增合理 baseline：local / partition methods

#### A. Graph-kNN local GP

每个 test point 选择 graph neighbors：

\[
\mathcal D_v=\{(x_u,y_u):u\in\mathcal N_G(v)\}
\]

训练 local GP 或使用 local subset prediction。

输入可以是：

- raw \(x\)；
- PCA \(x\)；
- known graph-neighbor aggregate features。

UQ：天然 GP variance。

这是最直接挑战 DJGP-graph 的 baseline。预期它在低维或 local data 足够多时会很强；在 \(D\gg n_j\) 时变弱。

#### B. Graph-kNN local GP + PCA

为了公平，给 local GP 一个 unsupervised dimensionality reduction baseline：

\[
z=\operatorname{PCA}(x)
\]

然后 local GP on \(z\)。

这测试 DJGP 的 supervised/shared \(W\) 是否优于简单 PCA。

#### C. Tree partition GP / Treed GP

用 tree 或 random forest 划分输入空间，然后每个 leaf fit GP。

输入：

\[
[x,\phi_G(v),\operatorname{Agg}_{\mathcal N_G}x]
\]

UQ：leaf GP variance。

它能处理非平稳和 discontinuity，但通常 partition 来自 feature splits，不天然尊重 graph relation。它是非常合理的 reviewer baseline。

#### D. Mixture-of-experts GP

全局 gate + 多个 GP experts：

\[
p(y|x)=\sum_k \pi_k(x)p_k(y|x).
\]

Graph-aware version:

\[
\pi_k=\pi_k(x,\phi_G(v)).
\]

这个 baseline 会直接挑战 JGP/DJGP 的 expert structure。如果实现成本太高，可以列为 second-stage baseline。

---

### 3.5 UQ 指标

因为 DJGP/JGP 是 probabilistic surrogate，只看 RMSE 不够。建议报告：

Point prediction:

- RMSE；
- MAE；
- \(R^2\)；
- near-boundary RMSE；
- jump-region RMSE。

Uncertainty:

- NLPD / NLL；
- CRPS；
- 90% prediction interval coverage；
- 90% interval width；
- calibration error；
- sharpness vs coverage curve。

Regime-specific:

- no-jump region RMSE；
- jump region RMSE；
- boundary band RMSE；
- held-out region RMSE；
- few-shot region RMSE。

Graph-locality diagnostics:

- feature distance vs graph distance correlation；
- graph-kNN overlap with feature-kNN；
- performance vs graph radius \(h\)；
- performance vs local sample size \(n_j\)。

---

## 4. 具体实验矩阵

### 4.1 Dataset variants

建议先定义 6 个主 variants。

#### Dataset A: Graph-local smooth, latent, D=30

目的：基础 sanity check。

\[
D=30,\quad q=3,\quad \text{no jump}.
\]

真实 locality 来自 graph。没有 jump，主要测试 graph-locality 和 representation sharing。

#### Dataset B: Graph-local jump, latent, D=30

目的：测试 jump advantage。

\[
D=30,\quad q=3,\quad \text{hard jump}.
\]

#### Dataset C: Graph-local smooth, latent, D=100

目的：测试 high-dimensional sample efficiency。

\[
D=100,\quad q=5,\quad \text{no jump}.
\]

#### Dataset D: Graph-local jump, latent, D=100

目的：主 stress test。

\[
D=100,\quad q=5,\quad \text{hard jump}.
\]

#### Dataset E: Misleading feature-kNN, D=100

目的：证明 feature locality 和 graph locality 分离。

设置 graph distance 与 feature distance 低相关：

\[
\operatorname{corr}(\|x_i-x_j\|,\operatorname{dist}_G(i,j))\approx 0.
\]

#### Dataset F: No-clean-latent, graph-local jump, D=100

目的：检查 synthetic 是否过度偏向 linear projection。

使用：

\[
z=\tanh(W^\star x)
\]

或 sparse nonlinear \(B x\)，而不是 clean linear latent。

---

### 4.2 Baseline matrix

1-seed 初筛不要一次跑全部 baseline。建议分层。

#### Core baselines for 1-seed screen

必须先跑：

1. XGBoost raw \(x\)
2. XGBoost graph features
3. Sparse GP / DKL raw \(x\)
4. DKL graph features
5. DMGP raw \(x\)
6. DMGP graph features
7. Graph-kNN local GP raw \(x\)
8. Graph-kNN local GP + PCA
9. JGP-feature
10. JGP-graph
11. DJGP-feature
12. DJGP-graph no-jump
13. DJGP-graph jump

#### Add after core signal appears

第二批：

14. GCN/GraphSAGE MC dropout
15. GNN ensemble
16. Graph-kernel GP
17. Tree partition GP with graph features
18. MoE-GP or neural MoE

---

## 5. 1-seed 实验顺序

### Stage 0: data sanity diagnostics

先不跑模型，只生成数据并检查：

1. \(y\) distribution；
2. graph degree distribution；
3. jump proportion；
4. feature distance vs graph distance correlation；
5. feature-kNN 与 graph-kNN overlap；
6. train/test split sizes；
7. held-out region 是否真的 distribution shift；
8. signal-to-noise ratio。

通过标准：

- jump proportion 在 20%-40%；
- feature-graph correlation 不高于 0.3；
- graph-kNN 与 feature-kNN overlap 不应太高；
- 每个 local expert 有足够样本；
- \(y\) 不应被 noise 淹没。

---

### Stage 1: easiest sanity dataset

先跑 Dataset A：

- SBM graph；
- \(D=30\)；
- \(q=3\)；
- no jump；
- random node-time split。

目的：确认 DJGP-graph 至少不比 JGP-graph 差，graph-aware methods 能正常训练。

期望：

\[
\text{graph-aware baselines} > \text{non-graph baselines}
\]

\[
\text{DJGP-graph} \ge \text{JGP-graph}
\]

如果这里失败，说明 implementation 或 data generation 有问题。

---

### Stage 2: graph locality test

跑 Dataset E：

- misleading feature-kNN；
- \(D=30\) 或 \(D=100\)；
- no jump；
- random split。

关键比较：

\[
\text{JGP-graph} > \text{JGP-feature}
\]

\[
\text{DJGP-graph} > \text{DJGP-feature}
\]

如果这个差异不明显，说明 feature-kNN 仍然能间接恢复 graph locality，需要降低 feature-graph correlation 或增强 graph-local response。

---

### Stage 3: high-dimensional sharing test

跑 Dataset C：

- \(D=100\)；
- \(q=5\)；
- no jump。

关键比较：

\[
\text{DJGP-graph} > \text{JGP-graph}
\]

\[
\text{DJGP-graph} > \text{Graph-kNN local GP}
\]

如果 JGP-graph/local GP 已经很强，说明 local sample size 太大或高维噪声不够。可以减少 per-neighborhood samples、提高 \(D\)、提高 \(\sigma_x\)。

---

### Stage 4: jump test

跑 Dataset D：

- \(D=100\)；
- \(q=5\)；
- hard jump。

关键比较：

\[
\text{DJGP-graph-jump} > \text{DJGP-graph-nojump}
\]

\[
\text{DJGP-graph-jump} > \text{global graph-aware DKL/DMGP}
\]

重点看：

- jump-region RMSE；
- boundary band RMSE；
- NLPD；
- calibration near boundary。

如果 jump/nojump 差异不明显，可以提高 \(\Delta_r\)、降低 noise、或把 threshold 设到更明确的 quantile。

---

### Stage 5: held-out region / few-shot test

跑 Dataset D 或 F：

- held-out block/community split；
- zero-shot and few-shot。

关键比较：

\[
\text{DJGP-graph} > \text{pure local graph GP}
\]

\[
\text{DJGP-graph} \approx \text{or} > \text{GNN under few-shot}
\]

这个结果最适合论文，因为它体现 shared \(W\) 的价值。

---

## 6. 多 seed 扩展标准

不要一开始跑多 seeds。先在 1 seed 上看是否满足以下条件。

### 6.1 必须满足的核心信号

至少满足 3 个：

1. DJGP-graph 明显优于 DJGP-feature；
2. DJGP-graph 明显优于 JGP-graph in high-D；
3. DJGP-graph-jump 明显优于 DJGP-graph-nojump in jump region；
4. DJGP-graph 在 NLPD / coverage 上优于 XGBoost/GNN point baseline；
5. DJGP-graph 在 held-out/few-shot region 上优于 local GP 或 graph-kNN local GP。

### 6.2 多 seed 配置

如果 1 seed 信号满意，跑：

\[
\text{seeds}=\{0,1,2,3,4\}
\]

正式论文最好：

\[
\text{seeds}=\{0,\dots,9\}
\]

每个 dataset 报：

\[
\text{mean} \pm \text{standard error}
\]

或 median + IQR。

### 6.3 多 seed 优先级

优先跑：

1. Dataset B: D=30 jump；
2. Dataset D: D=100 jump；
3. Dataset E: misleading feature-kNN；
4. Dataset F: no-clean-latent。

如果资源有限，主表用 Dataset D/E/F，Dataset A/B/C 放 appendix。

---

## 7. 预期结果表格设计

### Table 1: Main RMSE/MAE/NLPD

Rows:

- XGBoost raw
- XGBoost graph features
- DKL raw
- DKL graph features
- DMGP raw
- DMGP graph features
- GNN MC dropout
- Graph-kNN local GP
- JGP-feature
- JGP-graph
- DJGP-feature
- DJGP-graph no-jump
- DJGP-graph jump

Columns:

- Dataset A
- Dataset B
- Dataset D
- Dataset E
- Dataset F

Metrics:

- RMSE
- NLPD

---

### Table 2: Ablation table

Rows:

- feature-kNN vs graph-kNN；
- raw high-D vs PCA vs learned \(W\)；
- no-jump vs jump；
- local only vs shared \(W\)。

Columns:

- RMSE all；
- RMSE jump；
- RMSE non-jump；
- NLPD；
- 90% coverage；
- interval width。

---

### Figure 1: Graph locality diagnostic

Plot:

- feature distance vs graph distance；
- feature-kNN / graph-kNN overlap；
- graph layout colored by true regime；
- graph layout colored by prediction error。

---

### Figure 2: Boundary/jump performance

Plot:

- predicted mean vs true \(y\) around threshold；
- uncertainty near threshold；
- RMSE as function of distance to jump boundary。

---

### Figure 3: Few-shot held-out region

x-axis:

\[
\text{fraction of calibration points in held-out region}
=
\{0,1\%,5\%,10\%,20\%\}
\]

y-axis:

- RMSE；
- NLPD；
- coverage error。

Expected:

\[
\text{DJGP-graph} \text{ improves faster than pure local GP}.
\]

---

## 8. Implementation notes

### 8.1 Data generator API

建议实现：

```python
generate_graph_local_surrogate(
    n_nodes: int = 500,
    n_time: int = 20,
    D: int = 100,
    q: int = 5,
    graph_type: str = "sbm",
    n_blocks: int = 8,
    graph_radius: int = 1,
    jump: bool = True,
    soft_jump: bool = False,
    misleading_features: bool = True,
    latent_type: str = "linear",  # linear, tanh, sparse
    noise_x: float = 0.5,
    noise_y: float = 0.1,
    seed: int = 0,
)
```

Return:

```python
{
    "X": X,                     # [N, D]
    "y": y,                     # [N]
    "node_id": node_id,         # [N]
    "time_id": time_id,         # [N]
    "edge_index": edge_index,   # [2, E]
    "graph_dist": graph_dist,   # optional
    "block_id": block_id,       # [n_nodes]
    "regime_id": regime_id,     # [N]
    "jump_indicator": jump_ind, # [N]
    "Z_true": Z,                # [N, q], for diagnostics only
    "M_true": M,                # [N, q], for diagnostics only
}
```

---

### 8.2 Neighborhood operator abstraction

把 neighborhood 单独抽象出来：

```python
class NeighborhoodProvider:
    def neighbors(self, node_id, mode: str):
        ...
```

Modes:

- `"feature_knn"`
- `"projected_feature_knn"`
- `"graph_hop"`
- `"graph_weighted"`
- `"random_control"`

这样同一套 JGP/DJGP 可以切换 locality source。

---

### 8.3 Fairness rules for baselines

1. 如果 DJGP-graph 用 graph，那么 graph-aware baselines 也必须能用 graph。
2. 如果 DJGP 学 \(W\)，local GP 至少要有 PCA reduction baseline。
3. 如果 DJGP 报 UQ，GNN/XGBoost 也应至少有 ensemble/conformal/MC dropout UQ。
4. 如果 synthetic 有 jump，baseline 应该允许 tree/MoE 处理 discontinuity。
5. 不要只比较 raw global RBF GP。

---

## 9. 可能的 reviewer challenge 与回应

### Challenge 1: “This dataset is designed for your method.”

回应：

我们提供 no-clean-latent variant、graph-aware global baselines、GNN、graph-kernel GP、tree partition GP 和 local GP+PCA。实验不是只比较 weak baselines。Dataset 的目的是 isolate three real distributed-system phenomena: graph-constrained locality, high-dimensional noisy state, and regime-dependent transitions.

### Challenge 2: “GNNs can naturally handle graph locality.”

回应：

同意，所以 GNN 是主 baseline。我们的 claim 不是 GNN 不能用 graph，而是在 small-to-medium data、高维 noisy features、local heterogeneity、需要 calibrated UQ 的 surrogate setting 下，DJGP 的 GP-style local uncertainty 和 shared representation 可能更合适。

### Challenge 3: “Why not global GP with graph kernel?”

回应：

这也是 baseline。Graph-kernel GP uses topology in covariance but still imposes a globally coupled response mechanism unless extended with local experts. DJGP uses topology to define local response models while sharing representation globally.

### Challenge 4: “Why not local GP with graph neighborhood?”

回应：

这也是 baseline。Graph-local GP has the correct locality but poor sample efficiency when each neighborhood has limited samples and \(D\) is large. DJGP shares \(W\) across neighborhoods to reduce effective dimension.

### Challenge 5: “Why jump?”

回应：

Jump models congestion threshold, queue saturation, failure mode, scheduling switch, and autoscaling triggers. These are natural in distributed systems. We also include no-jump variants to show when jump modeling is or is not necessary.

---

## 10. Recommended first coding milestone

### Milestone 1: generator + diagnostics

Implement:

- SBM graph；
- latent high-dimensional feature generation；
- graph-neighborhood aggregation；
- no-jump and hard-jump outputs；
- random split and held-out-block split；
- diagnostics.

Do not train models yet.

### Milestone 2: minimum model screen

Run on seed 0:

- XGBoost raw；
- XGBoost graph features；
- DKL/DMGP raw；
- DKL/DMGP graph features；
- Graph-kNN local GP；
- JGP-feature；
- JGP-graph；
- DJGP-feature；
- DJGP-graph no-jump；
- DJGP-graph jump。

Datasets:

- A: D=30 no-jump；
- B: D=30 jump；
- D: D=100 jump；
- E: misleading feature-kNN。

### Milestone 3: add graph neural baseline

Add GraphSAGE/GCN with MC dropout or ensemble. Report point metrics and UQ metrics.

### Milestone 4: formal multi-seed

Run 5 seeds on selected datasets after core signal is confirmed.

---

## 11. Minimal success criteria

A convincing 1-seed result would look like:

1. On misleading graph-local data:

\[
\text{DJGP-graph} > \text{DJGP-feature}
\]

2. On high-dimensional local data:

\[
\text{DJGP-graph} > \text{JGP-graph} \approx \text{Graph-local GP}
\]

3. On jump data:

\[
\text{DJGP-graph-jump} > \text{DJGP-graph-nojump}
\]

especially in boundary/jump-region RMSE and NLPD.

4. Against graph-aware global baselines:

DJGP-graph does not need to dominate everywhere, but should win or be clearly competitive in at least one of:

- held-out region；
- few-shot region；
- jump boundary；
- UQ calibration；
- small training size；
- high \(D\)/low local sample setting。

If these are not observed, adjust data difficulty before scaling to multi-seed.

---

## 12. References and real-world motivation anchors

These references are not required for the synthetic generator, but they support the realism of distributed-system surrogate needs and public cloud traces.

1. Google Borg cluster traces: Google publicly provides Borg workload traces from compute cells, and the 2019 dataset contains traces from multiple clusters/cells with workload and scheduling information.
   - https://github.com/google/cluster-data
   - https://arxiv.org/pdf/2308.02358

2. Alibaba Cluster Trace Program: Alibaba cluster-trace-v2018 contains about 4000 machines over 8 days and includes DAG information for production batch workloads.
   - https://github.com/alibaba/clusterdata
   - https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md

3. Microsoft Azure Public Dataset: Azure has released public VM and cloud workload traces used for cloud workload analysis.
   - https://github.com/Azure/AzurePublicDataset

4. Graph neural network surrogate models: GNNs are a natural graph-aware surrogate baseline for networked systems and infrastructure flow models.
   - https://re.public.polimi.it/retrieve/e04e634f-4dde-446c-a117-5447d3fc3f5e/A_Graph_Neural_Network_Surrogate_Model_for_Critical_Infrastructure_Network_Flow_Optimisation.pdf

---

## 13. One-paragraph paper-ready framing

Distributed infrastructure induces surrogate modeling problems in which information flow and statistical dependence are governed by system relations rather than by Euclidean feature similarity. A node's relevant neighborhood may be determined by network connectivity, service dependency, rack/cell hierarchy, or task DAG constraints, while the observed local state can be high-dimensional and noisy. Moreover, performance responses often exhibit regime transitions caused by congestion, queue saturation, failures, or scheduling switches. We therefore construct a graph-local high-dimensional surrogate benchmark where the response depends on a shared low-dimensional representation of node states, a graph-defined neighborhood aggregate, and region-specific smooth or jump response functions. This benchmark isolates the setting where purely global graph-aware models may over-pool heterogeneous responses, while purely local graph-GP models suffer from limited local sample size. DJGP is designed for this intermediate regime by sharing a representation across neighborhoods while preserving graph-local response modeling and uncertainty quantification.
