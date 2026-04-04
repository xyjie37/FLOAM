 ```markdown
# FedMTL: 自适应多教师知识蒸馏联邦持续学习算法详解

## 1. 问题定义

联邦类持续学习（FCCL）场景涉及 $n$ 个参与客户端 $C = \{C_1, C_2, \ldots, C_n\}$，包含 $m$ 个顺序任务 $T = \{T_1, T_2, \ldots, T_m\}$。

**关键约束**：
- 任务 $T_t$ 的类别空间 $y^t \subset Y$，且 $y^i \cap y^j = \varnothing, \forall i \neq j$
- 客户端 $C_i$ 在任务 $T_t$ 的数据集为 $D_t^i$，任务完成后禁止访问历史数据
- 每任务包含 $R$ 轮通信，客户端提取知识 $k_t^i$，全局融合知识 $K_g^t = \mathcal{F}_{KN}(\{k_t^1, \ldots, k_t^n\})$
- 最终目标：跨任务知识融合 $K_g = \mathcal{F}_{CKN}(\{K_g^1, \ldots, K_g^m\})$

---

## 2. 客户端：任务知识提取

每个任务阶段，客户端接收批次数据，训练两个模型：

### 2.1 分类模型训练

- **初始化**：每轮使用全局模型初始化本地模型
- **优化目标**：交叉熵损失
- **上传**：训练完成后上传分类模型至服务器
- **服务器聚合**：FedAvg 算法
  $$M_g^t = \sum_{i=1}^{m} w_i M_t^i$$

### 2.2 生成模型训练（CVAE）

采用条件变分自编码器捕获局部分布：

**编码器**：将样本 $x$ 映射至潜在向量 $z$
**解码器**：根据类别条件 $c$ 从 $z$ 重构样本

**损失函数**：
$$\mathcal{L}_t = \mathbb{E}_{q_\phi(z|x,c)}[\log p_\theta(x|z,c)] - \mathrm{KL}(q_\phi(z|x,c) \| p_\theta(z|c))$$

其中：
- 第一项：重构损失
- 第二项：KL散度约束潜在空间
- $\theta, \phi$：分别为解码器和编码器参数

**关键设计**：新任务开始时，客户端**不使用上一任务的全局模型初始化**，而是采用全新初始化，确保任务间参数隔离。

---

## 3. 服务器：跨任务知识融合

任务完成后，服务器执行两阶段操作：

### 3.1 任务数据集构建（双阈值策略）

**Step 1: 模型聚合**
$$G_g = \sum_{i=1}^{m} w_i G_i, \quad C_g = \sum_{i=1}^{m} w_i C_i$$

**Step 2: 样本生成与质量评估**
- 使用全局生成模型 $G_g$ 按类别生成样本：
  $$D_k = \mathrm{GenerateSamples}(G_g, k), \quad k \in [1, K]$$
- 全局分类器 $C_g$ 计算分类概率：
  $$\{P_1, \ldots, P_k\} \leftarrow \mathrm{OutputProbs}(\{D_1, \ldots, D_k\}; C_g)$$
- 取最大概率作为样本质量分数 $p_i(x)$

**Step 3: 双阈值采样**

设定最大阈值 $V_{\max}$ 和最小阈值 $V_{\min}$，划分样本集：
$$
\begin{cases}
D_H = \{x \in P_i \mid p_i(x) \geq V_{\max}\} \\
D_L = \{x \in P_i \mid V_{\min} \leq p_i(x) < V_{\max}\}
\end{cases}
$$

**Step 4: 数据集构建**
每类别目标样本数 $N$，最终选择策略：
$$
S_t^c = 
\begin{cases}
\mathrm{sort}(D_H)[:N], & \text{if } |D_H| \geq N \\
\mathrm{sort}(D_H) \cup \mathrm{sort}(D_L)[:R], & \text{if } |D_H| < N
\end{cases}
$$

其中 $R = N - |D_H|$。若不足则重复生成迭代。

---

### 3.2 自适应多教师知识蒸馏

**Step 1: 跨任务数据集构建**

采用均匀采样策略，确保类别平衡：
$$D_z^t = \mathrm{MergeDataset}(\{D_1^s, D_2^s, \ldots, D_{t-1}^s\})$$

**Step 2: 自适应蒸馏系数计算**

**类别质量分数（CQS）**：
$$\mathrm{CQS}_k = \frac{1}{N_k} \sum_{i \in D_k} \left( z_{i,k}^T - \frac{1}{C-1} \sum_{c \neq k} z_{i,c}^T \right)$$

其中 $z_{i,k}^T$ 为教师模型对样本 $i$ 属于类别 $k$ 的预测概率，$C$ 为总类别数。

**Sigmoid变换**：
$$\sigma_k = \frac{1}{1 + e^{-\beta \cdot (\mathrm{Norm}(\mathrm{CQS}_k) - \gamma)}}$$

参数设置：$\beta = 5, \gamma = 0.5$

**阈值约束**：
$$\alpha_k = v_{\min} + (1 - v_{\min}) \cdot \sigma_k$$

其中 $v_{\min}$ 为下界（如F-MNIST上设为0.7），防止交叉熵系数过小。

**Step 3: 多教师蒸馏**

教师模型选择：根据样本类别 $c$ 选择对应任务训练的教师模型 $M_t$

**KL蒸馏损失**（温度 $T$）：
$$\mathcal{L}_{\mathrm{KD}} = \mathrm{KL}\left( \sigma\left( \frac{Z_t^c}{T} \right) \Big\| \sigma\left( \frac{Z_s^c}{T} \right) \right)$$

**交叉熵损失**：
$$\mathcal{L}_{\mathrm{CE}} = \mathrm{CE}(y, \sigma(z_s))$$

**总损失**：
$$\mathcal{L}_{\mathrm{total}} = \alpha_k \cdot \mathcal{L}_{\mathrm{CE}} + (1 - \alpha_k) \cdot \mathcal{L}_{\mathrm{KD}}$$

---

## 4. 完整算法流程

```
算法 1: FedMTL 工作流程

输入: 客户端数据集 D^i = {D_1^i, D_2^i, ..., D_t^i}, i ∈ [1,k]
      任务数: T

输出: 最终模型 M_g

/* 客户端进程 */
1: for 任务 t = 1 to T do
2:   for 客户端 k = 1 to K do
3:     M_t^k ← TrainClassifyModel(D_t^k; M_g^t)      // 训练分类模型
4:     G_t^k ← TrainGenerateModel(D_t^k; G_g^t)      // 训练CVAE生成模型
5:     服务器 ← UploadModel({M_t^k, G_t^k})          // 上传双模型
6:   end for
7: end for

/* 服务器进程 */
8: for 任务 t = 1 to T do
9:   M_g^t ← FusionClientModel({M_t^1, M_t^2, ..., M_t^k})    // FedAvg聚合
10:  G_g^t ← FusionGenerateModel({G_t^1, G_t^2, ..., G_t^k}) // 生成器聚合
11:  {C_1, C_2, ..., C_m} ← DistributeModel({M_g^t, G_g^t})   // 分发模型
12: end for

13: D_g^t ← GenerateDataset(G_g^t)                    // 生成样本（第3.4节）
14: D_s^t ← FilterAndSelectDataset(D_g^t; M_g^t)      // 双阈值筛选
15: D_z^t ← MergeDataset({D_1^s, D_2^s, ..., D_{t-1}^s}) // 构建跨任务集
16: M_g ← TrainModelByAKD(M_g^t, D_z^t)               // 自适应知识蒸馏（第3.5节）
```

---

## 5. 关键设计对比

| 特性 | 传统联邦学习 | FedMTL |
|:---|:---|:---|
| **知识载体** | 模型参数 | 生成样本 + 教师模型输出 |
| **任务间关系** | 参数覆盖 | 蒸馏继承 |
| **异构性处理** | 加权平均 | 双阈值质量筛选 |
| **自适应机制** | 静态聚合 | CQS动态调整蒸馏强度 |
| **通信内容** | 分类模型 | 分类模型 + 生成模型 |
```