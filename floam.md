以下是完整的客户端算法描述，包含数学定义、损失函数推导及伪代码。

---

## 1. 符号定义与初始化

**全局输入**（来自服务器）：
- $\mathbf{A} \in \mathbb{R}^{C \times D}$：全局文本-视觉锚点矩阵，第 $c$ 行 $\mathbf{a}_c$ 为类别 $c$ 的原型中心；
- $\theta$：当前模型参数（视觉编码器 $\mathcal{E}_v$ 与分类头）。

**客户端本地维护**：
- $\mathcal{Q} = \{Q_c\}_{c=1}^C$：类别特定 Memory Bank，每个 $Q_c \in \mathbb{R}^{M \times D}$ 为 FIFO 队列，长度 $M$；
- $\theta^t$：教师模型参数（EMA 更新：$\theta^t \leftarrow \mu\theta^t + (1-\mu)\theta$）。

**超参数**：
- $\tau$：对比学习温度系数；
- $\varepsilon$：熵正则化强度（Sinkhorn 近似）；
- $\lambda_1, \lambda_2$：损失权重（固定值，无 GradNorm）；
- $\alpha \in [0,1]$：本地锚点聚合时当前 batch 与 Memory Bank 的融合系数；
- $\gamma$：队列更新时的动量系数（可选）。

---

## 2. 核心损失函数

### 2.1 对比学习损失（Memory Bank Based InfoNCE）

对于批次样本特征 $\mathbf{z}_i \in \mathbb{R}^D$（归一化后）及其标签 $y_i$：

**正样本对**：$(\mathbf{z}_i, \mathbf{a}_{y_i})$，其中 $\mathbf{a}_{y_i}$ 为全局锚点第 $y_i$ 行。

**负样本池**：$\mathcal{N}_i = \bigcup_{c \neq y_i} Q_c$，即所有异类队列中存储的历史特征。

**损失计算**：
$$
\mathcal{L}_{\text{con}} = -\frac{1}{B}\sum_{i=1}^B \log \frac{\exp(\mathbf{z}_i^\top \mathbf{a}_{y_i} / \tau)}{\exp(\mathbf{z}_i^\top \mathbf{a}_{y_i} / \tau) + \sum_{\mathbf{m} \in \mathcal{N}_i} \exp(\mathbf{z}_i^\top \mathbf{m} / \tau)}
$$

*注：所有特征均预先归一化，故 $\mathbf{z}^\top \mathbf{a}$ 即余弦相似度。*

### 2.2 熵正则化蒸馏损失（替代 Sinkhorn OT）

设学生模型输出 logits 为 $\mathbf{f}^s = \mathcal{E}_v(\mathbf{X}; \theta)$，教师模型输出为 $\mathbf{f}^t = \mathcal{E}_v(\mathbf{X}; \theta^t)$。

**概率分布**：
$$
\mathbf{p}^s = \text{softmax}(\mathbf{f}^s / T), \quad \mathbf{p}^t = \text{softmax}(\mathbf{f}^t / T)
$$
其中 $T$ 为蒸馏温度。

**成本矩阵**（Batch-wise）：
$$
C_{ij} = \|\mathbf{p}^s_i - \mathbf{p}^t_j\|_2^2, \quad \mathbf{C} \in \mathbb{R}^{C \times C}
$$

**熵正则化近似传输矩阵**（零次迭代 Sinkhorn）：
$$
\mathbf{T} = \text{softmax}\left(-\frac{\mathbf{C}}{\varepsilon}\right) \in \mathbb{R}^{C \times C}
$$
（沿行归一化，即 $\mathbf{T}_{ij} = \frac{\exp(-C_{ij}/\varepsilon)}{\sum_k \exp(-C_{ik}/\varepsilon)}$）

**蒸馏损失**（Wasserstein 距离近似）：
$$
\mathcal{L}_{\text{distill}} = \langle \mathbf{T}, \mathbf{C} \rangle_F = \sum_{i=1}^C \sum_{j=1}^C T_{ij} \cdot C_{ij}
$$

### 2.3 总训练目标

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{con}} + \lambda_2 \mathcal{L}_{\text{distill}}
$$

其中 $\mathcal{L}_{\text{CE}}$ 为标准交叉熵损失，**无 GradNorm 动态加权**。

---

## 3. Memory Bank 更新与锚点聚合

### 3.1 队列更新（FIFO）

对于批次中类别 $c$ 的特征集合 $\mathcal{Z}_c = \{\mathbf{z}_i | y_i = c\}$：
$$
Q_c \leftarrow \text{Enqueue}(Q_c, \mathcal{Z}_c)
$$
若 $|Q_c| > M$，则按时间顺序移除最早的 $|\mathcal{Z}_c|$ 个特征。

### 3.2 融合 Memory Bank 的本地锚点聚合

客户端完成本地训练后，计算返回给服务器的更新后锚点 $\mathbf{A}^{\text{new}}$：

对于每个类别 $c$：
$$
\mathbf{g}_c^{\text{batch}} = \frac{1}{|\mathcal{Z}_c|} \sum_{\mathbf{z} \in \mathcal{Z}_c} \mathbf{z}, \quad 
\mathbf{g}_c^{\text{queue}} = \frac{1}{|Q_c|} \sum_{\mathbf{m} \in Q_c} \mathbf{m}
$$

**融合策略**：
$$
\mathbf{a}_c^{\text{new}} = \text{Normalize}\left( \alpha \cdot \mathbf{g}_c^{\text{batch}} + (1-\alpha) \cdot \mathbf{g}_c^{\text{queue}} \right)
$$

若类别 $c$ 在当前轮次未出现（$\mathcal{Z}_c = \emptyset$），则：
$$
\mathbf{a}_c^{\text{new}} = \text{Normalize}\left( \mathbf{g}_c^{\text{queue}} \right)
$$

---

## 4. 客户端算法伪代码

```
算法 1: 联邦客户端本地训练与锚点更新 (FedACD-MB)

输入: 全局锚点矩阵 A ∈ ℝ^(C×D), 本地数据集 D, 学习率 η,
      队列长度 M, 损失权重 λ₁, λ₂, 融合系数 α, 熵正则 ε
      
输出: 更新后的本地锚点 A^new

1:  初始化: 接收 A, 令 Q_c ← [a_c; a_c; ...; a_c] (复制 M 次), ∀c ∈ [C]
2:  初始化: 教师网络 θ^t ← θ
3:  
4:  for epoch = 1 to E do
5:      for each batch (X, Y) in D do
6:          // 前向传播
7:          Z ← Normalize(E_v(X; θ))          // 特征提取并归一化, Z ∈ ℝ^(B×D)
8:          F^s ← Classifier(Z)                // 学生 logits
9:          F^t ← Classifier(Z; θ^t)           // 教师 logits (无梯度)
10:         
11:         // 1. 对比学习损失 (Memory Bank)
12:         L_con ← 0
13:         for i = 1 to B do
14:             y ← Y[i], z ← Z[i]
15:             pos ← exp(z^T · A[y] / τ)
16:             neg_set ← ∪_{c≠y} Q_c          // 异类队列并集
17:             neg_sum ← Σ_{m∈neg_set} exp(z^T · m / τ)
18:             L_con ← L_con - log(pos / (pos + neg_sum))
19:         end for
20:         L_con ← L_con / B
21:         
22:         // 2. 熵正则化蒸馏 (无 Sinkhorn 迭代)
23:         P^s ← softmax(F^s / T), P^t ← softmax(F^t / T)
24:         C_ij ← ||P^s_i - P^t_j||²₂          // 成本矩阵
25:         T ← softmax(-C / ε)                 // 行方向 softmax
26:         L_distill ← Σ_{i,j} T_ij · C_ij     // Frobenius 内积
27:         
28:         // 3. 分类损失
29:         L_ce ← CrossEntropy(F^s, Y)
30:         
31:         // 总损失 (固定权重, 无 GradNorm)
32:         L ← L_ce + λ₁·L_con + λ₂·L_distill
33:         
34:         // 反向传播与更新
35:         θ ← θ - η·∇_θ L
36:         
37:         // 更新 Memory Bank (类别特定 FIFO)
38:         for c = 1 to C do
39:             Z_c ← {Z[i] | Y[i] = c}
40:             Q_c ← Enqueue(Q_c, Z_c)
41:             if |Q_c| > M then
42:                 Q_c ← Dequeue_oldest(Q_c, |Z_c|)
43:             end if
44:         end for
45:         
46:         // 更新教师网络 (EMA)
47:         θ^t ← μ·θ^t + (1-μ)·θ
48:     end for
49: end for
50: 
51: // 融合 Memory Bank 的锚点聚合
52: A^new ← ∅
53: for c = 1 to C do
54:     G_batch ← Mean({z | z ∈ current batch ∧ class=z_c})
55:     G_queue ← Mean(Q_c)
56:     if G_batch exists then
57:         a_c^new ← Normalize(α·G_batch + (1-α)·G_queue)
58:     else
59:         a_c^new ← Normalize(G_queue)
60:     end if
61:     Append A^new, a_c^new
62: end for
63: 
64: return A^new
```

---

## 5. 关键设计说明

1. **Memory Bank 的类别隔离**：第 38-44 行确保每个类别 $c$ 有独立队列 $Q_c$，杜绝假阴性。
2. **熵正则化近似**：第 25 行以单次 `softmax(-C/ε)` 替代 Sinkhorn 迭代，计算复杂度从 $O(C^2 \cdot \text{iter})$ 降至 $O(C^2)$。
3. **无 GradNorm**：第 32 行使用固定权重 $\lambda_1, \lambda_2$，避免三重前向传播开销。
4. **融合聚合**：第 54-61 行将历史队列特征纳入本地锚点更新，提升非 IID 场景下的原型稳定性。