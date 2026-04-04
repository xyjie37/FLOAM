我将基于上传的PDF文档，为您总结FedCLASS算法并迁移至标签空间固定的联邦学习场景。
让我基于您上传的PDF文档内容，为您总结FedCLASS算法并迁移至标签空间固定的联邦学习场景。

---

## FedCLASS 算法总结（迁移至固定标签空间版本）

### 一、核心思想

FedCLASS 通过**自蒸馏机制**实现联邦学习中的知识迁移。在固定标签空间场景下，移除"新类别扩充"相关组件，保留其核心概率建模框架，实现客户端间基于历史模型的知识一致性约束。

---

### 二、形式化定义

设总类别数为 $C$，标签空间 $\mathcal{Y} = \{1, 2, \dots, C\}$ 全局固定。

**符号系统：**
- $K$：客户端数量
- $T$：通信轮次总数
- $\tau$：本地训练轮数
- $\theta_t^{(k)}$：第 $t$ 轮第 $k$ 个客户端的模型参数
- $\tilde{\theta}_{t-1}$：第 $t-1$ 轮全局聚合后的历史模型（冻结）
- $\mathcal{D}_k$：第 $k$ 个客户端的本地数据集
- $p_{\theta}(y|x)$：模型 $\theta$ 对输入 $x$ 的类别概率分布

---

### 三、概率建模框架

#### 3.1 历史模型蒸馏分布

历史模型 $\tilde{\theta}_{t-1}$ 提供的软标签作为知识源：

$$q_t^{(k)}(x) = p_{\tilde{\theta}_{t-1}}(y|x) \in \mathbb{R}^C$$

#### 3.2 当前模型预测

当前模型 $\theta_t^{(k)}$ 的输出经温度缩放：

$$p_t^{(k)}(x) = \text{softmax}\left(\frac{z_{\theta_t^{(k)}}(x)}{\tau}\right) \in \mathbb{R}^C$$

其中 $z_{\theta}(x)$ 为 logits，$\tau > 0$ 为温度系数。

---

### 四、损失函数设计

#### 4.1 自蒸馏损失（固定标签空间适配版）

由于标签空间固定，无需区分新旧类别，统一对所有类别进行蒸馏：

$$\mathcal{L}_{\text{distill}}^{(k)} = \frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} \text{KL}\left( q_t^{(k)}(x) \,\|\, p_t^{(k)}(x) \right)$$

#### 4.2 监督交叉熵损失

$$\mathcal{L}_{\text{ce}}^{(k)} = -\frac{1}{|\mathcal{B}|} \sum_{(x,y) \in \mathcal{B}} \sum_{c=1}^{C} \mathbb{1}_{[y=c]} \log p_t^{(k)}(c|x)$$

#### 4.3 总体损失函数

$$\mathcal{L}^{(k)} = \mathcal{L}_{\text{ce}}^{(k)} + \lambda \cdot \mathcal{L}_{\text{distill}}^{(k)}$$

其中 $\lambda > 0$ 为蒸馏权重系数。

---

### 五、服务器端聚合

采用标准联邦平均（FedAvg）：

$$\tilde{\theta}_t = \sum_{k=1}^{K} \frac{n_k}{n} \theta_{t,\tau}^{(k)}$$

其中 $n_k = |\mathcal{D}_k|$，$n = \sum_{k=1}^{K} n_k$。

---

### 六、算法伪代码

```markdown
## Algorithm: FedCLASS-Fixed (Fixed Label Space Adaptation)

**Input:**  
- $K$: number of clients  
- $T$: total communication rounds  
- $\tau$: local epochs  
- $\mathcal{D}_k$: local dataset of client $k$  
- $\lambda$: distillation loss weight  
- $T_{\text{temp}}$: temperature for softmax  

**Output:**  
- $\tilde{\theta}_T$: final global model  

---

### Server Procedure:

1. **Initialize** global model $\tilde{\theta}_0$
2. **For** $t = 1, 2, \dots, T$ **do**:
   3. $\quad$ **Broadcast** $\tilde{\theta}_{t-1}$ to all clients $k \in [K]$
   4. $\quad$ **For each** client $k \in [K]$ **in parallel**:
   5. $\quad\quad$ $\theta_{t}^{(k)} \leftarrow \text{ClientUpdate}(k, \tilde{\theta}_{t-1})$
   6. $\quad$ **Aggregate**: $\tilde{\theta}_t \leftarrow \sum_{k=1}^{K} \frac{n_k}{n} \theta_{t}^{(k)}$
7. **Return** $\tilde{\theta}_T$

---

### ClientUpdate($k$, $\tilde{\theta}_{\text{hist}}$):

1. **Initialize** local model: $\theta \leftarrow \tilde{\theta}_{\text{hist}}$
2. **Freeze** historical model: $\theta_{\text{hist}} \leftarrow \tilde{\theta}_{\text{hist}}$ (no gradient)
3. **For** epoch $e = 1, 2, \dots, \tau$ **do**:
   4. $\quad$ **For each** batch $\mathcal{B} \subset \mathcal{D}_k$ **do**:
   5. $\quad\quad$ // Forward pass with current model
   6. $\quad\quad$ $p \leftarrow \text{softmax}(z_{\theta}(x) / T_{\text{temp}})$, for $x \in \mathcal{B}$
   7. $\quad\quad$ // Forward pass with frozen historical model
   8. $\quad\quad$ $q \leftarrow \text{softmax}(z_{\theta_{\text{hist}}}(x) / T_{\text{temp}})$, for $x \in \mathcal{B}$
   9. $\quad\quad$ // Compute losses
   10. $\quad\quad$ $\mathcal{L}_{\text{ce}} \leftarrow -\frac{1}{|\mathcal{B}|} \sum_{(x,y) \in \mathcal{B}} \log p[y]$
   11. $\quad\quad$ $\mathcal{L}_{\text{distill}} \leftarrow \frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} \sum_{c=1}^{C} q[c] \cdot \log \frac{q[c]}{p[c]}$
   12. $\quad\quad$ $\mathcal{L} \leftarrow \mathcal{L}_{\text{ce}} + \lambda \cdot \mathcal{L}_{\text{distill}}$
   13. $\quad\quad$ **Update** $\theta$ via SGD/Adam on $\mathcal{L}$
14. **Return** $\theta$ to server
```

---

### 七、与原FedCLASS的关键差异（迁移说明）

| 组件 | 原FedCLASS（类增量） | FedCLASS-Fixed（固定标签） |
|:---|:---|:---|
| **标签空间** | 动态扩展 $C_{\text{old}} \to C_{\text{old}} + C_{\text{new}}$ | 固定 $C$ |
| **概率分解** | 需分解 $p_{\text{old}}$ 与 $p_{\text{new}}$ | 统一处理完整分布 $p \in \mathbb{R}^C$ |
| **蒸馏对象** | 历史模型仅提供旧类知识，需新类扩充 | 历史模型提供全类别知识 |
| **损失权重** | 新旧类分别加权 | 统一蒸馏权重 $\lambda$ |
| **应用场景** | 联邦类增量学习（FCIL） | 标准联邦学习（异构数据） |

---

### 八、理论性质（继承自原文）

**定理（固定标签空间版本）：** 在固定标签空间下，自蒸馏损失 $\mathcal{L}_{\text{distill}}$ 等价于对历史模型后验分布的极大似然估计，保证本地优化过程满足：

$$\mathbb{E}_{x \sim \mathcal{D}_k}\left[ \text{KL}\left( p_{\tilde{\theta}_{t-1}}(y|x) \,\|\, p_{\theta_t^{(k)}}(y|x) \right) \right] \leq \epsilon$$

当本地数据分布 $\mathcal{D}_k$ 与全局分布存在异构性时，该约束强制客户端模型向历史共识模型对齐，缓解客户端漂移（client drift）。

---

### 九、超参数设置建议

| 参数 | 推荐值 | 说明 |
|:---|:---|:---|
| $\lambda$ | $0.1 \sim 1.0$ | 蒸馏强度，数据异构度高时取大值 |
| $T_{\text{temp}}$ | $2 \sim 4$ | 温度系数，软化概率分布 |
| $\tau$ | $5 \sim 20$ | 本地轮数，平衡通信与计算成本 |

---

### 十、适用边界

本算法适用于：
- 标签空间全局已知且固定的横向联邦学习
- 客户端数据分布异构（Non-IID）场景
- 需利用历史全局知识约束本地优化的场景

不适用于：
- 类别动态增长的联邦类增量学习（需使用原FedCLASS）
- 标签空间不完全重叠的跨域联邦学习