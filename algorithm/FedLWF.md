# Learning without Forgetting (LwF) 算法详解
本算法为面向卷积神经网络（CNN）的增量学习算法，核心目标是**仅利用新任务训练数据，为预训练CNN添加新任务能力，同时保留原有任务的性能**，无需访问旧任务的训练数据，解决了深度网络的灾难性遗忘问题。以下为算法的完整符号定义、步骤、损失函数及伪代码，所有内容严格遵循原文公式与流程。

## 1 核心符号定义
| 符号          | 含义                                                                 |
|---------------|----------------------------------------------------------------------|
| $\theta_s$    | CNN的**共享参数**（如AlexNet的卷积层+前两层全连接层，VGG的非输出层）|
| $\theta_o$    | 旧任务的**任务特定参数**（如CNN的输出层权重，对应已学任务的分类器）|
| $\theta_n$    | 新任务的**任务特定参数**（随机初始化，对应新任务的分类器，与$\theta_o$共享$\theta_s$） |
| $X_n, Y_n$    | 新任务的训练数据与真实标签（仅需此数据完成整个算法训练）|
| $\hat{y}_o$   | 当前网络对新任务样本的**旧任务预测输出**（softmax概率）|
| $y_o$         | 原预训练网络对新任务样本的**旧任务记录输出**（softmax概率，训练前固定）|
| $\hat{y}_n$   | 当前网络对新任务样本的**新任务预测输出**（softmax概率）|
| $y_n$         | 新任务样本的**真实标签**（one-hot向量）|
| $T$           | 知识蒸馏的**温度参数**（用于放大小概率值的权重）|
| $\lambda_o$   | 旧任务损失的**平衡权重**（调节新旧任务损失的占比，默认$\lambda_o=1$）|
| $\mathcal{L}_{new}$ | 新任务损失函数 |
| $\mathcal{L}_{old}$  | 旧任务知识蒸馏损失函数 |
| $R(\theta)$   | 正则化项（权重衰减）|

## 2 算法核心前提
1. 存在一个**预训练完成的CNN**，已掌握若干旧任务，参数为$\theta_s$（共享）+$θ_o$（旧任务特定），且旧任务的训练数据不可用；
2. 仅能获取**新任务的训练数据**$X_n, Y_n$，无任何旧任务数据；
3. 新任务为分类任务（原文为多分类/多标签分类，可扩展至其他视觉任务），CNN的共享参数$\theta_s$为新旧任务共享，新任务仅新增任务特定参数$\theta_n$。

## 3 算法完整执行步骤
LwF算法分为**5个核心步骤**，所有步骤均基于新任务数据完成，无旧任务数据参与，步骤间严格衔接，公式均为原文LaTeX原版。

### 步骤1：预训练网络加载与新任务参数初始化
1. 加载预训练完成的CNN模型，固定此时的共享参数$\theta_s^0$和旧任务特定参数$\theta_o^0$（上标0表示初始预训练值）；
2. 为新任务**新增输出层节点**，节点数等于新任务的类别数，该层为新任务特定参数$\theta_n$，采用**Xavier初始化**随机初始化；
3. 新参数$\theta_n$的维度为：$N_{new} \times N_{last}$，其中$N_{new}$为新任务类别数，$N_{last}$为CNN最后一个共享层的节点数（$\theta_n$仅占总参数的极小比例）。

### 步骤2：记录新任务样本在原网络的旧任务输出
对所有新任务训练样本$x \in X_n$，通过**原预训练网络（$\theta_s^0, \theta_o^0$）** 进行前向传播，得到旧任务的输出概率$y_o(x)$，并**永久记录该值**（作为后续旧任务损失的监督信号，训练过程中不再修改）。
$$y_o(x) = \text{CNN}(x; \theta_s^0, \theta_o^0)$$
若存在**多个旧任务/多标签旧任务**，则对每个旧任务/标签分别记录$y_o^k(x)$（$k$为旧任务/标签索引），最终旧任务损失为所有$y_o^k(x)$的损失和。

### 步骤3：构建整体损失函数
整体损失为**新任务损失**、**旧任务知识蒸馏损失**与**正则化项**的加权和，是算法的核心，所有公式严格遵循原文(1)-(4)。
$$\mathcal{L}(\theta_s, \theta_o, \theta_n) = \mathcal{L}_{new}(y_n, \hat{y}_n) + \lambda_o \cdot \mathcal{L}_{old}(y_o, \hat{y}_o) + R(\theta_s, \theta_o, \theta_n)$$
其中：$\hat{y}_o = \text{CNN}(x; \theta_s, \theta_o)$，$\hat{y}_n = \text{CNN}(x; \theta_s, \theta_n)$，为当前网络的前向输出。

#### 子步骤3.1：新任务损失$\mathcal{L}_{new}$（多项逻辑损失/交叉熵）
新任务为多分类时，采用经典的**多项逻辑损失**，与CNN分类的标准交叉熵一致：
$$\mathcal{L}_{new }\left(y_{n}, \hat{y}_{n}\right)=-y_{n} \cdot \log \hat{y}_{n} \tag{1}$$
- $\hat{y}_n$为当前网络对新任务的**softmax输出概率**；
- $y_n$为新任务样本的**one-hot真实标签向量**；
- 若为**多标签新任务/多个新任务**，则损失为所有标签/新任务的损失之和。

#### 子步骤3.2：旧任务损失$\mathcal{L}_{old}$（知识蒸馏损失）
采用Hinton的**知识蒸馏损失**，通过温度参数$T$放大小概率值的权重，强制当前网络的旧任务输出与**步骤2记录的原网络输出$y_o$** 一致，公式为：
$$\mathcal{L}_{old }\left(y_{o}, \hat{y}_{o}\right) = -H\left(y_{o}', \hat{y}_{o}'\right) = -\sum_{i=1}^{l} y_{o}'^{(i)} \log \hat{y}_{o}'^{(i)} \tag{2,3}$$
其中$l$为旧任务的类别数，$y_o'^{(i)}$和$\hat{y}_o'^{(i)}$为**温度修正后的概率值**，修正公式为：
$$y_{o}'^{(i)}=\frac{\left(y_{o}^{(i)}\right)^{1 / T}}{\sum_{j}\left(y_{o}^{(j)}\right)^{1 / T}}, \quad \hat{y}_{o}'^{(i)}=\frac{\left(\hat{y}_{o}^{(i)}\right)^{1 / T}}{\sum_{j}\left(\hat{y}_{o}^{(j)}\right)^{1 / T}} \tag{4}$$
- 温度$T \geq 1$，原文通过网格搜索确定$T=2$，$T>1$时增强对类别间相似性的编码；
- 若存在**多个旧任务/多标签旧任务**，则损失为所有旧任务/标签的知识蒸馏损失之和。

#### 子步骤3.3：正则化项$R(\theta)$（权重衰减）
采用**L2权重衰减**作为正则化，防止过拟合，原文固定权重衰减系数为$0.0005$：
$$R(\theta_s, \theta_o, \theta_n) = 0.0005 \cdot \left( \|\theta_s\|_2^2 + \|\theta_o\|_2^2 + \|\theta_n\|_2^2 \right)$$

### 步骤4：分阶段训练优化（预热阶段+联合优化阶段）
采用**随机梯度下降（SGD）** 优化整体损失，动量为$0.9$，学习率远小于原网络训练的学习率（原文为原学习率的$0.1/0.02$倍），训练分为两个阶段，均在新任务数据$X_n,Y_n$上完成。

#### 子步骤4.1：预热阶段（Warm-up）
1. **冻结共享参数$\theta_s$和旧任务参数$\theta_o$**，即$\nabla_{\theta_s} \mathcal{L} = 0$，$\nabla_{\theta_o} \mathcal{L} = 0$；
2. 仅对新任务参数$\theta_n$进行梯度下降优化，最小化整体损失；
3. 训练至**新任务损失收敛**（以新任务验证集性能为指标）。
> 原文说明：预热阶段对LwF的性能提升非关键，但能显著提升对比方法（如fine-tuning）的旧任务性能，为公平对比，LwF保留此步骤。

#### 子步骤4.2：联合优化阶段（Joint-optimize）
1. **解冻所有参数**（$\theta_s, \theta_o, \theta_n$均参与优化）；
2. 以**预热阶段的收敛参数**为初始值，对整体损失进行梯度下降优化；
3. 训练过程中，当**新任务验证集性能趋于平稳**时，将学习率降低10倍，继续训练至收敛；
4. 若为**批次训练**，损失为批次内所有样本的平均损失。

### 步骤5：模型推理
对测试样本$x$，通过训练完成的网络分别前向传播得到：
- 旧任务输出：$\hat{y}_o = \text{CNN}(x; \theta_s, \theta_o)$；
- 新任务输出：$\hat{y}_n = \text{CNN}(x; \theta_s, \theta_n)$；
实现**单模型同时完成新旧任务的推理**，无需为不同任务单独构建模型。

## 4 训练优化的关键超参设置（原文固定）
1. 优化器：SGD + 动量$0.9$，全连接层启用Dropout；
2. 温度参数：$T=2$（知识蒸馏）；
3. 损失平衡权重：$\lambda_o=1$（默认，可调整以权衡新旧任务性能）；
4. 权重衰减：$0.0005$；
5. 学习率：为原网络训练学习率的$0.1/0.02$倍，预热阶段与联合优化阶段使用相同学习率，性能平稳时降为1/10；
6. 数据增强：与原预训练网络一致（随机裁剪、水平翻转、RGB值加噪声）；
7. 数据归一化：沿用原预训练网络的均值减法归一化。

## 5 LwF算法伪代码
伪代码严格遵循原文流程，包含**符号定义、初始化、输出记录、损失计算、分阶段训练**，可直接作为工程实现的参考，所有操作均基于新任务数据完成。

```plaintext
# Learning without Forgetting (LwF) 算法伪代码
# 输入：预训练模型(θ_s^0, θ_o^0)，新任务数据(X_n, Y_n)，温度T=2，损失平衡权重λ_o=1，权重衰减系数λ_r=0.0005
# 输出：融合新旧任务的模型(θ_s^*, θ_o^*, θ_n^*)
def LwF(θ_s^0, θ_o^0, X_n, Y_n, T, λ_o, λ_r):
    # 步骤1：初始化新任务参数θ_n
    θ_n = Xavier_Init(N_new, N_last)  # N_new:新任务类别数，N_last:最后一个共享层节点数
    θ_s, θ_o = θ_s^0, θ_o^0          # 初始化共享参数和旧任务参数
    batch_size = B                   # 批次大小，原文未指定，按需设置
    epochs_warmup = E1               # 预热阶段轮数，由验证集确定
    epochs_joint = E2                # 联合优化阶段轮数，由验证集确定
    lr = η                           # 学习率，原网络的0.1/0.02倍
    momentum = 0.9                   # SGD动量

    # 步骤2：记录新任务样本在原网络的旧任务输出y_o
    y_o = []
    for x in X_n:
        y_o_x = CNN_Forward(x, θ_s^0, θ_o^0)  # 原网络前向传播，输出旧任务概率
        y_o.append(y_o_x)
    y_o = np.array(y_o)  # 永久固定，训练过程不修改

    # 步骤3：定义整体损失函数
    def Loss(y_n, y_o, x, θ_s, θ_o, θ_n):
        # 前向传播得到当前输出
        ŷ_o = CNN_Forward(x, θ_s, θ_o)
        ŷ_n = CNN_Forward(x, θ_s, θ_n)
        # 新任务损失：多项逻辑损失
        Ln = -np.sum(y_n * np.log(ŷ_n + 1e-8))  # 加小值防止log(0)
        # 旧任务损失：知识蒸馏损失（温度修正）
        y_o_prime = (y_o ** (1/T)) / np.sum(y_o ** (1/T), axis=-1, keepdims=True)
        ŷ_o_prime = (ŷ_o ** (1/T)) / np.sum(ŷ_o ** (1/T), axis=-1, keepdims=True)
        Lo = -np.sum(y_o_prime * np.log(ŷ_o_prime + 1e-8))
        # 正则化项：L2权重衰减
        R = λ_r * (np.linalg.norm(θ_s)**2 + np.linalg.norm(θ_o)**2 + np.linalg.norm(θ_n)**2)
        # 整体损失
        total_L = Ln + λ_o * Lo + R
        return total_L, Ln, Lo

    # 步骤4.1：预热阶段——冻结θ_s, θ_o，仅训练θ_n
    for epoch in 1 to epochs_warmup:
        shuffle(X_n, Y_n)  # 每轮打乱数据
        for i in 0 to len(X_n) step batch_size:
            # 取批次数据
            x_batch = X_n[i:i+batch_size]
            y_n_batch = Y_n[i:i+batch_size]
            y_o_batch = y_o[i:i+batch_size]
            # 计算损失与梯度（仅对θ_n求导）
            total_L, _, _ = Loss(y_n_batch, y_o_batch, x_batch, θ_s, θ_o, θ_n)
            grad_θn = Gradient(total_L, θ_n)
            # SGD更新θ_n
            θ_n = θ_n - lr * grad_θn + momentum * (θ_n_prev - θ_n)  # θ_n_prev为上一轮θ_n
        # 验证新任务性能，收敛则提前终止
        if Val_Performance(X_val, Y_val, θ_s, θ_o, θ_n) >= Threshold:
            break

    # 步骤4.2：联合优化阶段——解冻所有参数，联合训练θ_s, θ_o, θ_n
    for epoch in 1 to epochs_joint:
        shuffle(X_n, Y_n)
        for i in 0 to len(X_n) step batch_size:
            x_batch = X_n[i:i+batch_size]
            y_n_batch = Y_n[i:i+batch_size]
            y_o_batch = y_o[i:i+batch_size]
            # 计算损失与所有参数的梯度
            total_L, _, _ = Loss(y_n_batch, y_o_batch, x_batch, θ_s, θ_o, θ_n)
            grad_θs = Gradient(total_L, θ_s)
            grad_θo = Gradient(total_L, θ_o)
            grad_θn = Gradient(total_L, θ_n)
            # SGD更新所有参数
            θ_s = θ_s - lr * grad_θs + momentum * (θ_s_prev - θ_s)
            θ_o = θ_o - lr * grad_θo + momentum * (θ_o_prev - θ_o)
            θ_n = θ_n - lr * grad_θn + momentum * (θ_n_prev - θ_n)
        # 验证新任务性能，平稳则降低学习率
        if Val_Performance_Plateau(X_val, Y_val, θ_s, θ_o, θ_n):
            lr = lr / 10
        # 收敛则提前终止
        if Val_Performance(X_val, Y_val, θ_s, θ_o, θ_n) >= Threshold:
            break

    # 输出训练完成的模型参数
    return θ_s, θ_o, θ_n

# 辅助函数：CNN前向传播（根据具体网络结构实现，如AlexNet/VGG）
def CNN_Forward(x, θ_s, θ_task):
    features = Extract_Features(x, θ_s)  # 共享层提取特征
    logits = np.dot(features, θ_task[:-1].T) + θ_task[-1]  # 任务特定层（输出层）
    prob = Softmax(logits)  # softmax得到概率
    return prob

# 辅助函数：Softmax激活
def Softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 防止溢出
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

## 6 算法的核心特性（原文总结）
1. **无旧任务数据依赖**：全程仅用新任务数据$X_n,Y_n$，解决了旧任务数据因版权/存储/隐私无法获取的问题；
2. **无模型膨胀**：仅新增少量新任务参数$\theta_n$，共享参数$\theta_s$复用，无需为每个任务构建单独模型；
3. **双任务性能优化**：既保证新任务性能优于传统的特征提取/微调，又能大幅保留旧任务性能，避免灾难性遗忘；
4. **训练效率高**：训练速度略慢于微调，远快于联合训练，推理时单模型完成多任务，效率与原模型一致。