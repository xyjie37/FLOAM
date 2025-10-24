import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset
from tqdm import tqdm
import math
import pdb
import copy
from torch.optim import Optimizer
#from transformers import CLIPProcessor, CLIPModel

from utils.data_utils import load_train_data, load_test_data
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from collections import defaultdict
import torch.optim as optim
from scipy.stats import wasserstein_distance
from cvxopt import matrix, solvers


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class AnchorContrastiveLoss(nn.Module):
    """
    动态 Hard Negative Mining（稳定版）
    - 用 sparsemax 近似替代离散 topk，降低梯度路径跳变；
    - 对 s_avg 进行 EMA + Sigmoid 温度压缩，减小早期抖动灵敏度；
    - 对 k 使用 EMA（连续化），避免硬限速的边界振荡。
    """
    def __init__(self,
                 anchors,                    # [num_classes, feat_dim]
                 temperature=0.1,
                 device='cuda',
                 momentum_s_avg=0.9,         # s_avg 的 EMA 系数
                 s_avg_scale=3.0,            # Sigmoid 压缩强度：sigmoid((x-0.5)*scale)
                 momentum_k=0.9,             # k 的 EMA 系数
                 use_sparsemax=True):
        super().__init__()
        # 确保 anchors 本身不携带梯度
        self.register_buffer('anchors', anchors.detach().clone())
        self.temperature = temperature
        self.device = device

        # 平滑与温度压缩相关超参
        self.momentum_s_avg = momentum_s_avg
        self.s_avg_scale = s_avg_scale

        # k 的 EMA 超参
        self.momentum_k = momentum_k

        # 使用 sparsemax 替代 soft top-k
        self.use_sparsemax = use_sparsemax

        # =====  关键修改：buffer 初始到与 anchors 同一 device  =====
        dev = self.anchors.device          # 既支持 cuda 也支持 cpu
        self.register_buffer('s_avg_ema', torch.tensor(0.5, device=dev))
        self.register_buffer('k_ema', torch.tensor(0.0, device=dev))

        # 记录指标
        self.metrics = {'hard_neg_sim': 0.0, 'dynamic_k': 0, 's_avg_ema': 0.5}

    @staticmethod
    def sparsemax(input, dim=1):
        """
        Sparsemax implementation
        Args:
            input: Tensor of any shape
            dim: Dimension along which to apply sparsemax
        Returns:
            Tensor of same shape as input with sparsemax applied
        """
        # Get the number of elements in the specified dimension
        num_elements = input.size(dim)
        
        # Sort input in descending order
        input_sorted, _ = torch.sort(input, dim=dim, descending=True)
        
        # Calculate cumulative sum
        input_cumsum = torch.cumsum(input_sorted, dim=dim)
        
        # Create a range tensor [1, 2, ..., num_elements]
        k = torch.arange(1, num_elements + 1, device=input.device).view(1, -1)
        
        # Calculate the condition: 1 + k * z_k > sum(z_1:k)
        condition = 1 + k * input_sorted > input_cumsum
        
        # Find the largest k that satisfies the condition
        k_max = condition.sum(dim=dim, keepdim=True).float()
        
        # Calculate tau (threshold)
        tau = (input_cumsum.gather(dim, k_max.long() - 1) - 1) / k_max
        
        # Apply sparsemax: max(z - tau, 0)
        output = torch.clamp(input - tau, min=0)
        
        return output

    # 维度自适应（可选）
    def _adjust_anchor_dim(self, target_dim):
        cur = self.anchors.size(1)
        if cur == target_dim:
            return
        linear = nn.Linear(cur, target_dim, bias=False).to(self.anchors.device)
        with torch.no_grad():
            new_anchors = linear(self.anchors)
        # 防止带入梯度
        self.anchors = new_anchors.detach().clone()

    def forward(self, features, labels, k=5, alpha=0.8, adaptive_k=True):
        if features.size(1) != self.anchors.size(1):
            self._adjust_anchor_dim(features.size(1))

        # 余弦相似度 [B,K]
        sim = F.cosine_similarity(features.unsqueeze(1),
                                  self.anchors.unsqueeze(0), dim=2)
        logits = sim / self.temperature
        B, K = logits.shape

        # mask 正样本
        pos_mask = F.one_hot(labels, num_classes=K).bool()
        pos_scores = logits.gather(1, labels.view(-1, 1))  # [B,1]
        neg_logits = logits.masked_fill(pos_mask, -float('inf'))

        # ------- 计算稳定的 dynamic k -------
        if adaptive_k:
            with torch.no_grad():
                # 将余弦相似度 [-1,1] -> [0,1]
                s = ((sim.detach() + 1.0) * 0.5).mean()

                # EMA 平滑 s_avg
                self.s_avg_ema.mul_(self.momentum_s_avg).add_(
                    s * (1 - self.momentum_s_avg))

                # Sigmoid 压缩，降低灵敏度
                s_bar = torch.sigmoid((self.s_avg_ema - 0.5) * self.s_avg_scale)

                # 原始 k（连续域），随后再做 EMA
                raw_k = torch.clamp(
                    k + torch.round(s_bar * (K - 1)).to(self.s_avg_ema.dtype),
                    min=1, max=K - 1)

                # 初始化 k_ema
                if self.k_ema.item() == 0.0:
                    self.k_ema.copy_(raw_k)

                # EMA 连续化 k
                self.k_ema.mul_(self.momentum_k).add_(
                    raw_k * (1 - self.momentum_k))
                dynamic_k = int(
                    torch.clamp(torch.round(self.k_ema), 1, K - 1).item())
        else:
            dynamic_k = k

        # ------- sparsemax 近似（可导、连续） -------
        eps = 1e-12
        if self.use_sparsemax:
            # 使用 sparsemax 替代 soft top-k
            w = self.sparsemax(neg_logits, dim=1)  # [B,K]
            
            # 确保权重和为1，避免数值问题
            w_sum = w.sum(dim=1, keepdim=True)
            w = w / (w_sum + eps)
            
            # 计算稀疏加权的负样本logits
            lse_neg_sparsemax = torch.logsumexp(neg_logits + torch.log(w + eps), 
                                               dim=1, keepdim=True)  # [B,1]

            log_sum_hard = torch.logsumexp(
                torch.cat([pos_scores, lse_neg_sparsemax], dim=1),
                dim=1, keepdim=True)  # [B,1]
        else:
            hard_neg, _ = torch.topk(neg_logits, k=dynamic_k, dim=1)  # [B,k]
            log_sum_hard = torch.logsumexp(
                torch.cat([pos_scores, hard_neg], dim=1), dim=1, keepdim=True)

        # 所有负样本的 log-sum-exp（作为对照项）
        log_sum_all = torch.logsumexp(neg_logits, dim=1, keepdim=True)

        # 可选：自适应 alpha（平滑）
        if self.training and adaptive_k:
            with torch.no_grad():
                if self.use_sparsemax:
                    sim_neg = sim.masked_fill(pos_mask, -1e9)
                    w_sim = self.sparsemax(neg_logits, dim=1)
                    # 确保权重和为1
                    w_sim = w_sim / (w_sim.sum(dim=1, keepdim=True) + eps)
                    hard_stat = (w_sim * neg_logits.clamp(min=-30, max=30)
                                 ).sum(dim=1).mean()
                else:
                    hard_neg, _ = torch.topk(neg_logits, k=dynamic_k, dim=1)
                    hard_stat = hard_neg.mean()
                adapt_alpha = torch.sigmoid(hard_stat)
            alpha = 0.9 * alpha + 0.1 * adapt_alpha.item()

        # 最终损失（InfoNCE 的"正 vs 组合负"混合）
        loss = -(pos_scores -
                 (alpha * log_sum_hard + (1 - alpha) * log_sum_all)).mean()

        # 指标记录
        with torch.no_grad():
            self.metrics['dynamic_k'] = dynamic_k
            self.metrics['s_avg_ema'] = float(self.s_avg_ema.item())
            if self.use_sparsemax:
                sim_neg = sim.masked_fill(pos_mask, -1e9)
                w_sim = self.sparsemax(neg_logits, dim=1)
                # 确保权重和为1
                w_sim = w_sim / (w_sim.sum(dim=1, keepdim=True) + eps)
                self.metrics['hard_neg_sim'] = float(
                    (w_sim * sim_neg.clamp(min=-1, max=1)
                     ).sum(dim=1).mean().item())
            else:
                hard_neg_sim, _ = torch.topk(
                    sim.masked_fill(pos_mask, -1e9), k=dynamic_k, dim=1)
                self.metrics['hard_neg_sim'] = float(hard_neg_sim.mean().item())

        return loss


class LocalUpdateFedACD(object):
    def __init__(self, args, anchor=None, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.anchor = anchor.to(self.args.device) if anchor is not None \
                      else torch.randn(args.num_classes, 100).to(args.device)
        self.num_classes = args.num_classes

        # --------- GradNorm 解耦 ---------
        self.delay_steps = getattr(args, 'gradnorm_delay_steps', 1)
        self.loss_weights_queue = []          # 环形缓冲区
        self.loss_weights = torch.ones(3, requires_grad=True, device=self.args.device)
        self.optimizer_weights = torch.optim.Adam([self.loss_weights], lr=0.01)

    # ---------- 工具函数 ----------
    def _enqueue_weights(self, w):
        self.loss_weights_queue.append(w.detach().clone())
        if len(self.loss_weights_queue) > self.delay_steps:
            self.loss_weights_queue.pop(0)
    def _get_delayed_weights(self):
        """取出最早放进队列的权重供本轮训练使用"""
        if not self.loss_weights_queue:          # 队列空时返回均匀权重
            return torch.ones(3, device=self.args.device)
        return self.loss_weights_queue[0]        # 队首即“延迟权重”
    @torch.no_grad()
    def _snapshot_params(self, params):
        """返回参数的 detached 克隆，用于重新 forward"""
        return [p.clone() for p in params]

    def _grad_norm(self, loss_fn, inputs, targets, teacher_out=None):
        """
        重新 forward 一次，得到全新损失张量，再求梯度范数
        参数：
            loss_fn :  callable，接受 (logits, targets, teacher_out) 返回损失
            inputs  :  图像
            targets :  标签
            teacher_out : 教师输出（可选）
        """
        features = self.real_net.extract_features(inputs)
        logits = self.real_net.only_liner(features)
        loss = loss_fn(logits, targets, teacher_out)   # 全新张量，从未 backward
        grads = torch.autograd.grad(
            loss, self.body_params,
            create_graph=False, only_inputs=True, allow_unused=True
        )
        norms = [g.norm() for g in grads if g is not None]
        return torch.stack(norms).mean() if norms else torch.tensor(0.0, device=self.args.device)

    # ---------- 训练 ----------
    def train(self, net, teacher_net, lr, idx=-1, local_eps=None):
        net.train()
        teacher_net.eval()

        self.real_net = net.module if hasattr(net, 'module') else net
        self.body_params = [p for name, p in self.real_net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in self.real_net.named_parameters() if 'linear' in name]
        for p in head_params:
            p.requires_grad = False

        optimizer = torch.optim.SGD(self.body_params, lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        epoch_loss = []
        local_eps = self.args.local_ep_pretrain if self.pretrain \
                    else self.args.local_ep if local_eps is None else local_eps

        # 预热队列
        while len(self.loss_weights_queue) < self.delay_steps:
            self._enqueue_weights(torch.ones(3, device=self.args.device))

        for epoch in range(local_eps):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                # 1. 主 forward
                features = self.real_net.extract_features(images)
                logits = self.real_net.only_liner(features)

                loss_ce = self.loss_func(logits, labels)

                contrast_loss = AnchorContrastiveLoss(
                    anchors=self.anchor, temperature=0.5, device=self.args.device
                )(features=logits, labels=labels)

                with torch.no_grad():
                    t_net = teacher_net.module if hasattr(teacher_net, 'module') else teacher_net
                    teacher_out = t_net(images)
                distillation_loss = AnchorDistillationLoss(
                    logits, teacher_out, self.anchor, temperature=1.0)()

                # 2. 重新 forward 得到全新子损失，求梯度范数（绝无二此错误）
                grad_norms = torch.stack([
                    self._grad_norm(lambda logits, y, _: self.loss_func(logits, y),
                                    images, labels, None),
                    self._grad_norm(lambda logits, y, _: AnchorContrastiveLoss(
                        anchors=self.anchor, temperature=0.5, device=self.args.device
                    )(features=logits, labels=y),
                                    images, labels, None),
                    self._grad_norm(lambda logits, _, t: AnchorDistillationLoss(
                        logits, t, self.anchor, temperature=1.0)(),
                                    images, labels, teacher_out)
                ])

                # 3. 主损失 backward（图即释放）
                delayed_w = self._get_delayed_weights()
                loss = delayed_w[0]*loss_ce + delayed_w[1]*contrast_loss + delayed_w[2]*distillation_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 4. GradNorm 更新权重
                target_norm = grad_norms.mean()
                epsilon = 1e-6
                loss_ratios = (grad_norms + epsilon) / (target_norm + epsilon)
                grad_loss = loss_ratios - loss_ratios.mean()

                self.optimizer_weights.zero_grad()
                self.loss_weights.backward(gradient=grad_loss)
                self.optimizer_weights.step()

                with torch.no_grad():
                    self.loss_weights.data = 3 * self.loss_weights.data / (self.loss_weights.data.sum() + 1e-8)
                self._enqueue_weights(self.loss_weights)

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # ---------- 聚合原型 ----------
        agg_protos_label = {}
        with torch.no_grad():
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                features = self.real_net.extract_features(images)
                for label in labels.unique():
                    mask = labels == label
                    lbl = label.item()
                    feat = features[mask]
                    if feat.numel() == 0:
                        continue
                    weights = torch.softmax(feat.norm(dim=1), dim=0)
                    weighted = (feat.T @ weights).T
                    agg_protos_label[lbl] = agg_protos_label.get(lbl, 0) + weighted.cpu()
            for lbl in agg_protos_label:
                agg_protos_label[lbl] /= len(self.ldr_train)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), agg_protos_label

class AnchorDistillationLoss(nn.Module):
    def __init__(self, student_outputs, teacher_outputs, anchors, temperature=1.0, lambda_anchor=0.1, device='cuda'):
        """
        初始化锚点蒸馏损失。

        Args:
            student_outputs (torch.Tensor): 学生模型的原始输出 (logits)。
            teacher_outputs (torch.Tensor): 教师模型的原始输出 (logits)。
            anchors (torch.Tensor): 锚点张量。
            temperature (float): 知识蒸馏中的温度参数 τ。
            lambda_anchor (float): 锚点成本项的权重 λ。
            device (str): 计算设备 ('cuda' 或 'cpu')。
        """
        super(AnchorDistillationLoss, self).__init__()
        self.temperature = temperature
        self.lambda_anchor = lambda_anchor
        self.student_outputs = student_outputs
        self.teacher_outputs = teacher_outputs
        self.device = device
        
        # Sinkhorn-Knopp算法参数
        self.sinkhorn_iterations = 10
        self.sinkhorn_epsilon = 0.1
        
        # 确保 anchors 的维度是 [num_classes, feature_dim]
        # 并将其设置为不需要梯度的参数
        self.anchors = nn.Parameter(anchors, requires_grad=False)
        
        # 如果锚点的特征维度与类别数不匹配，则进行调整
        num_classes = student_outputs.size(1)
        if self.anchors.size(0) != num_classes:
            # 注意：这里的逻辑假设锚点的第一维是类别数，如果不是，需要调整
            # 原始代码中是 anchors.size(1)，这里根据上下文理解调整为 anchors.size(0)
            # 如果 anchors 的形状是 [num_anchors, feature_dim]，需要一个映射层
            # 这里我们假设 anchors 的形状是 [num_classes, feature_dim]
            pass # 根据实际的锚点维度设计调整逻辑
        
        # 注意：原adjust_anchors逻辑可能不适用于所有情况，此处保留但需谨慎使用
        # if anchors.size(1) != num_classes:
        #     self.anchors = self.adjust_anchors(anchors, num_classes)


    def adjust_anchors(self, anchors, num_classes):
        """
        通过一个线性层调整 anchors 的特征维度以匹配 num_classes。
        """
        # 这个函数在当前上下文中可能不是必需的，因为成本是基于锚点内部的距离计算的
        linear_transform = nn.Linear(anchors.size(1), num_classes).to(self.device)
        adjusted_anchors = linear_transform(anchors)
        return nn.Parameter(adjusted_anchors, requires_grad=False)

    def sinkhorn_knopp(self, cost_matrix):
        """
        Sinkhorn-Knopp算法实现，用于计算最优传输矩阵。
        输入是成本矩阵 C，算法内部会计算 K = exp(-C/ε)。

        Args:
            cost_matrix (torch.Tensor): 成本矩阵 [batch_size, num_classes, num_classes]
            
        Returns:
            transport_matrix (torch.Tensor): 最优传输矩阵 [batch_size, num_classes, num_classes]
        """
        batch_size, n, m = cost_matrix.shape
        
        # 初始化传输矩阵 K = exp(-C/ε)
        K = torch.exp(-cost_matrix / self.sinkhorn_epsilon)
        
        # 初始化行和列的缩放因子
        u = torch.ones(batch_size, n, 1, device=self.device) / n
        v = torch.ones(batch_size, m, 1, device=self.device) / m
        
        # Sinkhorn迭代
        for _ in range(self.sinkhorn_iterations):
            u = 1.0 / (torch.bmm(K, v) + 1e-8)
            v = 1.0 / (torch.bmm(K.transpose(1, 2), u) + 1e-8)
        
        # 计算最优传输矩阵 T = diag(u) * K * diag(v)
        transport_matrix = u * K * v.transpose(1, 2)
        
        return transport_matrix
    
    def compute_cost_matrix(self, student_probs, teacher_probs):
        """
        计算总成本矩阵。
        锚点项改用 cosine 距离（或归一化欧氏距离），其余不变。
        """
        batch_size, num_classes = student_probs.shape

        # ---------- 概率项：与原代码一致 ----------
        student_expanded = student_probs.unsqueeze(2)   # [B, C, 1]
        teacher_expanded = teacher_probs.unsqueeze(1)   # [B, 1, C]
        prob_cost = torch.pow(student_expanded - teacher_expanded, 2)   # [B, C, C]

        # ---------- 锚点项：改用 cosine 距离 ----------
        if self.lambda_anchor > 0 and self.anchors is not None:
            # self.anchors: [C, feat_dim]
            A = self.anchors                                   # [C, D]
            A_norm = F.normalize(A, p=2, dim=1)                # 单位向量
            # cosine 距离矩阵: 1 - cos(x,y)
            cosine_dist = 1.0 - torch.matmul(A_norm, A_norm.t())  # [C, C]
            # 扩展到 batch
            anchor_cost_term = self.lambda_anchor * cosine_dist.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            anchor_cost_term = 0.0

        total_cost_matrix = prob_cost + anchor_cost_term
        return total_cost_matrix

    def earth_movers_distance(self, cost_matrix, transport_matrix):
        """
        计算Earth Mover's Distance (Wasserstein Distance)。
        此函数被重构以直接接收成本矩阵，避免重复计算。

        Args:
            cost_matrix (torch.Tensor): 成本矩阵 [batch_size, num_classes, num_classes]
            transport_matrix (torch.Tensor): 最优传输矩阵 [batch_size, num_classes, num_classes]
            
        Returns:
            emd_loss (torch.Tensor): EMD损失的均值
        """
        # EMD = sum(T_ij * C_ij)，其中T是传输矩阵，C是成本矩阵
        emd = torch.sum(transport_matrix * cost_matrix, dim=[1, 2])
        
        return torch.mean(emd)

    def forward(self):
        """
        计算基于Sinkhorn-Knopp软对齐的蒸馏损失。
        流程已根据修改意见简化和明确。
        
        Returns:
            loss (torch.Tensor): 计算得到的最终蒸馏损失
        """
        # 1. 根据公式清晰定义学生和教师模型的概率分布 PS_τ 和 PT_τ
        # PS_τ(i, j) = exp(S_ij/τ) / Σ_k exp(S_ik/τ)
        student_probs = F.softmax(self.student_outputs / self.temperature, dim=1)
        teacher_probs = F.softmax(self.teacher_outputs / self.temperature, dim=1)
        
        # 2. 将锚点信息明确加入到蒸馏的成本矩阵中
        # C_total = ||PS_τ(i) - PT_τ(j)||_2^2 + λ * ||A'(i) - A'(j)||_2^2
        cost_matrix = self.compute_cost_matrix(student_probs, teacher_probs)
        
        # 3. 明确输入到 Sinkhorn-Knopp 算法的矩阵为 exp(-C_total)
        #    我们的 sinkhorn_knopp 函数接收 C_total，并在内部计算 K = exp(-C_total / ε)
        # T = Sinkhorn-Knopp(C_total)
        transport_matrix = self.sinkhorn_knopp(cost_matrix)
        
        # 4. 计算Earth Mover's Distance作为最终的蒸馏损失
        emd_loss = self.earth_movers_distance(cost_matrix, transport_matrix)
        
        return emd_loss
#fedavg
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)#读取数据集
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()

        # train and update
        
        # For ablation study
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logits = net(images)
                loss = self.loss_func(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
#fedprox

class LocalUpdateFedProx(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False, task = 0):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, body_lr, head_lr):
        net.train()
        g_net = copy.deepcopy(net)
        
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                
                # for fedprox
                fed_prox_reg = 0.0
                for l_param, g_param in zip(net.parameters(), g_net.parameters()):
                    fed_prox_reg += (0.1 / 2 * torch.norm((l_param - g_param)) ** 2)
                loss += fed_prox_reg
                
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) 
    
#icarl+fl
class LocalUpdateICARL(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.exemplar_sets = []  # 用于存储每个类的样本

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []

        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logits = net(images)
                loss = self.loss_func(logits, labels)

                # 添加蒸馏损失
                if self.exemplar_sets:
                    old_classes_logits = net(self.exemplar_sets)
                    distillation_loss = self.distillation_loss(logits, old_classes_logits)
                    loss += distillation_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # 更新样本集
        self.update_exemplar_sets(net)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def distillation_loss(self, new_logits, old_logits):
        # 计算蒸馏损失
        return nn.KLDivLoss()(F.log_softmax(new_logits, dim=1), F.softmax(old_logits, dim=1))

    def update_exemplar_sets(self, net):
        # 更新样本集
        new_exemplars = self.select_exemplars(net)
        self.exemplar_sets.extend(new_exemplars)

    def select_exemplars(self, net):
        # 基于herding选择样本
        exemplars = []
        for images, labels in self.ldr_train:
            images = images.to(self.args.device)
            features = net(images)
            exemplars.extend(self.herding(features, labels))
        return exemplars

    def herding(self, features, labels):
        # Herding算法选择样本
        exemplars = []
        for label in torch.unique(labels):
            class_features = features[labels == label]
            mean_feature = torch.mean(class_features, dim=0)
            distances = torch.norm(class_features - mean_feature, dim=1)
            exemplars.extend(class_features[torch.argsort(distances)[:self.args.m]])
        return exemplars

#fedntd
class NTD_Loss(nn.Module):
    def __init__(self, num_classes, tau=3, beta=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.tau = tau
        self.beta = beta
        self.num_classes = num_classes 
    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        loss = ce_loss + self.beta * ntd_loss

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        T = self.tau  
        dg_probs = F.softmax(dg_logits / T, dim=1)
        student_probs = F.softmax(logits / T, dim=1)
        kl_div_loss = self.KLDiv(F.log_softmax(logits / T, dim=1), dg_probs)
        kl_div_loss /= self.num_classes

        return kl_div_loss
def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits   


class LocalUpdateNTD(object):
    def __init__(self, args, dataset=None, task=0, idxs=None):
        self.args = args
        self.dataset = dataset
        self.idxs = idxs
        self.loss_func = NTD_Loss(num_classes=args.num_classes, tau=args.tau, beta=args.beta)
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)

    def train(self, net, lr=None):
        self.optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        net.train()
        
        epoch_loss = []
        num_updates = 0
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logits = net(images)
                dg_logits = net(images).detach()
                loss = self.loss_func(logits, labels, dg_logits)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_updates += 1
                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

#fedknow
class LocalUpdateFedKnow(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
      
        # FedKNOW components
        self.task_id = task
        self.signature_tasks = []  # Stores knowledge of previous tasks
        self.k = 5
        self.rho = 0.1
      
        # Gradient integration parameters
        self.epsilon = 1e-5  # Small constant for numerical stability
        solvers.options['show_progress'] = False  # Disable QP solver output
        self.task_features = []  # 存储每个任务的特征（平均梯度）
        self.task_weights = []   # 存储每个任务的权重知识

    '''def _extract_knowledge(self, net):
        """Extract top (1-rho)% important weights as task knowledge"""
        weights = []
        for param in net.parameters():
            if len(param.shape) > 1:  # Weight matrices only
                flattened = param.data.abs().flatten()
                threshold = torch.quantile(flattened, 1 - self.rho)
                mask = (param.data.abs() >= threshold).float()
                weights.append((param.data * mask, mask))
        return weights'''
    def _extract_knowledge(self, net, images):
        """提取知识并计算任务特征"""
        # 获取当前任务的平均梯度作为特征
        net.zero_grad()
        outputs = net(images)
        loss = self.loss_func(outputs, outputs.softmax(dim=1).argmax(dim=1))
        loss.backward()
        task_feature = torch.cat([p.grad.flatten().abs().mean().unsqueeze(0) for p in net.parameters()])

        # 原有权重提取逻辑
        weights = []
        for param in net.parameters():
            if len(param.shape) > 1:  # Weight matrices only
                flattened = param.data.abs().flatten()
                threshold = torch.quantile(flattened, 1 - self.rho)
                mask = (param.data.abs() >= threshold).float()
                weights.append((param.data * mask, mask))
        return weights, task_feature
    def _restore_gradients(self, net, images):
        """Restore gradients from signature tasks"""
        restored_grads = []
        for task_weights, masks in self.signature_tasks:
            # Set network to signature task weights
            idx = 0
            for param in net.parameters():
                if len(param.shape) > 1:
                    param.data = task_weights[idx].to(self.args.device)
                    idx += 1
          
            # Compute gradient w.r.t. current task data
            net.zero_grad()
            outputs = net(images)
            pseudo_labels = outputs.detach()
            loss = self.loss_func(outputs, pseudo_labels.softmax(dim=1).argmax(dim=1))
            loss.backward()
          
            # Apply original masks and store gradient
            grads = []
            idx = 0
            for param in net.parameters():
                if len(param.shape) > 1:
                    grad = param.grad * masks[idx].to(self.args.device)
                    grads.append(grad.flatten())
                    idx += 1
            restored_grads.append(torch.cat(grads))
        return restored_grads

    '''def _integrate_gradients(self, current_grad, restored_grads):
        """Quadratic programming for gradient integration"""
        if not restored_grads:
            return current_grad
      
        # Prepare constraints: G'*g >= 0
        G = torch.stack(restored_grads).cpu().numpy()
        G = -np.vstack([G, -np.eye(G.shape[1])])  # Non-negativity constraints
        h = np.zeros(G.shape[0])
      
        # Quadratic programming setup
        P = matrix(np.eye(len(current_grad)))
        q = matrix(-current_grad.cpu().numpy())
        G = matrix(G)
        h = matrix(h)
      
        # Solve QP problem
        try:
            sol = solvers.qp(P, q, G, h)
            v = np.array(sol['x']).flatten()
            integrated_grad = current_grad + torch.from_numpy(v).to(self.args.device)
            return integrated_grad
        except:
            return current_grad  # Fallback to original gradient'''
    def _integrate_gradients(self, current_grad, restored_grads):
        """修正后的梯度集成方法"""
        if not restored_grads:
            return current_grad

        # 1. 构建约束矩阵（论文公式3）
        G = torch.stack(restored_grads).cpu().numpy()
        h = np.zeros(len(restored_grads))  # Gg' >= 0

        # 2. 二次规划参数设置
        P = matrix(np.eye(len(current_grad)))
        q = matrix(-current_grad.cpu().numpy())
        G = matrix(-G)  # 转换为 <= 0 约束
        h = matrix(h)

        # 3. 求解QP问题
        try:
            sol = solvers.qp(P, q, G, h)
            v = np.array(sol['x']).flatten()
            return current_grad + torch.from_numpy(v).to(self.args.device)
        except:
            return current_grad

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
      
        # Extract knowledge from previous tasks
        if self.task_id > 0 and not self.pretrain:
            self.signature_tasks = self.signature_tasks[-self.k:]  # Keep only k recent
      
        epoch_loss = []
        local_eps = self.args.local_ep if local_eps is None else local_eps
      
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
              
                # 1. Compute current task gradient
                optimizer.zero_grad()
                outputs = net(images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                current_grad = torch.cat([p.grad.flatten() for p in net.parameters()])
              
                # 2. Restore gradients from signature tasks
                restored_grads = self._restore_gradients(net, images)
              
                # 3. Gradient integration
                integrated_grad = self._integrate_gradients(current_grad, restored_grads)
              
                # 4. Update parameters with integrated gradient
                idx = 0
                for param in net.parameters():
                    if param.grad is not None:
                        param.grad = integrated_grad[idx:idx+param.numel()].view(param.shape)
                        idx += param.numel()
                optimizer.step()
              
                batch_loss.append(loss.item())
          
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
      
        # Store current task knowledge
        if not self.pretrain:
            '''task_knowledge = self._extract_knowledge(net)
            self.signature_tasks.append(task_knowledge)
            self.task_id += 1'''
            task_knowledge, task_feature = self._extract_knowledge(net, images)
            self.task_weights.append(task_knowledge)
            self.task_features.append(task_feature)
            self.task_id += 1

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
#target
class LocalUpdateTARGET(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False, synthetic_data=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.synthetic_data = synthetic_data  # 添加合成数据

    def train(self, net, teacher_model, lr, idx=-1, local_eps=None):
        net.train()
        teacher_model.eval()  # 固定教师模型

        # 组合优化器
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                   momentum=self.args.momentum,
                                   weight_decay=self.args.wd)

        epoch_loss = []
        local_eps = self.args.local_ep if local_eps is None else local_eps

        for _ in range(local_eps):
            batch_loss = []
            
            if self.synthetic_data is not None:
                # 同时遍历真实数据和合成数据
                for (real_images, real_labels), (synth_images, _) in zip(self.ldr_train, self.synthetic_data):
                    # 当前任务数据
                    real_images, real_labels = real_images.to(self.args.device), real_labels.to(self.args.device)
                    
                    # 合成数据（旧任务）
                    synth_images = synth_images.to(self.args.device)
                    
                    # 前向传播
                    real_logits = net(real_images)
                    synth_logits = net(synth_images)
                    
                    # 教师模型输出
                    with torch.no_grad():
                        teacher_logits = teacher_model(synth_images)
                    
                    # 计算损失
                    ce_loss = self.loss_func(real_logits, real_labels)  # 当前任务损失
                    kl_loss = nn.KLDivLoss()(F.log_softmax(synth_logits, dim=1),
                                           F.softmax(teacher_logits, dim=1))  # 旧任务蒸馏损失
                    
                    total_loss = ce_loss + 0.1 * kl_loss  # 组合损失
                    
                    # 反向传播
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    batch_loss.append(total_loss.item())
            else:
                # 仅使用真实数据训练
                for real_images, real_labels in self.ldr_train:
                    real_images, real_labels = real_images.to(self.args.device), real_labels.to(self.args.device)
                    
                    # 前向传播
                    real_logits = net(real_images)
                    
                    # 计算损失
                    ce_loss = self.loss_func(real_logits, real_labels)  # 当前任务损失
                    
                    # 反向传播
                    optimizer.zero_grad()
                    ce_loss.backward()
                    optimizer.step()

                    batch_loss.append(ce_loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

#ReFed
class LocalUpdateReFed(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain

        # 初始化个性化信息模型 (PIM)
        self.pim = copy.deepcopy(args.global_model)  # 全局模型初始化 PIM
        self.cached_samples = []  # 缓存的重要样本

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()

        # 定义优化器
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []

        if local_eps is None:
            if self.pretrain:
                local_epochs = self.args.local_ep_pretrain
            else:
                local_epochs = self.args.local_ep
        else:
            local_epochs = local_eps

        for iter in range(local_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                # 前向传播
                logits = net(images)
                loss = self.loss_func(logits, labels)

                # 反向传播更新本地模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # 在本地训练结束后，更新 PIM 并计算样本重要性
        self.update_pim_and_cache_samples(net)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_pim_and_cache_samples(self, net):
        # 更新个性化信息模型 (PIM)
        self.pim.train()
        pim_optimizer = torch.optim.SGD(self.pim.parameters(), lr=self.args.lr_pim,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.wd)

        importance_scores = {}

        # 使用本地数据更新 PIM，并记录样本梯度范数
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            # 前向传播
            logits = self.pim(images)
            loss = self.loss_func(logits, labels)

            # 反向传播更新 PIM
            pim_optimizer.zero_grad()
            loss.backward()
            pim_optimizer.step()

            # 计算样本梯度范数作为重要性分数
            for i in range(len(images)):
                sample_grad_norm = torch.norm(self.pim.fc.weight.grad[i]).item()
                if (images[i], labels[i]) not in importance_scores:
                    importance_scores[(images[i], labels[i])] = 0
                importance_scores[(images[i], labels[i])] += sample_grad_norm

        # 根据重要性分数缓存样本
        sorted_samples = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        max_cache_size = self.args.max_cache_size
        self.cached_samples = [sample for sample, _ in sorted_samples[:max_cache_size]]

        # 将缓存的样本与新任务数据合并，用于下一次训练
        self.ldr_train = combine_cached_and_new_data(self.cached_samples, self.ldr_train)

def combine_cached_and_new_data(cached_samples, new_data_loader):
    """将缓存样本与新任务数据合并"""
    cached_dataset = CachedDataset(cached_samples)
    combined_dataset = ConcatDataset([cached_dataset, new_data_loader.dataset])
    return DataLoader(combined_dataset, batch_size=new_data_loader.batch_size, shuffle=True)

class CachedDataset(torch.utils.data.Dataset):
    """缓存样本的自定义数据集"""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
#EWC
class LocalUpdateEWC(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # 加载数据集
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        
        # EWC 相关初始化
        self.fisher = None  # 存储 Fisher 信息矩阵
        self.old_params = None  # 存储之前的模型参数

    def compute_fisher(self, net):
        """
        计算当前任务的 Fisher 信息矩阵。
        """
        fisher = {}
        for name, param in net.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        net.eval()
        total_samples = 0
        for images, labels in self.ldr_train:
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            logits = net(images)
            loss = self.loss_func(logits, labels)
            
            net.zero_grad()
            loss.backward()
            
            # 更新 Fisher 信息矩阵
            for name, param in net.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * len(labels)
            total_samples += len(labels)
        
        # 归一化 Fisher 信息矩阵
        for name in fisher:
            fisher[name] /= total_samples
        
        self.fisher = fisher

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()

        # 初始化优化器
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        # 如果未指定本地训练轮数，则根据是否预训练设置默认值
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                # 前向传播
                logits = net(images)
                ce_loss = self.loss_func(logits, labels)
                
                # EWC 正则化项
                ewc_loss = 0
                if self.fisher is not None and self.old_params is not None:
                    for name, param in net.named_parameters():
                        if name in self.fisher:
                            ewc_loss += torch.sum(self.fisher[name] * (param - self.old_params[name]).pow(2))
                
                # 总损失 = CE 损失 + EWC 损失
                total_loss = ce_loss +  ewc_loss
                
                # 反向传播和优化
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_loss.append(total_loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # 返回更新后的模型参数和平均损失
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_old_params(self, net):
        """
        保存当前模型参数作为旧参数。
        """
        self.old_params = {name: param.clone().detach() for name, param in net.named_parameters()}

#CGoFed
class LocalUpdateCGoFed(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # Load training data for the current task
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.task = task  # Store the current task index

    def train(self, net, lr, idx=-1, local_eps=None, historical_basis_vectors=None):
        net.train()

        # Initialize optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        task_representation_matrix = None  # To store the representation matrix for the current task

        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                # Forward pass
                logits = net(images)
                loss = self.loss_func(logits, labels)

                # Backward pass with relax-constrained gradient update
                optimizer.zero_grad()
                loss.backward()

                if self.task > 0:  # Only apply constraints for tasks after the first one
                    self.relax_constrained_gradient_update(net, self.task, historical_basis_vectors)

                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # Compute the representation matrix for the current task
        task_representation_matrix = self.compute_representation_matrix(net)

        # Return the updated model state, average loss, and task representation matrix
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), task_representation_matrix

    def relax_constrained_gradient_update(self, net, task, historical_basis_vectors):
        """
        Apply relax-constrained gradient update to balance stability and plasticity.
        This function restricts the gradient update direction based on historical tasks' gradient spaces.
        """
        # Retrieve the memory of basis vectors for historical tasks
        Mt = historical_basis_vectors.get(task, None)
        if Mt is None:
            # If no historical basis vectors are available, skip the constrained update
            return

        # Compute the relaxation coefficient μt
        μt = self.compute_relaxation_coefficient(task)

        # Project the gradients onto the orthogonal space of historical tasks with relaxation
        for param in net.parameters():
            if param.grad is not None:
                grad = param.grad.data
                projected_grad = μt * torch.matmul(torch.matmul(grad, Mt), Mt.T)
                param.grad.data -= projected_grad

    def compute_representation_matrix(self, net):
        """
        Compute the representation matrix for the current task using a subset of samples.
        This matrix represents the feature space of the current task.
        """
        # Randomly sample a subset of data from the current task
        sample_loader = self.sample_task_data(self.ldr_train)
        representations = []

        with torch.no_grad():
            for images, _ in sample_loader:
                images = images.to(self.args.device)
                features = net.extract_features(images)
                representations.append(features.cpu().numpy())

        # Concatenate all representations into a single matrix
        representation_matrix = np.concatenate(representations, axis=0)
        return representation_matrix

    def retrieve_historical_basis_vectors(self, task):
        """
        Retrieve the basis vectors of historical tasks from memory.
        These basis vectors represent the gradient spaces of previous tasks.
        """
        # Placeholder for retrieving historical basis vectors (e.g., from server or local memory)
        # Assume `self.memory` stores the basis vectors for all tasks
        return self.memory[task]

    def compute_relaxation_coefficient(self, task):
        """
        Compute the relaxation coefficient μt based on the forgetting threshold τ and decay rate α.
        """
        # Example computation of μt (adjust as needed)
        α = self.args.alpha  # Decay rate
        τ = self.args.tau  # Forgetting threshold
        AF = self.compute_average_forgetting()  # Compute average forgetting metric
        μt = α ** task if AF < τ else α ** (task - self.args.t_tau)
        return μt

    def compute_average_forgetting(self):
        """
        Compute the average forgetting metric across historical tasks.
        """
        # Placeholder for computing average forgetting (AF)
        # Assume this function uses historical accuracy metrics
        return 0.0  # Replace with actual implementation

    def sample_task_data(self, loader):
        """
        Sample a subset of data from the current task for representation matrix computation.
        """
        # Placeholder for sampling logic
        return DataLoader(loader.dataset, batch_size=self.args.sample_bs, shuffle=True)
    
#TagFed
class LocalUpdateTagFed(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.task = task  # 当前任务标识
        self.masks = {}  # 存储每个任务的掩码

    def apply_mask(self, net, mask):
        """应用掩码到模型参数"""
        for name, param in net.named_parameters():
            if name in mask:
                param.data = param.data * mask[name].to(param.device)

    def generate_mask(self, net, task_id):
        """为当前任务生成掩码"""
        mask = {}
        for name, param in net.named_parameters():
            if f"task_{task_id}" not in self.masks:
                # 初始化掩码，允许新任务使用未被占用的神经元
                mask[name] = torch.ones_like(param.data)
            else:
                # 如果是重复任务，复用之前的掩码
                mask[name] = self.masks[f"task_{task_id}"][name]
        return mask

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()

        # 定义优化器
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        feature_maps = []  # 存储特征图
        all_logits = []    # 存储logits

        # 动态设置本地训练轮数
        if local_eps is None:
            local_eps = self.args.local_ep_pretrain if self.pretrain else self.args.local_ep

        # 为当前任务生成掩码
        mask = self.generate_mask(net, self.task)
        self.apply_mask(net, mask)

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                # 前向传播
                logits = net(images)
                features = net.extract_features(images)
                loss = self.loss_func(logits, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 应用掩码，确保冻结的权重不被更新
                self.apply_mask(net, mask)

                batch_loss.append(loss.item())

                # 存储特征图和logits
                feature_maps.append(features.detach().cpu())  # 提取特征图
                all_logits.append(logits.detach().cpu())      # 提取logits

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # 更新当前任务的掩码
        self.masks[f"task_{self.task}"] = mask

        # 返回特征图、logits 和平均损失
        return feature_maps, all_logits, sum(epoch_loss) / len(epoch_loss)
    
class LocalUpdateMFCL(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.task = task  # 当前任务标识
        from utils.train_utils import get_data, get_model
        self.device = args.device
        
    # 客户端训练
    def train(self, server_model, generator, w_ft=1,
                     w_kd=1, local_eps = None):
        # client_model = self.local_model
        client_model = server_model
        device = self.args.device
        task_classes = self.args.num_classes
        client_model.train()
        optimizer = optim.SGD(client_model.parameters(), lr=0.001)
        criterion_ce = nn.CrossEntropyLoss()
        
         # 冻结特征提取部分，只训练分类头
        # for param in client_model.parameters():
        #     param.requires_grad = False
        # client_model.linear.requires_grad = True
        
        # 动态设置本地训练轮数
        if local_eps is None:
            local_eps = self.args.local_ep_pretrain if self.pretrain else self.args.local_ep
            
        # 获取合成数据
        synthetic_images = None
        if self.task > 0:
            noise = torch.randn(len(self.ldr_train), 100).to(self.device)
            synthetic_images = generator(noise)
            
        
        epoch_loss = []
        # 训练当前任务
        for epoch in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                # 前向传播
                logits = client_model(images)
                loss = self.loss_func(logits, labels)
                
                # 如果有合成数据，添加合成数据的损失
                if self.task > 0 and synthetic_images is not None:
                    synthetic_outputs = client_model(synthetic_images)
                    synthetic_labels = torch.randint(0, self.args.num_classes, (synthetic_images.size(0),)).to(self.device)
                    synthetic_loss = criterion_ce(synthetic_outputs, synthetic_labels)
                    total_loss = loss + synthetic_loss
                else:
                    total_loss = loss

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward(retain_graph = True)
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            
        client_data = self.ldr_train
        # 使用合成数据和真实数据训练以克服遗忘
#         noise = torch.randn(len(client_data), 100).to(device)
#         synthetic_images = generator(noise)
#         client_data_tensors = []
#         client_labels_tensors = []
#         for batch in client_data:
#             data, labels = batch
#             client_data_tensors.append(data)
#             client_labels_tensors.append(labels)

#         # 将所有批次的数据和标签拼接成一个大张量
#         client_data_tensor = torch.cat(client_data_tensors, dim=0).to(device)
#         client_labels_tensor = torch.cat(client_labels_tensors, dim=0).to(device)
#         all_data = torch.cat([client_data_tensor, synthetic_images], dim=0)
#         all_labels = torch.cat([client_labels_tensor, client_labels_tensor], dim=0)

#         for param in client_model.parameters():
#             param.requires_grad = False
#         client_model.linear.requires_grad = True

#         optimizer = optim.SGD(client_model.linear.parameters(), lr=0.001)
#         optimizer.zero_grad()
#         ft_outputs = client_model(all_data)
#         ft_loss = criterion_ce(ft_outputs, all_labels)
#         ft_loss.backward()
#         optimizer.step()

        # 重要性加权特征蒸馏
#         with torch.no_grad():
#             client_features = client_model.features(client_data)
#             server_features = server_model.features(client_data)
#         kd_loss = torch.mean((client_features - server_features) ** 2)

#         optimizer.zero_grad()
#         kd_loss.backward()
#         optimizer.step()
        
        return client_model.state_dict(), sum(epoch_loss) / len(epoch_loss)
