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

    import torch
import torch.nn as nn
import torch.nn.functional as F

class AnchorContrastiveLoss(nn.Module):
    def __init__(self, anchors, temperature=0.1, device='cuda'):
        """
        初始化函数
        :param anchors: 预定义的锚点张量 [num_classes, feature_dim]
        :param temperature: 温度系数，用于缩放logits
        :param device: 计算设备 ('cuda' or 'cpu')
        """
        super(AnchorContrastiveLoss, self).__init__()
        # 将anchors注册为非训练参数
        self.anchors = nn.Parameter(anchors, requires_grad=False)
        self.temperature = temperature
        self.device = device

        # 用于在训练过程中记录一些度量指标
        self.metrics = {
            'hard_neg_sim': 0.0,
            'dynamic_k': 0
        }

    def forward(self, features, labels):
        """
        前向传播函数
        :param features: 输入的特征张量 [Batch_size, feature_dim]
        :param labels: 对应的标签 [Batch_size]
        :return: 计算出的损失值
        """
        # 如果输入特征的维度与锚点维度不匹配，则动态调整锚点维度
        if features.size(1) != self.anchors.size(1):
            self._adjust_anchor_dim(features.size(1))

        # 计算特征与所有锚点的余弦相似度
        anchor_sim = F.cosine_similarity(
            features.unsqueeze(1),  # 形状变为 [B, 1, D]
            self.anchors.unsqueeze(0),  # 形状变为 [1, K, D]
            dim=2
        )  # 输出形状为 [B, K]

        # 调用动态困难负样本挖掘的损失函数
        return self.dynamic_hard_neg_loss(anchor_sim, labels)

    def dynamic_hard_neg_loss(self, anchor_similarity, labels, 
                             k=5, alpha=0.8, adaptive_k=True):
        """
        带有动态困难负样本挖掘的对比损失函数
        :param anchor_similarity: 特征与锚点的余弦相似度 [B, K]
        :param labels: 真实标签 [B]
        :param k: 基础的负样本挖掘数量
        :param alpha: 混合权重，用于平衡困难负样本和所有样本
        :param adaptive_k: 是否启用自适应k值
        :return: 损失值
        """
        logits = anchor_similarity / self.temperature
        batch_size, num_anchors = logits.shape

        # 创建正样本的掩码
        pos_mask = F.one_hot(labels, num_classes=num_anchors).bool()

        # 将正样本位置的logits填充为负无穷，以便后续只关注负样本
        neg_logits = logits.masked_fill(pos_mask, -float('inf'))

        if adaptive_k:
            with torch.no_grad():
                # --- START: 修改部分 ---
                # 步骤 1: 将余弦相似度从[-1, 1]归一化到[0, 1]
                # (S_b,c + 1) / 2
                normalized_sim = (anchor_similarity.detach() + 1) / 2
                
                # 步骤 2: 计算归一化后的平均相似度 s_avg
                # savg = (1 / (B * C)) * sum( (S_b,c + 1) / 2 )
                s_avg = torch.mean(normalized_sim)
                
                # 步骤 3: 根据 s_avg 计算动态的 k 值
                # k_dynamic = min(k_base + floor(10 * s_avg), C - 1)
                # 使用 int() 来实现 floor 功能
                dynamic_k = min(k + int(s_avg * 10), num_anchors - 1)
                # --- END: 修改部分 ---
        else:
            dynamic_k = k
        
        # 从负样本中选取topk个最困难的（即相似度最高的）
        hard_neg, _ = torch.topk(neg_logits, k=dynamic_k, dim=1)
        
        # 获取正样本的分数
        pos_scores = logits.gather(1, labels.view(-1, 1))

        # 计算包含所有样本的log_sum_exp
        log_sum_all = torch.logsumexp(logits, dim=1, keepdim=True)

        # 将正样本分数和困难负样本分数合并，用于计算困难样本的log_sum_exp
        combined = torch.cat([pos_scores, hard_neg], dim=1)
        log_sum_hard = torch.logsumexp(combined, dim=1, keepdim=True)

        # 动态调整alpha值，使得模型在不同阶段更关注困难样本
        if self.training and adaptive_k:
            with torch.no_grad():
                hard_sim = torch.mean(hard_neg)
                adapt_alpha = torch.sigmoid(hard_sim * 5)
                alpha = alpha * 0.9 + adapt_alpha * 0.1
        
        # 计算最终的损失函数
        # 结合了 InfoNCE loss 和 hard negative mining 的思想
        loss = - (pos_scores - (alpha * log_sum_hard + (1 - alpha) * log_sum_all)).mean()
        
        # 如果在训练模式，则记录相关指标
        if self.training:
            self.metrics['hard_neg_sim'] = hard_neg.mean().item()
            self.metrics['dynamic_k'] = dynamic_k
            
        return loss

    def _adjust_anchor_dim(self, target_dim):
        """
        当输入特征维度变化时，动态调整锚点的维度。
        这在模型结构变化或迁移学习场景中可能有用。
        """
        current_dim = self.anchors.size(1)
        #print(f"Adjusting anchor dimensions from {current_dim} to {target_dim}.")
        # 使用一个简单的线性层来做维度映射
        linear = nn.Linear(current_dim, target_dim).to(self.device)
        # 更新锚点参数（不参与梯度计算）
        self.anchors = nn.Parameter(linear(self.anchors.detach()), requires_grad=False)
        return self.anchors

'''class AnchorContrastiveLoss(nn.Module):
    def __init__(self, anchors, temperature=0.1, device='cuda'):

        super(AnchorContrastiveLoss, self).__init__()
        self.anchors = nn.Parameter(anchors, requires_grad=False)
        self.temperature = temperature
        self.device = device

        self.metrics = {
            'hard_neg_sim': 0.0,
            'dynamic_k': 0
        }

    def forward(self, features, labels):

        if features.size(1) != self.anchors.size(1):
            self._adjust_anchor_dim(features.size(1))

        anchor_sim = F.cosine_similarity(
            features.unsqueeze(1),  # [B, 1, D]
            self.anchors.unsqueeze(0),  # [1, K, D]
            dim=2
        )  # [B, K]

        return self.dynamic_hard_neg_loss(anchor_sim, labels)

    def dynamic_hard_neg_loss(self, anchor_similarity, labels, 
                             k=5, alpha=0.8, adaptive_k=True):
        logits = anchor_similarity / self.temperature
        batch_size, num_anchors = logits.shape

        pos_mask = F.one_hot(labels, num_classes=num_anchors).bool()

        neg_logits = logits.masked_fill(pos_mask, -float('inf'))

        if adaptive_k:
            with torch.no_grad():
                avg_sim = torch.mean(anchor_similarity.detach())
                dynamic_k = min(k + int(avg_sim * 10), num_anchors-1)
        else:
            dynamic_k = k
        
        hard_neg, _ = torch.topk(neg_logits, k=dynamic_k, dim=1)
        
        pos_scores = logits.gather(1, labels.view(-1, 1))

        log_sum_all = torch.logsumexp(logits, dim=1, keepdim=True)

        combined = torch.cat([pos_scores, hard_neg], dim=1)
        log_sum_hard = torch.logsumexp(combined, dim=1, keepdim=True)

        if self.training and adaptive_k:
            with torch.no_grad():
                hard_sim = torch.mean(hard_neg)
                adapt_alpha = torch.sigmoid(hard_sim * 5)
                alpha = alpha * 0.9 + adapt_alpha * 0.1
        
        loss = - (pos_scores - (alpha*log_sum_hard + (1-alpha)*log_sum_all)).mean()
        
        if self.training:
            self.metrics['hard_neg_sim'] = hard_neg.mean().item()
            self.metrics['dynamic_k'] = dynamic_k
            
        return loss

    def _adjust_anchor_dim(self, target_dim):
        """动态调整锚点维度"""
        linear = nn.Linear(self.anchors.size(1), target_dim).to(self.device)
        self.anchors = nn.Parameter(linear(self.anchors), requires_grad=False)
        return self.anchors'''


'''class LocalUpdateFedACD(object):
    def __init__(self, args, anchor=None, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.anchor = anchor.to(self.args.device)
        self.num_classes = args.num_classes

        # 初始化元学习率为可训练参数，并使用Adam优化器
        self.base_meta = nn.Parameter(torch.tensor(args.meta_lr))  # 初始元学习率来自args
        self.meta_optimizer = torch.optim.Adam([self.base_meta], lr=0.001)  # 配置Adam优化器
        self.min_lr = getattr(args, 'min_meta_lr', 0.8)    # 最小元学习率
        self.max_lr = getattr(args, 'max_meta_lr', 1.0)    # 最大元学习率

    def train(self, net, teacher_net, lr, idx=-1, local_eps=None):
        net.train()
        teacher_net.eval()
        num_classes, feat_dim = self.anchor.size()
      
        # 调整教师模型输出维度
        if self.args.dataset == 'fmnist':
            input_tensor = torch.randn(1, 1, 28, 28).to(self.args.device)
        elif self.args.dataset == 'tinyimagenet':
            input_tensor = torch.randn(1, 3, 64, 64).to(self.args.device)
        else:
            input_tensor = torch.randn(1, 3, 32, 32).to(self.args.device)
        with torch.no_grad():
            teacher_output_size = teacher_net(input_tensor).size(1)
        self.teacher_output_adjuster = nn.Linear(teacher_output_size, 100).to(self.args.device)
        initial_body_state = {name: param.clone().detach() for name, param in net.named_parameters() if 'linear' not in name}
      
        # 仅训练body部分
        body_params = [p for name, p in net.module.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.module.named_parameters() if 'linear' in name]
        for para in head_params:
            para.requires_grad = False
        optimizer = torch.optim.SGD(body_params, lr=lr, momentum=self.args.momentum, weight_decay=self.args.wd)
        epoch_loss = []
        local_eps = self.args.local_ep_pretrain if self.pretrain else self.args.local_ep if local_eps is None else local_eps
        
        # GradNorm相关参数
        loss_weights = torch.ones(3).to(self.args.device)  # 初始化CE、对比损失、蒸馏损失的权重为1
        loss_weights.requires_grad = True
        optimizer_weights = torch.optim.Adam([loss_weights], lr=0.01)  # 使用Adam优化器更新权重
      
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                features = net.module.extract_features(images)
                logits = net.module.only_liner(features)
                loss_ce = self.loss_func(logits, labels)
              
                contrast_loss = AnchorContrastiveLoss(
                    anchors=self.anchor,
                    temperature=0.5,
                    device=self.args.device
                )(features=logits, labels=labels)
              
                with torch.no_grad():
                    teacher_outputs = teacher_net(images)
                    adjusted_teacher_outputs = self.teacher_output_adjuster(teacher_outputs)

                distillation_loss = AnchorDistillationLoss(logits, adjusted_teacher_outputs, self.anchor, temperature=1.0)()
              
                loss = loss_weights[0] * loss_ce + loss_weights[1] * contrast_loss + loss_weights[2] * distillation_loss
                optimizer.zero_grad()
                loss.backward(retain_graph=True)  # 保留计算图以便后续计算各损失项的梯度
                grads = [p.grad.clone().detach() for p in body_params]
                
                # 计算每个损失项的梯度范数
                grad_norms = []
                for i, loss_i in enumerate([loss_ce, contrast_loss, distillation_loss]):
                    optimizer.zero_grad()
                    loss_i.backward(retain_graph=True)
                    grad_i = [p.grad.clone().detach() for p in body_params]
                    grad_norm_i = torch.stack([g.norm() for g in grad_i]).mean()
                    grad_norms.append(grad_norm_i)
                
                # GradNorm: 更新损失权重
                grad_norms = torch.stack(grad_norms)
                target_norm = grad_norms.mean()  # 目标梯度范数为平均值
                loss_ratios = grad_norms / target_norm  # 计算各损失项的梯度比例
                loss_weights_grad = loss_ratios - loss_ratios.mean()  # 计算权重的梯度
                optimizer_weights.zero_grad()
                loss_weights.backward(gradient=loss_weights_grad)  # 更新权重
                optimizer_weights.step()
                
                # 归一化损失权重，使其和为3（保持与初始权重尺度一致）
                loss_weights.data = loss_weights.data / loss_weights.data.sum() * 3.0
                
                # 重新计算总损失并优化
                loss = loss_weights[0] * loss_ce + loss_weights[1] * contrast_loss + loss_weights[2] * distillation_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
      
        # 应用元学习率更新参数
        with torch.no_grad():
            for name, param in net.module.named_parameters():
                if 'linear' not in name:
                    initial_p = initial_body_state[name]
                    param.data = initial_p + (param.data - initial_p) * self.base_meta.item()

        # 计算元损失并更新元学习率
        meta_loss = 0.0
        net.eval()
        with torch.enable_grad():
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                #with torch.no_grad():
                    
                    #teacher_outputs = teacher_net(images)
                    #adjusted_teacher_outputs = self.teacher_output_adjuster(teacher_outputs)
                features = net.module.extract_features(images)
                #logits = net.only_liner(features)
                # 使用对比损失作为元损失，可替换为其他目标
                meta_loss += AnchorContrastiveLoss(
                    anchors=self.anchor,
                    temperature=0.5,
                    device=self.args.device
                )(features=net.module.only_liner(features), labels=labels)
                #meta_loss += AnchorDistillationLoss(logits, adjusted_teacher_outputs, self.anchor, temperature=1.0)()
        meta_loss /= len(self.ldr_train)
      
        # Adam优化器更新元学习率
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        self.base_meta.data.clamp_(min=self.min_lr, max=self.max_lr)  # 限制学习率范围

        # 聚合原型
        agg_protos_label = {}
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                features = net.module.extract_features(images)
                uniq_l = labels.unique()
                for label in uniq_l:
                    lbl = label.item()
                    mask = labels == lbl
                    label_features = features[mask]
                    weights = torch.softmax(label_features.norm(dim=1), dim=0)
                    weighted_features = (label_features.T @ weights).T
                    if lbl in agg_protos_label:
                        agg_protos_label[lbl] += weighted_features.cpu()
                    else:
                        agg_protos_label[lbl] = weighted_features.cpu()
            for lbl in agg_protos_label:
                agg_protos_label[lbl] /= len(agg_protos_label[lbl])

        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), agg_protos_label'''

class LocalUpdateFedACD(object):
    def __init__(self, args, anchor=None, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.anchor = anchor.to(self.args.device) if anchor is not None else torch.randn(args.num_classes, 100).to(args.device)
        self.num_classes = args.num_classes

        # 初始化元学习率为可训练参数，并使用Adam优化器
        self.base_meta = nn.Parameter(torch.tensor(args.meta_lr))  # 初始元学习率来自args
        self.meta_optimizer = torch.optim.Adam([self.base_meta], lr=0.001)  # 配置Adam优化器
        self.min_lr = getattr(args, 'min_meta_lr', 0.8)    # 最小元学习率
        self.max_lr = getattr(args, 'max_meta_lr', 1.0)    # 最大元学习率

    def train(self, net, teacher_net, lr, idx=-1, local_eps=None):
        net.train()
        teacher_net.eval()
        # num_classes, feat_dim = self.anchor.size() # 这行可以保留，也可以删除，因为后面没用到 feat_dim

        # --- 删除开始 ---
        # 这部分代码是问题的根源，它创建了一个将教师模型输出固定映射到100维的层
        # if self.args.dataset == 'fmnist':
        #     input_tensor = torch.randn(1, 1, 28, 28).to(self.args.device)
        # elif self.args.dataset == 'tinyimagenet':
        #     input_tensor = torch.randn(1, 3, 64, 64).to(self.args.device)
        # else:
        #     input_tensor = torch.randn(1, 3, 32, 32).to(self.args.device)
        # with torch.no_grad():
        #     teacher_output_size = teacher_net(input_tensor).size(1)
        # self.teacher_output_adjuster = nn.Linear(teacher_output_size, 100).to(self.args.device)
        # --- 删除结束 ---

        initial_body_state = {name: param.clone().detach() for name, param in net.module.named_parameters() if 'linear' not in name}

        # 仅训练body部分
        body_params = [p for name, p in net.module.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.module.named_parameters() if 'linear' in name]
        for para in head_params:
            para.requires_grad = False
        optimizer = torch.optim.SGD(body_params, lr=lr, momentum=self.args.momentum, weight_decay=self.args.wd)
        epoch_loss = []
        local_eps = self.args.local_ep_pretrain if self.pretrain else self.args.local_ep if local_eps is None else local_eps

        # GradNorm相关参数
        loss_weights = torch.ones(3, requires_grad=True, device=self.args.device)
        optimizer_weights = torch.optim.Adam([loss_weights], lr=0.01)

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                features = net.module.extract_features(images)
                logits = net.module.only_liner(features)
                loss_ce = self.loss_func(logits, labels)

                contrast_loss = AnchorContrastiveLoss(
                    anchors=self.anchor,
                    temperature=0.5,
                    device=self.args.device
                )(features=logits, labels=labels)

                with torch.no_grad():
                    teacher_outputs = teacher_net(images)
                    # --- 删除开始 ---
                    # 不再使用 adjuster 对教师模型的输出进行调整
                    # adjusted_teacher_outputs = self.teacher_output_adjuster(teacher_outputs)
                    # --- 删除结束 ---

                # --- 修改后 ---
                # 直接将原始的 teacher_outputs 传入，确保其维度与 logits 一致 (都为 num_classes)
                distillation_loss = AnchorDistillationLoss(logits, teacher_outputs, self.anchor, temperature=1.0)()
                # --- 修改结束 ---

                # 第一次前向传播，计算带权重的总损失，用于后续计算梯度
                loss = loss_weights[0] * loss_ce + loss_weights[1] * contrast_loss + loss_weights[2] * distillation_loss
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                # 计算每个损失项的梯度范数
                grad_norms = []
                for i, loss_i in enumerate([loss_ce, contrast_loss, distillation_loss]):
                    optimizer.zero_grad()
                    loss_i.backward(retain_graph=True)
                    grad_i = [p.grad.clone().detach() for p in body_params if p.grad is not None]
                    if not grad_i:
                        grad_norm_i = torch.tensor(0.0, device=self.args.device)
                    else:
                        grad_norm_i = torch.stack([g.norm() for g in grad_i]).mean()
                    grad_norms.append(grad_norm_i)

                # GradNorm: 更新损失权重
                grad_norms = torch.stack(grad_norms)
                target_norm = grad_norms.mean()

                epsilon = 1e-6
                loss_ratios = (grad_norms + epsilon) / (target_norm + epsilon)

                loss_weights_grad = loss_ratios - loss_ratios.mean()

                optimizer_weights.zero_grad()
                loss_weights.backward(gradient=loss_weights_grad)
                optimizer_weights.step()

                with torch.no_grad():
                    loss_weights.data = 3 * torch.softmax(loss_weights.data, dim=0)

                # 使用更新后的权重重新计算总损失并进行模型参数优化
                loss = loss_weights[0] * loss_ce + loss_weights[1] * contrast_loss + loss_weights[2] * distillation_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # 应用元学习率更新参数
        with torch.no_grad():
            for name, param in net.module.named_parameters():
                if 'linear' not in name:
                    initial_p = initial_body_state[name]
                    param.data = initial_p + (param.data - initial_p) * self.base_meta.item()

        # 计算元损失并更新元学习率
        meta_loss = 0.0
        net.eval()
        with torch.enable_grad():
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                features = net.module.extract_features(images)
                meta_loss += AnchorContrastiveLoss(
                    anchors=self.anchor,
                    temperature=0.5,
                    device=self.args.device
                )(features=net.module.only_liner(features), labels=labels)
        meta_loss /= len(self.ldr_train)

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        self.base_meta.data.clamp_(min=self.min_lr, max=self.max_lr)

        # 聚合原型
        agg_protos_label = {}
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                features = net.module.extract_features(images)
                uniq_l = labels.unique()
                for label in uniq_l:
                    lbl = label.item()
                    mask = labels == lbl
                    label_features = features[mask]
                    if len(label_features) == 0: continue
                    weights = torch.softmax(label_features.norm(dim=1), dim=0)
                    weighted_features = (label_features.T @ weights).T
                    if lbl in agg_protos_label:
                        agg_protos_label[lbl] += weighted_features.cpu()
                    else:
                        agg_protos_label[lbl] = weighted_features.cpu()

            for lbl in agg_protos_label:
                agg_protos_label[lbl] /= len(self.ldr_train)

        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), agg_protos_label
    
'''class AnchorDistillationLoss(nn.Module):
    def __init__(self, student_outputs, teacher_outputs, anchors, temperature=1.0, device='cuda'):
        super(AnchorDistillationLoss, self).__init__()
        self.anchors = nn.Parameter(anchors, requires_grad=False)  # 锚点不应梯度下降
        self.temperature = temperature
        self.student_outputs = student_outputs
        self.teacher_outputs = teacher_outputs
        self.device = device

        # 确保 anchors 的维度是 [num_classes, num_classes]
        num_classes = student_outputs.size(1)
        if anchors.size(1) != num_classes:
            # 如果 anchors 的第二维度与 num_classes 不匹配，则进行调整
            # 使用线性变换来调整 anchors 的维度
            self.anchors = self.adjust_anchors(anchors, num_classes)

    def adjust_anchors(self, anchors, num_classes):
        """
        调整 anchors 的维度以匹配 num_classes。
        """
        # 假设我们使用一个简单的线性变换来调整 anchors 的大小
        linear_transform = nn.Linear(anchors.size(1), num_classes).to(self.device)
        adjusted_anchors = linear_transform(anchors)
        return nn.Parameter(adjusted_anchors, requires_grad=False)

    def forward(self):
        """
        计算蒸馏损失。
        返回:
        - loss: 计算得到的蒸馏损失
        """
        # 将锚点的形状调整为 [1, C, C]，以便可以批次运算
        anchors_expanded = self.anchors.unsqueeze(0)
        
        # 计算学生模型的softmax概率
        student_probs = F.softmax(self.student_outputs / self.temperature, dim=1)
        
        # 计算学生模型在锚点上的特征表示
        student_features = torch.matmul(student_probs, anchors_expanded.squeeze(0))
        
        # 确保 student_features 的形状与 teacher_outputs 匹配
        if student_features.size(1) != self.teacher_outputs.size(1):
            # 使用线性变换来调整 student_features 的形状
            linear_transform = nn.Linear(student_features.size(1), self.teacher_outputs.size(1)).to(self.device)
            student_features = linear_transform(student_features)

        # 使用teacher_outputs作为真实分布，student_features作为预测分布
        # 计算蒸馏损失
        loss = -torch.sum(self.teacher_outputs * F.log_softmax(student_features / self.temperature, dim=1), dim=1)
        
        # 取所有样本损失的平均值作为批次的蒸馏损失
        loss = torch.mean(loss)
        
        return loss'''

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
        根据修改方案，清晰地计算总成本矩阵。
        C_total(i, j) = ||PS_τ(i) - PT_τ(j)||_2^2 + λ * ||A'(i) - A'(j)||_2^2

        Args:
            student_probs (torch.Tensor): 学生模型概率分布 [batch_size, num_classes]
            teacher_probs (torch.Tensor): 教师模型概率分布 [batch_size, num_classes]
            
        Returns:
            total_cost_matrix (torch.Tensor): 总成本矩阵 [batch_size, num_classes, num_classes]
        """
        batch_size, num_classes = student_probs.shape
        
        # 第一部分：计算概率分布之间的成本
        # ||PS_τ(i) - PT_τ(j)||_2^2
        # student_probs[b, i] 和 teacher_probs[b, j] 之间的差的平方
        student_expanded = student_probs.unsqueeze(2)  # [batch_size, num_classes, 1]
        teacher_expanded = teacher_probs.unsqueeze(1)  # [batch_size, 1, num_classes]
        prob_cost = torch.pow(student_expanded - teacher_expanded, 2)
        
        # 第二部分：计算锚点之间的成本
        # λ * ||A'(i) - A'(j)||_2^2
        # A' 是 self.anchors
        if self.lambda_anchor > 0 and self.anchors is not None:
            # 计算锚点间的L2距离
            anchor_dist = torch.cdist(self.anchors, self.anchors, p=2)
            # 计算L2距离的平方
            anchor_cost_squared = torch.pow(anchor_dist, 2)
            # 乘以权重λ，并扩展到与prob_cost相同的批次大小
            anchor_cost_term = self.lambda_anchor * anchor_cost_squared.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            anchor_cost_term = 0.0

        # 总成本矩阵
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

'''class AnchorDistillationLoss(nn.Module):
    def __init__(self, student_outputs, teacher_outputs, anchors, temperature=1.0, device='cuda'):
        super(AnchorDistillationLoss, self).__init__()
        self.anchors = nn.Parameter(anchors, requires_grad=False)
        self.temperature = temperature
        self.student_outputs = student_outputs
        self.teacher_outputs = teacher_outputs
        self.device = device
        
        # Sinkhorn-Knopp算法参数
        self.sinkhorn_iterations = 10
        self.sinkhorn_epsilon = 0.1
        
        # 确保 anchors 的维度是 [num_classes, num_classes]
        num_classes = student_outputs.size(1)
        if anchors.size(1) != num_classes:
            self.anchors = self.adjust_anchors(anchors, num_classes)

    def adjust_anchors(self, anchors, num_classes):
        """
        调整 anchors 的维度以匹配 num_classes。
        """
        linear_transform = nn.Linear(anchors.size(1), num_classes).to(self.device)
        adjusted_anchors = linear_transform(anchors)
        return nn.Parameter(adjusted_anchors, requires_grad=False)

    def sinkhorn_knopp(self, cost_matrix, num_iterations=None, epsilon=None):
        """
        Sinkhorn-Knopp算法实现，用于计算最优传输矩阵
        
        Args:
            cost_matrix: 成本矩阵 [batch_size, num_classes, num_classes]
            num_iterations: Sinkhorn迭代次数
            epsilon: 正则化参数
            
        Returns:
            transport_matrix: 最优传输矩阵
        """
        if num_iterations is None:
            num_iterations = self.sinkhorn_iterations
        if epsilon is None:
            epsilon = self.sinkhorn_epsilon
            
        batch_size, n, m = cost_matrix.shape
        
        # 初始化传输矩阵 K = exp(-C/ε)
        K = torch.exp(-cost_matrix / epsilon)
        
        # 初始化行和列的缩放因子
        u = torch.ones(batch_size, n, 1, device=self.device) / n
        v = torch.ones(batch_size, m, 1, device=self.device) / m
        
        # Sinkhorn迭代
        for _ in range(num_iterations):
            # 更新行缩放因子
            u = 1.0 / (torch.bmm(K, v) + 1e-8)
            # 更新列缩放因子
            v = 1.0 / (torch.bmm(K.transpose(1, 2), u) + 1e-8)
        
        # 计算最优传输矩阵
        transport_matrix = u * K * v.transpose(1, 2)
        
        return transport_matrix

    def compute_cost_matrix(self, student_probs, teacher_probs):
        """
        计算学生和教师概率分布之间的成本矩阵
        
        Args:
            student_probs: 学生模型概率分布 [batch_size, num_classes]
            teacher_probs: 教师模型概率分布 [batch_size, num_classes]
            
        Returns:
            cost_matrix: 成本矩阵 [batch_size, num_classes, num_classes]
        """
        batch_size, num_classes = student_probs.shape
        
        # 扩展维度用于计算成本矩阵
        student_expanded = student_probs.unsqueeze(2)  # [batch_size, num_classes, 1]
        teacher_expanded = teacher_probs.unsqueeze(1)  # [batch_size, 1, num_classes]
        
        # 计算L2距离作为成本
        cost_matrix = torch.pow(student_expanded - teacher_expanded, 2)
        
        # 可选：加入锚点信息到成本计算中
        if hasattr(self, 'anchors') and self.anchors is not None:
            # 使用锚点调节成本矩阵
            anchor_cost = torch.cdist(self.anchors, self.anchors, p=2)
            anchor_cost = anchor_cost.unsqueeze(0).expand(batch_size, -1, -1)
            cost_matrix = cost_matrix + 0.1 * anchor_cost  # 可调节权重
        
        return cost_matrix

    def earth_movers_distance(self, student_probs, teacher_probs, transport_matrix):
        """
        计算Earth Mover's Distance (Wasserstein Distance)
        
        Args:
            student_probs: 学生模型概率分布
            teacher_probs: 教师模型概率分布  
            transport_matrix: 最优传输矩阵
            
        Returns:
            emd_loss: EMD损失
        """
        # 计算成本矩阵
        cost_matrix = self.compute_cost_matrix(student_probs, teacher_probs)
        
        # EMD = sum(T_ij * C_ij)，其中T是传输矩阵，C是成本矩阵
        emd_loss = torch.sum(transport_matrix * cost_matrix, dim=[1, 2])
        
        return torch.mean(emd_loss)

    def forward(self):
        """
        计算基于Sinkhorn-Knopp软对齐的蒸馏损失。
        
        Returns:
            loss: 计算得到的蒸馏损失
        """
        # 计算学生和教师模型的softmax概率
        student_probs = F.softmax(self.student_outputs / self.temperature, dim=1)
        teacher_probs = F.softmax(self.teacher_outputs / self.temperature, dim=1)
        
        # 将锚点信息融入学生特征
        anchors_expanded = self.anchors.unsqueeze(0)
        student_features = torch.matmul(student_probs, anchors_expanded.squeeze(0))
        
        # 确保特征维度匹配
        if student_features.size(1) != teacher_probs.size(1):
            linear_transform = nn.Linear(student_features.size(1), teacher_probs.size(1)).to(self.device)
            student_features = linear_transform(student_features)
        
        # 重新计算调整后的学生概率
        student_probs_adjusted = F.softmax(student_features / self.temperature, dim=1)
        
        # 计算成本矩阵
        cost_matrix = self.compute_cost_matrix(student_probs_adjusted, teacher_probs)
        
        # 使用Sinkhorn-Knopp算法计算最优传输矩阵
        transport_matrix = self.sinkhorn_knopp(cost_matrix)
        
        # 计算Earth Mover's Distance作为蒸馏损失
        emd_loss = self.earth_movers_distance(student_probs_adjusted, teacher_probs, transport_matrix)
        
        return emd_loss'''
    
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
