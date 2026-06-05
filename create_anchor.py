import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_



def create_anchor(class_num=10, dim=32):
    # 正交初始化（转置后处理，使行向量正交）
    anchors = torch.nn.Parameter(torch.empty(class_num, dim))
    torch.nn.init.orthogonal_(anchors.T)  # 对转置矩阵正交化，使原始行向量正交
    return anchors.detach()
'''def create_anchor(class_num=10, dim=32):
    # 初始化参数
    anchors = torch.nn.Parameter(torch.randn(class_num, dim))
    optimizer = torch.optim.AdamW([anchors], lr=0.03, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # 动态参数配置
    base_margin = 1.2
    decay_rate = 0.995
    
    for epoch in range(1000):
        # 动态调整参数
        current_margin = base_margin * (decay_rate ** epoch)
        
        # 距离矩阵计算
        dist = torch.cdist(anchors, anchors, p=2)
        eye_mask = torch.eye(class_num, dtype=bool)
        
        # 修正后的损失计算
        # 类内距离最小化（对角线元素即同类）
        intra_loss = torch.mean(dist[eye_mask] ** 2)
        
        # 类间距离最大化（保留margin机制）
        inter_loss = torch.mean(torch.relu(current_margin - dist[~eye_mask]) ** 2)
        
        # 组合损失（调整权重平衡）
        loss = intra_loss + 3.0 * inter_loss
        
        # 优化步骤
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([anchors], 1.0)
        optimizer.step()
        scheduler.step()
        
        # 监控输出
        if (epoch+1) % 100 == 0:
            with torch.no_grad():
                avg_intra = dist[eye_mask].mean().item()
                avg_inter = dist[~eye_mask].mean().item()
            print(f"Epoch {epoch+1}/1000 | Margin:{current_margin:.2f} | Intra:{avg_intra:.2f} | Inter:{avg_inter:.2f}")
    
    return anchors.detach()'''





def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        # if len(proto_list) > 1:
            # proto = 0 * proto_list[0].data
            # for i in proto_list:
            #     proto += i.data
        protos[label] = proto_list.mean(dim=0)
        # else:
        #     protos[label] = proto_list[0]

    return protos


def proto_aggregation(local_protos_list, local_counts_list=None, mode='client_balanced'):
    agg_protos_label = dict()
    if mode == 'sample_weighted' and local_counts_list is not None:
        class_items = dict()
        for idx in local_protos_list:
            local_protos = local_protos_list[idx]
            local_counts = local_counts_list.get(idx, {})
            for label, proto in local_protos.items():
                count = local_counts.get(label, 1)
                class_items.setdefault(label, []).append((proto, count))
        for label, items in class_items.items():
            total = sum(c for _, c in items)
            weighted = sum(proto * (c / total) for proto, c in items)
            agg_protos_label[label] = weighted
        return agg_protos_label

    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label] = torch.cat(
                    (agg_protos_label[label], torch.unsqueeze(local_protos[label], 0)), dim=0)
            else:
                agg_protos_label[label] = torch.unsqueeze(local_protos[label], 0)

    for k in agg_protos_label.keys():
        agg_protos_label[k] = torch.mean(agg_protos_label[k], dim=0)

    return agg_protos_label

