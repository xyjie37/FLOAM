import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_



def create_anchor(class_num=10, dim=32):
    # Orthogonal init (transpose trick so rows are orthogonal)
    anchors = torch.nn.Parameter(torch.empty(class_num, dim))
    torch.nn.init.orthogonal_(anchors.T)  # orthogonalize transpose so original rows are orthogonal
    return anchors.detach()
'''def create_anchor(class_num=10, dim=32):
    # Init params
    anchors = torch.nn.Parameter(torch.randn(class_num, dim))
    optimizer = torch.optim.AdamW([anchors], lr=0.03, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # Dynamic margin config
    base_margin = 1.2
    decay_rate = 0.995
    
    for epoch in range(1000):
        # Adjust margin
        current_margin = base_margin * (decay_rate ** epoch)
        
        # Distance matrix
        dist = torch.cdist(anchors, anchors, p=2)
        eye_mask = torch.eye(class_num, dtype=bool)
        
        # Intra-class: minimize diagonal distances
        intra_loss = torch.mean(dist[eye_mask] ** 2)
        
        # Inter-class: maximize with margin
        inter_loss = torch.mean(torch.relu(current_margin - dist[~eye_mask]) ** 2)
        
        # Combined loss
        loss = intra_loss + 3.0 * inter_loss
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([anchors], 1.0)
        optimizer.step()
        scheduler.step()
        
        # Log progress
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

