#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

"""
FedNH: Tackling Data Heterogeneity in Federated Learning with Class Prototypes
(Dai et al., AAAI 2023)

Server-side: FedAvg on the backbone + smoothed prototype aggregation (Eq. 4).
Client-side: freeze orthonormal class prototypes, train the backbone with a
cosine classifier, then upload class-wise mean embeddings.

Protocol matches FLOAM baselines: task cycle every 10 rounds, client sampling,
broadcast to all users, and the same evaluation / logging format.
"""

import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.options import args_parser
from utils.train_utils import get_model
from models.Update import (
    LocalUpdateFedNH,
    wrap_fednh_model,
    _fednh_unwrap,
    _fednh_get_head,
    _fednh_is_head_key,
)
from models.test import test_img, test_img_local_all, compute_smi_tdi_for_task


def _aggregate_prototypes(global_proto, local_protos, local_counts, rho):
    """
    Sample-weighted prototype fusion with smoothing (FedNH Eq. 4):
        W <- rho * W + (1 - rho) * sum_k (n_{k,c} / n_c) mu_{k,c}
    then re-normalize each row.
    """
    device = global_proto.device
    counts = torch.stack([c.to(device) for c in local_counts], dim=0).sum(dim=0)
    weighted = torch.zeros_like(global_proto)
    for proto, cnt in zip(local_protos, local_counts):
        weighted = weighted + proto.to(device) * cnt.to(device).unsqueeze(1)

    avg = global_proto.clone()
    valid = counts > 0
    avg[valid] = weighted[valid] / counts[valid].unsqueeze(1)
    avg = F.normalize(avg, p=2, dim=1)
    updated = rho * global_proto + (1.0 - rho) * avg
    return F.normalize(updated, p=2, dim=1)


def _write_prototypes(net, prototypes):
    head = _fednh_get_head(net)
    with torch.no_grad():
        head.weight.copy_(prototypes.to(head.weight.device, dtype=head.weight.dtype))
        if head.bias is not None:
            head.bias.zero_()
        real = _fednh_unwrap(net)
        if hasattr(real, '_freeze_head'):
            real._freeze_head()


if __name__ == '__main__':
    args = args_parser()
    dataset_path = args.datasetpath
    task_num = args.task_num
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = './save/{}/{}_num{}_C{}_le{}_bs{}_round{}_m{}_lr{}/{}/'.format(
        dataset_path, args.model, args.num_users, args.frac, args.local_ep,
        args.local_bs, args.epochs, args.momentum, args.lr, args.results_save)
    algo_dir = 'fednh'
    save_folder = './results/fednh'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    net_glob = wrap_fednh_model(get_model(args), args)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net_glob = nn.DataParallel(net_glob)
    net_glob.to(args.device)
    net_glob.train()

    net_local_list = []
    for _ in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))

    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')

    loss_train = []
    net_best = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []
    prev_client_centroids = None
    current_smi = np.nan
    current_tdi = np.nan
    rho = getattr(args, 'fednh_rho', 0.9)

    for iter in range(args.epochs):
        w_glob = None
        loss_locals = []
        local_protos = []
        local_counts = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        task = (iter // 10) % task_num
        print('Current task: ', task)

        for idx in idxs_users:
            local = LocalUpdateFedNH(args=args, dataset=dataset_path, idxs=idx, task=task)
            net_local = copy.deepcopy(net_local_list[idx])
            w_local, loss, prototypes, proto_counts = local.train(
                net=net_local.to(args.device), lr=lr
            )

            loss_locals.append(copy.deepcopy(loss))
            local_protos.append(prototypes)
            local_counts.append(proto_counts)

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]

        for k in w_glob.keys():
            if _fednh_is_head_key(k):
                continue
            w_glob[k] = torch.div(w_glob[k], m)

        global_proto = _fednh_get_head(net_glob).weight.detach()
        new_proto = _aggregate_prototypes(global_proto, local_protos, local_counts, rho)

        for user_idx in range(args.num_users):
            net_local_list[user_idx].load_state_dict(w_glob, strict=False)
            _write_prototypes(net_local_list[user_idx], new_proto)
        net_glob.load_state_dict(w_glob, strict=False)
        _write_prototypes(net_glob, new_proto)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % args.test_freq == 0:
            acc_test, acc_test_var, loss_test = test_img_local_all(
                net_local_list, args, dataset_test=dataset_path, task=task, return_all=False
            )

            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

            all_acc, all_loss = test_img(
                net_glob, datatest=dataset_path, args=args, epoch=iter,
                class_num=args.num_classes, save_folder=save_folder
            )

            print('All Test Data: Average loss: {:.4f}, Accuracy: {:.2f}% '.format(
                all_loss, all_acc))

            if best_acc is None or all_acc > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = all_acc
                best_epoch = iter
                best_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
                torch.save(net_best.state_dict(), best_save_path)

            if (iter + 1) % 10 == 0:
                current_smi, current_tdi, prev_client_centroids = compute_smi_tdi_for_task(
                    net_local_list=net_local_list,
                    args=args,
                    dataset_test=dataset_path,
                    task=task,
                    prev_client_centroids=prev_client_centroids,
                    num_classes=args.num_classes
                )
                tdi_str = 'nan' if np.isnan(current_tdi) else '{:.6f}'.format(current_tdi)
                print('Task {:3d} SMI: {:.6f}, TDI: {}'.format(task, current_smi, tdi_str))
            else:
                current_smi, current_tdi = np.nan, np.nan

            results.append(np.array([
                iter, task, loss_avg, loss_test, acc_test, all_acc, best_acc,
                current_smi, current_tdi
            ]))
            final_results = pd.DataFrame(
                np.array(results),
                columns=['epoch', 'task', 'loss_avg', 'loss_test', 'acc_test',
                         'all_acc', 'best_acc', 'smi', 'tdi']
            )
            final_results.to_csv(results_save_path, index=False)

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
