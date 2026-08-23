#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedTA: Federated Tail Anchor on ResNet backbones.

Warmup phase: standard FedAvg to obtain a shared backbone.
FedTA phase: freeze backbone, train IE + TA with SIKF and BGPS on server.
"""

import copy
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.Update import LocalUpdateFedTA, LocalUpdateFedTAWarmup, _unwrap_module
from models.test import compute_smi_tdi_for_task, test_img
from utils.data_utils import read_client_data
from utils.options import args_parser
from utils.train_utils import get_fedta_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fedavg_weights(w_locals):
    """Average every float tensor; integer buffers are copied from the first client.

    Frozen parameters are identical across clients, so averaging them is a no-op
    and the same routine works for both the warmup and the FedTA phase.
    """
    w_avg = copy.deepcopy(w_locals[0])
    for key in w_avg.keys():
        if not torch.is_floating_point(w_avg[key]):
            continue
        stacked = torch.stack([w[key].float() for w in w_locals], dim=0)
        w_avg[key] = stacked.mean(dim=0).to(dtype=w_locals[0][key].dtype)
    return w_avg


def _average_ie_states(ie_states):
    if not ie_states:
        return None
    avg = {'ie_bank': None, 'ie_keys': None}
    avg['ie_bank'] = torch.stack([s['ie_bank'] for s in ie_states]).mean(dim=0)
    avg['ie_keys'] = torch.stack([s['ie_keys'] for s in ie_states]).mean(dim=0)
    return avg


def build_surrogate_loader(dataset_path, num_users, num_classes, per_class, seed, batch_size=32):
    rng = np.random.default_rng(seed)
    samples_by_class = defaultdict(list)

    for client_id in range(num_users):
        data = read_client_data(dataset_path, client_id, task=0, is_train=True)
        for x, y in data:
            y_val = int(y.item()) if torch.is_tensor(y) else int(y)
            samples_by_class[y_val].append(x)

    xs, ys = [], []
    for cls in range(num_classes):
        pool = samples_by_class.get(cls, [])
        if not pool:
            continue
        count = min(per_class, len(pool))
        indices = rng.choice(len(pool), size=count, replace=False)
        for idx in indices:
            xs.append(pool[idx])
            ys.append(cls)

    if not xs:
        return None

    X = torch.stack(xs)
    Y = torch.tensor(ys, dtype=torch.long)
    if len(X.shape) == 3:
        X = X.unsqueeze(1)
    return DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=False, drop_last=False)


def sikf_fuse(server_ie, client_ie_list, net, surrogate_loader, device, steps=5, lr=1e-3):
    """Selective Input Knowledge Fusion via surrogate MSE distillation (Eq. 8)."""
    if not client_ie_list:
        return server_ie
    if server_ie is None:
        return _average_ie_states(client_ie_list)
    if surrogate_loader is None or len(client_ie_list) < 2:
        return _average_ie_states([server_ie] + client_ie_list)

    real = _unwrap_module(net)
    was_training = net.training
    net.eval()

    # Cache surrogate batches once so client/server features stay aligned.
    surrogate_batches = [data.to(device) for data, _ in surrogate_loader]
    if not surrogate_batches:
        return _average_ie_states([server_ie] + client_ie_list)

    with torch.no_grad():
        ref_features = []
        for kb in client_ie_list:
            real.load_state_dict_ie(kb)
            ref_features.append([
                real.extract_backbone_output(data).detach() for data in surrogate_batches
            ])

    target_bank = server_ie['ie_bank'].detach().to(device).clone().requires_grad_(True)
    target_keys = server_ie['ie_keys'].detach().to(device).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([target_bank, target_keys], lr=lr)

    for _ in range(steps):
        total_loss = 0.0
        count = 0
        for batch_idx, data in enumerate(surrogate_batches):
            feat_i = real.extract_backbone_output(data, ie_bank=target_bank, ie_keys=target_keys)
            for client_feats in ref_features:
                total_loss = total_loss + F.mse_loss(feat_i, client_feats[batch_idx])
                count += 1
        if count == 0:
            break
        loss = total_loss / count
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    fused = {
        'ie_bank': target_bank.detach().cpu(),
        'ie_keys': target_keys.detach().cpu(),
    }
    real.load_state_dict_ie(server_ie)
    if was_training:
        net.train()
    return fused


def bgps_select(local_protos_list, fixed_prototypes, thr, round_idx, thr_min_round, device):
    """Best Global Prototype Selection on the full prototype set (Eq. 9-10)."""
    entries = []
    for _, protos in local_protos_list.items():
        for y, proto in protos.items():
            entries.append((int(y), proto))

    if not entries:
        return {}, fixed_prototypes

    protos = torch.stack([
        F.normalize(item[1].to(device).float(), dim=0, eps=1e-6) for item in entries
    ], dim=0)
    labels = [item[0] for item in entries]
    K = protos.size(0)
    M = torch.mm(protos, protos.t())

    for i in range(K):
        for j in range(K):
            if labels[i] == labels[j]:
                M[i, j] = 1.0

    G = {}
    for y in sorted(set(labels)):
        if y in fixed_prototypes:
            G[y] = fixed_prototypes[y].to(device)
            continue

        indices = [idx for idx, lbl in enumerate(labels) if lbl == y]
        if len(indices) == 1:
            G[y] = protos[indices[0]]
            continue

        best_idx = None
        best_mean = float('inf')
        for idx in indices:
            mean_sim = M[idx].mean().item()
            if mean_sim < best_mean:
                best_mean = mean_sim
                best_idx = idx

        selected = protos[best_idx]
        if round_idx >= thr_min_round and best_mean < thr:
            fixed_prototypes[y] = selected.detach().cpu().clone()
        G[y] = selected

    return G, fixed_prototypes


def update_global_prototypes(global_prototypes, new_G, gamma, num_classes, feat_dim, device):
    if global_prototypes is None:
        global_prototypes = torch.zeros(num_classes, feat_dim, device=device)

    for y, proto in new_G.items():
        g_norm = F.normalize(proto.unsqueeze(0), dim=1, eps=1e-6).squeeze(0)
        if global_prototypes[y].norm() < 1e-6:
            global_prototypes[y] = g_norm
        else:
            updated = (1.0 - gamma) * global_prototypes[y] + gamma * g_norm
            global_prototypes[y] = F.normalize(updated.unsqueeze(0), dim=1, eps=1e-6).squeeze(0)
    return global_prototypes


def _get_feat_dim(net):
    real = _unwrap_module(net)
    return real.feat_dim


def _clear_fedta_cache(net):
    real = _unwrap_module(net)
    if hasattr(real, 'clear_runtime_cache'):
        real.clear_runtime_cache()


def _clone_to_device(net, device):
    """CPU-safe clone that avoids deepcopy of non-leaf runtime tensors."""
    _clear_fedta_cache(net)
    cloned = copy.deepcopy(net)
    _clear_fedta_cache(cloned)
    return cloned.to(device)


def _state_to_cpu(state_dict):
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in state_dict.items()}


def _eval_local_all(net_local_list, args, dataset_path, task):
    """Evaluate each client on GPU one-at-a-time to avoid OOM."""
    from models.test import test_img_local

    acc_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    sample_per_client = np.zeros(args.num_users)

    for idx in range(args.num_users):
        _clear_fedta_cache(net_local_list[idx])
        net = net_local_list[idx].to(args.device)
        net.eval()
        try:
            a, b, data_client = test_img_local(net, dataset_path, task, args, user_idx=idx)
        finally:
            _clear_fedta_cache(net)
            net_local_list[idx] = net.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        acc_test_local[idx] = a
        loss_test_local[idx] = b
        sample_per_client[idx] = data_client

    data_ratio_local = sample_per_client / max(sample_per_client.sum(), 1.0)
    return acc_test_local.mean(), (acc_test_local * data_ratio_local).sum(), loss_test_local.mean()


if __name__ == '__main__':
    args = args_parser()
    set_seed(args.seed)

    dataset_path = args.datasetpath
    task_num = args.task_num
    warmup_rounds = getattr(args, 'fedta_warmup_rounds', 10)
    freeze_level = getattr(args, 'fedta_freeze_level', 'partial')
    ta_agg = getattr(args, 'fedta_ta_agg', 'fedavg')
    gamma = getattr(args, 'fedta_gamma', 0.2)
    thr = getattr(args, 'fedta_thr', 0.5)
    thr_min_round = getattr(args, 'fedta_thr_min_round', 20)
    sikf_steps = getattr(args, 'fedta_sikf_steps', 5)
    surrogate_per_class = getattr(args, 'fedta_surrogate_per_class', 20)

    if args.gpu != '-1':
        # Respect launcher-provided visibility (e.g. CUDA_VISIBLE_DEVICES=1).
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = './save/{}/{}_num{}_C{}_le{}_bs{}_round{}_m{}_lr{}/{}/'.format(
        dataset_path, args.model, args.num_users, args.frac, args.local_ep,
        args.local_bs, args.epochs, args.momentum, args.lr, args.results_save)
    algo_dir = 'fedta'
    save_folder = './results/fedta'
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    net_glob = get_fedta_model(args)
    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs')
        net_glob = nn.DataParallel(net_glob)
    net_glob.to(args.device)
    net_glob.train()

    feat_dim = _get_feat_dim(net_glob)
    surrogate_loader = build_surrogate_loader(
        dataset_path, args.num_users, args.num_classes,
        surrogate_per_class, args.seed, batch_size=min(32, args.local_bs)
    )

    ie_glob = None
    ta_glob = None
    fixed_prototypes = {}
    global_prototypes = None
    backbone_frozen = False

    net_local_list = [copy.deepcopy(net_glob).cpu() for _ in range(args.num_users)]
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')

    best_acc = None
    best_epoch = None
    results = []
    client_ta_states = {}
    prev_client_centroids = None
    current_smi = np.nan
    current_tdi = np.nan

    for epoch in range(args.epochs):
        task = (epoch // 10) % task_num
        in_warmup = epoch < warmup_rounds
        print('Round {:3d}, Task: {}, Phase: {}'.format(
            epoch, task, 'warmup' if in_warmup else 'fedta'))

        if not in_warmup and not backbone_frozen:
            for model in [net_glob] + net_local_list:
                real = _unwrap_module(model)
                real.freeze_backbone(level=freeze_level)
                real.ta_enabled = True
            backbone_frozen = True
            print('Entering FedTA phase at round {} (freeze_level={})'.format(
                epoch, freeze_level))

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        w_locals = []
        ie_locals = []
        ta_locals = []
        local_protos_list = {}
        local_counts_list = {}
        loss_locals = []

        for idx in idxs_users:
            net_local = _clone_to_device(net_local_list[idx], args.device)

            if in_warmup:
                local = LocalUpdateFedTAWarmup(args=args, dataset=dataset_path, idxs=idx, task=task)
                w_local, loss = local.train(net=net_local, lr=args.lr)
                w_locals.append(_state_to_cpu(w_local))
                loss_locals.append(loss)
            else:
                local = LocalUpdateFedTA(args=args, dataset=dataset_path, idxs=idx, task=task)
                w_local, ie_state, ta_state, prototypes, proto_counts, loss = local.train(
                    net=net_local,
                    global_prototypes=global_prototypes,
                    ie_state=ie_glob,
                    ta_state=ta_glob if ta_agg == 'fedavg' else None,
                    lr=args.lr,
                    round_idx=epoch,
                )
                w_locals.append(_state_to_cpu(w_local))
                ie_locals.append({k: v.detach().cpu() if torch.is_tensor(v) else v
                                  for k, v in ie_state.items()})
                ta_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v
                          for k, v in ta_state.items()}
                ta_locals.append(ta_cpu)
                client_ta_states[idx] = ta_cpu
                local_protos_list[idx] = prototypes
                local_counts_list[idx] = proto_counts
                loss_locals.append(loss)

            _clear_fedta_cache(net_local)
            del net_local
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        w_glob = _fedavg_weights(w_locals)
        if not in_warmup:
            ie_glob = sikf_fuse(
                server_ie=ie_glob,
                client_ie_list=ie_locals,
                net=net_glob,
                surrogate_loader=surrogate_loader,
                device=args.device,
                steps=sikf_steps,
            )
            if ta_agg == 'fedavg' and ta_locals:
                ta_glob = {
                    'tail_anchors': torch.stack([s['tail_anchors'] for s in ta_locals]).mean(dim=0),
                    'ta_keys': torch.stack([s['ta_keys'] for s in ta_locals]).mean(dim=0),
                    'logit_scale': torch.stack([s['logit_scale'] for s in ta_locals]).mean(dim=0),
                }

            G, fixed_prototypes = bgps_select(
                local_protos_list, fixed_prototypes, thr, epoch, thr_min_round, args.device
            )
            global_prototypes = update_global_prototypes(
                global_prototypes, G, gamma, args.num_classes, feat_dim, args.device
            )

        net_glob.load_state_dict(w_glob, strict=False)
        real_glob = _unwrap_module(net_glob)
        if ie_glob is not None:
            real_glob.load_state_dict_ie(ie_glob)
        if ta_glob is not None and ta_agg == 'fedavg':
            real_glob.load_state_dict_ta(ta_glob)

        for i in range(args.num_users):
            net_local_list[i].load_state_dict(w_glob, strict=False)
            real_local = _unwrap_module(net_local_list[i])
            if ie_glob is not None:
                real_local.load_state_dict_ie(ie_glob)
            if ta_agg == 'fedavg':
                if ta_glob is not None:
                    real_local.load_state_dict_ta(ta_glob)
            elif i in client_ta_states:
                # 'local' ablation: undo the FedAvg over TA keys for this client.
                real_local.load_state_dict_ta(client_ta_states[i])
            if backbone_frozen:
                real_local.freeze_backbone(level=freeze_level)
                real_local.ta_enabled = True

        loss_avg = sum(loss_locals) / max(len(loss_locals), 1)

        if (epoch + 1) % args.test_freq == 0:
            acc_test, _, loss_test = _eval_local_all(
                net_local_list, args, dataset_path, task
            )
            all_acc, all_loss = test_img(
                net_glob, datatest=dataset_path, args=args, epoch=epoch,
                class_num=args.num_classes, save_folder=save_folder
            )
            print('Round {:3d}, Loss {:.3f}, Test loss {:.3f}, Test acc {:.2f}'.format(
                epoch, loss_avg, loss_test, acc_test))
            print('All Test: loss {:.4f}, acc {:.2f}%'.format(all_loss, all_acc))

            if best_acc is None or all_acc > best_acc:
                best_acc = all_acc
                best_epoch = epoch
                torch.save(net_glob.state_dict(), os.path.join(base_dir, algo_dir, 'best_model.pt'))

            if (epoch + 1) % 10 == 0:
                # Move clients to GPU one-by-one inside metric collection via temporary list
                gpu_locals = []
                try:
                    for net in net_local_list:
                        _clear_fedta_cache(net)
                        gpu_locals.append(net.to(args.device))
                    current_smi, current_tdi, prev_client_centroids = compute_smi_tdi_for_task(
                        net_local_list=gpu_locals,
                        args=args,
                        dataset_test=dataset_path,
                        task=task,
                        prev_client_centroids=prev_client_centroids,
                        num_classes=args.num_classes,
                    )
                finally:
                    for i, net in enumerate(gpu_locals):
                        _clear_fedta_cache(net)
                        net_local_list[i] = net.cpu()
                    del gpu_locals
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                tdi_str = 'nan' if np.isnan(current_tdi) else '{:.6f}'.format(current_tdi)
                print('Task {:3d} SMI: {:.6f}, TDI: {}'.format(task, current_smi, tdi_str))
            else:
                current_smi, current_tdi = np.nan, np.nan

            results.append([
                epoch, task, loss_avg, loss_test, acc_test, all_acc, best_acc,
                current_smi, current_tdi,
            ])
            pd.DataFrame(
                np.array(results),
                columns=['epoch', 'task', 'loss_avg', 'loss_test', 'acc_test',
                         'all_acc', 'best_acc', 'smi', 'tdi'],
            ).to_csv(results_save_path, index=False)

    print('Best model at epoch {}, acc {:.2f}%'.format(best_epoch, best_acc))
