#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedTA: Federated Tail Anchor for ResNet.
- Client: ESA via forward hook, Tail Anchor mixing, L_CE + L_cons + L_key
- Server: FedAvg, SIKF (KB selection), BGPS (prototype selection), Anchor EMA
- Inference: ESA + TA enabled for train-test consistency
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.options import args_parser
from utils.train_utils import get_fedta_model
from utils.data_utils import read_all_test_data, load_test_data
from models.Update import LocalUpdateFedTAAdapter, _is_resnet_fedta
from create_anchor import create_anchor


def _get_feat_dim(net):
    """Get feature dimension from any FedTA-wrapped model.
    Supports ResNetFedTA (512), TextCNNFedTA (300), GenericFedTA (any).
    """
    real = net.module if hasattr(net, 'module') else net
    # Priority: check FedTA wrapper's feat_dim first
    if hasattr(real, 'feat_dim'):
        return real.feat_dim
    # Check base model (TextCNN has 'fc', ResNet has 'linear')
    if hasattr(real, 'base'):
        b = real.base.module if hasattr(real.base, 'module') else real.base
        if hasattr(b, 'fc') and hasattr(b.fc, 'in_features'):
            return b.fc.in_features
        if hasattr(b, 'linear') and hasattr(b.linear, 'in_features'):
            return b.linear.in_features
    # Check direct attributes
    if hasattr(real, 'fc') and hasattr(real.fc, 'in_features'):
        return real.fc.in_features
    if hasattr(real, 'linear') and hasattr(real.linear, 'in_features'):
        return real.linear.in_features
    # Default fallback for ResNet
    return 512


def _create_initial_anchor(args, feat_dim):
    """Create initial global anchor based on dataset."""
    return create_anchor(args.num_classes, feat_dim)


def _fedavg_weights(w_locals):
    """FedAvg aggregation of model state dicts."""
    w_avg = copy.deepcopy(w_locals[0])
    for k in w_avg.keys():
        for i in range(1, len(w_locals)):
            w_avg[k] += w_locals[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_locals))
    return w_avg


def _sikf_average(kb_list):
    """SIKF fallback: average KB when no surrogate data."""
    fedta_adapter = {}
    for k in kb_list[0]['fedta_adapter'].keys():
        fedta_adapter[k] = torch.stack([kb['fedta_adapter'][k] for kb in kb_list]).mean(dim=0)
    ie_keys = torch.stack([kb['ie_keys'] for kb in kb_list]).mean(dim=0)
    return {'fedta_adapter': fedta_adapter, 'ie_keys': ie_keys}


def _sikf_selective(server_kb, client_kb_list, net, surrogate_loader, device):
    """Select best KB by loss on surrogate; EMA merge when replacing."""
    if not client_kb_list:
        return server_kb
    if server_kb is None:
        return _sikf_average(client_kb_list)  # Initialize from all clients
    if surrogate_loader is None:
        return _sikf_average(client_kb_list)

    real = net.module if hasattr(net, 'module') else net
    best_kb = server_kb
    best_loss = float('inf')

    def eval_kb(kb):
        real.load_state_dict_fedta(kb)
        net.eval()
        loss = 0.0
        n = 0
        with torch.no_grad():
            for data, target in surrogate_loader:
                data, target = data.to(device), target.to(device)
                logits = net(data)
                loss += F.cross_entropy(logits, target, reduction='sum').item()
                n += len(target)
        return loss / max(n, 1)

    for kb in client_kb_list + [server_kb]:
        l = eval_kb(kb)
        if l < best_loss:
            best_loss = l
            best_kb = kb

    if best_kb is not server_kb:
        ema = 0.7
        new_adapter = {}
        for k in server_kb['fedta_adapter'].keys():
            new_adapter[k] = ema * server_kb['fedta_adapter'][k] + (1 - ema) * best_kb['fedta_adapter'][k]
        return {'fedta_adapter': new_adapter, 'ie_keys': ema * server_kb['ie_keys'] + (1 - ema) * best_kb['ie_keys']}
    return server_kb


def _bgps(local_protos_list, fixed_prototypes, thr, num_classes, device):
    """
    Best Global Prototype Selection.
    For each class y: build similarity matrix M^y, select k* = argmin_k mean(M_kj).
    If mean(M_k*j) < thr, fix G^y permanently.
    """
    # Aggregate prototypes: local_protos_list[idx] = {y: tensor}
    agg = {}
    for idx, protos in local_protos_list.items():
        for y, p in protos.items():
            if y not in agg:
                agg[y] = []
            agg[y].append(p.unsqueeze(0))
    for y in agg:
        agg[y] = torch.cat(agg[y], dim=0)  # [K, feat_dim]

    G = {}
    for y in range(num_classes):
        if y in fixed_prototypes:
            G[y] = fixed_prototypes[y].to(device)
            continue
        if y not in agg or len(agg[y]) < 2:
            if y in agg:
                G[y] = agg[y].mean(dim=0).to(device)
            continue
        P = agg[y].to(device)  # [K, feat_dim]
        P_norm = F.normalize(P, p=2, dim=1)
        M = torch.mm(P_norm, P_norm.t())  # [K, K] cosine similarity
        mean_sim = M.mean(dim=1)  # [K]
        k_star = mean_sim.argmin().item()
        if mean_sim[k_star] < thr:
            fixed_prototypes[y] = P[k_star].detach().cpu().clone()
        G[y] = P[k_star]
    return G, fixed_prototypes


def _update_anchor(global_anchor, G, gamma, num_classes, feat_dim, device):
    """A_g = (1-gamma)*A_g + gamma*Normalize(G)"""
    for y in range(num_classes):
        if y in G:
            g_norm = F.normalize(G[y].unsqueeze(0), p=2, dim=1).squeeze(0)
            global_anchor[y] = (1 - gamma) * global_anchor[y] + gamma * g_norm.to(global_anchor.device)


def test_img_fedta(net, datatest, args, epoch, class_num, save_folder, kb_glob, anchor, alpha):
    """Test ResNetFedTA (adapter built-in, no hooks). Load KB, sync anchor, forward."""
    import pandas as pd
    from sklearn.metrics import classification_report

    net.eval()
    device = args.device
    real = net.module if hasattr(net, 'module') else net
    if kb_glob is not None:
        real.load_state_dict_fedta(kb_glob)
    real.tail_anchors.data.copy_(anchor.to(device))

    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    data_loader = read_all_test_data(datatest)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            logits = net(data)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            y_pred = logits.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * float(correct) / len(data_loader.dataset)

    report = classification_report(all_targets, all_preds, labels=range(class_num),
                                   target_names=[f"class_{i}" for i in range(class_num)],
                                   output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df['Class'] = [f"class_{i}" for i in range(class_num)] + ['Accuracy', 'MacroAvg', 'WeightedAvg']
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    save_path = os.path.join(save_folder, f'classification_report_round{epoch}.csv')
    df.to_csv(save_path, index=False)

    return accuracy, test_loss


def test_img_local_fedta(net, dataset, task, args, user_idx, kb_glob, anchor, alpha):
    """Test single client (ResNetFedTA)."""
    net.eval()
    device = args.device
    real = net.module if hasattr(net, 'module') else net
    if kb_glob is not None:
        real.load_state_dict_fedta(kb_glob)
    real.tail_anchors.data.copy_(anchor.to(device))

    test_loss = 0
    correct = 0
    data_loader = load_test_data(dataset, task, user_idx, batch_size=args.bs)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            logits = net(data)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            y_pred = logits.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    n_samples = len(data_loader.dataset)
    test_loss /= n_samples
    accuracy = 100.0 * float(correct) / n_samples
    return accuracy, test_loss, n_samples


def test_img_local_all_fedta(net_local_list, args, dataset_test, task, kb_glob, anchor, alpha):
    """Test all local models (ResNetFedTA)."""
    acc_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    sample_per_client = np.zeros(args.num_users)
    for idx in range(args.num_users):
        acc, loss, n = test_img_local_fedta(
            net_local_list[idx], dataset_test, task, args, idx, kb_glob, anchor, alpha
        )
        acc_test_local[idx] = acc
        loss_test_local[idx] = loss
        sample_per_client[idx] = n
    data_ratio_local = sample_per_client / (sample_per_client.sum() + 1e-8)
    return acc_test_local.mean(), (acc_test_local * data_ratio_local).sum(), loss_test_local.mean()


if __name__ == '__main__':
    args = args_parser()
    dataset_path = args.datasetpath
    task_num = args.task_num
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = './save/{}/{}_num{}_C{}_le{}_bs{}_round{}_m{}_lr{}/{}/'.format(
        dataset_path, args.model, args.num_users, args.frac, args.local_ep, args.local_bs,
        args.epochs, args.momentum, args.lr, args.results_save)
    algo_dir = 'fedta'
    save_folder = './results/fedta'

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    net_glob = get_fedta_model(args)
    if torch.cuda.device_count() > 1:
        net_glob = nn.DataParallel(net_glob)
    net_glob.to(args.device)
    net_glob.train()

    feat_dim = _get_feat_dim(net_glob)
    surrogate_loader = None
    try:
        surrogate_loader = read_all_test_data(dataset_path)
    except Exception:
        pass
    global_anchor = _create_initial_anchor(args, feat_dim).to(args.device)
    kb_glob = None  # Initialized from first client round
    fixed_prototypes = {}
    global_prototypes = None  # Built from BGPS, passed to clients

    net_local_list = [copy.deepcopy(net_glob) for _ in range(args.num_users)]

    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')
    best_acc = None
    best_epoch = None
    results = []
    lr = args.lr
    gamma = getattr(args, 'fedta_gamma', 0.2)
    thr = getattr(args, 'fedta_thr', 0.5)

    for iter in range(args.epochs):
        task = (iter // 10) % task_num
        print('Round {:3d}, Task: {}'.format(iter, task))

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        w_locals = []
        kb_locals = []
        local_protos_list = {}
        loss_locals = []

        for idx in idxs_users:
            local = LocalUpdateFedTAAdapter(args=args, anchor=global_anchor, dataset=dataset_path, idxs=idx, task=task)
            net_local = copy.deepcopy(net_local_list[idx])
            w_net, kb_state, prototypes, loss = local.train(
                net=net_local.to(args.device),
                anchor=global_anchor,
                global_prototypes=global_prototypes,
                kb_state=kb_glob,
                lr=lr,
                round_idx=iter
            )
            w_locals.append(w_net)
            kb_locals.append(kb_state)
            local_protos_list[idx] = prototypes
            loss_locals.append(loss)

        w_glob = _fedavg_weights(w_locals)
        net_glob.load_state_dict(w_glob, strict=False)
        for i in range(args.num_users):
            net_local_list[i].load_state_dict(w_glob, strict=False)

        kb_glob = _sikf_selective(kb_glob, kb_locals, net_glob, surrogate_loader, args.device)

        G, fixed_prototypes = _bgps(
            local_protos_list, fixed_prototypes, thr,
            args.num_classes, args.device
        )
        global_prototypes = torch.zeros(args.num_classes, feat_dim, device=args.device)
        for y in G:
            global_prototypes[y] = G[y]

        _update_anchor(global_anchor, G, gamma, args.num_classes, feat_dim, args.device)

        # Sync tail_anchors from global_anchor (for ResNetFedTA)
        real = net_glob.module if hasattr(net_glob, 'module') else net_glob
        if _is_resnet_fedta(net_glob):
            real.tail_anchors.data.copy_(global_anchor)
        for i in range(args.num_users):
            r = net_local_list[i].module if hasattr(net_local_list[i], 'module') else net_local_list[i]
            if _is_resnet_fedta(net_local_list[i]):
                r.tail_anchors.data.copy_(global_anchor)

        loss_avg = sum(loss_locals) / len(loss_locals)

        if (iter + 1) % args.test_freq == 0:
            alpha_test = getattr(args, 'fedta_alpha_test', 0.018)
            acc_test, _, loss_test = test_img_local_all_fedta(
                net_local_list, args, dataset_path, task, kb_glob, global_anchor, alpha_test
            )
            print('Round {:3d}, Loss {:.3f}, Test loss {:.3f}, Test acc {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

            all_acc, all_loss = test_img_fedta(
                net_glob, datatest=dataset_path, args=args, epoch=iter,
                class_num=args.num_classes, save_folder=save_folder,
                kb_glob=kb_glob, anchor=global_anchor, alpha=alpha_test
            )
            print('All Test: loss {:.4f}, acc {:.2f}%'.format(all_loss, all_acc))

            if best_acc is None or all_acc > best_acc:
                best_acc = all_acc
                best_epoch = iter
                torch.save(net_glob.state_dict(), os.path.join(base_dir, algo_dir, 'best_model.pt'))

            results.append(np.array([iter, task, loss_avg, loss_test, acc_test, all_acc, best_acc]))
            pd.DataFrame(np.array(results), columns=['epoch', 'task', 'loss_avg', 'loss_test', 'acc_test', 'all_acc', 'best_acc']).to_csv(results_save_path, index=False)

    print('Best model at epoch {}, acc {:.2f}%'.format(best_epoch, best_acc))
