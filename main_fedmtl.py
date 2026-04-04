#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedMTL: Adaptive Multi-Teacher Knowledge Distillation for Federated Continual Learning.
- Client: trains classification model (CE) + CVAE (reconstruction + KL)
- Server: FedAvg aggregation, dual-threshold sample generation, adaptive multi-teacher distillation
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.options import args_parser
from utils.train_utils import get_model
from utils.data_utils import read_all_test_data
from models.Update import LocalUpdateFedMTL
from models.CVAE import CVAE, get_img_shape
from models.test import test_img, test_img_local_all
import os

# FedMTL hyperparameters
ROUNDS_PER_TASK = 10
V_MAX = 0.9
V_MIN = 0.5
SAMPLES_PER_CLASS = 100
CQS_BETA = 5.0
CQS_GAMMA = 0.5
V_MIN_ALPHA = 0.7
KD_TEMPERATURE = 4.0
AKD_EPOCHS = 5
AKD_LR = 0.01


def fedavg_state_dicts(state_dicts):
    """FedAvg aggregation of state dicts."""
    w_avg = copy.deepcopy(state_dicts[0])
    for k in w_avg.keys():
        for i in range(1, len(state_dicts)):
            w_avg[k] += state_dicts[i][k]
        w_avg[k] = torch.div(w_avg[k], len(state_dicts))
    return w_avg


def generate_samples(cvae, task_classes, samples_per_class, device, num_classes):
    """Generate samples per class using CVAE for given task classes."""
    cvae.eval()
    all_samples = []
    all_labels = []
    with torch.no_grad():
        for k in task_classes:
            c = F.one_hot(torch.tensor([k], device=device).long(), num_classes=num_classes).float()
            for batch_start in range(0, samples_per_class, 32):
                n = min(32, samples_per_class - batch_start)
                c_batch = c.repeat(n, 1)
                x = cvae.generate(c_batch, num_samples=n, device=device)
                all_samples.append(x)
                all_labels.append(torch.full((n,), k, device=device, dtype=torch.long))
    if all_samples:
        return torch.cat(all_samples, dim=0), torch.cat(all_labels, dim=0)
    return None, None


def dual_threshold_filter(net_glob, X, y, v_max, v_min, n_per_class, device, num_classes):
    """
    Dual-threshold filtering: D_H (high confidence), D_L (low confidence).
    Per class: select up to n_per_class samples, prefer D_H then fill from D_L.
    """
    net_glob.eval()
    with torch.no_grad():
        probs = F.softmax(net_glob(X.to(device)), dim=1)
    max_probs, preds = probs.max(dim=1)
    max_probs = max_probs.cpu()
    X, y = X.cpu(), y.cpu()

    selected_X, selected_y = [], []
    for k in range(num_classes):
        mask = (y == k)
        if not mask.any():
            continue
        X_k = X[mask]
        p_k = max_probs[mask]
        y_k = y[mask]

        d_h_mask = p_k >= v_max
        d_l_mask = (p_k >= v_min) & (p_k < v_max)

        X_h = X_k[d_h_mask]
        p_h = p_k[d_h_mask]
        X_l = X_k[d_l_mask]
        p_l = p_k[d_l_mask]

        # Sort by confidence descending
        if len(X_h) > 0:
            idx_h = torch.argsort(p_h, descending=True)
            X_h = X_h[idx_h]
        if len(X_l) > 0:
            idx_l = torch.argsort(p_l, descending=True)
            X_l = X_l[idx_l]

        # Select: prefer D_H, then D_L
        need = n_per_class
        if len(X_h) >= need:
            selected_X.append(X_h[:need])
            selected_y.append(y_k[d_h_mask][idx_h][:need])
        else:
            take_h = len(X_h)
            take_l = min(need - take_h, len(X_l))
            if take_h > 0:
                selected_X.append(X_h[:take_h])
                selected_y.append(y_k[d_h_mask][idx_h][:take_h])
            if take_l > 0:
                selected_X.append(X_l[:take_l])
                selected_y.append(y_k[d_l_mask][idx_l][:take_l])

    if not selected_X:
        return None, None
    return torch.cat(selected_X), torch.cat(selected_y)


def merge_datasets_uniform(task_datasets, num_classes, max_per_class=50):
    """Merge task datasets with uniform sampling for class balance."""
    from collections import defaultdict
    by_class = defaultdict(list)
    for X, y in task_datasets:
        if X is None:
            continue
        for i in range(len(X)):
            c = y[i].item()
            by_class[c].append((X[i], y[i]))

    # Uniform sample per class
    selected = []
    for c in range(num_classes):
        if c not in by_class:
            continue
        items = by_class[c]
        n = min(len(items), max_per_class)
        idx = np.random.choice(len(items), n, replace=False) if len(items) >= n else np.arange(len(items))
        for i in idx:
            selected.append(items[i])
    if not selected:
        return None, None
    X = torch.stack([s[0] for s in selected])
    y = torch.stack([s[1] for s in selected])
    return X, y


def compute_cqs(teacher_logits, labels, num_classes):
    """CQS_k = mean over D_k of (z_{i,k}^T - mean_{c!=k}(z_{i,c}^T))."""
    probs = F.softmax(teacher_logits, dim=1)
    cqs = torch.zeros(num_classes, device=teacher_logits.device)
    counts = torch.zeros(num_classes, device=teacher_logits.device)
    for k in range(num_classes):
        mask = (labels == k)
        if not mask.any():
            continue
        p_k = probs[mask, k]
        other_sum = (probs[mask].sum(dim=1) - probs[mask, k]) / (num_classes - 1) if num_classes > 1 else torch.zeros_like(p_k)
        cqs[k] = (p_k - other_sum).mean()
        counts[k] = mask.sum().float()
    return cqs, counts


def adaptive_kd_train(net_glob, teacher_models, D_z_X, D_z_y, args, classes_per_task,
                       beta=CQS_BETA, gamma=CQS_GAMMA, v_min_alpha=V_MIN_ALPHA,
                       temp=KD_TEMPERATURE, epochs=AKD_EPOCHS, lr=AKD_LR):
    """
    Adaptive multi-teacher knowledge distillation.
    For sample with class c: use teacher from task that trained class c.
    """
    if D_z_X is None or len(D_z_X) == 0:
        return
    device = args.device
    num_classes = args.num_classes

    # Map class -> task (teacher index)
    def class_to_task(c):
        return c // classes_per_task if classes_per_task > 0 else 0

    # Compute teacher logits for CQS (use first available teacher per sample's class)
    teacher_logits_list = []
    for t, teacher in enumerate(teacher_models):
        if teacher is None:
            continue
        teacher.eval()
        with torch.no_grad():
            # Get samples that belong to this task's classes
            task_classes = list(range(t * classes_per_task, min((t + 1) * classes_per_task, num_classes)))
            mask = torch.tensor([y.item() in task_classes for y in D_z_y], device=device)
            if mask.any():
                # D_z_X is on CPU; use mask.cpu() for indexing to avoid device mismatch
                logits = teacher(D_z_X[mask.cpu()].to(device))
                teacher_logits_list.append((mask, logits))

    # Build full teacher logits (each sample has one teacher)
    full_teacher_logits = torch.zeros(len(D_z_X), num_classes, device=device)
    for mask, logits in teacher_logits_list:
        full_teacher_logits[mask] = logits

    # CQS and alpha
    cqs, counts = compute_cqs(full_teacher_logits, D_z_y.to(device), num_classes)
    cqs_valid = cqs[counts > 0]
    if len(cqs_valid) == 0:
        return
    cqs_min, cqs_max = cqs_valid.min().item(), cqs_valid.max().item()
    norm_cqs = (cqs - cqs_min) / (cqs_max - cqs_min + 1e-8)
    sigma = torch.sigmoid(beta * (norm_cqs - gamma))
    alpha_k = v_min_alpha + (1 - v_min_alpha) * sigma

    # Train with AKD loss
    net_glob.train()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.wd)
    dataset = TensorDataset(D_z_X, D_z_y)
    loader = DataLoader(dataset, batch_size=args.local_bs, shuffle=True)

    for _ in range(epochs):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            student_logits = net_glob(batch_X)
            ce_loss = F.cross_entropy(student_logits, batch_y)

            # Teacher: per-sample, select by class
            teacher_logits_batch = torch.zeros_like(student_logits)
            for t, teacher in enumerate(teacher_models):
                if teacher is None:
                    continue
                task_classes = list(range(t * classes_per_task, min((t + 1) * classes_per_task, num_classes)))
                mask = torch.tensor([y.item() in task_classes for y in batch_y], device=device)
                if mask.any():
                    with torch.no_grad():
                        teacher_logits_batch[mask] = teacher(batch_X[mask])

            kd_loss_per_sample = F.kl_div(
                F.log_softmax(student_logits / temp, dim=1),
                F.softmax(teacher_logits_batch / temp, dim=1),
                reduction='none'
            ).sum(dim=1) * (temp ** 2)
            ce_loss_per_sample = F.cross_entropy(student_logits, batch_y, reduction='none')

            alphas = alpha_k[batch_y]
            loss = (alphas * ce_loss_per_sample + (1 - alphas) * kd_loss_per_sample).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


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
    algo_dir = 'fedmtl'
    save_folder = './results/fedmtl'

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    # Build global models
    net_glob = get_model(args)
    if torch.cuda.device_count() > 1:
        net_glob = nn.DataParallel(net_glob)
    net_glob.to(args.device)
    net_glob.train()

    cvae_glob = CVAE(args).to(args.device)

    net_local_list = [copy.deepcopy(net_glob) for _ in range(args.num_users)]
    cvae_local_list = [copy.deepcopy(cvae_glob) for _ in range(args.num_users)]

    task_synthetic_datasets = []
    teacher_models = []
    classes_per_task = args.num_classes // task_num if task_num > 0 else args.num_classes

    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')
    best_acc = None
    best_epoch = None
    results = []
    lr = args.lr

    for iter in range(args.epochs):
        task = (iter // ROUNDS_PER_TASK) % task_num
        prev_task = ((iter - 1) // ROUNDS_PER_TASK) % task_num if iter > 0 else -1

        # CVAE: fresh init when task changes (per FedMTL paper)
        if task != prev_task:
            cvae_glob = CVAE(args).to(args.device)
            for i in range(args.num_users):
                cvae_local_list[i] = copy.deepcopy(cvae_glob)

        # AKD at task switch: consolidate all completed tasks (incl. previous) before training new task
        # Paper is vague on AKD timing; running at task start avoids forgetting the just-finished task
        if task != prev_task and iter > 0 and len(task_synthetic_datasets) >= 1 and len(teacher_models) >= 1:
            D_z_X, D_z_y = merge_datasets_uniform(
                task_synthetic_datasets, args.num_classes, max_per_class=SAMPLES_PER_CLASS
            )
            if D_z_X is not None:
                adaptive_kd_train(
                    net_glob, teacher_models, D_z_X, D_z_y, args, classes_per_task
                )
                for i in range(args.num_users):
                    net_local_list[i].load_state_dict(net_glob.state_dict(), strict=False)

        print('Round {:3d}, Task: {}'.format(iter, task))

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        w_locals_net = []
        w_locals_cvae = []
        loss_locals = []

        for idx in idxs_users:
            local = LocalUpdateFedMTL(args=args, dataset=dataset_path, idxs=idx, task=task)
            net_local = copy.deepcopy(net_local_list[idx])
            cvae_local = copy.deepcopy(cvae_local_list[idx])
            w_net, w_cvae, loss = local.train(net=net_local.to(args.device), cvae=cvae_local.to(args.device), lr=lr)
            w_locals_net.append(w_net)
            w_locals_cvae.append(w_cvae)
            loss_locals.append(loss)

        w_glob_net = fedavg_state_dicts(w_locals_net)
        w_glob_cvae = fedavg_state_dicts(w_locals_cvae)

        net_glob.load_state_dict(w_glob_net, strict=False)
        cvae_glob.load_state_dict(w_glob_cvae, strict=False)
        for i in range(args.num_users):
            net_local_list[i].load_state_dict(w_glob_net, strict=False)
            cvae_local_list[i].load_state_dict(w_glob_cvae, strict=False)

        loss_avg = sum(loss_locals) / len(loss_locals)

        # Task completion: save teacher, generate and filter for current task (AKD runs at next task start)
        if (iter + 1) % ROUNDS_PER_TASK == 0:
            teacher_models.append(copy.deepcopy(net_glob))
            if torch.cuda.device_count() > 1 and hasattr(teacher_models[-1], 'module'):
                teacher_models[-1] = teacher_models[-1].module

            task_classes = list(range(task * classes_per_task, min((task + 1) * classes_per_task, args.num_classes)))
            if task_classes:
                X_gen, y_gen = generate_samples(cvae_glob, task_classes, SAMPLES_PER_CLASS, args.device, args.num_classes)
                if X_gen is not None:
                    X_filt, y_filt = dual_threshold_filter(
                        net_glob, X_gen, y_gen, V_MAX, V_MIN, SAMPLES_PER_CLASS, args.device, args.num_classes
                    )
                    if X_filt is not None:
                        task_synthetic_datasets.append((X_filt.cpu(), y_filt.cpu()))

        if (iter + 1) % args.test_freq == 0:
            acc_test, _, loss_test = test_img_local_all(
                net_local_list, args, dataset_test=dataset_path, task=task, return_all=False
            )
            print('Round {:3d}, Loss {:.3f}, Test loss {:.3f}, Test acc {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

            all_acc, all_loss = test_img(
                net_glob, datatest=dataset_path, args=args, epoch=iter,
                class_num=args.num_classes, save_folder=save_folder
            )
            print('All Test: loss {:.4f}, acc {:.2f}%'.format(all_loss, all_acc))

            if best_acc is None or all_acc > best_acc:
                best_acc = all_acc
                best_epoch = iter
                torch.save(net_glob.state_dict(), os.path.join(base_dir, algo_dir, 'best_model.pt'))

            results.append(np.array([iter, task, loss_avg, loss_test, acc_test, all_acc, best_acc]))
            pd.DataFrame(np.array(results), columns=['epoch', 'task', 'loss_avg', 'loss_test', 'acc_test', 'all_acc', 'best_acc']).to_csv(results_save_path, index=False)

    print('Best model at epoch {}, acc {:.2f}%'.format(best_epoch, best_acc))
