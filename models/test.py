#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pdb
from utils.data_utils import load_train_data, load_test_data, read_all_test_data
import csv


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
import pandas as pd
from sklearn.metrics import classification_report
import os

# def test_img(net_g, datatest, args, epoch, class_num):
#     net_g.eval()
#     test_loss = 0
#     correct = 0
#     all_preds = []
#     all_targets = []

#     data_loader = read_all_test_data(datatest)

#     for idx, (data, target) in enumerate(data_loader):
#         if args.gpu != -1:
#             data, target = data.to(args.device), target.to(args.device)
#         log_probs = net_g(data)

#         test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
#         correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

#         all_preds.extend(y_pred.cpu().numpy())
#         all_targets.extend(target.cpu().numpy())

#     test_loss /= len(data_loader.dataset)
#     accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    
#     print(all_targets)

#     # 计算每个类的分类报告
#     report = classification_report(all_targets, all_preds, labels=range(class_num), target_names=[f"class_{i}" for i in range(class_num)], output_dict=True, zero_division=0)
#     df = pd.DataFrame(report).transpose()
#     df['Class'] = [f"class_{i}" for i in range(class_num)] + ['Accuracy', 'MacroAvg', 'WeightedAvg']
#     # 将 Class 列移动到第一列
#     cols = df.columns.tolist()
#     cols = cols[-1:] + cols[:-1]
#     df = df[cols]
    
#     save_dir = './per_class_accuracy/'
#     # 保存到指定目录
#     save_path = os.path.join(save_dir, f'classification_report_round{epoch}.csv')
#     df.to_csv(save_path, index=False)

#     return accuracy, test_loss

'''def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = read_all_test_data(datatest)
    

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    
    return accuracy, test_loss'''

def test_img(net_g, datatest, args, epoch, class_num, save_folder):
    net_g.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    data_loader = read_all_test_data(datatest)

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)

        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        all_preds.extend(y_pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    
    #print(all_targets)

    # 计算每个类的分类报告
    report = classification_report(all_targets, all_preds, labels=range(class_num), target_names=[f"class_{i}" for i in range(class_num)], output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df['Class'] = [f"class_{i}" for i in range(class_num)] + ['Accuracy', 'MacroAvg', 'WeightedAvg']
    # 将 Class 列移动到第一列
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    
    #save_dir = './per_class_accuracy'
    '''if not os.path.exists(save_dir):
        os.makedirs(save_dir)'''
    # 保存到指定目录
    save_path = os.path.join(save_folder, f'classification_report_round{epoch}.csv')
    df.to_csv(save_path, index=False)

    return accuracy, test_loss

def test_img_local(net_g, dataset, task, args, user_idx=-1, idxs=None, return_features=False):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = load_test_data(dataset, task, user_idx, batch_size=args.bs)
    # data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.bs, shuffle=False)
    sample_num = len(data_loader)
        
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        
        if return_features:
            tmp_features = net_g.extract_features(data)
            
            if idx==0:
                features = tmp_features.detach().cpu()
                targets = target.detach().cpu()
            else:
                features = torch.cat([features, tmp_features.detach().cpu()], dim=0)
                targets = torch.cat([targets, target.detach().cpu()])

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    if args.verbose:
        print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            user_idx, test_loss, correct, len(data_loader.dataset), accuracy))
        
    if return_features:
        return accuracy, test_loss, features, targets
    else:
        return accuracy, test_loss, sample_num
    
# def ood_test_img_local(net_g, dataset, args, user_idx=-1, idxs=None, user_train_targets=None):
#     net_g.eval()
#     # testing
#     per_total = 0
#     per_correct = 0
#     ood_total = 0
#     ood_correct = 0
#     # data_loader = DataLoader(dataset, batch_size=args.bs)
#     data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.bs, shuffle=False)
#     l = len(data_loader)
    
#     for idx, (data, target) in enumerate(data_loader):
#         if args.gpu != -1:
#             data, target = data.to(args.device), target.to(args.device)
#             user_train_targets = user_train_targets.to(args.device)
#         log_probs = net_g(data)
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
        
#         # get the index of the max log-probability
#         target_dup = torch.cat([target.view(-1, 1)]*len(user_train_targets), dim=1)
#         user_train_targets_dup = torch.cat([user_train_targets.view(1, -1)]*len(target), dim=0)
#         per_ood = torch.sum(target_dup == user_train_targets_dup, dim=1)
        
#         per_idx = torch.where(per_ood == 1)
#         ood_idx = torch.where(per_ood == 0)
        
#         per_pred = y_pred[per_idx]
#         ood_pred = y_pred[ood_idx]
        
#         per_target = target[per_idx]
#         ood_target = target[ood_idx]
        
#         per_total += len(per_target)
#         ood_total += len(ood_target)
        
#         per_correct += per_pred.eq(per_target.data.view_as(per_pred)).long().cpu().sum()
#         ood_correct += ood_pred.eq(ood_target.data.view_as(ood_pred)).long().cpu().sum()

#     if args.verbose:
#         print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
#             user_idx, test_loss, correct, len(data_loader.dataset), accuracy))
#     else:
#         return per_correct.item()/per_total*100, ood_correct.item()/ood_total*100
    
def distance_test_img_local(net_g, dataset_train, dataset_test, args, user_idx=-1, train_idxs=None, test_idxs=None):
    net_g.eval()
    
    train_data_loader = DataLoader(DatasetSplit(dataset_train, train_idxs), batch_size=args.bs, shuffle=False)    
    for idx, (data, target) in enumerate(train_data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        feature = net_g.extract_features(data)
        
        if idx == 0:
            features, targets = feature.detach().cpu(), target.detach().cpu()
        else:
            features = torch.cat([features, feature.detach().cpu()])
            targets = torch.cat([targets, target.detach().cpu()])
            
    if args.model == 'cnn':
        template = -99 * torch.ones([10, 256])
    elif args.model == 'mobile':
        template = -99 * torch.ones([100, 1024])
    for i in range(len(template)):
        if i in targets:
            template[i] = torch.mean(features[targets==i], dim=0)
        
    test_data_loader = DataLoader(DatasetSplit(dataset_test, test_idxs), batch_size=args.bs, shuffle=False)    
    for idx, (data, target) in enumerate(test_data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        feature = net_g.extract_features(data)
        
        if idx == 0:
            features, targets = feature.detach().cpu(), target.detach().cpu()
        else:
            features = torch.cat([features, feature.detach().cpu()])
            targets = torch.cat([targets, target.detach().cpu()])
            
    predicted = torch.argmin(torch.cdist(features, template), dim=1)
    return sum(predicted==targets).item()
    
def test_img_local_all(net_local_list, args, dataset_test, task, return_all=False):
    acc_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    sample_per_client = np.zeros(args.num_users)
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        net_local.eval()
        a, b, data_client = test_img_local(net_local, dataset_test, task, args, user_idx=idx)

        acc_test_local[idx] = a
        loss_test_local[idx] = b
        sample_per_client[idx] = data_client
    data_ratio_local = sample_per_client/sample_per_client.sum()
    
    if return_all:
        return acc_test_local, loss_test_local
    return acc_test_local.mean(), (acc_test_local*data_ratio_local).sum(), loss_test_local.mean()
    # return (acc_test_local*data_ratio_local).sum(), (loss_test_local*data_ratio_local).sum()


def _extract_features_from_net(net_g, data):
    if hasattr(net_g, "extract_features"):
        return net_g.extract_features(data)

    # Fallback for models exposing features only via forward(returnFeature=True)
    try:
        outputs = net_g(data, returnFeature=True)
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            return outputs[1]
    except TypeError:
        pass

    raise AttributeError("Model must implement extract_features() or forward(..., returnFeature=True).")


def collect_client_class_centroids(net_local_list, args, dataset_test, task, num_classes=None):
    if num_classes is None:
        num_classes = args.num_classes

    client_centroids = []
    for user_idx in range(args.num_users):
        net_local = net_local_list[user_idx]
        net_local.eval()
        data_loader = load_test_data(dataset_test, task, user_idx, batch_size=args.bs)

        feat_batches = []
        target_batches = []

        with torch.no_grad():
            for data, target in data_loader:
                if args.gpu != -1:
                    data, target = data.to(args.device), target.to(args.device)

                feat = _extract_features_from_net(net_local, data)
                # L2-normalize each sample's feature vector
                feat = F.normalize(feat, p=2, dim=1)
                feat_batches.append(feat.detach().cpu())
                target_batches.append(target.detach().cpu())

        if len(feat_batches) == 0:
            client_centroids.append([None for _ in range(num_classes)])
            continue

        features = torch.cat(feat_batches, dim=0)
        targets = torch.cat(target_batches, dim=0)

        class_centroids = []
        for y in range(num_classes):
            cls_mask = targets == y
            if torch.any(cls_mask):
                # Mean of normalized features, then L2-normalize the prototype
                proto = features[cls_mask].mean(dim=0)
                proto = F.normalize(proto.unsqueeze(0), p=2, dim=1).squeeze(0)
                class_centroids.append(proto)
            else:
                class_centroids.append(None)

        client_centroids.append(class_centroids)

    return client_centroids


def compute_smi_from_centroids(client_centroids, task_classes):
    per_class_smi = []
    for y in task_classes:
        class_mu = [client_centroids[k][y] for k in range(len(client_centroids)) if client_centroids[k][y] is not None]
        n_clients = len(class_mu)
        if n_clients < 2:
            continue

        class_mu = torch.stack(class_mu, dim=0)
        # Cosine dissimilarity: 1 - p^T q
        cos_sim = torch.mm(class_mu, class_mu.t())
        tri_idx = torch.triu_indices(n_clients, n_clients, offset=1)
        dissimilarities = 1 - cos_sim[tri_idx[0], tri_idx[1]]
        per_class_smi.append(dissimilarities.mean().item())

    if len(per_class_smi) == 0:
        return np.nan
    return float(np.mean(per_class_smi))


def compute_tdi_from_centroids(prev_client_centroids, curr_client_centroids, task_classes):
    if prev_client_centroids is None:
        return np.nan

    per_pair_drift = []
    n_clients = min(len(prev_client_centroids), len(curr_client_centroids))
    for k in range(n_clients):
        for y in task_classes:
            prev_mu = prev_client_centroids[k][y]
            curr_mu = curr_client_centroids[k][y]
            if prev_mu is None or curr_mu is None:
                continue
            # Cosine dissimilarity: 1 - p^T q
            cos_sim = torch.dot(prev_mu, curr_mu).item()
            per_pair_drift.append(1 - cos_sim)

    if len(per_pair_drift) == 0:
        return np.nan
    return float(np.mean(per_pair_drift))


def compute_smi_tdi_for_task(net_local_list, args, dataset_test, task, prev_client_centroids=None, num_classes=None):
    if num_classes is None:
        num_classes = args.num_classes

    # Determine the set of classes belonging to the current task
    task_num = getattr(args, 'task_num', 1)
    if task_num > 1:
        classes_per_task = num_classes // task_num
        task_classes = list(range(task * classes_per_task, min((task + 1) * classes_per_task, num_classes)))
    else:
        task_classes = list(range(num_classes))

    curr_client_centroids = collect_client_class_centroids(
        net_local_list=net_local_list,
        args=args,
        dataset_test=dataset_test,
        task=task,
        num_classes=num_classes,
    )

    smi = compute_smi_from_centroids(curr_client_centroids, task_classes)
    tdi = compute_tdi_from_centroids(prev_client_centroids, curr_client_centroids, task_classes)

    return smi, tdi, curr_client_centroids

def test_img_avg_all(net_glob, net_local_list, args, dataset_test, return_net=False):
    net_glob_temp = copy.deepcopy(net_glob)
    w_keys_epoch = net_glob.state_dict().keys()
    w_glob_temp = {}
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        w_local = net_local.state_dict()
        
        if len(w_glob_temp) == 0:
            w_glob_temp = copy.deepcopy(w_local)
        else:
            for k in w_keys_epoch:
                w_glob_temp[k] += w_local[k]

    for k in w_keys_epoch:
        w_glob_temp[k] = torch.div(w_glob_temp[k], args.num_users)
    net_glob_temp.load_state_dict(w_glob_temp)
    acc_test_avg, loss_test_avg = test_img(net_glob_temp, dataset_test, args)

    if return_net:
        return acc_test_avg, loss_test_avg, net_glob_temp
    return acc_test_avg, loss_test_avg

criterion = nn.CrossEntropyLoss()

def test_img_ensemble_all(net_local_list, args, dataset_test):
    probs_all = []
    preds_all = []
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        net_local.eval()
        # _, _, probs = test_img(net_local, dataset_test, args, return_probs=True, user_idx=idx)
        acc, loss, probs = test_img(net_local, dataset_test, args, return_probs=True, user_idx=idx)
        # print('Local model: {}, loss: {}, acc: {}'.format(idx, loss, acc))
        probs_all.append(probs.detach())

        preds = probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
        preds_all.append(preds)

    labels = np.array(dataset_test.targets)
    preds_probs = torch.mean(torch.stack(probs_all), dim=0)

    # ensemble (avg) metrics
    preds_avg = preds_probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
    loss_test = criterion(preds_probs, torch.tensor(labels).to(args.device)).item()
    acc_test_avg = (preds_avg == labels).mean() * 100

    # ensemble (maj)
    preds_all = np.array(preds_all).T
    preds_maj = stats.mode(preds_all, axis=1)[0].reshape(-1)
    acc_test_maj = (preds_maj == labels).mean() * 100

    return acc_test_avg, loss_test, acc_test_maj


def test_img_local_all_our(net_local_list, args, dataset_test, loca_head_list):
    acc_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    per_acc_test_local = np.zeros(args.num_users)
    head = copy.deepcopy(net_local_list[0].linear)
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        head.load_state_dict(loca_head_list[idx], strict=True)
        head.eval()
        net_local.eval()
        a, b, c = test_img_local_our(net_local, dataset_test, args, user_idx=idx, loca_head = head)

        acc_test_local[idx] = a
        loss_test_local[idx] = b
        per_acc_test_local[idx] = c
        
    return acc_test_local.mean(), per_acc_test_local.mean(), loss_test_local.mean()


def test_img_local_our(net_g, dataset, args, user_idx=-1, loca_head=None):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    correct_per = 0
    data_loader = load_test_data(dataset, user_idx, batch_size=args.bs)
   
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        rep = net_g.extract_features(data)
        # rep = rep.view((rep.size(0), -1))
        global_logits = net_g.linear(rep)
        per_logits = loca_head(rep)
        
        test_loss += F.cross_entropy(global_logits, target, reduction='sum').item()
        
        y_pred = global_logits.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        y_pred_per = per_logits.data.max(1, keepdim=True)[1]
        correct_per += y_pred_per.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    global_accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    per_accuracy = 100.00 * float(correct_per) / len(data_loader.dataset)
    
    return global_accuracy, test_loss, per_accuracy