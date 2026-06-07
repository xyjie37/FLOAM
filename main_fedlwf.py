#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

"""
Federated Learning with Learning without Forgetting (FedLWF)

This algorithm combines FedAvg aggregation with LwF client-side training.
LwF uses knowledge distillation to preserve old task knowledge while learning new tasks,
without requiring access to old task data.

Key features:
- FedAvg server-side aggregation
- LwF client-side training with knowledge distillation
- Temperature-scaled softmax for distillation
- Automatic detection of task switches
"""

import copy
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdateLWF
from models.test import test_img, test_img_local, test_img_local_all, compute_smi_tdi_for_task
import os

import pdb
from collections import defaultdict


if __name__ == '__main__':
    # parse args
    args = args_parser()
    dataset_path = args.datasetpath
    # Seed
    # torch.manual_seed(args.seed)#seed=1
    # torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # np.random.seed(args.seed)
    task_num = args.task_num
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    base_dir = './save/{}/{}_num{}_C{}_le{}_bs{}_round{}_m{}_lr{}/{}/'.format(
        dataset_path, args.model, args.num_users, args.frac, args.local_ep, args.local_bs, args.epochs, args.momentum, args.lr, args.results_save)
    algo_dir = 'fedlwf'
    save_folder = './results/fedlwf'
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    # build a global model
    net_glob = get_model(args)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net_glob = nn.DataParallel(net_glob)
    net_glob.to(args.device)
    net_glob.train()

    # build local models
    net_local_list = []
    for user_idx in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))
    
    # Store old model states for each client (for knowledge distillation)
    # This is updated when task switches
    old_models_store = [None for _ in range(args.num_users)]
    old_task_classes_store = [None for _ in range(args.num_users)]  # Store old task's class indices
    
    # training
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []
    prev_client_centroids = None
    current_smi = np.nan
    current_tdi = np.nan
    
    prev_task = 0
    
    for iter in range(args.epochs):
        w_glob = None
        loss_locals = []
        
        # Client Sampling
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))
        
        task = (iter // 10) % task_num  # Task switch every 10 rounds
        
        # Compute current task's class range
        classes_per_task = args.num_classes // task_num
        current_task_classes = list(range(task * classes_per_task, (task + 1) * classes_per_task))
        
        # Detect task switch - record old model before learning new task
        if task != prev_task and prev_task is not None:
            print(f'Task switch detected: {prev_task} -> {task}')
            print('Recording old model for knowledge distillation...')
            
            # Store old model state from each client
            for user_idx in range(args.num_users):
                old_models_store[user_idx] = copy.deepcopy(net_local_list[user_idx].state_dict())
            
            # Store old task's class indices
            prev_task_classes = list(range(prev_task * classes_per_task, (prev_task + 1) * classes_per_task))
            for user_idx in range(args.num_users):
                old_task_classes_store[user_idx] = prev_task_classes.copy()
        
        prev_task = task
        print('Current task: ', task)
        
        # Local Updates with LwF
        for idx in idxs_users:
            # Create local update instance
            local = LocalUpdateLWF(args=args, dataset=dataset_path, idxs=idx, task=task)
            
            # Transfer old model if available (for knowledge distillation)
            if old_models_store[idx] is not None and task > 0:
                local.old_model_state = old_models_store[idx].copy()
                local.old_task_classes = old_task_classes_store[idx].copy()
            
            net_local = copy.deepcopy(net_local_list[idx])
            w_local, loss = local.train(net=net_local.to(args.device), lr=lr)
                
            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
        
        # FedAvg Aggregation
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)
        
        # Broadcast
        update_keys = list(w_glob.keys())
        w_glob = {k: v for k, v in w_glob.items() if k in update_keys}
        for user_idx in range(args.num_users):
            net_local_list[user_idx].load_state_dict(w_glob, strict=False)
        net_glob.load_state_dict(w_glob, strict=False)
        # if (iter + 1) == 50:
        #     lr = 0.01
        # elif (iter + 1) ==75:
        #     lr = 0.001

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % args.test_freq == 0:
            acc_test, acc_test_var, loss_test = test_img_local_all(net_local_list, args, dataset_test=dataset_path, task=task, return_all=False)
            
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))
            
            #all_acc, all_loss = test_img(net_glob, datatest=dataset_path, args=args, epoch=iter, class_num=args.num_classes)
            
            all_acc, all_loss = test_img(net_glob, datatest=dataset_path, args=args, epoch = iter, class_num=args.num_classes, save_folder = save_folder)
            
            print('All Test Data: Average loss: {:.4f}, Accuracy: {:.2f}% '.format(
                all_loss, all_acc))

            if best_acc is None or all_acc > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = all_acc
                best_epoch = iter
                
                best_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
                
                torch.save(net_best.state_dict(), best_save_path)
                
#                 for user_idx in range(args.num_users):
#                     best_save_path = os.path.join(base_dir, algo_dir, 'best_local_{}.pt'.format(user_idx))
#                     torch.save(net_local_list[user_idx].state_dict(), best_save_path)

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

            results.append(np.array([iter,task, loss_avg, loss_test, acc_test, all_acc, best_acc, current_smi, current_tdi]))
            #results.append(np.array([iter, task, loss_avg, all_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch','task', 'loss_avg', 'loss_test', 'acc_test',  'all_acc','best_acc', 'smi', 'tdi'])
            #final_results = pd.DataFrame(final_results, columns=['epoch','task', 'loss_avg', 'all_acc'])
            final_results.to_csv(results_save_path, index=False)

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
