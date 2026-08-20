#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import json
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img, test_img_local, test_img_local_all, compute_smi_tdi_for_task

import pdb
from collections import defaultdict


if __name__ == '__main__':
    # parse args
    args = args_parser()
    dataset_path = args.datasetpath
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Reproducibility: seed each random source used by the training pipeline.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    client_rng = np.random.default_rng(args.seed)
    task_num = args.task_num
    args.device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    base_dir = './save/{}/{}_num{}_C{}_le{}_bs{}_round{}_m{}_lr{}/{}/'.format(
        dataset_path, args.model, args.num_users, args.frac, args.local_ep, args.local_bs, args.epochs, args.momentum, args.lr, args.results_save)
    algo_dir = 'fedavg'
    run_dir = os.path.join(base_dir, algo_dir, 'seed_{}'.format(args.seed))
    save_folder = './results/fedavg'
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    os.makedirs(run_dir, exist_ok=True)

    # Freeze the complete client schedule before training starts.
    clients_per_round = max(int(args.frac * args.num_users), 1)
    client_schedule = []
    for round_idx in range(args.epochs):
        clients = client_rng.choice(
            args.num_users, clients_per_round, replace=False).tolist()
        client_schedule.append({
            'round': round_idx,
            'task': (round_idx // 10) % task_num,
            'clients': clients,
        })

    schedule_path = os.path.join(run_dir, 'client_schedule.json')
    with open(schedule_path, 'w', encoding='utf-8') as schedule_file:
        json.dump(client_schedule, schedule_file, indent=2)

    run_config = vars(args).copy()
    run_config['device'] = str(args.device)
    run_config['clients_per_round'] = clients_per_round
    config_path = os.path.join(run_dir, 'run_config.json')
    with open(config_path, 'w', encoding='utf-8') as config_file:
        json.dump(run_config, config_file, indent=2, sort_keys=True)

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
    
    # training
    results_save_path = os.path.join(run_dir, 'results.csv')

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
    final_metrics = None
    
    for iter in range(args.epochs):
        w_glob = None
        loss_locals = []
        
        # Client Sampling
        m = clients_per_round
        idxs_users = np.asarray(client_schedule[iter]['clients'], dtype=int)
        
        task=(iter//10)%task_num  # Task switch every 10 rounds
        print('Round {}, task {}, selected clients: {}'.format(
            iter, task, idxs_users.tolist()))
        # Local Updates
        for idx in idxs_users:
            # Dataset name, index
            local = LocalUpdate(args=args, dataset=dataset_path, idxs=idx, task = task)
            net_local = copy.deepcopy(net_local_list[idx])
            w_local, loss = local.train(net=net_local.to(args.device), lr=lr)
                
            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
        
        # Aggregation
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

        # Always evaluate the final round, even when test_freq does not divide
        # the total number of rounds.
        if (iter + 1) % args.test_freq == 0 or (iter + 1) == args.epochs:
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
                
                best_save_path = os.path.join(run_dir, 'best_model.pt')
                
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

            final_metrics = {
                'round': int(iter),
                'task': int(task),
                'loss_avg': float(loss_avg),
                'loss_test': float(loss_test),
                'acc_test': float(acc_test),
                'all_acc': float(all_acc),
                'smi': None if np.isnan(current_smi) else float(current_smi),
                'tdi': None if np.isnan(current_tdi) else float(current_tdi),
            }

    final_model_path = os.path.join(run_dir, 'final_model.pt')
    torch.save(net_glob.state_dict(), final_model_path)

    if final_metrics is not None:
        final_metrics_path = os.path.join(run_dir, 'final_metrics.json')
        with open(final_metrics_path, 'w', encoding='utf-8') as metrics_file:
            json.dump(final_metrics, metrics_file, indent=2, sort_keys=True)
        print('Final round {}, task {}, all-data accuracy: {:.2f}%'.format(
            final_metrics['round'], final_metrics['task'],
            final_metrics['all_acc']))
    else:
        print('No final metrics were produced because no training round ran.')

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
