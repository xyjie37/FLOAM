#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import gc
import copy
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdateFedACD
from models.test import test_img, test_img_local_all
from create_anchor import create_anchor, agg_func, proto_aggregation

class FedACDServer:
    def __init__(self, args):
        self.args = args
        if args.gpu != '-1':
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args.device = self.device
        self.dataset_path = args.datasetpath
        self.task_num = args.task_num
      
        # 初始化全局模型
        self.net_glob = get_model(args)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net_glob = nn.DataParallel(self.net_glob)
        self.net_glob.to(self.device)
        self.net_glob.train()
      
        # 创建全局锚点
        self.global_anchor = self._create_initial_anchor()
        self.global_anchor = self.global_anchor.to(self.device)
      
        # 初始化本地模型列表
        self.net_local_list = [copy.deepcopy(self.net_glob) for _ in range(args.num_users)]
      
        # 创建保存目录
        self.base_dir = self._create_save_directory()
        self.results_save_path = os.path.join(self.base_dir, 'results.csv')
      
        # 训练状态跟踪
        self.best_acc = None
        self.best_epoch = None
        self.results = []

    def _create_initial_anchor(self):
        """根据数据集创建初始锚点"""
        if self.args.dataset == 'fmnist':
            return create_anchor(10, 32)
        elif self.args.dataset == 'cifar10':
            return create_anchor(10, 256)
        elif self.args.dataset == 'cinic10':
            return create_anchor(10, 256)
        elif self.args.dataset == 'cifar100':
            return create_anchor(100, 512)
        elif self.args.dataset == 'miniimagenet':
            return create_anchor(100, 512)
        elif self.args.dataset == 'tinyimagenet':
            return create_anchor(200, 2048)
        elif self.args.dataset == 'speechcommands':
            return create_anchor(30, 512)
        elif self.args.dataset == 'yahooanswers':
            return create_anchor(10, 300)
        elif self.args.dataset == 'agnews':
            return create_anchor(4, 100)
        return create_anchor(10, 32)  # 默认值

    def _create_save_directory(self):
        """创建结果保存目录"""
        base_dir = f'./save/{self.dataset_path}/{self.args.model}_num{self.args.num_users}_C{self.args.frac}_le{self.args.local_ep}_bs{self.args.local_bs}_round{self.args.epochs}_m{self.args.momentum}_lr{self.args.lr}/{self.args.results_save}/'
        algo_dir = 'fedacd'
        full_path = os.path.join(base_dir, algo_dir)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def _aggregate_weights(self, w_locals):
        """聚合客户端权重"""
        w_glob = None
        for k in w_locals[0].keys():
            for w in w_locals:
                if w_glob is None:
                    w_glob = copy.deepcopy(w)
                    for key in w_glob:
                        w_glob[key] = w_glob[key] * 0
                w_glob[k] += w[k]
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], len(w_locals))
        return w_glob

    def _update_global_anchor(self, local_protos):
        """更新全局锚点"""
        new_anchor = proto_aggregation(local_protos)
        for i in range(self.args.num_classes):
            if i in new_anchor:
                self.global_anchor[i] = 0.2 * new_anchor[i].to(self.device) + 0.8 * self.global_anchor[i]

    def _save_results(self):
        """保存训练结果"""
        final_results = pd.DataFrame(
            np.array(self.results),
            columns=['epoch', 'task', 'loss_avg', 'loss_test', 'acc_test', 'all_acc', 'best_acc']
        )
        final_results.to_csv(self.results_save_path, index=False)

    def train(self):
        save_folder = './results/fedacd'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        """执行联邦学习训练过程"""
        for epoch in range(self.args.epochs):
            # 客户端采样
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
          
            # 当前任务周期
            task = (epoch // 10) % self.task_num
            print('current task:', task)
            # 本地训练
            w_locals = []
            local_protos = {}
            loss_locals = []
          
            for idx in idxs_users:
                local = LocalUpdateFedACD(
                    args=self.args,
                    anchor=self.global_anchor,
                    dataset=self.dataset_path,
                    idxs=idx,
                    task=task
                )
                w_local, loss, reps = local.train(
                    net=copy.deepcopy(self.net_glob).to(self.device),
                    teacher_net=self.net_glob,
                    lr=self.args.lr
                )
                local_protos[idx] = agg_func(reps)
                w_locals.append(w_local)
                loss_locals.append(loss)

            # 聚合更新
            w_glob = self._aggregate_weights(w_locals)
            self._update_global_anchor(local_protos)
          
            # 更新全局模型
            self.net_glob.load_state_dict(w_glob)
            for net_local in self.net_local_list:
                net_local.load_state_dict(w_glob)

            # 评估和保存
            if (epoch + 1) % self.args.test_freq == 0:
                acc_test, _, loss_test = test_img_local_all(
                    self.net_local_list, self.args, self.dataset_path, task
                )
                #all_acc, all_loss = test_img(self.net_glob, self.dataset_path, self.args)
                all_acc, all_loss = test_img(self.net_glob, datatest=self.dataset_path, args=self.args, epoch = epoch, class_num=self.args.num_classes, save_folder = save_folder)
                loss_avg = sum(loss_locals)/len(loss_locals)
                print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                        epoch, loss_avg, loss_test, acc_test))
                print('All Test Data: Average loss: {:.4f}, Accuracy: {:.2f}% '.format(all_loss, all_acc))
              
                if self.best_acc is None or all_acc > self.best_acc:
                    self.best_acc = all_acc
                    self.best_epoch = epoch
                    torch.save(self.net_glob.state_dict(), os.path.join(self.base_dir, 'best_model.pt'))
                
                self.results.append([
                    epoch, task, 
                    sum(loss_locals)/len(loss_locals), 
                    loss_test, 
                    acc_test, 
                    all_acc, 
                    self.best_acc
                ])
                self._save_results()

            gc.collect()
            torch.cuda.empty_cache()

        print(f'Best model at epoch {self.best_epoch}, accuracy {self.best_acc}')

if __name__ == '__main__':
    args = args_parser()
    server = FedACDServer(args)
    server.train()
