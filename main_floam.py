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
from models.test import test_img, test_img_local_all, compute_smi_tdi_for_task
from create_anchor import create_anchor, agg_func, proto_aggregation
from utils.runtime_utils import RoundTimer

class FedACDServer:
    def __init__(self, args):
        self.args = args
        if args.gpu != '-1':
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args.device = self.device
        self.dataset_path = args.datasetpath
        self.task_num = args.task_num
      
        # Initialize global model
        self.net_glob = get_model(args)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net_glob = nn.DataParallel(self.net_glob)
        self.net_glob.to(self.device)
        self.net_glob.train()
      
        # Create global anchor
        self.global_anchor = self._create_initial_anchor()
        self.global_anchor = self.global_anchor.to(self.device)
      
        # Initialize local model list
        self.net_local_list = [copy.deepcopy(self.net_glob) for _ in range(args.num_users)]
      
        # Create save directory
        self.base_dir = self._create_save_directory()
        self.results_save_path = os.path.join(self.base_dir, 'results.csv')
      
        # Training state tracking
        self.best_acc = None
        self.best_epoch = None
        self.results = []
        self.prev_client_centroids = None
        self.current_smi = np.nan
        self.current_tdi = np.nan
        self.round_timer = RoundTimer(device=self.device) if self.args.benchmark_runtime else None
        if self.args.benchmark_runtime and self.args.skip_eval is False:
            self.args.skip_eval = True

    def _create_initial_anchor(self):
        """Create initial anchor based on dataset."""
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
        elif self.args.dataset == 'imagenet100':
            return create_anchor(100, 512)
        elif self.args.dataset == 'speechcommands':
            return create_anchor(30, 512)
        elif self.args.dataset == 'yahooanswers':
            return create_anchor(10, 300)
        elif self.args.dataset == 'agnews':
            return create_anchor(4, 100)
        elif self.args.dataset == '20newsgroup':
            return create_anchor(20, 300)
        return create_anchor(10, 32)  # default

    def _create_save_directory(self):
        """Create result save directory."""
        base_dir = f'./save/{self.dataset_path}/{self.args.model}_num{self.args.num_users}_C{self.args.frac}_le{self.args.local_ep}_bs{self.args.local_bs}_round{self.args.epochs}_m{self.args.momentum}_lr{self.args.lr}/{self.args.results_save}/'
        algo_dir = 'fedacd'
        full_path = os.path.join(base_dir, algo_dir)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def _aggregate_weights(self, w_locals):
        """Aggregate client weights."""
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

    def _update_global_anchor(self, local_protos, local_counts=None):
        """Update global anchor (normalized EMA, paper Eq. 24)."""
        new_anchor = proto_aggregation(
            local_protos,
            local_counts_list=local_counts,
            mode=getattr(self.args, 'anchor_agg', 'client_balanced'),
        )
        for i in range(self.args.num_classes):
            if i in new_anchor:
                updated = 0.2 * new_anchor[i].to(self.device) + 0.8 * self.global_anchor[i]
                self.global_anchor[i] = updated / (updated.norm() + 1e-8)

    def _save_results(self):
        """Save training results."""
        final_results = pd.DataFrame(
            np.array(self.results),
            columns=['epoch', 'task', 'loss_avg', 'loss_test', 'acc_test', 'all_acc', 'best_acc', 'smi', 'tdi']
        )
        final_results.to_csv(self.results_save_path, index=False)

    def train(self):
        save_folder = './results/fedacd'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        """Run federated learning training."""
        for epoch in range(self.args.epochs):
            if self.round_timer is not None:
                self.round_timer.begin_round(epoch)

            # Client sampling
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
          
            # Current task cycle
            task = (epoch // 10) % self.task_num
            print('current task:', task)
            # Local training
            w_locals = []
            local_protos = {}
            local_counts = {}
            loss_locals = []
          
            for idx in idxs_users:
                if self.round_timer is not None:
                    self.round_timer.start_client(idx)
                local = LocalUpdateFedACD(
                    args=self.args,
                    anchor=self.global_anchor,
                    dataset=self.dataset_path,
                    idxs=idx,
                    task=task
                )
                w_local, loss, reps, proto_counts = local.train(
                    net=copy.deepcopy(self.net_glob).to(self.device),
                    teacher_net=self.net_glob,
                    lr=self.args.lr
                )
                if self.round_timer is not None:
                    self.round_timer.end_client(idx)
                local_protos[idx] = agg_func(reps)
                local_counts[idx] = proto_counts
                w_locals.append(w_local)
                loss_locals.append(loss)

            if self.round_timer is not None:
                self.round_timer.start_server()

            # Aggregate updates
            w_glob = self._aggregate_weights(w_locals)
            self._update_global_anchor(local_protos, local_counts)
          
            # Update global model
            self.net_glob.load_state_dict(w_glob)
            for net_local in self.net_local_list:
                net_local.load_state_dict(w_glob)

            if self.round_timer is not None:
                self.round_timer.end_server()
                record = self.round_timer.finish_round()
                print('Round {:3d} runtime: max_client={:.3f}s, server={:.3f}s, total={:.3f}s'.format(
                    epoch, record['max_client_time'], record['server_time'], record['round_time']))

            # Evaluate and save
            if self.args.skip_eval:
                gc.collect()
                torch.cuda.empty_cache()
                continue

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

                if (epoch + 1) % 10 == 0:
                    self.current_smi, self.current_tdi, self.prev_client_centroids = compute_smi_tdi_for_task(
                        net_local_list=self.net_local_list,
                        args=self.args,
                        dataset_test=self.dataset_path,
                        task=task,
                        prev_client_centroids=self.prev_client_centroids,
                        num_classes=self.args.num_classes
                    )
                    tdi_str = 'nan' if np.isnan(self.current_tdi) else '{:.6f}'.format(self.current_tdi)
                    print('Task {:3d} SMI: {:.6f}, TDI: {}'.format(task, self.current_smi, tdi_str))
                else:
                    self.current_smi, self.current_tdi = np.nan, np.nan

                self.results.append([
                    epoch, task, 
                    sum(loss_locals)/len(loss_locals), 
                    loss_test, 
                    acc_test, 
                    all_acc, 
                    self.best_acc,
                    self.current_smi,
                    self.current_tdi
                ])
                self._save_results()

            gc.collect()
            torch.cuda.empty_cache()

        if self.round_timer is not None:
            runtime_csv = self.args.runtime_csv or os.path.join(self.base_dir, 'runtime.csv')
            self.round_timer.save_csv(runtime_csv)
            self.round_timer.print_summary()
            print('Runtime CSV saved to: {}'.format(runtime_csv))

        print(f'Best model at epoch {self.best_epoch}, accuracy {self.best_acc}')

if __name__ == '__main__':
    args = args_parser()
    server = FedACDServer(args)
    server.train()
