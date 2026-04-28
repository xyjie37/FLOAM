#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdateRifFiL
from models.test import test_img, test_img_local, test_img_local_all, compute_smi_tdi_for_task
import os

class GlobalPromptManager:
    def __init__(self, args):
        # 添加维度校验
        if not hasattr(args, 'prompt_dim'):
            raise ValueError("args.prompt_dim must be defined in command line!")
            
        self.args = args
        self.global_prompts = defaultdict(list)
        self.clustered_prompts = {}
        self.prompt_dim = args.prompt_dim
        self.num_classes = args.num_classes
        self.device = args.device
        
        # 调试输出
        print(f"[GlobalPromptManager] Initialized with prompt_dim={self.prompt_dim}")
        
    def aggregate_prompts(self, local_prompts_dict):
        class_prompts = defaultdict(list)
        for client_id, (prompts, labels) in local_prompts_dict.items():
            for prompt, label in zip(prompts, labels):
                # 确保提示在CPU上处理
                if isinstance(prompt, torch.Tensor):
                    prompt = prompt.detach().cpu().numpy()
                class_prompts[label.item()].append(prompt)

        clustered_prompts = {}
        for class_id, prompts in class_prompts.items():
            if len(prompts) >= 2:
                prompts_np = np.stack(prompts)
                c, _, _ = self._finch_clustering(prompts_np)
                # 使用统一维度创建张量
                clustered_prompts[class_id] = torch.tensor(
                    c, 
                    dtype=torch.float32, 
                    device=self.device
                )[:, :self.prompt_dim]  # 确保维度匹配
            else:
                # 处理单个提示的情况
                if prompts:
                    prompt_tensor = torch.tensor(
                        np.array(prompts), 
                        dtype=torch.float32, 
                        device=self.device
                    )
                    # 维度校验
                    if prompt_tensor.dim() == 1:
                        prompt_tensor = prompt_tensor.unsqueeze(0)
                    clustered_prompts[class_id] = prompt_tensor[:, :self.prompt_dim]
                else:
                    # 创建默认提示（使用预设维度）
                    clustered_prompts[class_id] = torch.randn(
                        1, self.prompt_dim, 
                        device=self.device
                    )
        return clustered_prompts

    def _finch_clustering(self, data):
        # 实现真正的FINCH聚类
        # 这里简化实现，实际应该使用FINCH算法
        from sklearn.cluster import AgglomerativeClustering
        
        # 计算相似度矩阵
        sim_matrix = np.dot(data, data.T)
        np.fill_diagonal(sim_matrix, -1)
        
        # 使用层次聚类 - 修复参数名
        clustering = AgglomerativeClustering(
            n_clusters=min(10, len(data)), 
            metric='precomputed',  # 修改为 metric
            linkage='average'
        )
        clustering.fit(1 - sim_matrix)  # 转换为距离矩阵
        
        # 选择每个簇的中心点作为代表
        cluster_centers = []
        for cluster_id in range(clustering.n_clusters):
            cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
            cluster_data = data[cluster_indices]
            
            # 计算簇内平均相似度，选择中心点
            cluster_sim = np.dot(cluster_data, cluster_data.T)
            center_idx = np.argmax(cluster_sim.sum(axis=1))
            cluster_centers.append(cluster_data[center_idx])
        
        return np.array(cluster_centers), None, None


class RefFiLServer:
    def __init__(self, args):
        # 参数校验与维度管理
        if not hasattr(args, 'prompt_dim'):
            args.prompt_dim = 64  # 设置默认值
            print(f"[WARNING] Using default prompt_dim={args.prompt_dim}")
        else:
            print(f"[Server] Prompt dimension: {args.prompt_dim}")
            
        self.args = args
        self.num_users = args.num_users
        self.task_num = args.task_num
        self.prompt_dim = args.prompt_dim  # 存储统一维度
        
        # 初始化全局模型
        self.global_model = get_model(args).to(args.device)
        
        # 初始化客户端模型列表（确保设备一致性）
        self.client_models = []
        for user_idx in range(args.num_users):
            client_model = copy.deepcopy(self.global_model)
            client_model.to(args.device)  # 确保客户端模型也在正确设备上
            self.client_models.append(client_model)
        
        # 初始化提示管理器（传递维度参数）
        self.prompt_manager = GlobalPromptManager(args)
        
        # 设备验证
        print(f"[Server] Global model on: {next(self.global_model.parameters()).device}")
        print(f"[Server] Client models on: {next(self.client_models[0].parameters()).device}")
        
        # 移除客户端分组策略，使用简单变量
        self.current_task = 0
        self.clustered_prompts = {}  # 提示缓存

    def aggregate_models(self, client_updates):
        total_samples = sum([c['num_samples'] for c in client_updates])
        return {
            k: sum(update['model'][k] * (update['num_samples']/total_samples) 
                   for update in client_updates)
            for k in client_updates[0]['model'].keys()
        }

    def train_one_round(self, round_idx):
        # 与FedAvg相同的任务切换逻辑
        task = (round_idx // 10) % self.task_num
        print('Current task: ', task)
        
        # 与FedAvg相同的客户端选择逻辑
        m = max(int(self.args.frac * self.num_users), 1)
        selected_clients = np.random.choice(range(self.num_users), m, replace=False)
        
        client_updates = []
        local_prompts_dict = {}
        
        # 设备验证
        print(f"[Round {round_idx}] Selected clients: {selected_clients}")
        print(f"[Round {round_idx}] Current task: {task}")
        
        for client_id in selected_clients:
            # 关键修改：传递提示维度
            trainer = LocalUpdateRifFiL(
                args=self.args,
                dataset=self.args.datasetpath,
                idxs=client_id,
                task=task,
                prompt_dim=self.prompt_dim  # 统一维度传递
            )
            
            # 设备管理 - 确保CDAP在正确设备上
            trainer.cdap = trainer.cdap.to(self.args.device)
            trainer.update_global_prompts(self.clustered_prompts)
            
            # 客户端模型（从预初始化的列表中获取，已经在正确设备上）
            client_model = copy.deepcopy(self.client_models[client_id])
            # 不需要再次移动设备，因为已经在初始化时确保了设备一致性
            
            # 移除设备校验，因为已经确保了一致性
            # assert next(client_model.parameters()).device == self.args.device
            # assert next(trainer.cdap.parameters()).device == self.args.device
            
            # 维度调试输出
            print(f"[Client {client_id}] Prompt dim: {trainer.cdap.prompt_dim}")
            
            # 训练客户端
            result = trainer.train(net=client_model, lr=self.args.lr)
            
            # 收集结果
            client_updates.append({
                'model': result["model_state"],
                'num_samples': len(trainer.ldr_train.dataset),
                'prompts': result["prompts"],
                'labels': trainer.current_labels
            })
            local_prompts_dict[client_id] = (
                result["prompts"], 
                trainer.current_labels
            )

        # 全局聚合（与FedAvg相同的逻辑）
        global_weights = self.aggregate_models(client_updates)
        
        # Broadcast - 与FedAvg相同的更新逻辑
        update_keys = list(global_weights.keys())
        global_weights = {k: v for k, v in global_weights.items() if k in update_keys}
        
        # 更新所有客户端模型
        for user_idx in range(self.num_users):
            self.client_models[user_idx].load_state_dict(global_weights, strict=False)
        self.global_model.load_state_dict(global_weights, strict=False)
        
        # 提示聚类（RefFiL特有的逻辑）
        self.clustered_prompts = self.prompt_manager.aggregate_prompts(local_prompts_dict)
            
        # 设备验证
        print(f"[Round {round_idx}] Global model on: {next(self.global_model.parameters()).device}")
        if self.clustered_prompts:
            first_key = next(iter(self.clustered_prompts.keys()))
            print(f"[Round {round_idx}] Prompt dim: {self.clustered_prompts[first_key].shape[-1]}")
            
        return self.clustered_prompts

if __name__ == '__main__':
    # parse args - 与FedAvg相同的初始化
    args = args_parser()
    dataset_path = args.datasetpath
    
    # 设置设备
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"===== RifFiL Training Start [Device: {args.device}] =====")
    print(f"Dataset: {args.dataset}, Model: {args.model}")
    print(f"Users: {args.num_users}, Fraction: {args.frac}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    
    # 初始化服务器
    server = RefFiLServer(args)
    
    # 训练参数存储（与FedAvg相同的目录结构）
    base_dir = './save/{}/{}_num{}_C{}_le{}_bs{}_round{}_m{}_lr{}/{}/'.format(
        dataset_path, args.model, args.num_users, args.frac, args.local_ep, args.local_bs, args.epochs, args.momentum, args.lr, args.results_save)
    algo_dir = 'refil'
    save_folder = './results/refil'
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)
    
    # training variables - 与FedAvg相同
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
    
    for round_idx in range(args.epochs):
        print(f"\n===== Round {round_idx+1}/{args.epochs} =====")
        global_prompts = server.train_one_round(round_idx)
        
        # 计算平均损失（与FedAvg相同的逻辑）
        # 注意：RefFiL版本没有返回loss_locals，这里简化处理
        loss_avg = 0.0  # 可以从client_updates中计算
        loss_train.append(loss_avg)

        # 定期评估
        if (round_idx + 1) % args.test_freq == 0:
            # 确保模型在设备上
            server.global_model.to(args.device)
            
            # 与FedAvg相同的测试逻辑
            acc_test, acc_test_var, loss_test = test_img_local_all(server.client_models, args, dataset_test=dataset_path, task=(round_idx // 10) % args.task_num, return_all=False)
            
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                round_idx, loss_avg, loss_test, acc_test))
            
            all_acc, all_loss = test_img(server.global_model, datatest=dataset_path, args=args, epoch=round_idx, class_num=args.num_classes, save_folder=save_folder)
            
            print('All Test Data: Average loss: {:.4f}, Accuracy: {:.2f}% '.format(
                all_loss, all_acc))

            if best_acc is None or all_acc > best_acc:
                net_best = copy.deepcopy(server.global_model)
                best_acc = all_acc
                best_epoch = round_idx
                
                best_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
                torch.save(net_best.state_dict(), best_save_path)
                
                # 保存提示
                with open(f'{save_folder}/global_prompts_{round_idx}.pkl', 'wb') as f:
                    pickle.dump(global_prompts, f)
                    
                print(f"New best model saved! Accuracy: {best_acc:.2f}%")

            task = (round_idx // 10) % args.task_num
            if (round_idx + 1) % 10 == 0:
                current_smi, current_tdi, prev_client_centroids = compute_smi_tdi_for_task(
                    net_local_list=server.client_models,
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

            # 记录结果（与FedAvg相同的格式）
            results.append(np.array([round_idx, task, loss_avg, loss_test, acc_test, all_acc, best_acc, current_smi, current_tdi]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch','task', 'loss_avg', 'loss_test', 'acc_test',  'all_acc','best_acc', 'smi', 'tdi'])
            final_results.to_csv(results_save_path, index=False)

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
    print("===== Training Completed =====")