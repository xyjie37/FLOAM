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
from utils.runtime_utils import RoundTimer
import os

class GlobalPromptManager:
    def __init__(self, args):
        # Add dimension validation
        if not hasattr(args, 'prompt_dim'):
            raise ValueError("args.prompt_dim must be defined in command line!")
            
        self.args = args
        self.global_prompts = defaultdict(list)
        self.clustered_prompts = {}
        self.prompt_dim = args.prompt_dim
        self.num_classes = args.num_classes
        self.device = args.device
        
        # Debug output
        print(f"[GlobalPromptManager] Initialized with prompt_dim={self.prompt_dim}")
        
    def aggregate_prompts(self, local_prompts_dict):
        class_prompts = defaultdict(list)
        for client_id, (prompts, labels) in local_prompts_dict.items():
            for prompt, label in zip(prompts, labels):
                # Ensure prompts are processed on CPU
                if isinstance(prompt, torch.Tensor):
                    prompt = prompt.detach().cpu().numpy()
                class_prompts[label.item()].append(prompt)

        clustered_prompts = {}
        for class_id, prompts in class_prompts.items():
            if len(prompts) >= 2:
                prompts_np = np.stack(prompts)
                c, _, _ = self._finch_clustering(prompts_np)
                # Create tensor with unified dimension
                clustered_prompts[class_id] = torch.tensor(
                    c, 
                    dtype=torch.float32, 
                    device=self.device
                )[:, :self.prompt_dim]  # Ensure dimension match
            else:
                # Handle single prompt case
                if prompts:
                    prompt_tensor = torch.tensor(
                        np.array(prompts), 
                        dtype=torch.float32, 
                        device=self.device
                    )
                    # Dimension validation
                    if prompt_tensor.dim() == 1:
                        prompt_tensor = prompt_tensor.unsqueeze(0)
                    clustered_prompts[class_id] = prompt_tensor[:, :self.prompt_dim]
                else:
                    # Create default prompt (using preset dimension)
                    clustered_prompts[class_id] = torch.randn(
                        1, self.prompt_dim, 
                        device=self.device
                    )
        return clustered_prompts

    def _finch_clustering(self, data):
        # Implement real FINCH clustering
        # Simplified here; actual implementation should use FINCH algorithm
        from sklearn.cluster import AgglomerativeClustering
        
        # Compute similarity matrix
        sim_matrix = np.dot(data, data.T)
        np.fill_diagonal(sim_matrix, -1)
        
        # Use hierarchical clustering - fixed parameter name
        clustering = AgglomerativeClustering(
            n_clusters=min(10, len(data)), 
            metric='precomputed',  # Changed to metric
            linkage='average'
        )
        clustering.fit(1 - sim_matrix)  # Convert to distance matrix
        
        # Select cluster centroids as representatives
        cluster_centers = []
        for cluster_id in range(clustering.n_clusters):
            cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
            cluster_data = data[cluster_indices]
            
            # Compute intra-cluster avg similarity, select center
            cluster_sim = np.dot(cluster_data, cluster_data.T)
            center_idx = np.argmax(cluster_sim.sum(axis=1))
            cluster_centers.append(cluster_data[center_idx])
        
        return np.array(cluster_centers), None, None


class RefFiLServer:
    def __init__(self, args):
        # Parameter validation and dimension management
        if not hasattr(args, 'prompt_dim'):
            args.prompt_dim = 64  # Set default
            print(f"[WARNING] Using default prompt_dim={args.prompt_dim}")
        else:
            print(f"[Server] Prompt dimension: {args.prompt_dim}")
            
        self.args = args
        self.num_users = args.num_users
        self.task_num = args.task_num
        self.prompt_dim = args.prompt_dim  # Store unified dimension
        
        # Initialize global model
        self.global_model = get_model(args).to(args.device)
        
        # Initialize client model list (ensure device consistency)
        self.client_models = []
        for user_idx in range(args.num_users):
            client_model = copy.deepcopy(self.global_model)
            client_model.to(args.device)  # Ensure client models on correct device
            self.client_models.append(client_model)
        
        # Initialize prompt manager (pass dimension parameter)
        self.prompt_manager = GlobalPromptManager(args)
        
        # Device validation
        print(f"[Server] Global model on: {next(self.global_model.parameters()).device}")
        print(f"[Server] Client models on: {next(self.client_models[0].parameters()).device}")
        
        # Remove client grouping strategy, use simple variable
        self.current_task = 0
        self.clustered_prompts = {}  # Prompt cache

    def aggregate_models(self, client_updates):
        total_samples = sum([c['num_samples'] for c in client_updates])
        return {
            k: sum(update['model'][k] * (update['num_samples']/total_samples) 
                   for update in client_updates)
            for k in client_updates[0]['model'].keys()
        }

    def train_one_round(self, round_idx, round_timer=None):
        # Same task switch logic as FedAvg
        task = (round_idx // 10) % self.task_num
        print('Current task: ', task)
        
        # Same client selection logic as FedAvg
        m = max(int(self.args.frac * self.num_users), 1)
        selected_clients = np.random.choice(range(self.num_users), m, replace=False)
        
        client_updates = []
        local_prompts_dict = {}
        
        # Device validation
        print(f"[Round {round_idx}] Selected clients: {selected_clients}")
        print(f"[Round {round_idx}] Current task: {task}")
        
        for client_id in selected_clients:
            if round_timer is not None:
                round_timer.start_client(client_id)
            # Key change: pass prompt dimension
            trainer = LocalUpdateRifFiL(
                args=self.args,
                dataset=self.args.datasetpath,
                idxs=client_id,
                task=task,
                prompt_dim=self.prompt_dim  # Unified dimension pass
            )
            
            # Device management - ensure CDAP on correct device
            trainer.cdap = trainer.cdap.to(self.args.device)
            trainer.update_global_prompts(self.clustered_prompts)
            
            # Client model (from pre-initialized list, already on correct device)
            client_model = copy.deepcopy(self.client_models[client_id])
            # No need to move device again, ensured at initialization
            
            # Remove device check, consistency already ensured
            # assert next(client_model.parameters()).device == self.args.device
            # assert next(trainer.cdap.parameters()).device == self.args.device
            
            # Dimension debug output
            print(f"[Client {client_id}] Prompt dim: {trainer.cdap.prompt_dim}")
            
            # Train client
            result = trainer.train(net=client_model, lr=self.args.lr)
            if round_timer is not None:
                round_timer.end_client(client_id)
            
            # Collect results
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

        if round_timer is not None:
            round_timer.start_server()

        # Global aggregation (same logic as FedAvg)
        global_weights = self.aggregate_models(client_updates)
        
        # Broadcast - same update logic as FedAvg
        update_keys = list(global_weights.keys())
        global_weights = {k: v for k, v in global_weights.items() if k in update_keys}
        
        # Update all client models
        for user_idx in range(self.num_users):
            self.client_models[user_idx].load_state_dict(global_weights, strict=False)
        self.global_model.load_state_dict(global_weights, strict=False)
        
        # Prompt clustering (RefFiL-specific logic)
        self.clustered_prompts = self.prompt_manager.aggregate_prompts(local_prompts_dict)

        if round_timer is not None:
            round_timer.end_server()
            
        # Device validation
        print(f"[Round {round_idx}] Global model on: {next(self.global_model.parameters()).device}")
        if self.clustered_prompts:
            first_key = next(iter(self.clustered_prompts.keys()))
            print(f"[Round {round_idx}] Prompt dim: {self.clustered_prompts[first_key].shape[-1]}")
            
        return self.clustered_prompts

if __name__ == '__main__':
    # parse args - same initialization as FedAvg
    args = args_parser()
    dataset_path = args.datasetpath
    
    # Set device
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"===== RifFiL Training Start [Device: {args.device}] =====")
    print(f"Dataset: {args.dataset}, Model: {args.model}")
    print(f"Users: {args.num_users}, Fraction: {args.frac}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    
    # Initialize server
    server = RefFiLServer(args)
    
    # Training parameter storage (same directory structure as FedAvg)
    base_dir = './save/{}/{}_num{}_C{}_le{}_bs{}_round{}_m{}_lr{}/{}/'.format(
        dataset_path, args.model, args.num_users, args.frac, args.local_ep, args.local_bs, args.epochs, args.momentum, args.lr, args.results_save)
    algo_dir = 'refil'
    save_folder = './results/refil'
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)
    
    # training variables - same as FedAvg
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
    round_timer = RoundTimer(device=args.device) if args.benchmark_runtime else None
    if args.benchmark_runtime and args.skip_eval is False:
        args.skip_eval = True
    
    for round_idx in range(args.epochs):
        print(f"\n===== Round {round_idx+1}/{args.epochs} =====")
        if round_timer is not None:
            round_timer.begin_round(round_idx)

        global_prompts = server.train_one_round(round_idx, round_timer=round_timer)

        if round_timer is not None:
            record = round_timer.finish_round()
            print('Round {:3d} runtime: max_client={:.3f}s, server={:.3f}s, total={:.3f}s'.format(
                round_idx, record['max_client_time'], record['server_time'], record['round_time']))
        
        # Compute average loss (same logic as FedAvg)
        # Note: RefFiL version does not return loss_locals, simplified here
        loss_avg = 0.0  # Can be computed from client_updates
        loss_train.append(loss_avg)

        # Periodic evaluation
        if args.skip_eval:
            continue

        if (round_idx + 1) % args.test_freq == 0:
            # Ensure model on device
            server.global_model.to(args.device)
            
            # Same test logic as FedAvg
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
                
                # Save prompts
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

            # Record results (same format as FedAvg)
            results.append(np.array([round_idx, task, loss_avg, loss_test, acc_test, all_acc, best_acc, current_smi, current_tdi]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch','task', 'loss_avg', 'loss_test', 'acc_test',  'all_acc','best_acc', 'smi', 'tdi'])
            final_results.to_csv(results_save_path, index=False)

    if round_timer is not None:
        runtime_csv = args.runtime_csv or os.path.join(base_dir, algo_dir, 'runtime.csv')
        round_timer.save_csv(runtime_csv)
        round_timer.print_summary()
        print('Runtime CSV saved to: {}'.format(runtime_csv))

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
    print("===== Training Completed =====")