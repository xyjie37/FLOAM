#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import hashlib
import json
import os
import pickle
import platform
import random
import shlex
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import (
    test_img,
    test_img_local,
    test_img_local_all,
    compute_smi_tdi_for_task,
    test_global_model_on_task,
)
from utils.runtime_utils import sync_device

import pdb
from collections import defaultdict


def sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, 'rb') as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo_dir, arguments):
    try:
        completed = subprocess.run(
            ['git'] + arguments,
            cwd=repo_dir,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def collect_environment(dataset_path):
    """Collect the code, data, software, and hardware identity of a run."""
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.abspath(os.path.join(
        dataset_path, 'SHA256SUMS.txt'))
    git_status = git_output(repo_dir, ['status', '--porcelain'])

    environment = {
        'platform': platform.platform(),
        'working_directory': os.getcwd(),
        'python_executable': sys.executable,
        'python_version': platform.python_version(),
        'pytorch_version': str(torch.__version__),
        'torchvision_version': str(torchvision.__version__),
        'cuda_available': bool(torch.cuda.is_available()),
        'cuda_runtime_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'git_branch': git_output(
            repo_dir, ['rev-parse', '--abbrev-ref', 'HEAD']),
        'git_commit': git_output(repo_dir, ['rev-parse', 'HEAD']),
        'git_dirty': None if git_status is None else bool(git_status),
        'dataset_path': os.path.abspath(dataset_path),
        'dataset_manifest_path': manifest_path,
        'dataset_manifest_sha256': (
            sha256_file(manifest_path)
            if os.path.isfile(manifest_path) else None
        ),
    }

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(device_index)
        environment.update({
            'gpu_visible_device_index': int(device_index),
            'gpu_name': device_properties.name,
            'gpu_total_memory_bytes': int(device_properties.total_memory),
            'gpu_compute_capability': '{}.{}'.format(
                device_properties.major, device_properties.minor),
        })
    else:
        environment.update({
            'gpu_visible_device_index': None,
            'gpu_name': None,
            'gpu_total_memory_bytes': None,
            'gpu_compute_capability': None,
        })

    return environment


def compute_continual_metrics(task_accuracy_matrix):
    """Compute unambiguous continual-learning metrics from an ACC matrix."""
    matrix = np.asarray(task_accuracy_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('The task accuracy matrix must be square.')

    task_num = matrix.shape[1]
    final_task_accuracies = matrix[-1, :]
    if np.isnan(final_task_accuracies).any():
        raise ValueError(
            'The final matrix row must contain an accuracy for every task.')

    final_acc = float(np.mean(final_task_accuracies))
    forgetting_details = []

    # The final task has no later stage in which forgetting can be observed.
    for task_id in range(task_num - 1):
        prior_history = matrix[task_id:-1, task_id]
        valid_history = prior_history[~np.isnan(prior_history)]
        if valid_history.size == 0:
            raise ValueError(
                'No pre-final accuracy found for task {}.'.format(task_id))

        prior_peak = float(np.max(valid_history))
        final_task_acc = float(final_task_accuracies[task_id])
        signed_forgetting = prior_peak - final_task_acc
        forgetting_details.append({
            'task': int(task_id),
            'prior_peak_acc_percent': prior_peak,
            'final_acc_percent': final_task_acc,
            'signed_forgetting_percent_points': signed_forgetting,
        })

    if forgetting_details:
        mean_signed_forgetting = float(np.mean([
            row['signed_forgetting_percent_points']
            for row in forgetting_details
        ]))
    else:
        mean_signed_forgetting = None

    metrics = {
        'final_acc_task_macro_percent': final_acc,
        'mean_signed_forgetting_percent_points': mean_signed_forgetting,
        'forgetting_definition': (
            'For each non-final task: maximum accuracy after it was learned '
            'and before the final stage, minus its final-stage accuracy. '
            'Negative values are retained.'
        ),
        'arf_status': 'not_computed_formula_not_confirmed',
    }
    return metrics, pd.DataFrame(forgetting_details)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    program_wall_start = time.perf_counter()
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
    os.makedirs(run_dir, exist_ok=True)
    save_folder = os.path.join(run_dir, 'classification_reports')
    os.makedirs(save_folder, exist_ok=True)

    # Preserve the exact interpreter and command-line arguments used for this
    # run.  Use the platform's native quoting so the saved command can be
    # copied back into the same kind of shell.
    command_parts = [sys.executable] + sys.argv
    if os.name == 'nt':
        command_text = subprocess.list2cmdline(command_parts)
    else:
        command_text = shlex.join(command_parts)
    command_path = os.path.join(run_dir, 'run_command.txt')
    with open(command_path, 'w', encoding='utf-8') as command_file:
        command_file.write(command_text + '\n')

    environment = collect_environment(dataset_path)
    environment_path = os.path.join(run_dir, 'environment.json')
    with open(environment_path, 'w', encoding='utf-8') as environment_file:
        json.dump(environment, environment_file, indent=2, sort_keys=True)
    print('Run identity: commit={}, dirty={}, manifest_sha256={}'.format(
        environment['git_commit'],
        environment['git_dirty'],
        environment['dataset_manifest_sha256']))
    print('Environment: Python {}, PyTorch {}, CUDA {}, GPU {}'.format(
        environment['python_version'],
        environment['pytorch_version'],
        environment['cuda_runtime_version'],
        environment['gpu_name']))

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
    if args.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
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
    task_matrix_save_path = os.path.join(
        run_dir, 'task_accuracy_matrix.csv')
    stage_metrics_save_path = os.path.join(run_dir, 'stage_metrics.csv')
    continual_metrics_save_path = os.path.join(
        run_dir, 'continual_metrics.json')
    forgetting_details_save_path = os.path.join(
        run_dir, 'forgetting_details.csv')
    round_runtime_save_path = os.path.join(run_dir, 'round_runtime.csv')
    stage_runtime_save_path = os.path.join(run_dir, 'stage_runtime.csv')
    resource_metrics_save_path = os.path.join(
        run_dir, 'resource_metrics.json')

    model_state_bytes = int(sum(
        tensor.numel() * tensor.element_size()
        for tensor in net_glob.state_dict().values()
    ))
    trainable_parameter_bytes = int(sum(
        parameter.numel() * parameter.element_size()
        for parameter in net_glob.parameters()
        if parameter.requires_grad
    ))
    model_upload_bytes_per_round = clients_per_round * model_state_bytes
    model_download_bytes_per_round = clients_per_round * model_state_bytes

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
    task_accuracy_matrix = np.full((task_num, task_num), np.nan)
    stage_metrics = []
    round_runtime_records = []
    stage_runtime_records = []

    sync_device(args.device)
    training_loop_start = time.perf_counter()

    for iter in range(args.epochs):
        sync_device(args.device)
        round_wall_start = time.perf_counter()
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
        sync_device(args.device)
        round_training_seconds = time.perf_counter() - round_wall_start

        # At the end of each continual-learning stage, evaluate the global
        # model separately on every task seen so far.  These raw values form
        # the accuracy matrix needed for final ACC and forgetting metrics.
        if (iter + 1) % 10 == 0:
            stage_idx = task
            evaluated_samples = 0
            for eval_task in range(stage_idx + 1):
                task_acc, task_loss, task_samples = test_global_model_on_task(
                    net_g=net_glob,
                    dataset=dataset_path,
                    task=eval_task,
                    args=args,
                )
                task_accuracy_matrix[stage_idx, eval_task] = task_acc
                evaluated_samples += task_samples
                print(
                    'Stage {}, task {} global accuracy: {:.2f}% '
                    '(loss {:.4f}, samples {})'.format(
                        stage_idx, eval_task, task_acc, task_loss,
                        task_samples))

            stage_acc = float(np.nanmean(
                task_accuracy_matrix[stage_idx, :stage_idx + 1]))
            stage_metrics.append({
                'stage': int(stage_idx),
                'round': int(iter + 1),
                'stage_acc': stage_acc,
                'evaluated_tasks': int(stage_idx + 1),
                'evaluated_samples': int(evaluated_samples),
            })

            completed_matrix = pd.DataFrame(
                task_accuracy_matrix[:stage_idx + 1],
                columns=['task_{}'.format(i) for i in range(task_num)],
            )
            completed_matrix.insert(
                0, 'round', [(i + 1) * 10 for i in range(stage_idx + 1)])
            completed_matrix.insert(
                0, 'stage', list(range(stage_idx + 1)))
            completed_matrix.to_csv(task_matrix_save_path, index=False)
            pd.DataFrame(stage_metrics).to_csv(
                stage_metrics_save_path, index=False)
            print('Stage {} average accuracy: {:.2f}%'.format(
                stage_idx, stage_acc))

            if stage_idx == task_num - 1:
                continual_metrics, forgetting_details = \
                    compute_continual_metrics(task_accuracy_matrix)
                continual_metrics.update({
                    'final_stage': int(stage_idx),
                    'final_round': int(iter + 1),
                    'evaluated_tasks': int(task_num),
                })
                with open(
                        continual_metrics_save_path,
                        'w', encoding='utf-8') as metrics_file:
                    json.dump(
                        continual_metrics,
                        metrics_file,
                        indent=2,
                        sort_keys=True,
                    )
                forgetting_details.to_csv(
                    forgetting_details_save_path, index=False)
                print(
                    'Final task-macro ACC: {:.2f}%, mean signed forgetting: '
                    '{:.2f} percentage points'.format(
                        continual_metrics[
                            'final_acc_task_macro_percent'],
                        continual_metrics[
                            'mean_signed_forgetting_percent_points'],
                    ))

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

        sync_device(args.device)
        round_total_seconds = time.perf_counter() - round_wall_start
        round_evaluation_seconds = max(
            round_total_seconds - round_training_seconds, 0.0)
        round_runtime_records.append({
            'round': int(iter + 1),
            'task': int(task),
            'selected_clients': ','.join(
                str(client_id) for client_id in idxs_users.tolist()),
            'num_selected_clients': int(len(idxs_users)),
            'training_seconds': float(round_training_seconds),
            'evaluation_seconds': float(round_evaluation_seconds),
            'total_seconds': float(round_total_seconds),
            'model_upload_bytes': int(model_upload_bytes_per_round),
            'model_download_bytes': int(model_download_bytes_per_round),
            'model_total_bytes': int(
                model_upload_bytes_per_round
                + model_download_bytes_per_round),
            'prototype_extra_bytes': 0,
        })
        pd.DataFrame(round_runtime_records).to_csv(
            round_runtime_save_path, index=False)

        if (iter + 1) % 10 == 0:
            current_stage_rounds = round_runtime_records[-10:]
            stage_runtime_records.append({
                'stage': int(task),
                'start_round': int(iter - 8),
                'end_round': int(iter + 1),
                'training_seconds': float(sum(
                    row['training_seconds']
                    for row in current_stage_rounds)),
                'evaluation_seconds': float(sum(
                    row['evaluation_seconds']
                    for row in current_stage_rounds)),
                'total_seconds': float(sum(
                    row['total_seconds']
                    for row in current_stage_rounds)),
                'model_upload_bytes': int(sum(
                    row['model_upload_bytes']
                    for row in current_stage_rounds)),
                'model_download_bytes': int(sum(
                    row['model_download_bytes']
                    for row in current_stage_rounds)),
                'model_total_bytes': int(sum(
                    row['model_total_bytes']
                    for row in current_stage_rounds)),
                'prototype_extra_bytes': 0,
            })
            pd.DataFrame(stage_runtime_records).to_csv(
                stage_runtime_save_path, index=False)

        print(
            'Round {} timing: train={:.3f}s, eval={:.3f}s, total={:.3f}s'.format(
                iter,
                round_training_seconds,
                round_evaluation_seconds,
                round_total_seconds,
            ))

    sync_device(args.device)
    training_loop_seconds = time.perf_counter() - training_loop_start

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

    if args.device.type == 'cuda':
        sync_device(args.device)
        peak_memory_allocated_bytes = int(
            torch.cuda.max_memory_allocated(args.device))
        peak_memory_reserved_bytes = int(
            torch.cuda.max_memory_reserved(args.device))
    else:
        peak_memory_allocated_bytes = None
        peak_memory_reserved_bytes = None

    total_training_seconds = float(sum(
        row['training_seconds'] for row in round_runtime_records))
    total_evaluation_seconds = float(sum(
        row['evaluation_seconds'] for row in round_runtime_records))
    total_round_seconds = float(sum(
        row['total_seconds'] for row in round_runtime_records))
    total_upload_bytes = int(sum(
        row['model_upload_bytes'] for row in round_runtime_records))
    total_download_bytes = int(sum(
        row['model_download_bytes'] for row in round_runtime_records))

    resource_metrics = {
        'completed_rounds': int(len(round_runtime_records)),
        'program_wall_seconds': float(
            time.perf_counter() - program_wall_start),
        'training_loop_wall_seconds': float(training_loop_seconds),
        'summed_round_training_seconds': total_training_seconds,
        'summed_round_evaluation_seconds': total_evaluation_seconds,
        'summed_round_total_seconds': total_round_seconds,
        'peak_gpu_memory_allocated_bytes': peak_memory_allocated_bytes,
        'peak_gpu_memory_reserved_bytes': peak_memory_reserved_bytes,
        'model_state_payload_bytes': model_state_bytes,
        'trainable_parameter_bytes': trainable_parameter_bytes,
        'model_upload_bytes': total_upload_bytes,
        'model_download_bytes': total_download_bytes,
        'model_total_communication_bytes': int(
            total_upload_bytes + total_download_bytes),
        'prototype_extra_communication_bytes': 0,
        'timing_definition': (
            'Actual serial single-process wall-clock time on the recorded '
            'device. GPU synchronization is performed at timing boundaries. '
            'Per-round training excludes scheduled evaluation; per-round '
            'evaluation is the remaining measured round time.'
        ),
        'communication_definition': (
            'Logical federated communication: every selected client downloads '
            'one complete model state and uploads one complete model state per '
            'round. In-process broadcasts to unselected model copies are not '
            'counted. FedAvg has no prototype payload.'
        ),
    }
    with open(
            resource_metrics_save_path,
            'w', encoding='utf-8') as resource_file:
        json.dump(resource_metrics, resource_file, indent=2, sort_keys=True)
    print(
        'Resources: loop={:.3f}s, peak_allocated={}, model_comm_bytes={}'.format(
            training_loop_seconds,
            peak_memory_allocated_bytes,
            resource_metrics['model_total_communication_bytes'],
        ))

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
