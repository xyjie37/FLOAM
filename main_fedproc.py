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
from models.Update import LocalUpdateFedProc
from models.test import (
    test_img,
    test_img_local,
    test_img_local_all,
    compute_smi_tdi_for_task,
    test_global_model_on_task,
)
from utils.runtime_utils import sync_device
from utils.fedproc_validation import (
    FedProcValidationRecorder,
    preserve_rng_state,
)
from utils.continual_metrics import compute_continual_metrics

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


def prototype_payload_bytes(prototypes):
    """Return the logical payload size of a class-prototype dictionary."""
    return int(sum(
        torch.as_tensor(prototype).numel()
        * torch.as_tensor(prototype).element_size()
        for prototype in prototypes.values()
        if prototype is not None
    ))


def aggregate_global_prototypes(
        previous_global_prototypes, client_prototype_uploads, round_idx,
        phase='training'):
    """Equally average client prototypes class by class.

    Only clients that uploaded a class participate in that class average.
    Previously known classes with no upload in this round are retained.
    """
    next_global_prototypes = {
        int(class_label): torch.as_tensor(prototype).detach().cpu().clone()
        for class_label, prototype in previous_global_prototypes.items()
    }
    uploads_by_class = defaultdict(list)

    for client_id, local_prototypes in client_prototype_uploads:
        for class_label, prototype in local_prototypes.items():
            class_label = int(class_label)
            prototype = torch.as_tensor(prototype).detach().cpu().reshape(-1)
            uploads_by_class[class_label].append((int(client_id), prototype))

    status_records = []
    for class_label in sorted(uploads_by_class):
        class_uploads = uploads_by_class[class_label]
        uploaded_prototypes = [item[1] for item in class_uploads]
        expected_shape = uploaded_prototypes[0].shape
        if any(prototype.shape != expected_shape
               for prototype in uploaded_prototypes[1:]):
            raise ValueError(
                'Prototype shape mismatch among clients for class {}.'.format(
                    class_label))
        if (class_label in next_global_prototypes
                and next_global_prototypes[class_label].shape != expected_shape):
            raise ValueError(
                'Prototype shape changed for class {}: previous {}, new {}.'
                .format(
                    class_label,
                    tuple(next_global_prototypes[class_label].shape),
                    tuple(expected_shape),
                ))

        next_global_prototypes[class_label] = torch.stack(
            uploaded_prototypes, dim=0).mean(dim=0)
        uploading_clients = [item[0] for item in class_uploads]
        status_records.append({
            'round': int(round_idx),
            'phase': str(phase),
            'class': int(class_label),
            'status': 'updated',
            'uploading_client_count': int(len(uploading_clients)),
            'uploading_clients': ','.join(
                str(client_id) for client_id in uploading_clients),
        })

    retained_classes = sorted(
        set(next_global_prototypes) - set(uploads_by_class))
    for class_label in retained_classes:
        status_records.append({
            'round': int(round_idx),
            'phase': str(phase),
            'class': int(class_label),
            'status': 'retained',
            'uploading_client_count': 0,
            'uploading_clients': '',
        })

    return next_global_prototypes, status_records


def run_pre_round_bootstrap(
        args, dataset_path, net_glob, selected_clients, task,
        local_update_cls=LocalUpdateFedProc, preserve_rng=False):
    """Build initial global prototypes without updating the global model."""
    selected_clients = [int(client_id) for client_id in selected_clients]
    if not selected_clients:
        raise ValueError('Pre-round bootstrap requires at least one client.')

    client_prototype_uploads = []
    feature_forward_seconds = 0.0
    original_training_mode = net_glob.training
    with preserve_rng_state(preserve_rng):
        try:
            for client_id in selected_clients:
                local = local_update_cls(
                    args=args, dataset=dataset_path,
                    idxs=client_id, task=task)
                sync_device(getattr(args, 'device', None))
                feature_forward_start = time.perf_counter()
                local_prototypes = local.compute_local_prototypes(net_glob)
                sync_device(getattr(args, 'device', None))
                feature_forward_seconds += (
                    time.perf_counter() - feature_forward_start)
                client_prototype_uploads.append(
                    (client_id, local_prototypes))
        finally:
            net_glob.train(original_training_mode)

    global_prototypes, status_records = aggregate_global_prototypes(
        previous_global_prototypes={},
        client_prototype_uploads=client_prototype_uploads,
        round_idx=0,
        phase='bootstrap',
    )
    upload_bytes = int(sum(
        prototype_payload_bytes(local_prototypes)
        for _, local_prototypes in client_prototype_uploads
    ))
    return (
        global_prototypes, status_records, upload_bytes,
        feature_forward_seconds)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    if args.model != 'fedproc_resnet18':
        raise ValueError(
            'main_fedproc.py requires --model fedproc_resnet18.')
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
    prototype_transmission_enabled = (
        args.fedproc_ablation != 'no_proto_alpha0')
    alpha_override = (
        0.0 if args.fedproc_ablation != 'none' else None)

    base_dir = './save/{}/{}_num{}_C{}_le{}_bs{}_round{}_m{}_lr{}/{}/'.format(
        dataset_path, args.model, args.num_users, args.frac, args.local_ep, args.local_bs, args.epochs, args.momentum, args.lr, args.results_save)
    algo_dir = 'fedproc'
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
    run_config['pre_round_bootstrap_enabled'] = bool(
        prototype_transmission_enabled)
    run_config['pre_round_bootstrap_schedule_source'] = 'client_schedule[0]'
    run_config['prototype_transmission_enabled'] = bool(
        prototype_transmission_enabled)
    run_config['fedproc_alpha_override'] = alpha_override
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
    prototype_status_save_path = os.path.join(
        run_dir, 'prototype_round_status.csv')
    bootstrap_metrics_save_path = os.path.join(
        run_dir, 'bootstrap_metrics.json')
    bootstrap_prototypes_save_path = os.path.join(
        run_dir, 'bootstrap_global_prototypes.pt')
    validator = FedProcValidationRecorder(
        run_dir, args.fedproc_validation_logging)

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
    global_prototypes = {}
    prototype_status_records = []

    bootstrap_seconds = 0.0
    bootstrap_feature_forward_seconds = 0.0
    bootstrap_prototype_upload_bytes = 0
    bootstrap_metrics = {
        'executed': False,
        'execution_count': 0,
        'schedule_source': 'client_schedule[0]',
    }
    if args.epochs > 0 and prototype_transmission_enabled:
        bootstrap_clients = [
            int(client_id)
            for client_id in client_schedule[0]['clients']
        ]
        bootstrap_task = int(client_schedule[0]['task'])
        if bootstrap_task != 0:
            raise ValueError(
                'Pre-round bootstrap must use task 0, got task {}.'.format(
                    bootstrap_task))

        print(
            'Pre-round bootstrap, task {}, selected clients: {}'.format(
                bootstrap_task, bootstrap_clients))
        sync_device(args.device)
        bootstrap_wall_start = time.perf_counter()
        (
            global_prototypes,
            bootstrap_status_records,
            bootstrap_prototype_upload_bytes,
            bootstrap_feature_forward_seconds,
        ) = run_pre_round_bootstrap(
            args=args,
            dataset_path=dataset_path,
            net_glob=net_glob,
            selected_clients=bootstrap_clients,
            task=bootstrap_task,
            preserve_rng=args.fedproc_ablation != 'none',
        )
        sync_device(args.device)
        bootstrap_seconds = time.perf_counter() - bootstrap_wall_start

        prototype_status_records.extend(bootstrap_status_records)
        for status_row in bootstrap_status_records:
            print(
                'Pre-round bootstrap initialized global prototype class {} '
                'from {} client(s): [{}].'.format(
                    status_row['class'],
                    status_row['uploading_client_count'],
                    status_row['uploading_clients'],
                ))
        pd.DataFrame(prototype_status_records).to_csv(
            prototype_status_save_path, index=False)
        torch.save(global_prototypes, bootstrap_prototypes_save_path)

        bootstrap_metrics = {
            'executed': True,
            'execution_count': 1,
            'task': bootstrap_task,
            'selected_clients': bootstrap_clients,
            'selected_client_count': len(bootstrap_clients),
            'schedule_source': 'client_schedule[0]',
            'uses_untrained_unified_global_model': True,
            'local_training_performed': False,
            'future_task_data_used': False,
            'model_download_reused_by_round_zero': True,
            'extra_model_communication_bytes': 0,
            'prototype_upload_bytes': int(
                bootstrap_prototype_upload_bytes),
            'prototype_download_bytes': 0,
            'initial_global_prototype_class_count': int(
                len(global_prototypes)),
            'wall_seconds': float(bootstrap_seconds),
            'feature_forward_seconds': float(
                bootstrap_feature_forward_seconds),
        }
        print(
            'Pre-round bootstrap completed: classes={}, '
            'prototype_upload_bytes={}, feature_forward={:.3f}s, '
            'time={:.3f}s.'.format(
                len(global_prototypes),
                bootstrap_prototype_upload_bytes,
                bootstrap_feature_forward_seconds,
                bootstrap_seconds,
            ))
    elif args.epochs > 0:
        bootstrap_metrics.update({
            'disabled_reason': 'fedproc_ablation=no_proto_alpha0',
            'prototype_transmission_enabled': False,
        })
        print(
            'Pre-round bootstrap disabled by '
            '--fedproc_ablation no_proto_alpha0.')

    with open(
            bootstrap_metrics_save_path,
            'w', encoding='utf-8') as bootstrap_metrics_file:
        json.dump(
            bootstrap_metrics,
            bootstrap_metrics_file,
            indent=2,
            sort_keys=True,
        )

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
        round_prototype_download_bytes = int(
            len(idxs_users) * prototype_payload_bytes(global_prototypes))
        round_bootstrap_prototype_upload_bytes = int(
            bootstrap_prototype_upload_bytes if iter == 0 else 0)
        round_prototype_upload_bytes = int(
            round_bootstrap_prototype_upload_bytes)
        client_prototype_uploads = []

        task=(iter//10)%task_num  # Task switch every 10 rounds
        validator.log_downlink(
            iter, task, idxs_users, net_glob, global_prototypes)
        print('Round {}, task {}, selected clients: {}'.format(
            iter, task, idxs_users.tolist()))
        # Local Updates
        for idx in idxs_users:
            # Dataset name, index
            local = LocalUpdateFedProc(
                args=args, dataset=dataset_path, idxs=idx, task=task)
            net_local = copy.deepcopy(net_local_list[idx])
            w_local, loss, local_prototypes = local.train(
                net=net_local.to(args.device),
                lr=lr,
                global_prototypes=global_prototypes,
                global_round=iter,
                total_rounds=args.epochs,
                alpha_override=alpha_override,
                compute_prototypes=prototype_transmission_enabled,
                preserve_prototype_rng=args.fedproc_ablation != 'none',
            )

            loss_locals.append(copy.deepcopy(loss))
            client_prototype_uploads.append(
                (int(idx), local_prototypes))
            round_prototype_upload_bytes += prototype_payload_bytes(
                local_prototypes)

            validator.log_client(
                iter, task, idx, net_local, local, local_prototypes,
                {'ablation': args.fedproc_ablation,
                 **local.last_training_metrics},
                prototype_transmission_enabled, args.device)

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]

        # Aggregation
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)

        global_prototypes, round_prototype_status = (
            aggregate_global_prototypes(
                global_prototypes,
                client_prototype_uploads,
                round_idx=iter,
            )
        )
        prototype_status_records.extend(round_prototype_status)
        for status_row in round_prototype_status:
            if status_row['status'] == 'updated':
                print(
                    'Round {}, global prototype class {} updated from {} '
                    'client(s): [{}].'.format(
                        iter,
                        status_row['class'],
                        status_row['uploading_client_count'],
                        status_row['uploading_clients'],
                    ))
            else:
                print(
                    'Round {}, global prototype class {} retained: '
                    'no client uploaded this class.'.format(
                        iter, status_row['class']))
        pd.DataFrame(prototype_status_records).to_csv(
            prototype_status_save_path, index=False)

        validator.finish_round(iter)

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

            stage_continual_metrics, _ = compute_continual_metrics(
                task_accuracy_matrix, final_stage=stage_idx)
            stage_acc = stage_continual_metrics[
                'final_acc_task_macro_percent']
            stage_metrics.append({
                'stage': int(stage_idx),
                'round': int(iter + 1),
                'stage_acc': stage_acc,
                'stage_arf_clipped_absolute_percent_points': (
                    stage_continual_metrics[
                        'arf_clipped_absolute_percent_points']),
                'stage_mean_signed_forgetting_percent_points': (
                    stage_continual_metrics[
                        'mean_signed_forgetting_percent_points']),
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
            print(
                'Stage {}: ACC {:.2f}%, ARF {:.2f}, signed forgetting '
                '{:.2f} percentage points'.format(
                    stage_idx,
                    stage_acc,
                    stage_continual_metrics[
                        'arf_clipped_absolute_percent_points'],
                    stage_continual_metrics[
                        'mean_signed_forgetting_percent_points'],
                ))

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
                    'Final task-macro ACC: {:.2f}%, ARF: {:.2f}, '
                    'mean signed forgetting: {:.2f} percentage points'.format(
                        continual_metrics[
                            'final_acc_task_macro_percent'],
                        continual_metrics[
                            'arf_clipped_absolute_percent_points'],
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
            'bootstrap_wall_seconds': float(
                bootstrap_seconds if iter == 0 else 0.0),
            'bootstrap_feature_forward_seconds': float(
                bootstrap_feature_forward_seconds
                if iter == 0 else 0.0),
            'total_with_bootstrap_seconds': float(
                round_total_seconds
                + (bootstrap_seconds if iter == 0 else 0.0)),
            'model_upload_bytes': int(model_upload_bytes_per_round),
            'model_download_bytes': int(model_download_bytes_per_round),
            'model_total_bytes': int(
                model_upload_bytes_per_round
                + model_download_bytes_per_round),
            'prototype_upload_bytes': int(round_prototype_upload_bytes),
            'bootstrap_prototype_upload_bytes': int(
                round_bootstrap_prototype_upload_bytes),
            'prototype_download_bytes': int(round_prototype_download_bytes),
            'prototype_total_bytes': int(
                round_prototype_upload_bytes
                + round_prototype_download_bytes),
            'prototype_extra_bytes': int(
                round_prototype_upload_bytes
                + round_prototype_download_bytes),
            'total_communication_bytes': int(
                model_upload_bytes_per_round
                + model_download_bytes_per_round
                + round_prototype_upload_bytes
                + round_prototype_download_bytes),
            'global_prototype_class_count': int(len(global_prototypes)),
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
                'bootstrap_wall_seconds': float(sum(
                    row['bootstrap_wall_seconds']
                    for row in current_stage_rounds)),
                'bootstrap_feature_forward_seconds': float(sum(
                    row['bootstrap_feature_forward_seconds']
                    for row in current_stage_rounds)),
                'total_with_bootstrap_seconds': float(sum(
                    row['total_with_bootstrap_seconds']
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
                'prototype_upload_bytes': int(sum(
                    row['prototype_upload_bytes']
                    for row in current_stage_rounds)),
                'bootstrap_prototype_upload_bytes': int(sum(
                    row['bootstrap_prototype_upload_bytes']
                    for row in current_stage_rounds)),
                'prototype_download_bytes': int(sum(
                    row['prototype_download_bytes']
                    for row in current_stage_rounds)),
                'prototype_total_bytes': int(sum(
                    row['prototype_total_bytes']
                    for row in current_stage_rounds)),
                'prototype_extra_bytes': int(sum(
                    row['prototype_extra_bytes']
                    for row in current_stage_rounds)),
                'total_communication_bytes': int(sum(
                    row['total_communication_bytes']
                    for row in current_stage_rounds)),
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
    total_model_upload_bytes = int(sum(
        row['model_upload_bytes'] for row in round_runtime_records))
    total_model_download_bytes = int(sum(
        row['model_download_bytes'] for row in round_runtime_records))
    total_prototype_upload_bytes = int(sum(
        row['prototype_upload_bytes'] for row in round_runtime_records))
    total_prototype_download_bytes = int(sum(
        row['prototype_download_bytes'] for row in round_runtime_records))
    total_model_communication_bytes = int(
        total_model_upload_bytes + total_model_download_bytes)
    total_prototype_communication_bytes = int(
        total_prototype_upload_bytes + total_prototype_download_bytes)

    resource_metrics = {
        'fedproc_ablation': args.fedproc_ablation,
        'prototype_transmission_enabled': bool(
            prototype_transmission_enabled),
        'validation_logging_enabled': bool(
            args.fedproc_validation_logging),
        'completed_rounds': int(len(round_runtime_records)),
        'program_wall_seconds': float(
            time.perf_counter() - program_wall_start),
        'training_loop_wall_seconds': float(training_loop_seconds),
        'training_and_bootstrap_wall_seconds': float(
            training_loop_seconds + bootstrap_seconds),
        'summed_round_training_seconds': total_training_seconds,
        'summed_round_evaluation_seconds': total_evaluation_seconds,
        'summed_round_total_seconds': total_round_seconds,
        'pre_round_bootstrap_wall_seconds': float(bootstrap_seconds),
        'pre_round_bootstrap_feature_forward_seconds': float(
            bootstrap_feature_forward_seconds),
        'pre_round_bootstrap_execution_count': int(
            bootstrap_metrics['execution_count']),
        'pre_round_bootstrap_prototype_upload_bytes': int(
            bootstrap_prototype_upload_bytes),
        'peak_gpu_memory_allocated_bytes': peak_memory_allocated_bytes,
        'peak_gpu_memory_reserved_bytes': peak_memory_reserved_bytes,
        'model_state_payload_bytes': model_state_bytes,
        'trainable_parameter_bytes': trainable_parameter_bytes,
        'model_upload_bytes': total_model_upload_bytes,
        'model_download_bytes': total_model_download_bytes,
        'model_total_communication_bytes': total_model_communication_bytes,
        'prototype_upload_bytes': total_prototype_upload_bytes,
        'prototype_download_bytes': total_prototype_download_bytes,
        'prototype_total_communication_bytes': (
            total_prototype_communication_bytes),
        'prototype_extra_communication_bytes': (
            total_prototype_communication_bytes),
        'total_communication_bytes': int(
            total_model_communication_bytes
            + total_prototype_communication_bytes),
        'final_global_prototype_class_count': int(len(global_prototypes)),
        'timing_definition': (
            'Actual serial single-process wall-clock time on the recorded '
            'device. GPU synchronization is performed at timing boundaries. '
            'Per-round training excludes scheduled evaluation; per-round '
            'evaluation is the remaining measured round time. Bootstrap '
            'wall time and its feature-forward subset are recorded '
            'separately and included in training_and_bootstrap_wall_seconds.'
        ),
        'communication_definition': (
            'Logical federated communication: every selected client downloads '
            'one complete model state and uploads one complete model state per '
            'round. In-process broadcasts to unselected model copies are not '
            'counted. '
            + (
                'Every selected FedProc client also downloads all global '
                'prototypes available before its round and uploads only '
                'prototypes for classes present in its current local data. '
                'The one-time pre-round bootstrap reuses the round-zero model '
                'delivery, adds only its initial local-prototype uploads, and '
                'is included in round-zero prototype upload bytes.'
                if prototype_transmission_enabled else
                'Prototype bootstrap, download, upload, and aggregation are '
                'disabled by the no_proto_alpha0 validation ablation.'
            )
        ),
    }
    with open(
            resource_metrics_save_path,
            'w', encoding='utf-8') as resource_file:
        json.dump(resource_metrics, resource_file, indent=2, sort_keys=True)
    print(
        'Resources: loop={:.3f}s, peak_allocated={}, model_comm_bytes={}, '
        'prototype_comm_bytes={}, total_comm_bytes={}'.format(
            training_loop_seconds,
            peak_memory_allocated_bytes,
            resource_metrics['model_total_communication_bytes'],
            resource_metrics['prototype_total_communication_bytes'],
            resource_metrics['total_communication_bytes'],
        ))

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
