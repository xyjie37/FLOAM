"""Optional correctness-validation helpers for FedProc experiments."""

import csv
import json
import os
import random
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def preserve_rng_state(enabled=True):
    """Prevent auxiliary data passes from changing subsequent training."""
    if not enabled:
        yield
        return
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None,
    }
    try:
        yield
    finally:
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.random.set_rng_state(state['torch'])
        if state['cuda'] is not None:
            torch.cuda.set_rng_state_all(state['cuda'])


def tensor_shape_manifest(tensors):
    return {
        str(key): list(torch.as_tensor(value).shape)
        for key, value in sorted(tensors.items(), key=lambda item: str(item[0]))
        if value is not None
    }


def independently_recompute_prototypes(net, data_loader, device):
    """Recompute exact class means independently from the upload code."""
    original_mode = net.training
    sums, counts = {}, {}
    with preserve_rng_state():
        net.eval()
        try:
            with torch.no_grad():
                for images, labels in data_loader:
                    images, labels = images.to(device), labels.to(device)
                    z = net(images, returnFeature=True)[1]
                    for label in labels.unique():
                        class_id = int(label.item())
                        class_z = z[labels == label]
                        sums[class_id] = sums.get(
                            class_id, torch.zeros_like(class_z[0]).cpu()
                        ) + class_z.sum(dim=0).cpu()
                        counts[class_id] = counts.get(class_id, 0) + len(class_z)
        finally:
            net.train(original_mode)
    return {class_id: sums[class_id] / counts[class_id] for class_id in sums}


def prototype_max_abs_error(uploaded, expected):
    if set(uploaded) != set(expected):
        return float('inf')
    return max((
        float((uploaded[class_id].detach().cpu()
               - expected[class_id].detach().cpu())
              .abs().max().item())
        for class_id in uploaded
    ), default=0.0)


class FedProcValidationRecorder:
    """Write the detailed records requested by the correctness checklist."""

    def __init__(self, run_dir, enabled):
        self.enabled = bool(enabled)
        self.loss_records = []
        self.downlink_path = os.path.join(run_dir, 'validation_downlink.jsonl')
        self.upload_path = os.path.join(
            run_dir, 'validation_client_upload.jsonl')
        self.loss_path = os.path.join(
            run_dir, 'validation_loss_components.csv')
        if self.enabled:
            for path in (self.downlink_path, self.upload_path):
                open(path, 'w', encoding='utf-8').close()

    @staticmethod
    def _append_jsonl(path, record):
        with open(path, 'a', encoding='utf-8') as output_file:
            output_file.write(json.dumps(record, sort_keys=True) + '\n')

    def log_downlink(self, round_idx, task, clients, model, prototypes):
        if not self.enabled:
            return
        record = {
            'round': int(round_idx),
            'task': int(task),
            'selected_clients': [int(client) for client in clients],
            'model': tensor_shape_manifest(model.state_dict()),
            'global_prototypes': tensor_shape_manifest(prototypes),
        }
        self._append_jsonl(self.downlink_path, record)
        print('Validation downlink round {}: model={}, global_prototypes={}'
              .format(round_idx, json.dumps(record['model'], sort_keys=True),
                      json.dumps(record['global_prototypes'], sort_keys=True)))

    def log_client(self, round_idx, task, client_id, net, local_update,
                   prototypes, metrics, transmission_enabled, device):
        if not self.enabled:
            return
        expected = independently_recompute_prototypes(
            net, local_update.ldr_train, device
        ) if transmission_enabled else {}
        error = prototype_max_abs_error(prototypes, expected)
        detached = {
            str(class_id): bool(prototype.requires_grad)
            for class_id, prototype in sorted(prototypes.items())
        }
        if error > 1e-5 or any(detached.values()):
            raise AssertionError(
                'Client {} prototype validation failed (error={}).'.format(
                    client_id, error))
        record = {
            'round': int(round_idx),
            'task': int(task),
            'client': int(client_id),
            'local_prototypes': tensor_shape_manifest(prototypes),
            'prototype_requires_grad': detached,
            'independent_max_abs_error': float(error),
            'tolerance': 1e-5,
        }
        self._append_jsonl(self.upload_path, record)
        print('Validation upload round {}, client {}: local_prototypes={}, '
              'max_abs_error={:.3e}'.format(
                  round_idx, client_id,
                  json.dumps(record['local_prototypes'], sort_keys=True), error))
        self.loss_records.append({
            'round': int(round_idx), 'task': int(task),
            'client': int(client_id), **metrics,
        })

    def finish_round(self, round_idx):
        if not self.enabled:
            return
        with open(self.loss_path, 'w', newline='', encoding='utf-8') as output:
            writer = csv.DictWriter(output, fieldnames=self.loss_records[0])
            writer.writeheader()
            writer.writerows(self.loss_records)
        rows = [row for row in self.loss_records if row['round'] == round_idx]
        mean = lambda key: float(np.mean([row[key] for row in rows]))
        print('Validation loss round {}: alpha={:.6f}, alpha*L_gpc={:.6f}, '
              '(1-alpha)*L_ce={:.6f}, total={:.6f}'.format(
                  round_idx, mean('alpha_t'), mean('average_weighted_gpc'),
                  mean('average_weighted_ce'), mean('average_total_loss')))
