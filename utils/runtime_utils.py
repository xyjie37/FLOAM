#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import time

import numpy as np
import pandas as pd
import torch


def sync_device(device=None):
    """Ensure GPU kernels finish before measuring elapsed time."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, torch.device) and device.type == 'cuda':
        torch.cuda.synchronize(device)


class RoundTimer:
    """Track per-round client/server runtime for serial FL simulation."""

    def __init__(self, device=None):
        self.device = device
        self.rounds = []
        self._round_idx = None
        self._client_times = {}
        self._client_start = None
        self._server_start = None
        self._server_time = 0.0

    def begin_round(self, round_idx):
        self._round_idx = round_idx
        self._client_times = {}
        self._server_time = 0.0
        self._client_start = None
        self._server_start = None

    def start_client(self, client_id):
        sync_device(self.device)
        self._client_start = time.perf_counter()

    def end_client(self, client_id):
        sync_device(self.device)
        if self._client_start is None:
            raise RuntimeError(f'end_client({client_id}) called before start_client')
        self._client_times[int(client_id)] = time.perf_counter() - self._client_start
        self._client_start = None

    def start_server(self):
        sync_device(self.device)
        self._server_start = time.perf_counter()

    def end_server(self):
        sync_device(self.device)
        if self._server_start is None:
            raise RuntimeError('end_server() called before start_server()')
        self._server_time = time.perf_counter() - self._server_start
        self._server_start = None

    def finish_round(self):
        if self._round_idx is None:
            raise RuntimeError('finish_round() called before begin_round()')

        max_client_time = max(self._client_times.values()) if self._client_times else 0.0
        round_time = max_client_time + self._server_time
        selected_clients = sorted(self._client_times.keys())

        record = {
            'round': int(self._round_idx),
            'selected_clients': ','.join(str(c) for c in selected_clients),
            'num_selected_clients': len(selected_clients),
            'max_client_time': max_client_time,
            'server_time': self._server_time,
            'round_time': round_time,
            'client_times_json': json.dumps({str(k): v for k, v in self._client_times.items()}),
        }
        self.rounds.append(record)
        self._round_idx = None
        return record

    def summary(self):
        if not self.rounds:
            return {}

        df = pd.DataFrame(self.rounds)
        return {
            'num_rounds': len(df),
            'avg_max_client_time': float(df['max_client_time'].mean()),
            'avg_server_time': float(df['server_time'].mean()),
            'avg_round_time': float(df['round_time'].mean()),
            'std_round_time': float(df['round_time'].std(ddof=0)) if len(df) > 1 else 0.0,
        }

    def save_csv(self, csv_path):
        if not self.rounds:
            raise RuntimeError('No runtime records to save')

        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        df = pd.DataFrame(self.rounds)
        summary = self.summary()

        summary_row = {
            'round': 'AVG',
            'selected_clients': '',
            'num_selected_clients': summary['num_rounds'],
            'max_client_time': summary['avg_max_client_time'],
            'server_time': summary['avg_server_time'],
            'round_time': summary['avg_round_time'],
            'client_times_json': json.dumps({'std_round_time': summary['std_round_time']}),
        }
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        return summary

    def print_summary(self):
        summary = self.summary()
        if not summary:
            print('[Runtime] No timing records collected.')
            return summary

        print('[Runtime Summary]')
        print('  Rounds measured     : {}'.format(summary['num_rounds']))
        print('  Avg max client time : {:.4f}s'.format(summary['avg_max_client_time']))
        print('  Avg server time     : {:.4f}s'.format(summary['avg_server_time']))
        print('  Avg round time      : {:.4f}s'.format(summary['avg_round_time']))
        print('  Std round time      : {:.4f}s'.format(summary['std_round_time']))
        return summary
