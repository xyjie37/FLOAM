"""Shared continual-learning metrics for FedAvg and FedProc."""

import numpy as np
import pandas as pd


def compute_continual_metrics(task_accuracy_matrix, final_stage=None):
    """Compute task-macro ACC, clipped ARF, and signed forgetting."""
    matrix = np.asarray(task_accuracy_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('The task accuracy matrix must be square.')

    if final_stage is None:
        final_stage = matrix.shape[0] - 1
    if not 0 <= final_stage < matrix.shape[0]:
        raise ValueError('final_stage is outside the task accuracy matrix.')

    current_accuracies = matrix[final_stage, :final_stage + 1]
    if np.isnan(current_accuracies).any():
        raise ValueError(
            'The selected stage must contain every learned-task accuracy.')

    details = []
    for task_id in range(final_stage):
        history = matrix[task_id:final_stage, task_id]
        history = history[~np.isnan(history)]
        if history.size == 0:
            raise ValueError(
                'No prior accuracy found for task {}.'.format(task_id))
        prior_peak = float(np.max(history))
        current_acc = float(current_accuracies[task_id])
        signed_forgetting = prior_peak - current_acc
        details.append({
            'task': int(task_id),
            'prior_peak_acc_percent': prior_peak,
            'final_acc_percent': current_acc,
            'signed_forgetting_percent_points': signed_forgetting,
            'clipped_forgetting_percent_points': max(
                signed_forgetting, 0.0),
        })

    signed_values = [
        row['signed_forgetting_percent_points'] for row in details]
    clipped_values = [
        row['clipped_forgetting_percent_points'] for row in details]
    metrics = {
        'final_acc_task_macro_percent': float(np.mean(current_accuracies)),
        'arf_clipped_absolute_percent_points': (
            float(np.mean(clipped_values)) if clipped_values else 0.0),
        'mean_signed_forgetting_percent_points': (
            float(np.mean(signed_values)) if signed_values else 0.0),
        'arf_definition': (
            'Mean over previously learned tasks of max(0, prior peak '
            'accuracy minus current-stage accuracy).'),
        'forgetting_definition': (
            'Mean over previously learned tasks of prior peak accuracy '
            'minus current-stage accuracy; negative values are retained.'),
    }
    return metrics, pd.DataFrame(details)
