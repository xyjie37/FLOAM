import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.continual_metrics import compute_continual_metrics


def test_stage_and_final_metrics():
    matrix = np.array([
        [80.0, np.nan, np.nan],
        [70.0, 75.0, np.nan],
        [85.0, 65.0, 90.0],
    ])

    stage_zero, _ = compute_continual_metrics(matrix, final_stage=0)
    stage_one, _ = compute_continual_metrics(matrix, final_stage=1)
    final, details = compute_continual_metrics(matrix)

    assert stage_zero['arf_clipped_absolute_percent_points'] == 0.0
    assert stage_one['final_acc_task_macro_percent'] == 72.5
    assert stage_one['arf_clipped_absolute_percent_points'] == 10.0
    assert final['final_acc_task_macro_percent'] == 80.0
    assert final['arf_clipped_absolute_percent_points'] == 5.0
    assert final['mean_signed_forgetting_percent_points'] == 2.5
    assert details['signed_forgetting_percent_points'].tolist() == [-5.0, 10.0]

    print('stage_acc_and_arf_correct=True')
    print('final_acc_arf_signed_forgetting_correct=True')


if __name__ == '__main__':
    test_stage_and_final_metrics()
