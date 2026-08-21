import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from main_fedproc import (
    aggregate_global_prototypes,
    prototype_payload_bytes,
)


def test_classwise_equal_aggregation_and_retention():
    previous = {
        0: torch.tensor([2.0, 2.0]),
        1: torch.tensor([7.0, 9.0]),
    }
    client_uploads = [
        (3, {
            0: torch.tensor([1.0, 3.0]),
            2: torch.tensor([4.0, 6.0]),
        }),
        (8, {
            0: torch.tensor([5.0, 7.0]),
        }),
        (9, {
            2: torch.tensor([8.0, 10.0]),
        }),
    ]

    aggregated, status_records = aggregate_global_prototypes(
        previous, client_uploads, round_idx=4
    )
    status_by_class = {
        row["class"]: row for row in status_records
    }

    assert torch.allclose(aggregated[0], torch.tensor([3.0, 5.0]))
    assert torch.allclose(aggregated[1], torch.tensor([7.0, 9.0]))
    assert torch.allclose(aggregated[2], torch.tensor([6.0, 8.0]))
    assert torch.allclose(previous[0], torch.tensor([2.0, 2.0]))

    assert status_by_class[0]["status"] == "updated"
    assert status_by_class[0]["uploading_client_count"] == 2
    assert status_by_class[0]["uploading_clients"] == "3,8"
    assert status_by_class[1]["status"] == "retained"
    assert status_by_class[1]["uploading_client_count"] == 0
    assert status_by_class[2]["status"] == "updated"
    assert status_by_class[2]["uploading_client_count"] == 2
    assert status_by_class[2]["uploading_clients"] == "3,9"

    assert prototype_payload_bytes(previous) == 16
    uploaded_bytes = sum(
        prototype_payload_bytes(prototypes)
        for _, prototypes in client_uploads
    )
    assert uploaded_bytes == 32

    print("class_0_equal_client_mean=[3.0, 5.0]")
    print("class_1_retained_without_upload=[7.0, 9.0]")
    print("class_2_equal_client_mean=[6.0, 8.0]")
    print("classwise_uploading_client_counts_correct=True")
    print("prototype_payload_bytes_correct=True")


def test_all_previous_classes_retained_without_uploads():
    previous = {
        0: torch.tensor([1.0, 2.0]),
        4: torch.tensor([3.0, 4.0]),
    }
    aggregated, status_records = aggregate_global_prototypes(
        previous, [], round_idx=5
    )

    assert torch.allclose(aggregated[0], previous[0])
    assert torch.allclose(aggregated[4], previous[4])
    assert all(row["status"] == "retained" for row in status_records)
    assert all(
        row["uploading_client_count"] == 0 for row in status_records
    )

    print("all_missing_classes_retained=True")
    print("retained_status_logged=True")


if __name__ == "__main__":
    test_classwise_equal_aggregation_and_retention()
    test_all_previous_classes_retained_without_uploads()
