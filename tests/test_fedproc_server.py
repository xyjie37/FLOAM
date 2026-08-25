import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from main_fedproc import (
    aggregate_global_prototypes,
    prototype_payload_bytes,
    run_pre_round_bootstrap,
)


class FakeBootstrapLocalUpdate:
    init_calls = []
    model_ids = []

    def __init__(self, args, dataset, idxs, task):
        self.client_id = int(idxs)
        self.init_calls.append({
            "args": args,
            "dataset": dataset,
            "client_id": self.client_id,
            "task": int(task),
        })

    def compute_local_prototypes(self, net):
        self.model_ids.append(id(net))
        net.eval()
        torch.rand(1)
        if self.client_id == 4:
            return {
                0: torch.tensor([1.0, 3.0]),
                1: torch.tensor([2.0, 4.0]),
            }
        if self.client_id == 7:
            return {
                0: torch.tensor([5.0, 7.0]),
            }
        raise AssertionError("Unexpected bootstrap client.")

    def train(self, *args, **kwargs):
        raise AssertionError("Bootstrap must not perform local training.")


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


def test_pre_round_bootstrap_uses_frozen_round_zero_selection():
    FakeBootstrapLocalUpdate.init_calls = []
    FakeBootstrapLocalUpdate.model_ids = []
    net = nn.Linear(2, 2)
    net.train()
    state_before = {
        key: value.detach().clone()
        for key, value in net.state_dict().items()
    }
    torch_rng_before = torch.random.get_rng_state()

    prototypes, status_records, upload_bytes, forward_seconds = \
        run_pre_round_bootstrap(
        args=object(),
        dataset_path="task0-only-dataset",
        net_glob=net,
        selected_clients=[4, 7],
        task=0,
        local_update_cls=FakeBootstrapLocalUpdate,
        preserve_rng=True,
    )

    assert [
        call["client_id"] for call in FakeBootstrapLocalUpdate.init_calls
    ] == [4, 7]
    assert all(
        call["task"] == 0 for call in FakeBootstrapLocalUpdate.init_calls
    )
    assert all(
        call["dataset"] == "task0-only-dataset"
        for call in FakeBootstrapLocalUpdate.init_calls
    )
    assert len(set(FakeBootstrapLocalUpdate.model_ids)) == 1
    assert FakeBootstrapLocalUpdate.model_ids[0] == id(net)
    assert net.training
    assert all(
        torch.equal(net.state_dict()[key], value)
        for key, value in state_before.items()
    )
    assert torch.equal(torch.random.get_rng_state(), torch_rng_before)

    assert torch.allclose(prototypes[0], torch.tensor([3.0, 5.0]))
    assert torch.allclose(prototypes[1], torch.tensor([2.0, 4.0]))
    assert upload_bytes == 24
    assert forward_seconds >= 0.0
    assert all(row["phase"] == "bootstrap" for row in status_records)
    assert all(row["round"] == 0 for row in status_records)

    print("bootstrap_selected_clients=[4, 7]")
    print("bootstrap_task_0_only=True")
    print("bootstrap_uses_one_unified_model=True")
    print("bootstrap_local_training_performed=False")
    print("bootstrap_model_state_unchanged=True")
    print("bootstrap_model_mode_restored=True")
    print("bootstrap_rng_state_restored=True")
    print("bootstrap_equal_prototype_aggregation=True")
    print("bootstrap_upload_bytes=24")
    print("bootstrap_feature_forward_time_recorded=True")


if __name__ == "__main__":
    test_classwise_equal_aggregation_and_retention()
    test_all_previous_classes_retained_without_uploads()
    test_pre_round_bootstrap_uses_frozen_round_zero_selection()
