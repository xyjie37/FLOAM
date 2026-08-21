import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.Update import LocalUpdateFedProc
from models.FedProcNet import FedProcResNet18


class ToyFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.extract_calls_in_eval = []

    def extract_features(self, images):
        self.extract_calls_in_eval.append(not self.training)
        return images

    def forward(self, images, returnFeature=False):
        z = self.extract_features(images)
        logits = torch.zeros(images.size(0), 1, device=images.device)
        if returnFeature:
            return logits, z
        return logits


class ForwardOnlyWrapper(nn.Module):
    """Mimic wrappers such as DataParallel that expose forward but not helpers."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ToyFedProcNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def encode(self, images):
        r = images * self.scale
        return r, r

    def extract_features(self, images):
        _, z = self.encode(images)
        return z

    def forward(self, images, returnFeature=False, return_all=False):
        r, z = self.encode(images)
        logits = torch.stack([2.0 * z[:, 0], -1.5 * z[:, 1]], dim=1)
        if return_all:
            return logits, r, z
        if returnFeature:
            return logits, z
        return logits


def test_global_prototype_loss():
    z = torch.tensor(
        [
            [1.0, 0.0],  # Class 0 has a global prototype.
            [0.0, 1.0],  # Class 1 has no global prototype.
            [1.0, 1.0],  # Class 2 has a global prototype.
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 1, 2])
    global_prototypes = {
        2: torch.tensor([1.0, 1.0]),
        0: torch.tensor([1.0, 0.0]),
    }

    loss = LocalUpdateFedProc._global_prototype_loss(
        z, labels, global_prototypes
    )

    prototype_matrix = torch.stack(
        [global_prototypes[0], global_prototypes[2]]
    )
    valid_z = z[[0, 2]]
    similarities = F.cosine_similarity(
        valid_z.unsqueeze(1), prototype_matrix.unsqueeze(0), dim=2
    )
    expected = F.cross_entropy(similarities, torch.tensor([0, 1]))

    assert torch.allclose(loss, expected)
    loss.backward()
    assert z.grad[1].abs().sum().item() == 0.0

    no_valid_loss = LocalUpdateFedProc._global_prototype_loss(
        z[:1], torch.tensor([1]), global_prototypes
    )
    assert no_valid_loss is None

    print(f"gpc_loss={loss.item():.6f}")
    print("matches_formula=True")
    print("missing_class_gpc_grad=0.0")
    print("no_valid_batch_returns_none=True")


def test_local_prototype_recomputation():
    features = torch.tensor(
        [
            [1.0, 2.0],   # Class 0
            [10.0, 4.0],  # Class 1
            [3.0, 6.0],   # Class 0
            [14.0, 8.0],  # Class 1
            [5.0, 10.0],  # Class 0
        ]
    )
    labels = torch.tensor([0, 1, 0, 1, 0])

    updater = LocalUpdateFedProc.__new__(LocalUpdateFedProc)
    updater.args = SimpleNamespace(device="cpu")
    updater.ldr_train = DataLoader(
        TensorDataset(features, labels), batch_size=2, shuffle=False
    )

    base_net = ToyFeatureNet()
    net = ForwardOnlyWrapper(base_net)
    assert not hasattr(net, "extract_features")
    net.train()
    local_prototypes = updater._compute_local_prototypes(net)

    assert set(local_prototypes) == {0, 1}
    assert torch.allclose(local_prototypes[0], torch.tensor([3.0, 6.0]))
    assert torch.allclose(local_prototypes[1], torch.tensor([12.0, 6.0]))
    assert not net.training
    assert all(base_net.extract_calls_in_eval)

    print("prototype_class_0=", local_prototypes[0].tolist())
    print("prototype_class_1=", local_prototypes[1].tolist())
    print("prototype_uses_exact_sample_mean=True")
    print("prototype_recomputed_in_eval_mode=True")
    print("prototype_forward_wrapper_compatible=True")


def make_training_updater(images, labels):
    updater = LocalUpdateFedProc.__new__(LocalUpdateFedProc)
    updater.args = SimpleNamespace(
        device="cpu", momentum=0.0, wd=0.0, local_ep=1
    )
    updater.loss_func = nn.CrossEntropyLoss()
    updater.ldr_train = DataLoader(
        TensorDataset(images, labels), batch_size=len(labels), shuffle=False
    )
    return updater


def test_full_training_loss_and_fallback():
    images = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    global_prototypes = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([0.0, 1.0]),
    }

    updater = make_training_updater(images, labels)
    net = ToyFedProcNet()
    logits, _, z = net(images, return_all=True)
    ce_loss = F.cross_entropy(logits, labels)
    gpc_loss = LocalUpdateFedProc._global_prototype_loss(
        z, labels, global_prototypes
    )
    expected_loss = 0.6 * gpc_loss + 0.4 * ce_loss

    weights, average_loss, local_prototypes = updater.train(
        net=net,
        lr=0.0,
        global_prototypes=global_prototypes,
        global_round=4,
        total_rounds=10,
        local_eps=1,
    )

    assert abs(average_loss - expected_loss.item()) < 1e-6
    assert net.scale.grad is not None
    assert net.scale.grad.abs().item() > 0.0
    assert "scale" in weights
    assert set(local_prototypes) == {0, 1}

    fallback_updater = make_training_updater(images, labels)
    fallback_net = ToyFedProcNet()
    fallback_logits = fallback_net(images)
    expected_ce = F.cross_entropy(fallback_logits, labels).item()
    _, fallback_loss, _ = fallback_updater.train(
        net=fallback_net,
        lr=0.0,
        global_prototypes={2: torch.tensor([1.0, 1.0])},
        global_round=0,
        total_rounds=10,
        local_eps=1,
    )

    assert abs(fallback_loss - expected_ce) < 1e-6

    print("dynamic_alpha_t=0.6")
    print("combined_loss_matches=True")
    print("combined_loss_backpropagates=True")
    print("no_valid_positive_uses_full_ce=True")


def test_real_fedproc_network_integration():
    torch.manual_seed(0)
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 1, 2])
    updater = make_training_updater(images, labels)
    updater.ldr_train = DataLoader(
        TensorDataset(images, labels), batch_size=2, shuffle=False
    )

    net = FedProcResNet18(num_classes=10, z_dim=256)
    projection_before = net.projection_head[0].weight.detach().clone()
    global_prototypes = {
        0: torch.randn(256),
        2: torch.randn(256),
    }

    weights, average_loss, local_prototypes = updater.train(
        net=net,
        lr=0.001,
        global_prototypes=global_prototypes,
        global_round=3,
        total_rounds=10,
        local_eps=1,
    )

    assert average_loss > 0.0
    assert torch.isfinite(torch.tensor(average_loss))
    assert not torch.allclose(
        projection_before, net.projection_head[0].weight.detach()
    )
    assert "projection_head.0.weight" in weights
    assert "classifier.weight" in weights
    assert set(local_prototypes) == {0, 1, 2}
    assert all(prototype.shape == (256,) for prototype in local_prototypes.values())
    assert all(torch.isfinite(prototype).all() for prototype in local_prototypes.values())

    print("real_fedproc_train_step=True")
    print("projection_head_updated=True")
    print("returned_local_prototype_classes=[0, 1, 2]")
    print("returned_local_prototype_dim=256")


if __name__ == "__main__":
    test_global_prototype_loss()
    test_local_prototype_recomputation()
    test_full_training_loss_and_fallback()
    test_real_fedproc_network_integration()
