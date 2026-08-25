"""FedProc network for CIFAR-style inputs.

The model follows the architecture required by the FedProc task:

    x -> ResNet18 backbone -> r -> two-layer projection head -> z
      -> classifier -> logits

The projected representation ``z`` is the representation used for class
prototypes and the global prototypical contrastive loss.  The default
``forward`` contract remains compatible with the other FLOAM models by
returning logits only.
"""

import torch.nn as nn

from models.ResNet import ResNet18


class FedProcNet(nn.Module):
    """ResNet18-based FedProc model.

    Args:
        num_classes: Number of output classes.
        z_dim: Dimension of the projected representation ``z``.
        projection_hidden_dim: Hidden dimension of the two-layer projection
            head.  When omitted, it is set to the backbone feature dimension.
    """

    def __init__(
            self,
            num_classes=10,
            z_dim=256,
            projection_hidden_dim=None):
        super().__init__()

        if num_classes <= 0:
            raise ValueError('num_classes must be positive.')
        if z_dim <= 0:
            raise ValueError('z_dim must be positive.')

        self.backbone = ResNet18(num_classes=num_classes)
        self.r_dim = self.backbone.linear.in_features

        # The original ResNet classifier is not part of FedProc.  Replacing it
        # with Identity removes its unused parameters from aggregation and
        # communication while retaining the backbone's feature extractor.
        self.backbone.linear = nn.Identity()

        if projection_hidden_dim is None:
            projection_hidden_dim = self.r_dim
        if projection_hidden_dim <= 0:
            raise ValueError('projection_hidden_dim must be positive.')

        self.z_dim = z_dim
        self.projection_head = nn.Sequential(
            nn.Linear(self.r_dim, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, z_dim),
        )
        self.classifier = nn.Linear(z_dim, num_classes)

    def extract_base_features(self, x):
        """Return the backbone representation ``r``."""
        return self.backbone.extract_features(x)

    def project_features(self, r):
        """Project backbone representations ``r`` into prototype space."""
        return self.projection_head(r)

    def encode(self, x):
        """Return both the backbone representation ``r`` and projection ``z``."""
        r = self.extract_base_features(x)
        z = self.project_features(r)
        return r, z

    def extract_features(self, x):
        """Return ``z`` for FLOAM feature/prototype compatibility."""
        _, z = self.encode(x)
        return z

    def only_liner(self, features):
        """Classify projected features ``z`` using the FedProc classifier."""
        return self.classifier(features)

    def forward(self, x, returnFeature=False, return_all=False):
        r, z = self.encode(x)
        logits = self.classifier(z)

        if return_all:
            return logits, r, z
        if returnFeature:
            return logits, z
        return logits


def FedProcResNet18(
        num_classes=10,
        z_dim=256,
        projection_hidden_dim=None):
    """Construct the default ResNet18-based FedProc network."""
    return FedProcNet(
        num_classes=num_classes,
        z_dim=z_dim,
        projection_hidden_dim=projection_hidden_dim,
    )
