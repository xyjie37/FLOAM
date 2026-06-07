"""
ResNet18 for 224x224 ImageNet-style input.
Uses torchvision ResNet18 with custom classifier head.
"""
import torch.nn as nn
from torchvision import models


class ResNet18_ImageNet(nn.Module):
    """
    ResNet18 for 224x224 ImageNet-style input.
    Provides extract_features, only_liner, and returnFeature for FLOAM compatibility.
    """

    def __init__(self, num_classes=100):
        super().__init__()
        try:
            self.backbone = models.resnet18(weights=None)
        except TypeError:
            self.backbone = models.resnet18(pretrained=False)
        self.feat_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()
        self.linear = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, returnFeature=False):
        features = self.backbone(x)
        logits = self.linear(features)
        if returnFeature:
            return logits, features
        return logits

    def extract_features(self, x):
        return self.backbone(x)

    def only_liner(self, features):
        return self.linear(features)
