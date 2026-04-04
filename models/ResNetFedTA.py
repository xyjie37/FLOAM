"""
ResNetFedTA: ResNet with explicit Bottleneck Adapter (no hooks).
Replaces ESA hook with built-in adapter; Tail Anchor applied at the
feature layer (after GAP, before classifier) per FedTA paper formula (6):
    F_TA = (1 - alpha) * F_out + alpha * TA_s
where TA_s is the soft-selected anchor via cosine-similarity Query-Key matching.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetFedTA(nn.Module):
    """
    ResNet with explicit FedTA adapters (no hooks).
    Adapter: 1x1 Conv bottleneck, zero-init last layer.
    Tail Anchor: feature-level blending (FedTA paper formula 6).
        F_TA = (1 - alpha) * F_out + alpha * TA_s
    TA_s is soft-selected from tail_anchors via cosine similarity with F_out.
    The blended F_TA is then fed into the classifier head.
    """
    def __init__(self, base_resnet, num_classes, num_ie=10):
        super().__init__()
        self.base = base_resnet
        self.num_classes = num_classes
        self.num_ie = num_ie

        # Bottleneck Adapter (zero-init last layer)
        self.fedta_adapter = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 1, bias=False)
        )
        nn.init.zeros_(self.fedta_adapter[2].weight)

        self.ie_keys = nn.Parameter(torch.randn(num_ie, 64) * 0.01)
        self.feat_dim = self._get_feat_dim()
        self.tail_anchors = nn.Parameter(torch.zeros(num_classes, self.feat_dim))
        self.beta = nn.Parameter(torch.tensor(-4.0))

    def _get_feat_dim(self):
        real = self.base.module if hasattr(self.base, 'module') else self.base
        if hasattr(real, 'linear') and hasattr(real.linear, 'in_features'):
            return real.linear.in_features
        return 512

    def _stem_forward(self, x):
        """Stem: conv1 -> bn1 -> relu [-> maxpool]"""
        real = self.base.module if hasattr(self.base, 'module') else self.base
        if hasattr(real, 'conv1'):
            x = F.relu(real.bn1(real.conv1(x)))
            if hasattr(real, 'maxpool') and real.maxpool is not None:
                x = real.maxpool(x)
        elif hasattr(real, 'backbone'):
            x = real.backbone.conv1(x)
            x = real.backbone.bn1(x)
            x = F.relu(x)
            x = real.backbone.maxpool(x)
        return x

    def _layers_forward(self, x):
        """layer1 -> layer2 -> layer3 -> layer4 -> pool -> flatten"""
        real = self.base.module if hasattr(self.base, 'module') else self.base
        if hasattr(real, 'layer1'):
            x = real.layer1(x)
            x = real.layer2(x)
            x = real.layer3(x)
            x = real.layer4(x)
            x = F.avg_pool2d(x, 4)
        elif hasattr(real, 'backbone'):
            x = real.backbone.layer1(x)
            x = real.backbone.layer2(x)
            x = real.backbone.layer3(x)
            x = real.backbone.layer4(x)
            x = real.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _classifier(self, f):
        real = self.base.module if hasattr(self.base, 'module') else self.base
        return real.linear(f)

    def forward(self, x, return_features=False):
        # Stem
        x = self._stem_forward(x)  # [B, 64, H, W]

        # Adapter residual (zero-init, no effect initially)
        adapter_out = self.fedta_adapter(x)
        x = x + 0.1 * adapter_out

        # Layers + GAP  →  F_out ∈ R^512
        f = self._layers_forward(x)  # [B, 512]

        # Tail Anchor: feature-level blending (FedTA paper formula 6)
        # TA_s selected via cosine-similarity Query-Key matching (soft, differentiable)
        alpha = torch.sigmoid(self.beta)
        f_norm = F.normalize(f, dim=1, eps=1e-6)               # [B, 512]
        A_norm = F.normalize(self.tail_anchors, dim=1, eps=1e-6)  # [C, 512]
        sim = torch.mm(f_norm, A_norm.t())                      # [B, C]
        w = F.softmax(sim * 10.0, dim=1)                        # [B, C] soft weights
        TA_s = torch.mm(w, self.tail_anchors)                   # [B, 512] soft-selected anchor
        f_ta = (1 - alpha) * f + alpha * TA_s                   # F_TA = (1-α)F_out + α·TA_s
        logits = self._classifier(f_ta)

        if return_features:
            return logits, f_ta
        return logits

    def extract_features(self, x):
        x = self._stem_forward(x)
        adapter_out = self.fedta_adapter(x)
        x = x + 0.1 * adapter_out
        return self._layers_forward(x)

    def only_liner(self, features):
        return self._classifier(features)

    def state_dict_fedta(self):
        """FedTA-specific state for KB/adapter aggregation."""
        return {
            'fedta_adapter': {k: v.clone() for k, v in self.fedta_adapter.state_dict().items()},
            'ie_keys': self.ie_keys.data.clone(),
        }

    def load_state_dict_fedta(self, state):
        if 'fedta_adapter' in state:
            self.fedta_adapter.load_state_dict(state['fedta_adapter'], strict=False)
        if 'ie_keys' in state:
            self.ie_keys.data.copy_(state['ie_keys'])
