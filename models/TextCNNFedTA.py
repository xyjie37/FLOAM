"""
TextCNNFedTA: FedTA wrapper specifically for TextCNN.
Uses Linear adapter since TextCNN produces 1D feature vectors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNNFedTA(nn.Module):
    """
    TextCNN with FedTA adapter and Tail Anchor.
    - Adapter: Linear bottleneck (works with 1D text features)
    - Tail Anchor: feature-level blending
    """
    def __init__(self, base_textcnn, num_classes, num_ie=10):
        super().__init__()
        self.base = base_textcnn
        self.num_classes = num_classes
        self.num_ie = num_ie

        # Get feature dimension from base TextCNN
        self.feat_dim = self._get_feat_dim_from_base()

        # Linear Adapter for 1D features
        # TextCNN feature dim is 300 (100 channels * 3 kernels)
        hidden_dim = max(64, self.feat_dim // 2)
        self.fedta_adapter = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.feat_dim, bias=False)
        )
        # Zero-init last layer for stable training
        nn.init.zeros_(self.fedta_adapter[2].weight)

        # Interest Example keys
        self.ie_keys = nn.Parameter(torch.randn(num_ie, self.feat_dim) * 0.01)

        # Tail Anchors (one per class)
        self.tail_anchors = nn.Parameter(torch.zeros(num_classes, self.feat_dim))

        # Learnable blending parameter
        self.beta = nn.Parameter(torch.tensor(-4.0))

    def _get_feat_dim_from_base(self):
        """Extract feature dimension from base TextCNN."""
        real = self.base.module if hasattr(self.base, 'module') else self.base

        # TextCNN has 'fc' layer
        if hasattr(real, 'fc') and hasattr(real.fc, 'in_features'):
            return real.fc.in_features

        # Check for explicit feat_dim
        if hasattr(real, 'feat_dim'):
            return real.feat_dim

        # Fallback: check fc in base (if wrapped)
        if hasattr(real, 'base'):
            b = real.base.module if hasattr(real.base, 'module') else real.base
            if hasattr(b, 'fc') and hasattr(b.fc, 'in_features'):
                return b.fc.in_features

        # Default for TextCNN
        return 300  # 100 channels * 3 kernel sizes

    def forward(self, x, return_features=False):
        # Extract features from base TextCNN
        if hasattr(self.base, 'extract_features'):
            features = self.base.extract_features(x)
        else:
            raise ValueError("Base model must have extract_features() method")

        # Apply adapter (residual connection)
        adapter_out = self.fedta_adapter(features)
        f = features + 0.1 * adapter_out

        # Tail Anchor: feature-level blending (FedTA paper formula 6)
        alpha = torch.sigmoid(self.beta)
        f_norm = F.normalize(f, dim=1, eps=1e-6)
        A_norm = F.normalize(self.tail_anchors, dim=1, eps=1e-6)
        sim = torch.mm(f_norm, A_norm.t())  # [B, num_classes]
        w = F.softmax(sim * 10.0, dim=1)  # Soft weights
        TA_s = torch.mm(w, self.tail_anchors)  # Soft-selected anchors
        f_ta = (1 - alpha) * f + alpha * TA_s

        # Classification using base model's fc layer
        logits = self.base.only_liner(f_ta)

        if return_features:
            return logits, f_ta
        return logits

    def extract_features(self, x):
        """Extract features without Tail Anchor modification."""
        return self.base.extract_features(x)

    def only_liner(self, features):
        """Apply classifier to features without Tail Anchor."""
        return self.base.only_liner(features)

    def state_dict_fedta(self):
        """FedTA-specific state for KB/adapter aggregation."""
        return {
            'fedta_adapter': {k: v.clone() for k, v in self.fedta_adapter.state_dict().items()},
            'ie_keys': self.ie_keys.data.clone(),
        }

    def load_state_dict_fedta(self, state):
        """Load FedTA state from server KB."""
        if 'fedta_adapter' in state:
            self.fedta_adapter.load_state_dict(state['fedta_adapter'], strict=False)
        if 'ie_keys' in state:
            self.ie_keys.data.copy_(state['ie_keys'])