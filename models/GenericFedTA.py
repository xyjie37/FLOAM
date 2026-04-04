"""
GenericFedTA: Generic FedTA wrapper for any backbone model.
Supports arbitrary base models (ResNet, SpeechResNet, TinyResNet, etc.) with
FedTA's Adapter and Tail Anchor mechanisms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GenericFedTA(nn.Module):
    """
    Generic FedTA wrapper that works with any base model.
    - Adapter: Linear bottleneck (works with any 1D feature dimension)
    - Tail Anchor: feature-level blending (FedTA paper formula 6)
    """
    def __init__(self, base_model, num_classes, num_ie=10):
        super().__init__()
        self.base = base_model
        self.num_classes = num_classes
        self.num_ie = num_ie

        # Get feature dimension from base model
        self.feat_dim = self._get_feat_dim_from_base()

        # Linear Adapter (works with any 1D features)
        # Bottleneck: feat_dim -> feat_dim//2 -> feat_dim
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
        """Extract feature dimension from base model."""
        real = self.base.module if hasattr(self.base, 'module') else self.base

        # Check for explicit feat_dim attribute
        if hasattr(real, 'feat_dim'):
            return real.feat_dim

        # Check for linear/fc layer
        if hasattr(real, 'linear') and hasattr(real.linear, 'in_features'):
            return real.linear.in_features
        if hasattr(real, 'fc') and hasattr(real.fc, 'in_features'):
            return real.fc.in_features
        if hasattr(real, 'classifier'):
            if hasattr(real.classifier, 'in_features'):
                return real.classifier.in_features
            # Handle sequential classifier
            for layer in real.classifier:
                if hasattr(layer, 'in_features'):
                    return layer.in_features

        # Check in base module
        if hasattr(real, 'base'):
            b = real.base.module if hasattr(real.base, 'module') else real.base
            if hasattr(b, 'linear') and hasattr(b.linear, 'in_features'):
                return b.linear.in_features

        # Fallback: test with dummy input
        return 512  # Default fallback

    def forward(self, x, return_features=False):
        # Extract features from base model
        if hasattr(self.base, 'extract_features'):
            features = self.base.extract_features(x)
        else:
            # For models that don't have extract_features, do forward pass
            # and get features before classifier
            if hasattr(self.base, 'forward'):
                # Try to get features by checking return type
                out = self.base.forward(x)
                if isinstance(out, tuple):
                    features = out[1]  # (logits, features) format
                else:
                    # If no features returned, need base model to support it
                    raise ValueError("Base model must have extract_features() or return (logits, features)")
            else:
                raise ValueError("Base model must have forward() or extract_features()")

        # Apply adapter (residual connection)
        adapter_out = self.fedta_adapter(features)
        f = features + 0.1 * adapter_out  # Small residual for stability

        # Tail Anchor: feature-level blending (FedTA paper formula 6)
        alpha = torch.sigmoid(self.beta)
        f_norm = F.normalize(f, dim=1, eps=1e-6)
        A_norm = F.normalize(self.tail_anchors, dim=1, eps=1e-6)
        sim = torch.mm(f_norm, A_norm.t())  # [B, num_classes]
        w = F.softmax(sim * 10.0, dim=1)  # Soft weights
        TA_s = torch.mm(w, self.tail_anchors)  # Soft-selected anchors
        f_ta = (1 - alpha) * f + alpha * TA_s  # Blended features

        # Classification
        if hasattr(self.base, 'only_liner'):
            logits = self.base.only_liner(f_ta)
        else:
            # Fallback: use a simple linear layer
            if not hasattr(self, 'classifier'):
                self.classifier = nn.Linear(self.feat_dim, self.num_classes).to(f_ta.device)
            logits = self.classifier(f_ta)

        if return_features:
            return logits, f_ta
        return logits

    def extract_features(self, x):
        """Extract features without Tail Anchor modification."""
        if hasattr(self.base, 'extract_features'):
            return self.base.extract_features(x)
        else:
            raise NotImplementedError("Base model must have extract_features() method")

    def only_liner(self, features):
        """Apply classifier to features without Tail Anchor."""
        if hasattr(self.base, 'only_liner'):
            return self.base.only_liner(features)
        else:
            if not hasattr(self, 'classifier'):
                self.classifier = nn.Linear(self.feat_dim, self.num_classes)
            return self.classifier(features)

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