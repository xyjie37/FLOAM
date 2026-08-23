"""
FedTAWrapper: unified FedTA adapter for ResNet / TextCNN backbones.

Implements Input Enhancement (IE) at the stem and Tail Anchor (TA) mixing in
normalized feature space, following FedTA (CVPR 2025).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FedTAWrapper(nn.Module):
    """Wrap a base model with FedTA Input Enhancement and Tail Anchor."""

    def __init__(self, base, num_classes, num_ie=10, num_ta=100,
                 topn=3, alpha=0.5, tau=0.1):
        super().__init__()
        self.base = base
        self.num_classes = num_classes
        self.num_ie = num_ie
        self.num_ta = num_ta
        self.topn = min(topn, num_ie)
        self.alpha = alpha
        self.tau = tau
        self.backbone_frozen = False

        self.model_type = self._detect_model_type()
        self.stem_channels = self._get_stem_channels()
        self.feat_dim = self._get_feat_dim()

        # Input Enhancement bank (zero-init => identity at start)
        if self.model_type == 'text':
            self.ie_bank = nn.Parameter(torch.zeros(num_ie, self.stem_channels))
        else:
            self.ie_bank = nn.Parameter(torch.zeros(num_ie, self.stem_channels, 1, 1))
        self.ie_keys = nn.Parameter(torch.randn(num_ie, self.stem_channels) * 0.01)

        # Tail Anchor bank (independent of num_classes)
        self.tail_anchors = nn.Parameter(torch.randn(num_ta, self.feat_dim) * 0.01)
        self.ta_keys = nn.Parameter(torch.randn(num_ta, self.feat_dim) * 0.01)

        # Temperature-scaled classifier on unit-norm features
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

        self._last_ie_query = None
        self._last_ie_indices = None
        self._last_ta_query = None
        self._last_ta_index = None

    # ------------------------------------------------------------------
    # Model introspection helpers
    # ------------------------------------------------------------------
    def _real_base(self):
        return self.base.module if hasattr(self.base, 'module') else self.base

    def _detect_model_type(self):
        real = self._real_base()
        if hasattr(real, 'embedding'):
            return 'text'
        return 'conv'

    def _get_stem_channels(self):
        real = self._real_base()
        if self.model_type == 'text':
            return real.embedding.embedding_dim
        if hasattr(real, 'conv1'):
            return real.conv1.out_channels
        if hasattr(real, 'backbone') and hasattr(real.backbone, 'conv1'):
            return real.backbone.conv1.out_channels
        return 64

    def _get_feat_dim(self):
        real = self._real_base()
        if hasattr(real, 'feat_dim'):
            return real.feat_dim
        if hasattr(real, 'linear') and hasattr(real.linear, 'in_features'):
            return real.linear.in_features
        if hasattr(real, 'fc') and hasattr(real.fc, 'in_features'):
            return real.fc.in_features
        if hasattr(real, 'backbone') and hasattr(real.backbone, 'fc'):
            return real.backbone.fc.in_features
        return 512

    def _backbone_param_prefixes(self):
        if self.model_type == 'text':
            return ['embedding', 'conv1', 'conv2', 'conv3', 'dropout']
        real = self._real_base()
        if hasattr(real, 'backbone'):
            return ['backbone']
        return ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

    def _head_param_names(self):
        names = []
        for name, _ in self._real_base().named_parameters():
            if 'linear' in name or name == 'fc.weight' or name == 'fc.bias':
                names.append(name)
        return names

    # ------------------------------------------------------------------
    # Stem / backbone forward paths
    # ------------------------------------------------------------------
    def _prepare_text(self, x):
        if isinstance(x, (list, tuple)):
            text = x[0]
        else:
            text = x
        if text.dtype in (torch.float32, torch.float64):
            text = text.long()
        return text

    def _stem_forward(self, x):
        real = self._real_base()
        if self.model_type == 'text':
            text = self._prepare_text(x)
            return real.embedding(text).permute(0, 2, 1)  # [B, C, L]

        if hasattr(real, 'backbone'):
            out = real.backbone.conv1(x)
            out = real.backbone.bn1(out)
            out = F.relu(out)
            if hasattr(real.backbone, 'maxpool') and real.backbone.maxpool is not None:
                out = real.backbone.maxpool(out)
            return out

        return F.relu(real.bn1(real.conv1(x)))

    def _backbone_forward(self, x_stem):
        real = self._real_base()
        if self.model_type == 'text':
            conv_out1 = real.conv1(x_stem).squeeze(2)
            conv_out2 = real.conv2(x_stem).squeeze(2)
            conv_out3 = real.conv3(x_stem).squeeze(2)
            features = torch.cat((conv_out1, conv_out2, conv_out3), dim=1)
            return real.dropout(features)

        if hasattr(real, 'backbone'):
            out = real.backbone.layer1(x_stem)
            out = real.backbone.layer2(out)
            out = real.backbone.layer3(out)
            out = real.backbone.layer4(out)
            out = real.backbone.avgpool(out)
            return torch.flatten(out, 1)

        out = real.layer1(x_stem)
        out = real.layer2(out)
        out = real.layer3(out)
        out = real.layer4(out)
        if hasattr(real, 'avgpool'):
            out = real.avgpool(out)
        else:
            out = F.avg_pool2d(out, 4)
        return torch.flatten(out, 1)

    # ------------------------------------------------------------------
    # Input Enhancement
    # ------------------------------------------------------------------
    def _apply_input_enhancement_with_params(self, x_stem, ie_bank, ie_keys):
        if self.model_type == 'text':
            query = x_stem.mean(dim=2)
        else:
            query = F.adaptive_avg_pool2d(x_stem, 1).view(x_stem.size(0), -1)

        q_norm = F.normalize(query, p=2, dim=1, eps=1e-6)
        k_norm = F.normalize(ie_keys, p=2, dim=1, eps=1e-6)
        sim = torch.mm(q_norm, k_norm.t()) / self.tau

        topn = min(self.topn, ie_bank.size(0))
        top_vals, top_idx = torch.topk(sim, topn, dim=1)
        weights = F.softmax(top_vals, dim=1)

        if self.model_type == 'text':
            selected = ie_bank[top_idx]
            enhancement = (selected * weights.unsqueeze(-1)).sum(dim=1)
            enhancement = enhancement.unsqueeze(-1).expand_as(x_stem)
        else:
            selected = ie_bank[top_idx]
            enhancement = (selected * weights.view(-1, topn, 1, 1, 1)).sum(dim=1)
            enhancement = enhancement.expand_as(x_stem)

        return x_stem + enhancement, query, top_idx

    def _apply_input_enhancement(self, x_stem):
        enhanced, query, top_idx = self._apply_input_enhancement_with_params(
            x_stem, self.ie_bank, self.ie_keys
        )
        self._last_ie_query = query.detach()
        self._last_ie_indices = top_idx.detach()
        return enhanced

    def ie_key_loss(self):
        """L_key for Input Enhancement (Eq. 3 surrogate term)."""
        if self._last_ie_query is None or self._last_ie_indices is None:
            return torch.tensor(0.0, device=self.ie_bank.device)
        q = self._last_ie_query
        idx = self._last_ie_indices[:, 0]
        keys = self.ie_keys[idx]
        return F.mse_loss(q, keys)

    # ------------------------------------------------------------------
    # Tail Anchor
    # ------------------------------------------------------------------
    def _select_tail_anchor(self, f_hat):
        """Top-1 hard TA selection with straight-through estimator."""
        q_norm = f_hat.detach()
        k_norm = F.normalize(self.ta_keys, p=2, dim=1, eps=1e-6)
        sim = torch.mm(q_norm, k_norm.t())
        idx = sim.argmax(dim=1)
        self._last_ta_query = f_hat.detach()
        self._last_ta_index = idx.detach()

        ta_hard = self.tail_anchors[idx]
        ta_soft = torch.mm(F.softmax(sim, dim=1), self.tail_anchors)
        ta = ta_hard + (ta_soft - ta_soft.detach())
        return F.normalize(ta, p=2, dim=1, eps=1e-6)

    def ta_key_loss(self):
        """Key-query alignment for the selected Tail Anchor."""
        if self._last_ta_query is None or self._last_ta_index is None:
            return torch.tensor(0.0, device=self.tail_anchors.device)
        q = self._last_ta_query
        keys = self.ta_keys[self._last_ta_index]
        return F.mse_loss(q, keys)

    def _mix_tail_anchor(self, f_out, alpha=None, use_ta=True):
        f_hat = F.normalize(f_out, p=2, dim=1, eps=1e-6)
        if not use_ta:
            return f_hat
        alpha = self.alpha if alpha is None else alpha
        if alpha <= 0:
            return f_hat
        ta = self._select_tail_anchor(f_hat)
        mixed = (1.0 - alpha) * f_hat + alpha * ta
        return F.normalize(mixed, p=2, dim=1, eps=1e-6)

    # ------------------------------------------------------------------
    # Public forward API
    # ------------------------------------------------------------------
    def extract_backbone_output(self, x, ie_bank=None, ie_keys=None):
        """Frozen-backbone features before TA (for SIKF distillation)."""
        x_stem = self._stem_forward(x)
        if ie_bank is None and ie_keys is None:
            x_stem = self._apply_input_enhancement(x_stem)
        else:
            bank = self.ie_bank if ie_bank is None else ie_bank
            keys = self.ie_keys if ie_keys is None else ie_keys
            enhanced, _, _ = self._apply_input_enhancement_with_params(x_stem, bank, keys)
            x_stem = enhanced
        return self._backbone_forward(x_stem)

    def forward(self, x, returnFeature=False, alpha=None, use_ta=True):
        f_out = self.extract_backbone_output(x)
        f_ta = self._mix_tail_anchor(f_out, alpha=alpha, use_ta=use_ta)
        logits = self._classifier(f_ta)
        if returnFeature:
            return logits, f_ta
        return logits

    def extract_features(self, x):
        """Return TA-mixed normalized features (for metrics / evaluation)."""
        return self.forward(x, returnFeature=True, use_ta=True)[1]

    def only_liner(self, features):
        return self._classifier(features)

    def _classifier(self, features):
        real = self._real_base()
        scale = self.logit_scale.clamp(min=1.0, max=100.0)
        if hasattr(real, 'only_liner'):
            return real.only_liner(features * scale)
        if hasattr(real, 'linear'):
            return real.linear(features * scale)
        if hasattr(real, 'fc'):
            return real.fc(features * scale)
        raise AttributeError('Base model has no classifier head')

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def clear_runtime_cache(self):
        """Drop forward caches so deepcopy / CPU-GPU moves stay safe."""
        self._last_ie_query = None
        self._last_ie_indices = None
        self._last_ta_query = None
        self._last_ta_index = None

    def freeze_backbone(self):
        """Freeze backbone parameters and keep BN statistics fixed."""
        self.backbone_frozen = True
        prefixes = self._backbone_param_prefixes()
        for name, param in self._real_base().named_parameters():
            if any(name == pref or name.startswith(pref + '.') for pref in prefixes):
                param.requires_grad = False
        self.set_backbone_bn_eval()

    def unfreeze_backbone(self):
        self.backbone_frozen = False
        for param in self._real_base().parameters():
            param.requires_grad = True

    def set_backbone_bn_eval(self):
        if not self.backbone_frozen:
            return
        for module in self._real_base().modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.set_backbone_bn_eval()
        return self

    # ------------------------------------------------------------------
    # State dict helpers for server aggregation
    # ------------------------------------------------------------------
    def state_dict_ie(self):
        return {
            'ie_bank': self.ie_bank.detach().clone(),
            'ie_keys': self.ie_keys.detach().clone(),
        }

    def load_state_dict_ie(self, state):
        if state is None:
            return
        if 'ie_bank' in state:
            self.ie_bank.data.copy_(state['ie_bank'])
        if 'ie_keys' in state:
            self.ie_keys.data.copy_(state['ie_keys'])

    def state_dict_ta(self):
        return {
            'tail_anchors': self.tail_anchors.detach().clone(),
            'ta_keys': self.ta_keys.detach().clone(),
            'logit_scale': self.logit_scale.detach().clone(),
        }

    def load_state_dict_ta(self, state):
        if state is None:
            return
        if 'tail_anchors' in state:
            self.tail_anchors.data.copy_(state['tail_anchors'])
        if 'ta_keys' in state:
            self.ta_keys.data.copy_(state['ta_keys'])
        if 'logit_scale' in state:
            self.logit_scale.data.copy_(state['logit_scale'])
