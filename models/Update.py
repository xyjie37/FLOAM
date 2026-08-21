import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset
from tqdm import tqdm
import math
import pdb
import copy
from torch.optim import Optimizer
#from transformers import CLIPProcessor, CLIPModel

from utils.data_utils import load_train_data, load_test_data
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from collections import defaultdict
import torch.optim as optim
from scipy.stats import wasserstein_distance
try:
    from cvxopt import matrix, solvers
except ImportError:
    matrix = None
    solvers = None
from collections import Counter
import torchvision.transforms.functional as TF
from collections import OrderedDict
from typing import Callable
import multiprocessing as mp
import atexit
import threading
from queue import Queue
from copy import deepcopy
import gc

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class AnchorContrastiveLoss(nn.Module):
    """
    Dynamic Hard Negative Mining (stable version).
    - sparsemax approximates discrete topk, reducing gradient path jumps;
    - EMA + Sigmoid temperature compression on s_avg reduces early-stage sensitivity;
    - EMA on k (continuous) avoids boundary oscillation from hard clipping.
    """
    def __init__(self,
                 anchors,                    # [num_classes, feat_dim]
                 temperature=0.1,
                 device='cuda',
                 momentum_s_avg=0.9,         # EMA coeff for s_avg
                 s_avg_scale=3.0,            # Sigmoid compression strength: sigmoid((x-0.5)*scale)
                 momentum_k=0.9,             # EMA coeff for k
                 use_sparsemax=True):
        super().__init__()
        # Ensure anchors themselves do not carry gradients
        self.register_buffer('anchors', anchors.detach().clone())
        self.temperature = temperature
        self.device = device

        # Smoothing and temperature compression hyperparams
        self.momentum_s_avg = momentum_s_avg
        self.s_avg_scale = s_avg_scale

        # EMA hyperparams for k
        self.momentum_k = momentum_k

        # Use sparsemax instead of soft top-k
        self.use_sparsemax = use_sparsemax

        # Key fix: initialize buffers on the same device as anchors
        dev = self.anchors.device          # Supports both cuda and cpu
        self.register_buffer('s_avg_ema', torch.tensor(0.5, device=dev))
        self.register_buffer('k_ema', torch.tensor(0.0, device=dev))

        # Metrics
        self.metrics = {'hard_neg_sim': 0.0, 'dynamic_k': 0, 's_avg_ema': 0.5}

    @staticmethod
    def sparsemax(input, dim=1):
        """
        Sparsemax implementation
        Args:
            input: Tensor of any shape
            dim: Dimension along which to apply sparsemax
        Returns:
            Tensor of same shape as input with sparsemax applied
        """
        # Get the number of elements in the specified dimension
        num_elements = input.size(dim)
        
        # Sort input in descending order
        input_sorted, _ = torch.sort(input, dim=dim, descending=True)
        
        # Calculate cumulative sum
        input_cumsum = torch.cumsum(input_sorted, dim=dim)
        
        # Create a range tensor [1, 2, ..., num_elements]
        k = torch.arange(1, num_elements + 1, device=input.device).view(1, -1)
        
        # Calculate the condition: 1 + k * z_k > sum(z_1:k)
        condition = 1 + k * input_sorted > input_cumsum
        
        # Find the largest k that satisfies the condition
        k_max = condition.sum(dim=dim, keepdim=True).float()
        
        # Calculate tau (threshold)
        tau = (input_cumsum.gather(dim, k_max.long() - 1) - 1) / k_max
        
        # Apply sparsemax: max(z - tau, 0)
        output = torch.clamp(input - tau, min=0)
        
        return output

    # Optional: adaptive anchor dimension
    def _adjust_anchor_dim(self, target_dim):
        cur = self.anchors.size(1)
        if cur == target_dim:
            return
        linear = nn.Linear(cur, target_dim, bias=False).to(self.anchors.device)
        with torch.no_grad():
            new_anchors = linear(self.anchors)
        # Prevent gradient leakage
        self.anchors = new_anchors.detach().clone()

    def forward(self, features, labels, k=5, alpha=0.8, adaptive_k=True):
        if features.size(1) != self.anchors.size(1):
            self._adjust_anchor_dim(features.size(1))

        # Cosine similarity [B,K]
        sim = F.cosine_similarity(features.unsqueeze(1),
                                  self.anchors.unsqueeze(0), dim=2)
        logits = sim / self.temperature
        B, K = logits.shape

        # Mask positive samples
        pos_mask = F.one_hot(labels, num_classes=K).bool()
        pos_scores = logits.gather(1, labels.view(-1, 1))  # [B,1]
        neg_logits = logits.masked_fill(pos_mask, -float('inf'))

        # Compute stable dynamic k
        if adaptive_k:
            with torch.no_grad():
                # Map cosine similarity [-1,1] -> [0,1]
                s = ((sim.detach() + 1.0) * 0.5).mean()

                # EMA smooth s_avg
                self.s_avg_ema.mul_(self.momentum_s_avg).add_(
                    s * (1 - self.momentum_s_avg))

                # Sigmoid compression to reduce sensitivity
                s_bar = torch.sigmoid((self.s_avg_ema - 0.5) * self.s_avg_scale)

                # Raw k (continuous), then EMA
                raw_k = torch.clamp(
                    k + torch.round(s_bar * (K - 1)).to(self.s_avg_ema.dtype),
                    min=1, max=K - 1)

                # Initialize k_ema
                if self.k_ema.item() == 0.0:
                    self.k_ema.copy_(raw_k)

                # EMA continuous k
                self.k_ema.mul_(self.momentum_k).add_(
                    raw_k * (1 - self.momentum_k))
                dynamic_k = int(
                    torch.clamp(torch.round(self.k_ema), 1, K - 1).item())
        else:
            dynamic_k = k

        # Sparsemax approximation (differentiable, continuous)
        eps = 1e-12
        if self.use_sparsemax:
            # sparsemax selects hard-negative support, not used as probability weights
            w = self.sparsemax(neg_logits, dim=1)  # [B,K]

            # support: negative classes with positive sparsemax weight
            support_mask = (w > 0)  # [B, K]

            # log-sum-exp on support, preserving hard-negative strength
            neg_logits_sparse = neg_logits.masked_fill(~support_mask, -float('inf'))
            lse_neg_sparsemax = torch.logsumexp(neg_logits_sparse,
                                               dim=1, keepdim=True)  # [B,1]

            log_sum_hard = torch.logsumexp(
                torch.cat([pos_scores, lse_neg_sparsemax], dim=1),
                dim=1, keepdim=True)  # [B,1]
        else:
            hard_neg, _ = torch.topk(neg_logits, k=dynamic_k, dim=1)  # [B,k]
            log_sum_hard = torch.logsumexp(
                torch.cat([pos_scores, hard_neg], dim=1), dim=1, keepdim=True)

        # All negatives log-sum-exp (baseline)
        log_sum_all = torch.logsumexp(neg_logits, dim=1, keepdim=True)

        # Optional: adaptive alpha (smooth)
        if self.training and adaptive_k:
            with torch.no_grad():
                if self.use_sparsemax:
                    w_sim = self.sparsemax(neg_logits, dim=1)
                    # Uniform average over sparsemax support
                    support_mask_sim = (w_sim > 0)  # [B, K]
                    support_size = support_mask_sim.sum(dim=1).clamp(min=1).float()  # [B]
                    neg_logits_clamped = neg_logits.clamp(min=-30, max=30)
                    hard_stat = (
                        (neg_logits_clamped * support_mask_sim.float()).sum(dim=1) / support_size
                    ).mean()
                else:
                    hard_neg, _ = torch.topk(neg_logits, k=dynamic_k, dim=1)
                    hard_stat = hard_neg.mean()
                adapt_alpha = torch.sigmoid(hard_stat)
            alpha = 0.9 * alpha + 0.1 * adapt_alpha.item()

        # Final loss: InfoNCE-style "positive vs combined negatives"
        loss = -(pos_scores -
                 (alpha * log_sum_hard + (1 - alpha) * log_sum_all)).mean()

        # Metrics
        with torch.no_grad():
            self.metrics['dynamic_k'] = dynamic_k
            self.metrics['s_avg_ema'] = float(self.s_avg_ema.item())
            if self.use_sparsemax:
                sim_neg = sim.masked_fill(pos_mask, -1e9)
                w_sim = self.sparsemax(neg_logits, dim=1)
                # Uniform average over support, consistent with hard-negative definition
                support_mask_m = (w_sim > 0)
                support_size_m = support_mask_m.sum(dim=1).clamp(min=1).float()
                self.metrics['hard_neg_sim'] = float(
                    ((sim_neg.clamp(min=-1, max=1) * support_mask_m.float()).sum(dim=1) / support_size_m
                     ).mean().item())
            else:
                hard_neg_sim, _ = torch.topk(
                    sim.masked_fill(pos_mask, -1e9), k=dynamic_k, dim=1)
                self.metrics['hard_neg_sim'] = float(hard_neg_sim.mean().item())

        return loss


class LocalUpdateFedACD(object):         # Full client version
    def __init__(self, args, anchor=None, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.anchor = anchor.to(self.args.device) if anchor is not None \
                      else torch.randn(args.num_classes, 100).to(args.device)
        self.num_classes = args.num_classes

        # GradNorm decoupling
        self.delay_steps = getattr(args, 'gradnorm_delay_steps', 1)
        self.loss_weights_queue = []          # Ring buffer
        self.loss_weights = torch.ones(3, requires_grad=True, device=self.args.device)
        self.optimizer_weights = torch.optim.Adam([self.loss_weights], lr=0.01)

    # Utility functions
    def _enqueue_weights(self, w):
        self.loss_weights_queue.append(w.detach().clone())
        if len(self.loss_weights_queue) > self.delay_steps:
            self.loss_weights_queue.pop(0)
    def _get_delayed_weights(self):
        """Return the oldest weights in queue for current round."""
        if not self.loss_weights_queue:          # Return uniform if empty
            return torch.ones(3, device=self.args.device)
        return self.loss_weights_queue[0]        # Front = delayed weights
    @torch.no_grad()
    def _snapshot_params(self, params):
        """Detached clone for re-forward."""
        return [p.clone() for p in params]

    def _grad_norm(self, loss_fn, inputs, targets, teacher_out=None):
        """
        Re-forward to get fresh loss tensor, then compute grad norm.
        Args:
            loss_fn : callable taking (logits, targets, teacher_out) -> loss
            inputs  : images
            targets : labels
            teacher_out : teacher output (optional)
        """
        features = self.real_net.extract_features(inputs)
        logits = self.real_net.only_liner(features)
        loss = loss_fn(logits, targets, teacher_out)   # Fresh tensor, never backwarded
        grads = torch.autograd.grad(
            loss, self.body_params,
            create_graph=False, only_inputs=True, allow_unused=True
        )
        norms = [g.norm() for g in grads if g is not None]
        return torch.stack(norms).mean() if norms else torch.tensor(0.0, device=self.args.device)

    def _get_classifier_layer(self, net):
        if hasattr(net, 'linear'):
            return net.linear
        if hasattr(net, 'fc'):
            return net.fc
        raise AttributeError('Model must expose linear or fc classifier head')

    def _get_classifier_weights(self, net):
        weights = self._get_classifier_layer(net).weight.detach()
        return F.normalize(weights, p=2, dim=1)

    @torch.no_grad()
    def _compute_local_logit_means(self, net):
        """Per-client class mean in logit space (same space as baseline contrastive)."""
        num_classes = self.num_classes
        class_sums = torch.zeros(num_classes, num_classes, device=self.args.device)
        class_counts = torch.zeros(num_classes, device=self.args.device)
        for images, labels in self.ldr_train:
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            features = net.extract_features(images)
            logits = net.only_liner(features)
            for label in labels.unique():
                mask = labels == label
                lbl = label.item()
                class_sums[lbl] += logits[mask].sum(dim=0)
                class_counts[lbl] += mask.sum().item()

        contrast_anchors = torch.eye(num_classes, device=self.args.device)
        classifier_rows = self._get_classifier_weights(net)
        if classifier_rows.size(1) == num_classes:
            contrast_anchors = classifier_rows.clone()
        for lbl in range(num_classes):
            if class_counts[lbl] > 0:
                mean_vec = class_sums[lbl] / class_counts[lbl]
                contrast_anchors[lbl] = mean_vec / (mean_vec.norm() + 1e-8)
        return contrast_anchors

    def _get_contrast_anchors(self, net):
        mode = getattr(self.args, 'contrast_target', 'shared')
        if mode == 'shared':
            return self.anchor
        if mode == 'classifier':
            return self._get_classifier_weights(net)
        return self._compute_local_logit_means(net)

    # Training
    def train(self, net, teacher_net, lr, idx=-1, local_eps=None):
        net.train()
        teacher_net.eval()

        self.real_net = net.module if hasattr(net, 'module') else net
        self.body_params = [p for name, p in self.real_net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in self.real_net.named_parameters() if 'linear' in name]
        for p in head_params:
            p.requires_grad = False

        optimizer = torch.optim.SGD(self.body_params, lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        epoch_loss = []
        local_eps = self.args.local_ep_pretrain if self.pretrain \
                    else self.args.local_ep if local_eps is None else local_eps

        # Warm up queue
        while len(self.loss_weights_queue) < self.delay_steps:
            self._enqueue_weights(torch.ones(3, device=self.args.device))

        ot_cost = getattr(self.args, 'ot_cost', 'anchor_geometry')

        for epoch in range(local_eps):
            contrast_anchors = self._get_contrast_anchors(self.real_net)
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                # 1. Main forward
                features = self.real_net.extract_features(images)
                logits = self.real_net.only_liner(features)

                loss_ce = self.loss_func(logits, labels)

                contrast_loss = AnchorContrastiveLoss(
                    anchors=contrast_anchors, temperature=0.5, device=self.args.device
                )(features=logits, labels=labels)

                with torch.no_grad():
                    t_net = teacher_net.module if hasattr(teacher_net, 'module') else teacher_net
                    teacher_out = t_net(images)
                distillation_loss = AnchorDistillationLoss(
                    logits, teacher_out, self.anchor, temperature=1.0, ot_cost=ot_cost)()

                # 2. Re-forward for fresh sub-losses, compute grad norms (no double-backward bug)
                grad_norms = torch.stack([
                    self._grad_norm(lambda logits, y, _: self.loss_func(logits, y),
                                    images, labels, None),
                    self._grad_norm(lambda logits, y, _: AnchorContrastiveLoss(
                        anchors=contrast_anchors, temperature=0.5, device=self.args.device
                    )(features=logits, labels=y),
                                    images, labels, None),
                    self._grad_norm(lambda logits, _, t: AnchorDistillationLoss(
                        logits, t, self.anchor, temperature=1.0, ot_cost=ot_cost)(),
                                    images, labels, teacher_out)
                ])

                # 3. Main loss backward (graph released)
                delayed_w = self._get_delayed_weights()
                loss = delayed_w[0]*loss_ce + delayed_w[1]*contrast_loss + delayed_w[2]*distillation_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 4. GradNorm update weights
                target_norm = grad_norms.mean()
                epsilon = 1e-6
                loss_ratios = (grad_norms + epsilon) / (target_norm + epsilon)
                grad_loss = loss_ratios - loss_ratios.mean()

                self.optimizer_weights.zero_grad()
                self.loss_weights.backward(gradient=grad_loss)
                self.optimizer_weights.step()

                with torch.no_grad():
                    self.loss_weights.data = 3 * self.loss_weights.data / (self.loss_weights.data.sum() + 1e-8)
                self._enqueue_weights(self.loss_weights)

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # Aggregate prototypes
        agg_protos_label = {}
        proto_counts = {}
        with torch.no_grad():
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                features = self.real_net.extract_features(images)
                for label in labels.unique():
                    mask = labels == label
                    lbl = label.item()
                    feat = features[mask]
                    if feat.numel() == 0:
                        continue
                    proto_counts[lbl] = proto_counts.get(lbl, 0) + mask.sum().item()
                    weights = torch.softmax(feat.norm(dim=1), dim=0)
                    weighted = (feat.T @ weights).T
                    agg_protos_label[lbl] = agg_protos_label.get(lbl, 0) + weighted.cpu()
            for lbl in agg_protos_label:
                agg_protos_label[lbl] /= len(self.ldr_train)
                # Normalize to unit direction before upload (paper Eq. 21)
                norm = agg_protos_label[lbl].norm() + 1e-8
                agg_protos_label[lbl] = agg_protos_label[lbl] / norm

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), agg_protos_label, proto_counts


class AnchorDistillationLoss(nn.Module):       # Original AD version
    def __init__(self, student_outputs, teacher_outputs, anchors, temperature=1.0,
                 lambda_anchor=0.1, device='cuda', ot_cost='anchor_geometry'):
        """
        Initialize anchor distillation loss.

        Args:
            student_outputs (torch.Tensor): Student model raw outputs (logits).
            teacher_outputs (torch.Tensor): Teacher model raw outputs (logits).
            anchors (torch.Tensor): Anchor tensor.
            temperature (float): Temperature parameter tau for distillation.
            lambda_anchor (float): Weight lambda for anchor cost term.
            device (str): Compute device ('cuda' or 'cpu').
            ot_cost (str): 'anchor_geometry' or 'uniform'.
        """
        super(AnchorDistillationLoss, self).__init__()
        self.temperature = temperature
        self.lambda_anchor = lambda_anchor
        self.student_outputs = student_outputs
        self.teacher_outputs = teacher_outputs
        self.device = device
        self.ot_cost = ot_cost
        
        # Sinkhorn-Knopp parameters
        self.sinkhorn_iterations = 10
        self.sinkhorn_epsilon = 0.1
        
        # Ensure anchors shape is [num_classes, feature_dim]
        # and set as non-trainable parameter
        self.anchors = nn.Parameter(anchors, requires_grad=False)
        
        # Adjust if anchor feature dim does not match num_classes
        num_classes = student_outputs.size(1)
        if self.anchors.size(0) != num_classes:
            # Logic assumes first dim is num_classes; adjust if needed
            pass
        
        # Note: original adjust_anchors may not apply to all cases; use with caution
        # if anchors.size(1) != num_classes:
        #     self.anchors = self.adjust_anchors(anchors, num_classes)


    def adjust_anchors(self, anchors, num_classes):
        """
        Adjust anchor feature dim to match num_classes via linear layer.
        """
        # May not be needed since cost is based on internal anchor distances
        linear_transform = nn.Linear(anchors.size(1), num_classes).to(self.device)
        adjusted_anchors = linear_transform(anchors)
        return nn.Parameter(adjusted_anchors, requires_grad=False)

    def sinkhorn_knopp(self, cost_matrix, p_s, p_t):
        """
        Sinkhorn-Knopp algorithm with marginal constraints matching p_s and p_t (paper Eq. 4).

        Args:
            cost_matrix (torch.Tensor): Anchor cost matrix [C, C]
            p_s (torch.Tensor): Student prediction distribution [B, C]
            p_t (torch.Tensor): Teacher prediction distribution [B, C]

        Returns:
            transport_matrix (torch.Tensor): Optimal transport matrix [B, C, C]
        """
        batch_size = p_s.size(0)

        # K = exp(-C/eps), expand to batch dim
        K = torch.exp(
            -cost_matrix.unsqueeze(0).expand(batch_size, -1, -1) / self.sinkhorn_epsilon
        )  # [B, C, C]

        # Initialize with actual marginals (paper U(p_S, p_T) constraint)
        u = p_s.unsqueeze(2)   # [B, C, 1]
        v = p_t.unsqueeze(2)   # [B, C, 1]

        for _ in range(self.sinkhorn_iterations):
            u = p_s.unsqueeze(2) / (torch.bmm(K, v) + 1e-8)
            v = p_t.unsqueeze(2) / (torch.bmm(K.transpose(1, 2), u) + 1e-8)

        transport_matrix = u * K * v.transpose(1, 2)
        return transport_matrix

    def compute_cost_matrix(self):
        """
        Pure anchor cosine distance cost matrix (paper Eq. 3): Psi(y,y') = 1 - cos(A_y, A_y')
        Uniform mode uses all-ones matrix, keeping OT framework but removing anchor geometry.
        """
        C = self.anchors.size(0)
        if self.ot_cost == 'uniform':
            return torch.ones(C, C, device=self.anchors.device)
        A_norm = F.normalize(self.anchors, p=2, dim=1)   # [C, D]
        cost = 1.0 - torch.matmul(A_norm, A_norm.t())    # [C, C]
        return cost

    def earth_movers_distance(self, cost_matrix, transport_matrix):
        """
        Compute Earth Mover's Distance (Wasserstein Distance).

        Args:
            cost_matrix (torch.Tensor): Cost matrix [B, C, C]
            transport_matrix (torch.Tensor): Optimal transport matrix [B, C, C]

        Returns:
            emd_loss (torch.Tensor): Mean EMD loss
        """
        emd = torch.sum(transport_matrix * cost_matrix, dim=[1, 2])
        return torch.mean(emd)

    def forward(self):
        """
        OT distillation loss based on anchor geometry (paper Eq. 3-5).
        Cost matrix from anchor cosine distance only; Sinkhorn matches p_S and p_T.
        """
        # Student/teacher softmax prediction distributions
        student_probs = F.softmax(self.student_outputs / self.temperature, dim=1)
        teacher_probs = F.softmax(self.teacher_outputs / self.temperature, dim=1)

        # Pure anchor cosine distance cost matrix [C, C]
        cost = self.compute_cost_matrix()

        # Sinkhorn optimal transport (marginal matching p_S, p_T) [B, C, C]
        transport_matrix = self.sinkhorn_knopp(cost, student_probs, teacher_probs)

        # EMD loss: sum Pi* o Psi, expand cost to batch
        batch_size = student_probs.size(0)
        cost_batch = cost.unsqueeze(0).expand(batch_size, -1, -1)
        emd_loss = self.earth_movers_distance(cost_batch, transport_matrix)

        return emd_loss

'''class AnchorDistillationLoss(nn.Module):       # Original AD version
    def __init__(self, student_outputs, teacher_outputs, anchors, temperature=1.0, lambda_anchor=0.1, device='cuda'):
        """
        Initialize anchor distillation loss.

        Args:
            student_outputs (torch.Tensor): Student model raw outputs (logits).
            teacher_outputs (torch.Tensor): Teacher model raw outputs (logits).
            anchors (torch.Tensor): Anchor tensor.
            temperature (float): Temperature parameter tau for distillation.
            lambda_anchor (float): Weight lambda for anchor cost term.
            device (str): Compute device ('cuda' or 'cpu').
        """
        super(AnchorDistillationLoss, self).__init__()
        self.temperature = temperature
        self.lambda_anchor = lambda_anchor
        self.student_outputs = student_outputs
        self.teacher_outputs = teacher_outputs
        self.device = device
        
        # Sinkhorn-Knopp parameters
        self.sinkhorn_iterations = 10
        self.sinkhorn_epsilon = 0.1
        
        # Ensure anchors shape is [num_classes, feature_dim]
        # and set as non-trainable parameter
        self.anchors = nn.Parameter(anchors, requires_grad=False)
        
        # Adjust if anchor feature dim does not match num_classes
        num_classes = student_outputs.size(1)
        if self.anchors.size(0) != num_classes:
            # Logic assumes first dim is num_classes; adjust if needed
            pass
        
        # Note: original adjust_anchors may not apply to all cases; use with caution
        # if anchors.size(1) != num_classes:
        #     self.anchors = self.adjust_anchors(anchors, num_classes)


    def adjust_anchors(self, anchors, num_classes):
        """
        Adjust anchor feature dim to match num_classes via linear layer.
        """
        # May not be needed since cost is based on internal anchor distances
        linear_transform = nn.Linear(anchors.size(1), num_classes).to(self.device)
        adjusted_anchors = linear_transform(anchors)
        return nn.Parameter(adjusted_anchors, requires_grad=False)

    def sinkhorn_knopp(self, cost_matrix):
        """
        Sinkhorn-Knopp algorithm implementation for computing optimal transport matrix.
        Input is cost matrix C; internally computes K = exp(-C/eps).

        Args:
            cost_matrix (torch.Tensor): Cost matrix [batch_size, num_classes, num_classes]
            
        Returns:
            transport_matrix (torch.Tensor): Optimal transport matrix [batch_size, num_classes, num_classes]
        """
        batch_size, n, m = cost_matrix.shape
        
        # Initialize transport matrix K = exp(-C/eps)
        K = torch.exp(-cost_matrix / self.sinkhorn_epsilon)
        
        # Initialize row and column scaling factors
        u = torch.ones(batch_size, n, 1, device=self.device) / n
        v = torch.ones(batch_size, m, 1, device=self.device) / m
        
        # Sinkhorn iterations
        for _ in range(self.sinkhorn_iterations):
            u = 1.0 / (torch.bmm(K, v) + 1e-8)
            v = 1.0 / (torch.bmm(K.transpose(1, 2), u) + 1e-8)
        
        # Compute optimal transport matrix T = diag(u) * K * diag(v)
        transport_matrix = u * K * v.transpose(1, 2)
        
        return transport_matrix
    
    def compute_cost_matrix(self, student_probs, teacher_probs):
        """
        Compute total cost matrix.
        Anchor term uses cosine distance (or normalized Euclidean), rest unchanged.
        """
        batch_size, num_classes = student_probs.shape

        # Probability term: same as original
        student_expanded = student_probs.unsqueeze(2)   # [B, C, 1]
        teacher_expanded = teacher_probs.unsqueeze(1)   # [B, 1, C]
        prob_cost = torch.pow(student_expanded - teacher_expanded, 2)   # [B, C, C]

        # Anchor term: use cosine distance
        if self.lambda_anchor > 0 and self.anchors is not None:
            # self.anchors: [C, feat_dim]
            A = self.anchors                                   # [C, D]
            A_norm = F.normalize(A, p=2, dim=1)                # unit vectors
            # cosine distance matrix: 1 - cos(x,y)
            cosine_dist = 1.0 - torch.matmul(A_norm, A_norm.t())  # [C, C]
            # Expand to batch
            anchor_cost_term = self.lambda_anchor * cosine_dist.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            anchor_cost_term = 0.0

        total_cost_matrix = prob_cost + anchor_cost_term
        return total_cost_matrix

    def earth_movers_distance(self, cost_matrix, transport_matrix):
        """
        Compute Earth Mover's Distance (Wasserstein Distance).
        Refactored to directly receive cost matrix, avoiding redundant computation.

        Args:
            cost_matrix (torch.Tensor): Cost matrix [batch_size, num_classes, num_classes]
            transport_matrix (torch.Tensor): Optimal transport matrix [batch_size, num_classes, num_classes]
            
        Returns:
            emd_loss (torch.Tensor): Mean EMD loss
        """
        # EMD = sum(T_ij * C_ij), where T is transport matrix and C is cost matrix
        emd = torch.sum(transport_matrix * cost_matrix, dim=[1, 2])
        
        return torch.mean(emd)

    def forward(self):
        """
        Compute distillation loss based on Sinkhorn-Knopp soft alignment.
        Simplified and clarified per revision notes.
        
        Returns:
            loss (torch.Tensor): Final distillation loss
        """
        # 1. Define student and teacher probability distributions PS_tau and PT_tau per formula
        # PS_tau(i, j) = exp(S_ij/tau) / sum_k exp(S_ik/tau)
        student_probs = F.softmax(self.student_outputs / self.temperature, dim=1)
        teacher_probs = F.softmax(self.teacher_outputs / self.temperature, dim=1)
        
        # 2. Incorporate anchor info into distillation cost matrix
        # C_total = ||PS_tau(i) - PT_tau(j)||_2^2 + lambda * ||A'(i) - A'(j)||_2^2
        cost_matrix = self.compute_cost_matrix(student_probs, teacher_probs)
        
        # 3. Input to Sinkhorn-Knopp is exp(-C_total)
        #    Our sinkhorn_knopp receives C_total and internally computes K = exp(-C_total / eps)
        # T = Sinkhorn-Knopp(C_total)
        transport_matrix = self.sinkhorn_knopp(cost_matrix)
        
        # 4. Compute Earth Mover's Distance as final distillation loss
        emd_loss = self.earth_movers_distance(cost_matrix, transport_matrix)
        
        return emd_loss'''


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)  # Load dataset
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()

        # For ablation study
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logits = net(images)
                loss = self.loss_func(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateFedProc(object):
    """FedProc client training with global-prototype contrastive loss."""

    def __init__(self, args, dataset=None, idxs=None, task=0):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(
            dataset, idxs, task, batch_size=self.args.local_bs
        )

    @staticmethod
    def _global_prototype_loss(z, labels, global_prototypes):
        """Return GPC loss, or None when the batch has no valid positive prototype."""
        if not global_prototypes:
            return None

        prototype_items = []
        for raw_label, prototype in global_prototypes.items():
            if prototype is None:
                continue
            class_label = int(raw_label)
            prototype = torch.as_tensor(
                prototype, device=z.device, dtype=z.dtype
            ).detach().reshape(-1)
            if prototype.numel() != z.size(1):
                raise ValueError(
                    "Global prototype dimension mismatch for class "
                    f"{class_label}: expected {z.size(1)}, got {prototype.numel()}."
                )
            prototype_items.append((class_label, prototype))

        if not prototype_items:
            return None

        prototype_items.sort(key=lambda item: item[0])
        prototype_labels = [item[0] for item in prototype_items]
        prototype_matrix = torch.stack([item[1] for item in prototype_items])
        label_to_column = {
            class_label: column
            for column, class_label in enumerate(prototype_labels)
        }

        batch_labels = [int(label) for label in labels.detach().cpu().tolist()]
        valid_mask = torch.tensor(
            [label in label_to_column for label in batch_labels],
            device=z.device,
            dtype=torch.bool,
        )
        if not valid_mask.any().item():
            return None

        valid_z = z[valid_mask]
        target_columns = torch.tensor(
            [label_to_column[label] for label in batch_labels if label in label_to_column],
            device=z.device,
            dtype=torch.long,
        )

        similarities = F.cosine_similarity(
            valid_z.unsqueeze(1), prototype_matrix.unsqueeze(0), dim=2
        )
        return F.cross_entropy(similarities, target_columns)

    def _compute_local_prototypes(self, net):
        """Recompute class means with the final local model in eval mode."""
        net.eval()
        prototype_sums = {}
        prototype_counts = {}

        with torch.no_grad():
            for images, labels in self.ldr_train:
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                _, z = net(images, returnFeature=True)

                for label in labels.unique():
                    class_label = int(label.item())
                    class_features = z[labels == label]
                    feature_sum = class_features.sum(dim=0).detach().cpu()
                    if class_label not in prototype_sums:
                        prototype_sums[class_label] = feature_sum
                        prototype_counts[class_label] = class_features.size(0)
                    else:
                        prototype_sums[class_label] += feature_sum
                        prototype_counts[class_label] += class_features.size(0)

        return {
            class_label: prototype_sums[class_label] / prototype_counts[class_label]
            for class_label in prototype_sums
        }

    def train(
            self, net, lr, global_prototypes, global_round, total_rounds,
            idx=-1, local_eps=None):
        if total_rounds <= 0:
            raise ValueError("total_rounds must be positive.")
        if global_round < 0 or global_round >= total_rounds:
            raise ValueError(
                "global_round must satisfy 0 <= global_round < total_rounds."
            )

        alpha_t = 1.0 - float(global_round) / float(total_rounds)
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=self.args.momentum,
            weight_decay=self.args.wd,
        )

        if local_eps is None:
            local_eps = self.args.local_ep

        epoch_loss = []
        for _ in range(local_eps):
            batch_loss = []
            for images, labels in self.ldr_train:
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                logits, _, z = net(images, return_all=True)

                ce_loss = self.loss_func(logits, labels)
                gpc_loss = self._global_prototype_loss(
                    z, labels, global_prototypes
                )
                if gpc_loss is None:
                    loss = ce_loss
                else:
                    loss = alpha_t * gpc_loss + (1.0 - alpha_t) * ce_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            if not batch_loss:
                raise ValueError("FedProc client received an empty local dataset.")
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if not epoch_loss:
            raise ValueError("local_eps must be positive.")

        local_prototypes = self._compute_local_prototypes(net)
        average_loss = sum(epoch_loss) / len(epoch_loss)
        return net.state_dict(), average_loss, local_prototypes
    
# FedProx

class LocalUpdateFedProx(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False, task = 0):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, body_lr, head_lr):
        net.train()
        g_net = copy.deepcopy(net)
        
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                
                # FedProx regularization
                fed_prox_reg = 0.0
                for l_param, g_param in zip(net.parameters(), g_net.parameters()):
                    fed_prox_reg += (0.1 / 2 * torch.norm((l_param - g_param)) ** 2)
                loss += fed_prox_reg
                
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) 
    

# FedKnow
class LocalUpdateFedKnow(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        if solvers is None:
            raise ImportError(
                "LocalUpdateFedKnow requires the optional dependency 'cvxopt'."
            )
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
      
        # FedKNOW components
        self.task_id = task
        self.signature_tasks = []  # Stores knowledge of previous tasks
        self.k = 5
        self.rho = 0.1
      
        # Gradient integration parameters
        self.epsilon = 1e-5  # Small constant for numerical stability
        solvers.options['show_progress'] = False  # Disable QP solver output
        self.task_features = []  # Store per-task features (avg gradients)
        self.task_weights = []   # Store per-task weight knowledge

    '''def _extract_knowledge(self, net):
        """Extract top (1-rho)% important weights as task knowledge"""
        weights = []
        for param in net.parameters():
            if len(param.shape) > 1:  # Weight matrices only
                flattened = param.data.abs().flatten()
                threshold = torch.quantile(flattened, 1 - self.rho)
                mask = (param.data.abs() >= threshold).float()
                weights.append((param.data * mask, mask))
        return weights'''
    def _extract_knowledge(self, net, images):
        """Extract knowledge and compute task features."""
    def _extract_knowledge(self, net, images):
        """Extract knowledge and compute task features."""
        # Use current task avg gradient as feature
        net.zero_grad()
        outputs = net(images)
        loss = self.loss_func(outputs, outputs.softmax(dim=1).argmax(dim=1))
        loss.backward()
        task_feature = torch.cat([p.grad.flatten().abs().mean().unsqueeze(0) for p in net.parameters()])

        # Original weight extraction logic
        weights = []
        for param in net.parameters():
            if len(param.shape) > 1:  # Weight matrices only
                flattened = param.data.abs().flatten()
                threshold = torch.quantile(flattened, 1 - self.rho)
                mask = (param.data.abs() >= threshold).float()
                weights.append((param.data * mask, mask))
        return weights, task_feature
    def _restore_gradients(self, net, images):
        """Restore gradients from signature tasks"""
        restored_grads = []
        for task_weights, masks in self.signature_tasks:
            # Set network to signature task weights
            idx = 0
            for param in net.parameters():
                if len(param.shape) > 1:
                    param.data = task_weights[idx].to(self.args.device)
                    idx += 1
          
            # Compute gradient w.r.t. current task data
            net.zero_grad()
            outputs = net(images)
            pseudo_labels = outputs.detach()
            loss = self.loss_func(outputs, pseudo_labels.softmax(dim=1).argmax(dim=1))
            loss.backward()
          
            # Apply original masks and store gradient
            grads = []
            idx = 0
            for param in net.parameters():
                if len(param.shape) > 1:
                    grad = param.grad * masks[idx].to(self.args.device)
                    grads.append(grad.flatten())
                    idx += 1
            restored_grads.append(torch.cat(grads))
        return restored_grads

    '''def _integrate_gradients(self, current_grad, restored_grads):
        """Quadratic programming for gradient integration"""
        if not restored_grads:
            return current_grad
      
        # Prepare constraints: G'*g >= 0
        G = torch.stack(restored_grads).cpu().numpy()
        G = -np.vstack([G, -np.eye(G.shape[1])])  # Non-negativity constraints
        h = np.zeros(G.shape[0])
      
        # Quadratic programming setup
        P = matrix(np.eye(len(current_grad)))
        q = matrix(-current_grad.cpu().numpy())
        G = matrix(G)
        h = matrix(h)
      
        # Solve QP problem
        try:
            sol = solvers.qp(P, q, G, h)
            v = np.array(sol['x']).flatten()
            integrated_grad = current_grad + torch.from_numpy(v).to(self.args.device)
            return integrated_grad
        except:
            return current_grad  # Fallback to original gradient'''
    def _integrate_gradients(self, current_grad, restored_grads):
        """Fixed gradient integration method."""
        if not restored_grads:
            return current_grad

        # 1. Build constraint matrix (paper Eq. 3)
        G = torch.stack(restored_grads).cpu().numpy()
        h = np.zeros(len(restored_grads))  # Gg' >= 0

        # 2. QP parameter setup
        P = matrix(np.eye(len(current_grad)))
        q = matrix(-current_grad.cpu().numpy())
        G = matrix(-G)  # Convert to <= 0 constraint
        h = matrix(h)

        # 3. Solve QP
        try:
            sol = solvers.qp(P, q, G, h)
            v = np.array(sol['x']).flatten()
            return current_grad + torch.from_numpy(v).to(self.args.device)
        except:
            return current_grad

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
      
        # Extract knowledge from previous tasks
        if self.task_id > 0 and not self.pretrain:
            self.signature_tasks = self.signature_tasks[-self.k:]  # Keep only k recent
      
        epoch_loss = []
        local_eps = self.args.local_ep if local_eps is None else local_eps
      
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
              
                # 1. Compute current task gradient
                optimizer.zero_grad()
                outputs = net(images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                current_grad = torch.cat([p.grad.flatten() for p in net.parameters()])
              
                # 2. Restore gradients from signature tasks
                restored_grads = self._restore_gradients(net, images)
              
                # 3. Gradient integration
                integrated_grad = self._integrate_gradients(current_grad, restored_grads)
              
                # 4. Update parameters with integrated gradient
                idx = 0
                for param in net.parameters():
                    if param.grad is not None:
                        param.grad = integrated_grad[idx:idx+param.numel()].view(param.shape)
                        idx += param.numel()
                optimizer.step()
              
                batch_loss.append(loss.item())
          
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
      
        # Store current task knowledge
        if not self.pretrain:
            '''task_knowledge = self._extract_knowledge(net)
            self.signature_tasks.append(task_knowledge)
            self.task_id += 1'''
            task_knowledge, task_feature = self._extract_knowledge(net, images)
            self.task_weights.append(task_knowledge)
            self.task_features.append(task_feature)
            self.task_id += 1

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
#TARGET: client-side update (ICCV'23, exemplar-free distillation via a server-side generator)
def _KD_loss(pred, soft, T):
    """Old-task knowledge distillation loss (official TARGET implementation)."""
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


class LocalUpdateTARGET(object):
    _KD_DEGENERATE_WARNED = False  # warn once per process when old-class KD is skipped

    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False, syn_loader=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.syn_loader = syn_loader  # unlabeled synthetic data from the server-side generator
        # classes covered by tasks [0, task): the old classes kept by distillation
        self.old_classes = task * (args.num_classes // args.task_num) if args.task_num > 0 else 0

    def train(self, net, teacher_model, lr, idx=-1, local_eps=None):
        net.train()
        if teacher_model is not None:
            teacher_model.eval()  # frozen snapshot of the global model at the last task switch

        # Combined optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                   momentum=self.args.momentum,
                                   weight_decay=self.args.wd)

        epoch_loss = []
        local_eps = self.args.local_ep if local_eps is None else local_eps

        use_kd = self.syn_loader is not None and teacher_model is not None
        if use_kd and self.old_classes < 2:
            # single old class makes the old-class softmax degenerate (e.g. 1 class/task splits)
            use_kd = False
            if not LocalUpdateTARGET._KD_DEGENERATE_WARNED:
                print('[TARGET] old_classes={} (<2): old-class KD degenerates, CE-only local training'.format(self.old_classes))
                LocalUpdateTARGET._KD_DEGENERATE_WARNED = True

        for _ in range(local_eps):
            batch_loss = []

            if use_kd:
                # Iterate real and synthetic data together
                for (images, labels), syn_batch in zip(self.ldr_train, self.syn_loader):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    syn_images = syn_batch[0].to(self.args.device)

                    # Current task loss (global labels on the full head, FLOAM protocol)
                    logits = net(images)
                    ce_loss = self.loss_func(logits, labels)

                    # Old task distillation on synthetic data against the frozen teacher
                    s_out = net(syn_images)
                    with torch.no_grad():
                        t_out = teacher_model(syn_images)
                    kd_loss = _KD_loss(s_out[:, :self.old_classes],
                                       t_out[:, :self.old_classes],
                                       self.args.target_kd_T)

                    loss = ce_loss + self.args.target_kd * kd_loss

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_loss.append(loss.item())
            else:
                # Train with real data only
                for images, labels in self.ldr_train:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)

                    # Forward pass
                    logits = net(images)
                    loss = self.loss_func(logits, labels)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

#ReFed
class LocalUpdateReFed(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain

        # Initialize personalized info model (PIM)
        self.pim = copy.deepcopy(args.global_model)  # Init PIM from global model
        self.cached_samples = []  # Cached important samples

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()

        # Define optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []

        if local_eps is None:
            if self.pretrain:
                local_epochs = self.args.local_ep_pretrain
            else:
                local_epochs = self.args.local_ep
        else:
            local_epochs = local_eps

        for iter in range(local_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                # Forward pass
                logits = net(images)
                loss = self.loss_func(logits, labels)

                # Backward pass to update local model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # After local training, update PIM and compute sample importance
        self.update_pim_and_cache_samples(net)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_pim_and_cache_samples(self, net):
        # Update personalized info model (PIM)
        self.pim.train()
        pim_optimizer = torch.optim.SGD(self.pim.parameters(), lr=self.args.lr_pim,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.wd)

        importance_scores = {}

        # Update PIM with local data and record sample grad norms
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            # Forward pass
            logits = self.pim(images)
            loss = self.loss_func(logits, labels)

            # Backward pass to update PIM
            pim_optimizer.zero_grad()
            loss.backward()
            pim_optimizer.step()

            # Compute sample grad norm as importance score
            for i in range(len(images)):
                sample_grad_norm = torch.norm(self.pim.fc.weight.grad[i]).item()
                if (images[i], labels[i]) not in importance_scores:
                    importance_scores[(images[i], labels[i])] = 0
                importance_scores[(images[i], labels[i])] += sample_grad_norm

        # Cache samples by importance score
        sorted_samples = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        max_cache_size = self.args.max_cache_size
        self.cached_samples = [sample for sample, _ in sorted_samples[:max_cache_size]]

        # Merge cached samples with new task data for next training
        self.ldr_train = combine_cached_and_new_data(self.cached_samples, self.ldr_train)

def combine_cached_and_new_data(cached_samples, new_data_loader):
    """Merge cached samples with new task data"""
    cached_dataset = CachedDataset(cached_samples)
    combined_dataset = ConcatDataset([cached_dataset, new_data_loader.dataset])
    return DataLoader(combined_dataset, batch_size=new_data_loader.batch_size, shuffle=True)

class CachedDataset(torch.utils.data.Dataset):
    """Custom dataset for cached samples"""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
#EWC
class LocalUpdateEWC(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # Load dataset
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        
        # EWC init
        self.fisher = None  # Store Fisher information matrix
        self.old_params = None  # Store previous model params

    def compute_fisher(self, net):
        """
        Compute Fisher information matrix for the current task.
        """
        fisher = {}
        for name, param in net.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        net.eval()
        total_samples = 0
        for images, labels in self.ldr_train:
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            logits = net(images)
            loss = self.loss_func(logits, labels)
            
            net.zero_grad()
            loss.backward()
            
            # Update Fisher info matrix
            for name, param in net.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * len(labels)
            total_samples += len(labels)
        
        # Normalize Fisher info matrix
        for name in fisher:
            fisher[name] /= total_samples
        
        self.fisher = fisher

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()

        # Init optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        # Set default local epochs if not specified
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                # Forward pass
                logits = net(images)
                ce_loss = self.loss_func(logits, labels)
                
                # EWC regularization
                ewc_loss = 0
                if self.fisher is not None and self.old_params is not None:
                    for name, param in net.named_parameters():
                        if name in self.fisher:
                            ewc_loss += torch.sum(self.fisher[name] * (param - self.old_params[name]).pow(2))
                
                # Total loss = CE + EWC
                total_loss = ce_loss +  ewc_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_loss.append(total_loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # Return updated model params and avg loss
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_old_params(self, net):
        """
        Save current model parameters as old parameters.
        """
        self.old_params = {name: param.clone().detach() for name, param in net.named_parameters()}

#CGoFed
class LocalUpdateCGoFed(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # Load training data for the current task
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.task = task  # Store the current task index

    def train(self, net, lr, idx=-1, local_eps=None, historical_basis_vectors=None):
        net.train()

        # Initialize optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        task_representation_matrix = None  # To store the representation matrix for the current task

        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                # Forward pass
                logits = net(images)
                loss = self.loss_func(logits, labels)

                # Backward pass with relax-constrained gradient update
                optimizer.zero_grad()
                loss.backward()

                if self.task > 0:  # Only apply constraints for tasks after the first one
                    self.relax_constrained_gradient_update(net, self.task, historical_basis_vectors)

                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # Compute the representation matrix for the current task
        task_representation_matrix = self.compute_representation_matrix(net)

        # Return the updated model state, average loss, and task representation matrix
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), task_representation_matrix

    def relax_constrained_gradient_update(self, net, task, historical_basis_vectors):
        """
        Apply relax-constrained gradient update to balance stability and plasticity.
        This function restricts the gradient update direction based on historical tasks' gradient spaces.
        """
        # Retrieve the memory of basis vectors for historical tasks
        Mt = historical_basis_vectors.get(task, None)
        if Mt is None:
            # If no historical basis vectors are available, skip the constrained update
            return

        # Compute the relaxation coefficient μt
        μt = self.compute_relaxation_coefficient(task)

        # Project the gradients onto the orthogonal space of historical tasks with relaxation
        for param in net.parameters():
            if param.grad is not None:
                grad = param.grad.data
                projected_grad = μt * torch.matmul(torch.matmul(grad, Mt), Mt.T)
                param.grad.data -= projected_grad

    def compute_representation_matrix(self, net):
        """
        Compute the representation matrix for the current task using a subset of samples.
        This matrix represents the feature space of the current task.
        """
        # Randomly sample a subset of data from the current task
        sample_loader = self.sample_task_data(self.ldr_train)
        representations = []

        with torch.no_grad():
            for images, _ in sample_loader:
                images = images.to(self.args.device)
                features = net.extract_features(images)
                representations.append(features.cpu().numpy())

        # Concatenate all representations into a single matrix
        representation_matrix = np.concatenate(representations, axis=0)
        return representation_matrix

    def retrieve_historical_basis_vectors(self, task):
        """
        Retrieve the basis vectors of historical tasks from memory.
        These basis vectors represent the gradient spaces of previous tasks.
        """
        # Placeholder for retrieving historical basis vectors (e.g., from server or local memory)
        # Assume `self.memory` stores the basis vectors for all tasks
        return self.memory[task]

    def compute_relaxation_coefficient(self, task):
        """
        Compute the relaxation coefficient μt based on the forgetting threshold τ and decay rate α.
        """
        # Example computation of μt (adjust as needed)
        α = self.args.alpha  # Decay rate
        τ = self.args.tau  # Forgetting threshold
        AF = self.compute_average_forgetting()  # Compute average forgetting metric
        μt = α ** task if AF < τ else α ** (task - self.args.t_tau)
        return μt

    def compute_average_forgetting(self):
        """
        Compute the average forgetting metric across historical tasks.
        """
        # Placeholder for computing average forgetting (AF)
        # Assume this function uses historical accuracy metrics
        return 0.0  # Replace with actual implementation

    def sample_task_data(self, loader):
        """
        Sample a subset of data from the current task for representation matrix computation.
        """
        # Placeholder for sampling logic
        return DataLoader(loader.dataset, batch_size=self.args.sample_bs, shuffle=True)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

class CDAPGenerator(nn.Module):
    def __init__(self, feat_dim=None, prompt_dim=64, num_tasks=5):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.feat_dim = feat_dim
        self.ln = None
        self.mlp = None
        self.ccda = nn.Linear(prompt_dim, prompt_dim)
        self.affine_predictor = None

        if self.feat_dim is not None:
            self._build_backbone(self.feat_dim)

    def _build_backbone(self, feat_dim):
        self.feat_dim = feat_dim
        self.ln = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, self.prompt_dim)
        )
        self.affine_predictor = nn.Sequential(
            nn.Linear(feat_dim, self.prompt_dim * 2),
            nn.GELU(),
            nn.Linear(self.prompt_dim * 2, self.prompt_dim * 2)
        )

    def ensure_initialized(self, feat_dim, device):
        if self.ln is None or self.feat_dim != feat_dim:
            self._build_backbone(feat_dim)
            self.to(device)
        
    def forward(self, features, task_id):
        self.ensure_initialized(features.size(-1), features.device)
        # Sample-level prompts
        x = self.ln(features)
        sample_prompts = self.mlp(x)  # [batch, prompt_dim]
        
        # Apply CCDA layer
        adapted_prompts = self.ccda(sample_prompts)
        
        # Predict LT parameters
        affine_params = self.affine_predictor(features)  # [batch, prompt_dim*2]
        alpha, lambda_ = torch.chunk(affine_params, 2, dim=1)  # each [batch, prompt_dim]
        
        # Apply LT transformation
        final_prompts = alpha * adapted_prompts + lambda_
        
        return final_prompts  # [batch, prompt_dim]



class LocalUpdateRifFiL(object):
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False, prompt_dim=None):  # add prompt_dim parameter
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        
        # RefFiL modules
        # Use provided prompt_dim or fallback to args.prompt_dim
        self.prompt_dim = prompt_dim if prompt_dim is not None else args.prompt_dim
        print(f"[Client] Using prompt_dim={self.prompt_dim}")
        
        self.cdap = CDAPGenerator(prompt_dim=self.prompt_dim)  # dynamically initialize based on actual feature dimension
        self.current_global_prompts = None
        self.temperature = nn.Parameter(torch.tensor(0.9))
        self.prompt_cache = {}
        self.current_task = task
        self.current_labels = []  # add label cache
        
    def compute_gpl_loss(self, logits_global, labels):
        return F.cross_entropy(logits_global, labels)
    
    def compute_dpcl_loss(self, local_prompts, global_prototypes, labels):
        """
        Reconstructed domain-specific prompt contrastive learning.
        Key improvement: use class prototypes instead of all global prompts.
        """
        # 1. Dimension check
        assert local_prompts.dim() == 2, f"Expected 2D tensor, got dim: {local_prompts.dim()}"
        assert global_prototypes.dim() == 2, f"Expected 2D tensor, got dim: {global_prototypes.dim()}"
        assert local_prompts.size(1) == global_prototypes.size(1), \
            f"Prompt dimension mismatch: local {local_prompts.size(1)} vs global {global_prototypes.size(1)}"
        
        # 2. Extract class-specific global prototypes
        target_prototypes = global_prototypes[labels]  # [batch, prompt_dim]
        
        # 3. Compute similarity between local prompts and class prototypes
        similarities = F.cosine_similarity(
            local_prompts,
            target_prototypes,
            dim=1
        )  # [batch]
        
        # 4. Temperature scaling and loss computation
        logits = similarities / self.temperature.clamp(min=0.3)
        return -logits.mean()  # maximize similarity


    def train(self, net, lr, idx=-1, local_eps=None):
        self.update_temperature(self.current_task)
        net.train()
        self.cdap.train()

        # Initialize CDAP with actual feature dim to avoid hardcoded LayerNorm size
        init_images, _ = next(iter(self.ldr_train))
        init_images = init_images.to(self.args.device)
        with torch.no_grad():
            init_features = net.extract_features(init_images)
        self.cdap.ensure_initialized(init_features.size(1), self.args.device)

        optimizer = torch.optim.SGD(
            list(net.parameters()) + list(self.cdap.parameters()),
            lr=lr,
            momentum=self.args.momentum,
            weight_decay=self.args.wd
        )

        epoch_loss = []
        local_eps = self.args.local_ep if local_eps is None else local_eps
        
        for _ in range(local_eps):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.current_labels = labels  # cache current batch labels
                
                # Feature extraction
                features = net.extract_features(images)
                logits_local = net.only_liner(features)
                
                # Generate local prompts
                task_ids = torch.full((len(images),), self.current_task, 
                                    device=self.args.device)
                local_prompts = self.cdap(features.detach(), task_ids)
                
                # Global prompt processing
                with torch.no_grad():
                    # Build global class-prototype matrix
                    global_prototypes = torch.zeros(
                        self.args.num_classes, 
                        self.prompt_dim,
                        device=self.args.device
                    )
                    
                    # Aggregate per-class mean prompts
                    for cls in range(self.args.num_classes):
                        if cls in self.current_global_prompts:
                            cls_prompts = self.current_global_prompts[cls]
                            global_prototypes[cls] = cls_prompts.mean(dim=0)
                        else:
                            # Fallback: random init
                            global_prototypes[cls] = torch.randn(
                                self.prompt_dim, 
                                device=self.args.device
                            )
                    
                    # Keep raw global prompts for GPL loss
                    all_prompt_features = []
                    all_prompt_labels = []
                    
                    # Collect global prompts
                    for label in labels.unique():
                        label_int = label.item()
                        if label_int in self.current_global_prompts:
                            class_prompts = self.current_global_prompts[label_int]
                            num_prompts = class_prompts.size(0)
                            
                            all_prompt_features.append(class_prompts)
                            all_prompt_labels.append(label.repeat(num_prompts))
                    
                    # Handle missing global prompts
                    if all_prompt_features:
                        global_prompts = torch.cat(all_prompt_features)
                        prompt_labels = torch.cat(all_prompt_labels)
                    else:
                        num_fallback = min(10, len(labels))
                        global_prompts = torch.randn(
                            num_fallback, self.prompt_dim,
                            device=self.args.device
                        )
                        prompt_labels = torch.randint(
                            0, self.args.num_classes, (num_fallback,), 
                            device=self.args.device
                        )
                    
                    # Get real sample features for GPL loss
                    real_global_features = []
                    valid_global_labels = []
                    
                    # Vectorized ops for efficiency
                    unique_labels = prompt_labels.unique()
                    label_mask_dict = {}
                    
                    for label in unique_labels:
                        mask = (labels == label)
                        if mask.any():
                            indices = torch.where(mask)[0]
                            label_mask_dict[label.item()] = indices
                    
                    for label in prompt_labels:
                        label_val = label.item()
                        if label_val in label_mask_dict and len(label_mask_dict[label_val]) > 0:
                            idx = torch.randint(0, len(label_mask_dict[label_val]), (1,))
                            real_idx = label_mask_dict[label_val][idx]
                            real_global_features.append(features[real_idx])
                            valid_global_labels.append(label)
                        else:
                            rand_idx = torch.randint(0, len(features), (1,))
                            real_global_features.append(features[rand_idx])
                            valid_global_labels.append(labels[rand_idx][0])
                    
                    # Convert results
                    real_global_features = torch.cat(real_global_features)
                    valid_global_labels = torch.tensor(valid_global_labels, 
                                                    device=self.args.device,
                                                    dtype=torch.long)
                    
                    # Compute global logits
                    logits_global = net.only_liner(real_global_features)
                
                # Loss computation
                ce_loss = self.loss_func(logits_local, labels)
                gpl_loss = self.compute_gpl_loss(logits_global, valid_global_labels)
                
                # DPCL loss with class prototypes (key fix #3)
                if len(local_prompts) == 0:
                    dpcl_loss = torch.tensor(0.0, device=self.args.device)
                else:
                    # Sanity-check dimensions
                    assert local_prompts.size(-1) == global_prototypes.size(-1), \
                        f"Prompt dimension mismatch: local {local_prompts.size(-1)} vs global prototype {global_prototypes.size(-1)}"
                    
                    # Use refactored DPCL loss
                    dpcl_loss = self.compute_dpcl_loss(
                        local_prompts, 
                        global_prototypes,  # use class prototypes instead of all prompts
                        labels
                    )
                
                total_loss = ce_loss + 0.5*gpl_loss + 0.3*dpcl_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                batch_loss.append(total_loss.item())
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return {
            "model_state": net.state_dict(),
            "prompts": local_prompts.detach().cpu(),
            "loss": sum(epoch_loss)/len(epoch_loss)
        }


    def update_global_prompts(self, global_prompts_dict):
        device = self.args.device
        self.current_global_prompts = {
            class_id: prompts.to(device) 
            for class_id, prompts in global_prompts_dict.items()
        }
        self.prompt_cache = {}
        
    def set_current_task(self, task_id):
        self.current_task = task_id
    def update_temperature(self, current_task):
    # Temperature decay schedule (from paper)
        tau_min = 0.3
        gamma = 0.1
        beta = 0.05
        tau = 0.9
        
        tau_prime = max(tau_min, tau * (1 - (gamma + (current_task - 1) * beta)))
        self.temperature.data = torch.tensor(tau_prime, device=self.args.device)


# FedMTL
class LocalUpdateFedMTL(object):
    """
    FedMTL client: trains both classification model and CVAE.
    - Classification: CE loss, initialized from global M_g
    - CVAE: reconstruction + KL loss, fresh initialization per task (per paper)
    """

    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.task = task

    def _cvae_loss(self, x_recon, x, mu, logvar):
        """CVAE loss: reconstruction + KL divergence."""
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def train(self, net, cvae, lr, idx=-1, local_eps=None):
        net.train()
        cvae.train()

        optimizer_net = torch.optim.SGD(net.parameters(), lr=lr,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.wd)
        cvae_lr = getattr(self.args, 'cvae_lr', 1e-3)
        optimizer_cvae = torch.optim.Adam(cvae.parameters(), lr=cvae_lr)

        epoch_loss = []
        if local_eps is None:
            local_eps = self.args.local_ep_pretrain if self.pretrain else self.args.local_ep

        for _ in range(local_eps):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                c = F.one_hot(labels, num_classes=self.args.num_classes).float()

                # 1. Train classification model (CE loss)
                logits = net(images)
                ce_loss = self.loss_func(logits, labels)
                optimizer_net.zero_grad()
                ce_loss.backward()
                optimizer_net.step()

                # 2. Train CVAE (reconstruction + KL)
                x_recon, mu, logvar = cvae(images, c)
                cvae_loss = self._cvae_loss(x_recon, images, mu, logvar) / images.size(0)
                optimizer_cvae.zero_grad()
                cvae_loss.backward()
                optimizer_cvae.step()

                total_loss = ce_loss.item() + cvae_loss.item()
                batch_loss.append(total_loss)

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), cvae.state_dict(), sum(epoch_loss) / len(epoch_loss)



# ==================== FedMTL ====================

    """
    Learning without Forgetting (LwF) Local Update for Federated Learning.
    
    Implements knowledge distillation to preserve old task knowledge while learning new tasks.
    The algorithm uses only new task data for training, without requiring old task data.
    
    Loss function:
    L = L_new + λ_o * L_old + R(θ)
    where:
    - L_new: Cross-entropy loss for new task
    - L_old: Knowledge distillation loss for old tasks
    - R(θ): Weight decay regularization
    """
    
    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        
        # LwF specific attributes
        self.old_model_state = None  # Store complete old model state dict
        self.old_task_classes = None  # Store class indices for old task
        self.temperature = getattr(args, 'lwf_temperature', 2)
        self.lambda_old = getattr(args, 'lwf_lambda', 1.0)
        self.warmup_epochs = getattr(args, 'lwf_warmup_epochs', 0)
        
    def record_old_outputs(self, net):
        """
        Record the old model state for knowledge distillation.
        This should be called before task switch using the old model.
        We store the entire model state rather than just outputs,
        because different tasks may have different output dimensions.
        
        Args:
            net: The old model (before learning new task)
        """
        # Store a copy of the old model's state dict
        self.old_model_state = copy.deepcopy(net.state_dict())
        
    def distillation_loss(self, old_logits, new_logits, class_mask=None):
        """
        Compute knowledge distillation loss (L_old).
        
        L_old = -sum(y_o' * log(y_hat_o'))
        
        where y_o' and y_hat_o' are temperature-scaled softmax outputs.
        Only computed on classes that belong to the old task.
        
        Args:
            old_logits: Old model logits (before temperature scaling)
            new_logits: New model logits
            class_mask: Boolean mask indicating which classes to include in distillation
            
        Returns:
            Knowledge distillation loss scaled by T^2
        """
        # Temperature-scaled softmax for both old and new
        old_log_probs = F.log_softmax(old_logits / self.temperature, dim=1)
        new_log_probs = F.log_softmax(new_logits / self.temperature, dim=1)
        
        # Apply class mask if provided (only compute distillation on old task's classes)
        if class_mask is not None:
            # Mask shape: [num_classes]
            # Expand to batch size: [batch_size, num_classes]
            mask_expanded = class_mask.unsqueeze(0).expand(old_log_probs.size(0), -1)
            
            # Zero out non-old-task classes
            old_log_probs = old_log_probs * mask_expanded
            new_log_probs = new_log_probs * mask_expanded
            
            # Normalize masked probabilities
            old_probs = torch.exp(old_log_probs)
            old_probs_sum = old_probs.sum(dim=1, keepdim=True) + 1e-8
            old_log_probs = torch.log(old_probs / old_probs_sum)
        
        # KL divergence between old and new distributions
        # D_KL(old || new) = sum(old * (log(old) - log(new)))
        kl_div = F.kl_div(new_log_probs, old_log_probs, reduction='batchmean', log_target=True)
        
        # Scale by T^2 as per Hinton's distillation paper
        return kl_div * (self.temperature ** 2)
    
    def train(self, net, lr, idx=-1, local_eps=None):
        """
        Train with LwF algorithm.
        
        L = L_new + λ_o * L_old + R(θ)
        
        Args:
            net: Local model
            lr: Learning rate
            idx: Client index
            local_eps: Number of local epochs
            
        Returns:
            Tuple of (model state dict, average loss)
        """
        net.train()
        
        # Initialize optimizer with weight decay (R(θ) in LwF)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        
        epoch_loss = []
        
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        
        # Check if we have an old model for distillation
        has_old_model = self.old_model_state is not None
        
        # Create old model if needed
        if has_old_model:
            from utils.train_utils import get_model
            old_net = get_model(self.args)
            old_net.load_state_dict(self.old_model_state)
            old_net.to(self.args.device)
            old_net.eval()
        
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                # Forward pass
                logits = net(images)
                
                # New task loss (L_new): Cross-entropy
                ce_loss = self.loss_func(logits, labels)
                
                # Old task loss (L_old): Knowledge distillation
                kd_loss = torch.tensor(0.0, device=self.args.device)
                if has_old_model:
                    with torch.no_grad():
                        old_logits = old_net(images)
                    # Create class mask for old task's classes
                    if self.old_task_classes is not None:
                        num_classes = self.args.num_classes
                        class_mask = torch.zeros(num_classes, dtype=torch.float32, device=self.args.device)
                        class_mask[self.old_task_classes] = 1.0
                        kd_loss = self.distillation_loss(old_logits, logits, class_mask)
                    else:
                        kd_loss = self.distillation_loss(old_logits, logits)
                
                # Total loss: L = L_new + λ_o * L_old
                total_loss = ce_loss + self.lambda_old * kd_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                batch_loss.append(total_loss.item())
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        
        # Clean up old model
        if has_old_model:
            del old_net
        
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


# MOON (Model-Contrastive Federated Learning, Li et al., ICML 2021)
class LocalUpdateMOON(object):
    """
    Client-side MOON update.

    L = L_sup + μ * L_con
    L_con uses NT-Xent on encoder representations:
      positive: current local model vs global model
      negative: current local model vs previous local model
    """

    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.mu = getattr(args, 'moon_mu', 1.0)
        self.temperature = getattr(args, 'moon_tau', 0.5)

    def _unwrap(self, net):
        return net.module if hasattr(net, 'module') else net

    def train(self, net, global_net, prev_net=None, lr=None, idx=-1, local_eps=None):
        net.train()
        global_net.eval()
        for param in global_net.parameters():
            param.requires_grad = False
        if prev_net is not None:
            prev_net.eval()
            for param in prev_net.parameters():
                param.requires_grad = False

        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        if local_eps is None:
            local_eps = self.args.local_ep_pretrain if self.pretrain else self.args.local_ep

        local_model = self._unwrap(net)
        glob_model = self._unwrap(global_net)
        prev_model = self._unwrap(prev_net) if prev_net is not None else None

        for _ in range(local_eps):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                pro1 = local_model.extract_features(images)
                logits = local_model.only_liner(pro1)
                loss_sup = self.loss_func(logits, labels)

                if prev_model is not None:
                    with torch.no_grad():
                        pro2 = glob_model.extract_features(images)
                        pro3 = prev_model.extract_features(images)

                    posi = F.cosine_similarity(pro1, pro2, dim=-1)
                    nega = F.cosine_similarity(pro1, pro3, dim=-1)
                    logits_con = torch.cat(
                        [posi.unsqueeze(1), nega.unsqueeze(1)], dim=1
                    ) / self.temperature
                    labels_con = torch.zeros(images.size(0), dtype=torch.long, device=self.args.device)
                    loss_con = self.loss_func(logits_con, labels_con)
                    loss = loss_sup + self.mu * loss_con
                else:
                    loss = loss_sup

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def _unwrap_module(net):
    return net.module if hasattr(net, 'module') else net


def _is_fedta_wrapper(net):
    real = _unwrap_module(net)
    return hasattr(real, 'ie_bank') and hasattr(real, 'tail_anchors')


class LocalUpdateFedTA(object):
    """
    FedTA client with two-stage local training:
      Stage 1: Input Enhancement (IE) + head
      Stage 2: Tail Anchor (TA) + head with global prototype contrastive loss
    """

    def __init__(self, args, dataset=None, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.task = task
        self.num_classes = args.num_classes

        self.lambda1 = getattr(args, 'fedta_lambda1', 0.1)
        self.lambda2 = getattr(args, 'fedta_lambda2', 0.1)
        self.lambda3 = getattr(args, 'fedta_lambda3', 0.01)
        self.tau_c = getattr(args, 'fedta_tau_c', 0.1)
        self.wd = getattr(args, 'fedta_wd', 5e-4)

        stage1 = getattr(args, 'fedta_stage1_ep', 0)
        self.stage1_ep = stage1 if stage1 > 0 else max(1, args.local_ep // 2)
        self.stage2_ep = max(1, args.local_ep - self.stage1_ep)

    def _head_params(self, real):
        prefixes = real._head_param_names()
        params = []
        for name, param in real.base.named_parameters():
            if name in prefixes:
                params.append(param)
        return params

    def _make_optimizer(self, adapter_params, decay_params, lr):
        """Adapter banks are bias-like, so they are exempt from weight decay."""
        groups = []
        if adapter_params:
            groups.append({'params': adapter_params, 'weight_decay': 0.0})
        if decay_params:
            groups.append({'params': decay_params, 'weight_decay': self.wd})
        return torch.optim.SGD(groups, lr=lr, momentum=self.args.momentum)

    def _run_stage(self, net, optimizer, global_prototypes, stage, num_epochs):
        device = self.args.device
        epoch_loss = []
        all_params = [p for group in optimizer.param_groups for p in group['params']]

        G = None
        valid_mask = None
        if stage == 'ta' and global_prototypes is not None:
            G = global_prototypes.to(device)
            valid_mask = G.norm(dim=1) > 1e-6

        for _ in range(num_epochs):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                logits, f_ta = net(images, returnFeature=True)
                loss = self.loss_func(logits, labels)

                real = _unwrap_module(net)
                if stage == 'ta':
                    if G is not None and valid_mask is not None:
                        sample_mask = valid_mask[labels]
                        if sample_mask.any():
                            logits_c = torch.mm(f_ta[sample_mask], G.t()) / self.tau_c
                            logits_c[:, ~valid_mask] = -1e9
                            l_cons = self.loss_func(logits_c, labels[sample_mask])
                            loss = loss + self.lambda2 * l_cons
                    loss = loss + self.lambda3 * real.ta_key_loss()
                else:
                    loss = loss + self.lambda1 * real.ie_key_loss()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)

    def _compute_prototypes(self, net, device):
        was_training = net.training
        net.eval()
        class_sums = {}
        class_counts = {}

        with torch.no_grad():
            for images, labels in self.ldr_train:
                images, labels = images.to(device), labels.to(device)
                _, f_ta = net(images, returnFeature=True)
                for y in labels.unique():
                    y_val = int(y.item())
                    mask = labels == y
                    if mask.sum() == 0:
                        continue
                    proto = f_ta[mask].mean(dim=0)
                    class_sums[y_val] = class_sums.get(y_val, 0) + proto
                    class_counts[y_val] = class_counts.get(y_val, 0) + mask.sum().item()

        prototypes = {}
        proto_counts = {}
        for y_val, summed in class_sums.items():
            proto = summed / class_counts[y_val]
            proto = F.normalize(proto.unsqueeze(0), p=2, dim=1).squeeze(0)
            prototypes[y_val] = proto.cpu()
            proto_counts[y_val] = class_counts[y_val]

        if was_training:
            net.train()
        return prototypes, proto_counts

    def train(self, net, global_prototypes=None, ie_state=None, ta_state=None,
              lr=None, round_idx=0, idx=-1, local_eps=None):
        if not _is_fedta_wrapper(net):
            raise ValueError('LocalUpdateFedTA requires a FedTAWrapper model')

        net.train()
        device = self.args.device
        lr = lr if lr is not None else self.args.lr
        real = _unwrap_module(net)

        if ie_state is not None:
            real.load_state_dict_ie(ie_state)
        if ta_state is not None:
            real.load_state_dict_ta(ta_state)

        head_params = self._head_params(real)
        # Deep blocks stay trainable unless the freeze level is 'full'.
        body_params = real.trainable_backbone_parameters()

        for param in real.parameters():
            param.requires_grad = False
        for param in head_params + body_params:
            param.requires_grad = True

        # Stage 1: Input Enhancement + head. TA is frozen but stays in the
        # forward path so the head never sees a shifting feature distribution.
        real.ie_bank.requires_grad = True
        real.ie_keys.requires_grad = True
        opt1 = self._make_optimizer([real.ie_bank, real.ie_keys],
                                    head_params + body_params, lr)
        loss1 = self._run_stage(net, opt1, global_prototypes=None,
                                stage='ie', num_epochs=self.stage1_ep)

        # Stage 2: Tail Anchor + head under the global-prototype contrastive loss.
        real.ie_bank.requires_grad = False
        real.ie_keys.requires_grad = False
        real.tail_anchors.requires_grad = True
        real.ta_keys.requires_grad = True
        real.logit_scale.requires_grad = True
        opt2 = self._make_optimizer([real.tail_anchors, real.ta_keys, real.logit_scale],
                                    head_params + body_params, lr)
        loss2 = self._run_stage(net, opt2, global_prototypes=global_prototypes,
                                stage='ta', num_epochs=self.stage2_ep)

        prototypes, proto_counts = self._compute_prototypes(net, device)
        avg_loss = (loss1 + loss2) / 2.0
        return net.state_dict(), real.state_dict_ie(), real.state_dict_ta(), prototypes, proto_counts, avg_loss


class LocalUpdateFedTAWarmup(object):
    """Standard local update used during FedTA backbone warmup."""

    def __init__(self, args, dataset=None, idxs=None, task=0):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = load_train_data(dataset, idxs, task, batch_size=self.args.local_bs)
        self.task = task

    def train(self, net, lr=None, idx=-1, local_eps=None):
        net.train()
        lr = lr if lr is not None else self.args.lr
        local_eps = local_eps or self.args.local_ep

        adapters, decay = [], []
        for name, param in net.named_parameters():
            if name.endswith(('ie_bank', 'ie_keys', 'tail_anchors', 'ta_keys', 'logit_scale')):
                adapters.append(param)
            else:
                decay.append(param)
        optimizer = torch.optim.SGD(
            [{'params': adapters, 'weight_decay': 0.0},
             {'params': decay, 'weight_decay': self.args.wd}],
            lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for _ in range(local_eps):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                logits = net(images)
                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
