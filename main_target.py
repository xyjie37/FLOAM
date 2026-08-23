#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

"""
TARGET: Federated Class-Continual Learning via Exemplar-Free Distillation (ICCV 2023)

Baseline reproduction for the FLOAM benchmark. The runtime protocol (task schedule,
rounds, aggregation, evaluation) is identical to main_fedavg.py; TARGET adds:
1. Server-side generator-based data synthesis at each task switch (data_generation),
   ported from the official implementation (zj-jayzhang/Federated-Class-Continual-Learning).
2. Client-side old-class knowledge distillation on the synthetic data against the
   frozen global model snapshot of the previous task (LocalUpdateTARGET in models/Update.py).

Adaptations w.r.t. the official code:
- kornia augmentation removed: the FLOAM npz training data is pre-normalized without
  augmentation, so only the dataset normalization is applied to generator outputs;
- the synthetic image pool is kept in memory (normalized tensors) instead of PNG files;
- devices are passed explicitly instead of hardcoded .cuda().
"""

import copy
import gc
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset

from utils.options import args_parser
from utils.train_utils import get_model
from models.Update import LocalUpdateTARGET
from models.test import test_img, test_img_local_all, compute_smi_tdi_for_task

# Dataset metadata: image size and the exact normalization used when building the
# npz client-task files (see dataset/split_*.py). Synthetic data must match it.
DATASET_META = {
    'cifar10':      (32, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    'cifar100':     (32, [0.507, 0.487, 0.441],    [0.267, 0.256, 0.276]),
    'cinic10':      (32, [0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]),
    'tinyimagenet': (64, [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
}


def normalize(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    return (tensor - mean[None, :, None, None]) / (std[None, :, None, None])


class Generator(nn.Module):
    """DCGAN-style image generator (official TARGET implementation)."""

    def __init__(self, nz=256, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    # return a copy of its own
    def clone(self, device):
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3])
        clone.load_state_dict(self.state_dict())
        return clone.to(device)


class DeepInversionHook:
    """Forward hook on BatchNorm layers matching input statistics to the running
    statistics of the frozen teacher (DeepInversion feature distribution regularization)."""

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (self.mmt_rate * mean_mmt + (1 - self.mmt_rate) * mean.data,
                        self.mmt_rate * var_mmt + (1 - self.mmt_rate) * var.data)

    def remove(self):
        self.hook.remove()


def fomaml_grad(src, tar, device):
    # first-order MAML: accumulate the fast generator's gradients into the global one
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = torch.zeros(p.size()).to(device)
        p.grad.data.add_(tar_p.grad.data)


def reptile_grad(src, tar, device):
    # REPTILE meta gradient
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = torch.zeros(p.size()).to(device)
        p.grad.data.add_(p.data - tar_p.data, alpha=67)


def reset_l0_fun(model):
    for n, m in model.named_modules():
        if n == 'l1.0' or n == 'conv_blocks.0':
            init.normal_(m.weight, 0.0, 0.02)
            init.constant_(m.bias, 0)


def weight_init(m):
    """Compact re-initialization for the data-free distillation student."""
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)


def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T * T)


class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super(KLDiv, self).__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)
        return data


class GlobalSynthesizer(object):
    """Server-side data synthesis: meta-learn a generator against the frozen global model."""

    def __init__(self, args, teacher, student, generator, num_classes, device, mean, std):
        self.args = args
        self.teacher = teacher
        self.student = student
        self.generator = generator.to(device).train()
        self.device = device
        self.num_classes = num_classes
        self.mean = mean
        self.std = std

        self.nz = args.target_nz
        self.iterations = args.target_g_steps
        self.lr_g = args.target_lr_g
        self.lr_z = args.target_lr_z
        self.adv = args.target_adv
        self.bn = args.target_bn
        self.oh = args.target_oh
        self.ismaml = args.target_is_maml
        self.synthesis_batch_size = args.target_syn_bs
        self.max_pool_size = args.target_num_syn
        self.reset_l0 = args.target_reset_l0

        self.ep = 0
        self.ep_start = args.target_warmup
        self.pool = []  # list of normalized synthetic batches (CPU tensors)

        self.meta_optimizer = torch.optim.Adam(
            self.generator.parameters(), self.lr_g * self.iterations, betas=[0.5, 0.999])

        self.hooks = []
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, args.target_bn_mmt))

    def synthesize(self):
        self.ep += 1
        self.teacher.eval()
        self.student.eval()
        best_cost = 1e6

        if (self.ep == 120 + self.ep_start) and self.reset_l0:
            reset_l0_fun(self.generator)

        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).to(self.device)
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.num_classes,
                                size=(self.synthesis_batch_size,)).to(self.device)

        fast_generator = self.generator.clone(self.device)
        optimizer = torch.optim.Adam([
            {'params': fast_generator.parameters()},
            {'params': [z], 'lr': self.lr_z}
        ], lr=self.lr_g, betas=[0.5, 0.999])
        for it in range(self.iterations):
            inputs = fast_generator(z)
            # match the FLOAM npz data distribution (pre-normalized, no augmentation)
            inputs_aug = normalize(inputs, self.mean, self.std)

            t_out = self.teacher(inputs_aug)
            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy(t_out, targets)
            if self.adv > 0 and (self.ep >= self.ep_start):
                s_out = self.student(inputs_aug)
                mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean()  # adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs_aug.data.cpu()  # keep the normalized version

            optimizer.zero_grad()
            loss.backward()

            if self.ismaml:
                if it == 0:
                    self.meta_optimizer.zero_grad()
                fomaml_grad(self.generator, fast_generator, self.device)
                if it == (self.iterations - 1):
                    self.meta_optimizer.step()

            optimizer.step()

        if self.args.target_bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        # REPTILE meta gradient
        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.generator, fast_generator, self.device)
            self.meta_optimizer.step()

        self.student.train()

        self.pool.append(best_inputs)
        total = sum(t.shape[0] for t in self.pool)
        while total > self.max_pool_size and len(self.pool) > 1:
            total -= self.pool[0].shape[0]
            self.pool.pop(0)  # drop the earliest batches

    def get_pool_dataset(self):
        return TensorDataset(torch.cat(self.pool, dim=0))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def kd_train(args, student, teacher, syn_dataset, criterion, optimizer, device):
    """Data-free distillation of the teacher into the student on the synthetic pool."""
    student.train()
    teacher.eval()
    loader = DataLoader(syn_dataset, batch_size=args.target_syn_bs, shuffle=True)
    data_iter = DataIter(loader)
    for i in range(args.target_kd_steps):
        images = data_iter.next()[0].to(device)
        with torch.no_grad():
            t_out = teacher(images)
        s_out = student(images.detach())
        loss_s = criterion(s_out, t_out.detach())
        optimizer.zero_grad()
        loss_s.backward()
        optimizer.step()


def data_generation(args, teacher, seen_classes, device):
    """Official TARGET data generation at each task switch.

    A fresh generator is meta-learned against the frozen global model (teacher);
    after warmup rounds a randomly re-initialized student is distilled from the
    teacher on the accumulated synthetic pool (inactive with the default
    syn_round=10 < warmup=20, matching the official CIFAR-100 configuration).
    Returns a TensorDataset of normalized synthetic images over seen_classes.
    """
    img_size, mean, std = DATASET_META[args.dataset]
    generator = Generator(nz=args.target_nz, ngf=args.target_ngf, img_size=img_size, nc=3)
    student = copy.deepcopy(teacher)
    student.apply(weight_init)

    synthesizer = GlobalSynthesizer(args, teacher=teacher, student=student, generator=generator,
                                    num_classes=seen_classes, device=device, mean=mean, std=std)

    criterion = KLDiv(T=args.target_T)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.2, weight_decay=0.0001,
                        momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=2e-4)

    for it in range(args.target_syn_round):
        synthesizer.synthesize()
        if it >= args.target_warmup:
            kd_train(args, student, teacher, synthesizer.get_pool_dataset(),
                     criterion, optimizer, device)
            scheduler.step()

    synthesizer.remove_hooks()
    syn_dataset = synthesizer.get_pool_dataset()
    print('[TARGET] data generation finished: {} synthetic images over {} seen classes'.format(
        len(syn_dataset), seen_classes))

    del student, generator, synthesizer
    gc.collect()
    torch.cuda.empty_cache()
    return syn_dataset


if __name__ == '__main__':
    # parse args
    args = args_parser()
    dataset_path = args.datasetpath
    if args.dataset not in DATASET_META:
        exit('Error: TARGET only supports image datasets {} (got {})'.format(
            sorted(DATASET_META.keys()), args.dataset))
    # Seed
    # torch.manual_seed(args.seed)#seed=1
    # torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # np.random.seed(args.seed)
    task_num = args.task_num
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    base_dir = './save/{}/{}_num{}_C{}_le{}_bs{}_round{}_m{}_lr{}/{}/'.format(
        dataset_path, args.model, args.num_users, args.frac, args.local_ep, args.local_bs, args.epochs, args.momentum, args.lr, args.results_save)
    algo_dir = 'target'
    save_folder = './results/target'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    # build a global model
    net_glob = get_model(args)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net_glob = nn.DataParallel(net_glob)
    net_glob.to(args.device)
    net_glob.train()

    # build local models
    net_local_list = []
    for user_idx in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))

    # TARGET state: frozen snapshot of the global model at the last task switch and
    # the unlabeled synthetic data loader built from the generator pool
    teacher_model = None
    syn_loader = None

    # training
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []
    prev_client_centroids = None
    current_smi = np.nan
    current_tdi = np.nan

    prev_task = None

    for iter in range(args.epochs):
        w_glob = None
        loss_locals = []

        # Client Sampling
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))

        task=(iter//10)%task_num  # Task switch every 10 rounds (FLOAM protocol)
        print('Current task: ', task)

        # Task switch: freeze the current global model as teacher and synthesize
        # old-class data with a freshly meta-learned generator
        if task != prev_task:
            classes_per_task = args.num_classes // task_num
            old_classes = task * classes_per_task
            if old_classes >= 2:
                print('[TARGET] Task switch -> {}: data generation over {} old classes'.format(task, old_classes))
                teacher_model = copy.deepcopy(net_glob).to(args.device)
                syn_pool = data_generation(args, teacher_model, old_classes, args.device)
                syn_loader = DataLoader(syn_pool, batch_size=args.local_bs, shuffle=True)
            else:
                # first task (or wrap-around to task 0): plain FedAvg phase
                teacher_model, syn_loader = None, None
        prev_task = task

        # Local Updates with TARGET
        for idx in idxs_users:
            # Dataset name, index
            local = LocalUpdateTARGET(args=args, dataset=dataset_path, idxs=idx, task=task,
                                      syn_loader=syn_loader)
            net_local = copy.deepcopy(net_local_list[idx])
            w_local, loss = local.train(net=net_local.to(args.device),
                                        teacher_model=teacher_model, lr=lr)

            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]

        # Aggregation
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)

        # Broadcast
        update_keys = list(w_glob.keys())
        w_glob = {k: v for k, v in w_glob.items() if k in update_keys}
        for user_idx in range(args.num_users):
            net_local_list[user_idx].load_state_dict(w_glob, strict=False)
        net_glob.load_state_dict(w_glob, strict=False)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % args.test_freq == 0:
            acc_test, acc_test_var, loss_test = test_img_local_all(net_local_list, args, dataset_test=dataset_path, task=task, return_all=False)

            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

            #all_acc, all_loss = test_img(net_glob, datatest=dataset_path, args=args, epoch=iter, class_num=args.num_classes)

            all_acc, all_loss = test_img(net_glob, datatest=dataset_path, args=args, epoch = iter, class_num=args.num_classes, save_folder = save_folder)

            print('All Test Data: Average loss: {:.4f}, Accuracy: {:.2f}% '.format(
                all_loss, all_acc))

            if best_acc is None or all_acc > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = all_acc
                best_epoch = iter

                best_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')

                torch.save(net_best.state_dict(), best_save_path)

            if (iter + 1) % 10 == 0:
                current_smi, current_tdi, prev_client_centroids = compute_smi_tdi_for_task(
                    net_local_list=net_local_list,
                    args=args,
                    dataset_test=dataset_path,
                    task=task,
                    prev_client_centroids=prev_client_centroids,
                    num_classes=args.num_classes
                )
                tdi_str = 'nan' if np.isnan(current_tdi) else '{:.6f}'.format(current_tdi)
                print('Task {:3d} SMI: {:.6f}, TDI: {}'.format(task, current_smi, tdi_str))
            else:
                current_smi, current_tdi = np.nan, np.nan

            results.append(np.array([iter,task, loss_avg, loss_test, acc_test, all_acc, best_acc, current_smi, current_tdi]))
            #results.append(np.array([iter, task, loss_avg, all_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch','task', 'loss_avg', 'loss_test', 'acc_test',  'all_acc','best_acc', 'smi', 'tdi'])
            #final_results = pd.DataFrame(final_results, columns=['epoch','task', 'loss_avg', 'all_acc'])
            final_results.to_csv(results_save_path, index=False)

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
