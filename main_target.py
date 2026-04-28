#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdateTARGET
from models.test import test_img, test_img_local, test_img_local_all, compute_smi_tdi_for_task
import os

import pdb
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Generator(nn.Module):
    """生成器网络，用于合成数据"""
    def __init__(self, latent_dim=100, img_shape=(3, 32, 32)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

'''def data_generation(teacher_model, generator, student_model, args):
    """数据生成和模型蒸馏阶段"""
    teacher_model.eval()
    generator.train()
    student_model.train()
    
    # 超参数设置
    latent_dim = 100
    num_batches = 100
    batch_size = 64
    epochs_g = 5  # 生成器训练轮次
    epochs_d = 3  # 蒸馏训练轮次
    
    # 优化器
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_s = optim.SGD(student_model.parameters(), lr=args.lr, momentum=0.9)
    
    synthetic_data = []
    
    # 数据生成阶段
    for _ in range(epochs_g):
        for _ in range(num_batches):
            # 生成随机噪声和伪标签
            z = torch.randn(batch_size, latent_dim).to(args.device)
            fake_labels = torch.randint(0, args.num_classes, (batch_size,)).to(args.device)
            
            # 生成图像
            fake_images = generator(z)
            
            # 计算损失
            with torch.no_grad():
                teacher_output = teacher_model(fake_images)
            
            # 交叉熵损失
            ce_loss = nn.CrossEntropyLoss()(teacher_output, fake_labels)
            
            # KL散度损失
            student_output = student_model(fake_images)
            kl_loss = nn.KLDivLoss()(F.log_softmax(student_output, dim=1),
                                    F.softmax(teacher_output, dim=1))
            
            # BN统计匹配损失
            bn_loss = 0
            for (m_teacher, m_student) in zip(teacher_model.modules(), student_model.modules()):
                if isinstance(m_teacher, nn.BatchNorm2d):
                    bn_loss += torch.norm(m_teacher.running_mean - m_student.running_mean, 2)
                    bn_loss += torch.norm(m_teacher.running_var - m_student.running_var, 2)
            
            # 总损失
            total_loss = ce_loss + 0.1 * kl_loss + 0.01 * bn_loss
            
            # 反向传播
            optimizer_g.zero_grad()
            total_loss.backward()
            optimizer_g.step()
            
            # 保存生成的batch
            synthetic_data.append((fake_images.detach(), fake_labels.detach()))
    
    # 模型蒸馏阶段
    for _ in range(epochs_d):
        for (synth_images, synth_labels) in synthetic_data:
            with torch.no_grad():
                teacher_logits = teacher_model(synth_images)
            
            student_logits = student_model(synth_images)
            loss = nn.KLDivLoss()(F.log_softmax(student_logits, dim=1),
                                F.softmax(teacher_logits, dim=1))
            
            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()
    
    return synthetic_data'''
def data_generation(teacher_model, generator, student_model, args):
    """数据生成和模型蒸馏阶段"""
    teacher_model.eval()  # 教师模型进入评估模式
    generator.train()     # 生成器进入训练模式
    student_model.train() # 学生模型进入训练模式
    
    # 超参数设置
    latent_dim = 100
    num_batches = 100
    batch_size = 64
    epochs_g = 5  # 生成器训练轮次
    epochs_d = 3  # 蒸馏训练轮次
    
    # 优化器
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_s = optim.SGD(student_model.parameters(), lr=args.lr, momentum=0.9)
    
    synthetic_data = []
    
    # 数据生成阶段
    for _ in range(epochs_g):
        for _ in range(num_batches):
            # 生成随机噪声和伪标签
            z = torch.randn(batch_size, latent_dim).to(args.device)
            fake_labels = torch.randint(0, args.num_classes, (batch_size,)).to(args.device)
            
            # 生成图像
            fake_images = generator(z)
            
            # 计算损失
            with torch.no_grad():
                teacher_output = teacher_model(fake_images)
            
            # 交叉熵损失
            ce_loss = nn.CrossEntropyLoss()(teacher_output, fake_labels)
            
            # KL散度损失
            student_output = student_model(fake_images)
            kl_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(student_output, dim=1),
                                                         F.softmax(teacher_output, dim=1))
            
            # BN统计匹配损失
            bn_loss = 0
            for (m_teacher, m_student) in zip(teacher_model.modules(), student_model.modules()):
                if isinstance(m_teacher, nn.BatchNorm2d) and isinstance(m_student, nn.BatchNorm2d):
                    if m_teacher.running_mean is not None and m_student.running_mean is not None:
                        bn_loss += torch.norm(m_teacher.running_mean - m_student.running_mean, 2)
                    if m_teacher.running_var is not None and m_student.running_var is not None:
                        bn_loss += torch.norm(m_teacher.running_var - m_student.running_var, 2)
            
            # 总损失
            total_loss = ce_loss + 0.1 * kl_loss + 0.01 * bn_loss
            
            # 反向传播
            optimizer_g.zero_grad()
            total_loss.backward()
            optimizer_g.step()
            
            # 保存生成的batch
            synthetic_data.append((fake_images.detach(), fake_labels.detach()))
    
    # 模型蒸馏阶段
    for _ in range(epochs_d):
        for (synth_images, synth_labels) in synthetic_data:
            with torch.no_grad():
                teacher_logits = teacher_model(synth_images)
            
            student_logits = student_model(synth_images)
            loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(student_logits, dim=1),
                                                     F.softmax(teacher_logits, dim=1))
            
            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()
    
    return synthetic_data



if __name__ == '__main__':
    # parse args
    args = args_parser()
    dataset_path = args.datasetpath
    # Seed
    # torch.manual_seed(args.seed)#seed=1
    # torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # np.random.seed(args.seed)
    task_num = args.task_num
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

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
    net_glob.train()

    # build local models
    net_local_list = []
    for user_idx in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))
    
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
    
    synthetic_data = None
    teacher_model = None
    generator = Generator().to(args.device)
    student_model = copy.deepcopy(net_glob).to(args.device)
    for iter in range(args.epochs):
        w_glob = None
        loss_locals = []
        
        # Client Sampling
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))
        
        task=(iter//10)%task_num#每过10个轮次进行任务切换
        print('Current task: ', task)

        # Local Updates
        for idx in idxs_users:
            #数据集名字，序号
            local = LocalUpdateTARGET(args=args, dataset=dataset_path, idxs=idx, task = task, synthetic_data=synthetic_data)
            net_local = copy.deepcopy(net_local_list[idx])
            teacher_model = copy.deepcopy(net_glob).eval().to(args.device)
            w_local, loss = local.train(net=net_local.to(args.device), teacher_model=teacher_model, lr=lr)
                
            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
        
        # Aggregation
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)
        net_glob.load_state_dict(w_glob)
        synthetic_data = data_generation(
            teacher_model=teacher_model,
            generator=generator,
            student_model=student_model,
            args=args
        )
        
        # Broadcast
        update_keys = list(w_glob.keys())
        w_glob = {k: v for k, v in w_glob.items() if k in update_keys}
        for user_idx in range(args.num_users):
            net_local_list[user_idx].load_state_dict(w_glob, strict=False)
        net_glob.load_state_dict(w_glob, strict=False)
        # if (iter + 1) == 50:
        #     lr = 0.01
        # elif (iter + 1) ==75:
        #     lr = 0.001

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % args.test_freq == 0:
            acc_test, acc_test_var, loss_test = test_img_local_all(net_local_list, args, dataset_test=dataset_path, task=task, return_all=False)
            
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))
            
            all_acc, all_loss = test_img(net_glob, datatest=dataset_path, args=args, epoch = iter, class_num=args.num_classes, save_folder = save_folder)

            print('All Test Data: Average loss: {:.4f}, Accuracy: {:.2f}% '.format(
                all_loss, all_acc))

            if best_acc is None or all_acc > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = all_acc
                best_epoch = iter
                
                best_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
                
                torch.save(net_best.state_dict(), best_save_path)
                
#                 for user_idx in range(args.num_users):
#                     best_save_path = os.path.join(base_dir, algo_dir, 'best_local_{}.pt'.format(user_idx))
#                     torch.save(net_local_list[user_idx].state_dict(), best_save_path)

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
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch','task', 'loss_avg', 'loss_test', 'acc_test',  'all_acc','best_acc', 'smi', 'tdi'])
            final_results.to_csv(results_save_path, index=False)

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))