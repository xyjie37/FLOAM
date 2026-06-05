#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=500, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--wd', type=float, default=0.99, help="weight decay (default: 0.0)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--local_ep_pretrain', type=int, default=0, help="the number of pretrain local ep")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
    parser.add_argument('--fl_alg', type=str, default='FedAvg', help="federated learning algorithm")
    #parser.add_argument('--tau', type=float, default=0.999, help="parameter for proximal local SGD")
    parser.add_argument('--global_ep', type=int, default=1, help="the number of global epochs: E")
    parser.add_argument('--task_num', type=int, default=1, help="number of task")
    parser.add_argument('--head_epoch', type=int, default=5, help="number of head epoch")
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name; use "auto" to auto-select based on dataset')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset (e.g. cifar10, cifar100, imagenet100, tinyimagenet)")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=str, default='0', help="To use cuda, set to a specific GPU ID. Default is 0. Set to -1 for CPU. For multi-gpu, use comma separated gpu ids")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--print_freq', type=int, default=100, help="print loss frequency during training")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--load_fed', type=str, default='', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='/', help='define fed results save folder')
    parser.add_argument('--start_saving', type=int, default=0, help='when to start saving models')
    parser.add_argument('--benchmark_runtime', action='store_true', help='measure per-round client/server runtime and save CSV')
    parser.add_argument('--runtime_csv', type=str, default='', help='output CSV path for runtime benchmark')
    parser.add_argument('--skip_eval', action='store_true', help='skip evaluation during training (auto-enabled with --benchmark_runtime)')
    
    # evaluation arguments
    parser.add_argument('--ft_ep', type=int, default=5, help="the number of epochs for fine-tuning")
    parser.add_argument('--fine_tuning', action='store_true', help='whether fine-tuning before evaluation')
    

    # additional arguments
    parser.add_argument('--local_upt_part', type=str, default=None, help='body, head, or full')
    parser.add_argument('--aggr_part', type=str, default=None, help='body, head, or full')
    parser.add_argument('--unbalanced', action='store_true', help='unbalanced data size')
    parser.add_argument('--num_batch_users', type=int, default=0, help='when unbalanced dataset setting, batch users (same data size)')
    parser.add_argument('--moved_data_size', type=int, default=0, help='when unbalanced dataset setting, moved data size')
    
    parser.add_argument('--server_data_ratio', type=float, default=0.0, help='The percentage of data that servers also have across data of all clients.')
    
    # arguments for a single model
    parser.add_argument('--opt', type=str, default='SGD', help="optimizer")
    parser.add_argument('--body_lr', type=float, default=None, help="learning rate for the body of the model")
    parser.add_argument('--head_lr', type=float, default=0.01, help="learning rate for the head of the model")
    parser.add_argument('--body_m', type=float, default=None, help="momentum for the body of the model")
    parser.add_argument('--head_m', type=float, default=None, help="momentum for the head of the model")
    parser.add_argument('--tau', type=float, default=3)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--datasetpath', type=str, default='None', help="which datasetpath")
    #parser.add_argument('--weight_decay', type=float, default=0.99, help='learning rate decay per global round')
    parser.add_argument('--lambda_anchor', type=float, default=0.1, help='anchor proximal term constant')
    parser.add_argument('--momentum_anchor', type=float, default=0.5, help="dynamic momentum update for feature anchor(default: 0.5)")
    parser.add_argument('--noise_std', type=float, default=0.1, help="")
    parser.add_argument('--mode', type=str, default = 'mode1')
    parser.add_argument('--m', type=int, default = '20')       #20 for cifar100
    parser.add_argument('--z_dim', type=int, default=100, help="Generator input dimension")
    parser.add_argument('--g_lr', type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument('--lambda1', type=float, default=0.1, help="Boundary loss weight")
    parser.add_argument('--lambda2', type=float, default=0.01, help="BatchNorm loss weight")
    parser.add_argument('--g_epochs', type=int, default=5, help="Number of generator training epochs")
    parser.add_argument('--d_epochs', type=int, default=3, help="Number of distillation training epochs")
    parser.add_argument('--syn_batch_size', type=int, default=256, help="Synthetic data batch size")
    parser.add_argument('--batch_size', type=int, default=64, help="Synthetic data generator learning rate")
    parser.add_argument('--classes_per_task ', type=int, default=10, help="Student learning rate")
    parser.add_argument('--meta-lr', type=float, default=0.9, help="Student learning rate")
    parser.add_argument('--sample_bs', type = int, default = 32)
    parser.add_argument('--cvae_lr', type=float, default=1e-3, help="CVAE learning rate for FedMTL")
    parser.add_argument('--prompt_dim', type=int, default=64, help='RefFiL prompt dimension')

    # FedTA arguments
    parser.add_argument('--num_ie', type=int, default=10, help='FedTA: Knowledge Base size M')
    parser.add_argument('--fedta_lambda1', type=float, default=0.1, help='FedTA: L_cons weight')
    parser.add_argument('--fedta_lambda2', type=float, default=0.01, help='FedTA: L_key weight')
    parser.add_argument('--fedta_tau', type=float, default=0.1, help='FedTA: IE softmax temperature')
    parser.add_argument('--fedta_tau_c', type=float, default=0.1, help='FedTA: contrastive temperature')
    parser.add_argument('--fedta_beta_init_logit', type=float, default=-4.0, help='FedTA: beta logit init (sigmoid(-4)≈0.018)')
    parser.add_argument('--fedta_alpha_test', type=float, default=0.018, help='FedTA: alpha for inference (sigmoid(-4)≈0.018)')
    parser.add_argument('--fedta_thr', type=float, default=0.5, help='FedTA: BGPS stability threshold')
    parser.add_argument('--fedta_gamma', type=float, default=0.2, help='FedTA: anchor EMA gamma')
    parser.add_argument('--fedta_progressive_rounds', type=int, default=0, help='FedTA: rounds to train only ESA first (0=train layer3-4 from start)')
    parser.add_argument('--fedta_proto_threshold', type=float, default=0.6, help='FedTA: BGPS confidence threshold for prototype selection')
    parser.add_argument('--fedta_wd', type=float, default=5e-4, help='FedTA: weight decay for full training')

    # LwF arguments
    parser.add_argument('--lwf_temperature', type=float, default=2, help='LwF: temperature for knowledge distillation')
    parser.add_argument('--lwf_lambda', type=float, default=1.0, help='LwF: balance weight for old task loss')
    parser.add_argument('--lwf_warmup_epochs', type=int, default=0, help='LwF: warm-up phase epochs (0=disabled)')

    # FLOAM ablation flags (each isolates ONE component; default = full FLOAM)
    parser.add_argument('--ot_cost', type=str, default='anchor_geometry',
                        choices=['anchor_geometry', 'uniform'],
                        help='OT cost: anchor cosine (full) vs uniform (no geometry)')
    parser.add_argument('--contrast_target', type=str, default='shared',
                        choices=['shared', 'local_mean', 'classifier'],
                        help='Contrastive positive target: global anchor vs local class mean vs classifier weights')
    parser.add_argument('--anchor_agg', type=str, default='client_balanced',
                        choices=['client_balanced', 'sample_weighted'],
                        help='Server anchor aggregation: equal per-client vs sample-weighted')

    args = parser.parse_args()
    return args
