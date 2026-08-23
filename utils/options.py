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
    parser.add_argument('--num_ie', type=int, default=10, help='FedTA: Input Enhancement bank size M')
    parser.add_argument('--num_ta', type=int, default=100, help='FedTA: Tail Anchor bank size')
    parser.add_argument('--fedta_alpha', type=float, default=0.5, help='FedTA: fixed TA mixing coefficient')
    parser.add_argument('--fedta_topn', type=int, default=3, help='FedTA: top-N IE selection')
    parser.add_argument('--fedta_lambda1', type=float, default=0.1, help='FedTA: IE key loss weight')
    parser.add_argument('--fedta_lambda2', type=float, default=0.1, help='FedTA: contrastive loss weight')
    parser.add_argument('--fedta_lambda3', type=float, default=0.01, help='FedTA: TA key loss weight')
    parser.add_argument('--fedta_tau', type=float, default=0.1, help='FedTA: IE softmax temperature')
    parser.add_argument('--fedta_tau_c', type=float, default=0.1, help='FedTA: contrastive temperature')
    parser.add_argument('--fedta_thr', type=float, default=0.5, help='FedTA: BGPS stability threshold')
    parser.add_argument('--fedta_gamma', type=float, default=0.2, help='FedTA: global prototype EMA gamma')
    parser.add_argument('--fedta_wd', type=float, default=5e-4, help='FedTA: weight decay')
    parser.add_argument('--fedta_warmup_rounds', type=int, default=10, help='FedTA: FedAvg warmup rounds before freezing backbone')
    parser.add_argument('--fedta_freeze_level', type=str, default='partial', choices=['full', 'partial', 'none'],
                        help='FedTA: backbone freezing after warmup. full=whole extractor (paper-faithful, '
                             'needs a strong pre-trained backbone), partial=stem+layer1/2 only, none=no freezing')
    parser.add_argument('--fedta_logit_scale', type=float, default=16.0, help='FedTA: initial cosine-classifier scale')
    parser.add_argument('--fedta_stage1_ep', type=int, default=0, help='FedTA: local epochs for IE stage (0=local_ep//2)')
    parser.add_argument('--fedta_ta_agg', type=str, default='fedavg', choices=['fedavg', 'local'], help='FedTA: TA aggregation mode')
    parser.add_argument('--fedta_surrogate_per_class', type=int, default=20, help='FedTA: surrogate samples per class for SIKF')
    parser.add_argument('--fedta_sikf_steps', type=int, default=5, help='FedTA: SIKF distillation steps')
    parser.add_argument('--fedta_thr_min_round', type=int, default=20, help='FedTA: earliest round to lock BGPS prototypes')

    # LwF arguments
    parser.add_argument('--lwf_temperature', type=float, default=2, help='LwF: temperature for knowledge distillation')
    parser.add_argument('--lwf_lambda', type=float, default=1.0, help='LwF: balance weight for old task loss')
    parser.add_argument('--lwf_warmup_epochs', type=int, default=0, help='LwF: warm-up phase epochs (0=disabled)')

    # MOON arguments
    parser.add_argument('--moon_mu', type=float, default=1.0, help='MOON: weight of model-contrastive loss')
    parser.add_argument('--moon_tau', type=float, default=0.5, help='MOON: temperature of model-contrastive loss')

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

    # TARGET arguments (ICCV'23 exemplar-free distillation, run via main_target.py)
    parser.add_argument('--target_kd', type=float, default=25.0, help='TARGET: weight of old-task KD loss on synthetic data')
    parser.add_argument('--target_kd_T', type=float, default=2.0, help='TARGET: temperature of old-task KD loss')
    parser.add_argument('--target_nz', type=int, default=256, help='TARGET: generator noise dimension')
    parser.add_argument('--target_ngf', type=int, default=64, help='TARGET: generator feature dimension')
    parser.add_argument('--target_syn_round', type=int, default=10, help='TARGET: synthesis rounds per task switch')
    parser.add_argument('--target_g_steps', type=int, default=10, help='TARGET: generator optimization steps per synthesis round')
    parser.add_argument('--target_kd_steps', type=int, default=400, help='TARGET: student distillation steps per synthesis round (after warmup)')
    parser.add_argument('--target_warmup', type=int, default=20, help='TARGET: warmup rounds before enabling adversarial loss and student distillation')
    parser.add_argument('--target_lr_g', type=float, default=0.002, help='TARGET: generator learning rate')
    parser.add_argument('--target_lr_z', type=float, default=0.01, help='TARGET: noise z learning rate')
    parser.add_argument('--target_syn_bs', type=int, default=256, help='TARGET: synthesis batch size')
    parser.add_argument('--target_num_syn', type=int, default=8000, help='TARGET: max number of synthetic images kept in the pool')
    parser.add_argument('--target_oh', type=float, default=0.5, help='TARGET: weight of CE loss on pseudo labels')
    parser.add_argument('--target_adv', type=float, default=1.0, help='TARGET: weight of boundary-support (adversarial) loss')
    parser.add_argument('--target_bn', type=float, default=10.0, help='TARGET: weight of BN statistics matching loss')
    parser.add_argument('--target_T', type=float, default=20.0, help='TARGET: temperature of student distillation')
    parser.add_argument('--target_is_maml', type=int, default=1, help='TARGET: 1=FOMAML, 0=REPTILE meta update for the generator')
    parser.add_argument('--target_bn_mmt', type=float, default=0.9, help='TARGET: momentum of BN statistics in the inversion hook')
    parser.add_argument('--target_reset_l0', type=int, default=1, help='TARGET: reset generator first layer at ep==120+warmup')

    args = parser.parse_args()
    return args
