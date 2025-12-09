# FLOAM: Modality-Agnostic Federated Continual Learning with Anchor-Guided Optimal Transport

Anonymous Authors.

## Abstract
Federated continual learning requires decentralized clients to adapt to evolving data streams while preserving prior knowledge and maintaining consistent representations across heterogeneous devices. These challenges intensify in non-IID streaming environments, where local optimization becomes unstable, and global alignment degrades. We introduce FLOAM, a geometry-aware framework that stabilizes representation learning in federated continual settings. FLOAM employs semantic anchors as shared class-level references and aligns client models through optimal transport distillation and anchor-centered contrastive regularization. Since its alignment mechanism relies solely on geometric representation rather than modality-specific assumptions, FLOAM remains modality-agnostic and applies to visual, textual, and acoustic data. A temporally smoothed gradient balancing strategy further supports stable multi-objective optimization under asynchronous client updates. Experiments across vision, text, and audio benchmarks show that FLOAM improves accuracy, reduces forgetting, and enhances cross-client consistency, highlighting its scalability for multimedia federated environments.

## Data Partition

Before running the main code, the data allocation program need to be executed. Data allocation methods are categorized into multi-task and class-incremental types, all data allocation code is located in the `./dataset` folder.

## Run Code

The code can be run as follows:

```shell
python main_floam.py --dataset cifar10 --model resnet18 --num_classes 10 --epochs 100 --lr 0.1 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 50 --results_save run0 --wd 0.0 --datasetpath ./dataset/cifar10-dir-0.1-task-10 --task_num 10
```

If you want to run other baseline algorithms, simply replace the main script with the corresponding one.

```shell
python main_[baseline].py --dataset cifar10 --model resnet18 --num_classes 10 --epochs 100 --lr 0.1 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 50 --results_save run0 --wd 0.0 --datasetpath ./dataset/cifar10-dir-0.1-task-10 --task_num 10
```

<!-- ```shell
python main_floam.py --dataset speechcommands --model speechresnet --num_classes 30 --epochs 100 --lr 0.1 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 50 --results_save run0 --wd 0.0 --datasetpath ./dataset/speechcommands-dir-0.1-task-10 --task_num 10
``` -->
