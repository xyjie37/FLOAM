# FLOAM: Anchor Memory Framework for Federated Multimedia Continual Learning

Anonymous Authors.

## Abstract
Federated continual learning aims to enable decentralized clients to adapt to evolving data streams while preserving privacy. However, real deployments remain hindered by catastrophic forgetting across tasks, inconsistent representations among heterogeneous clients, and unstable local optimization under non-IID streaming inputs. We introduce FLOAM, a unified framework that integrates semantic anchor memory, anchor-enhanced optimal transport distillation, anchor-driven contrastive learning, and temporally smoothed gradient balancing. Class-level anchors serve as shared semantic references for consistent cross-client alignment without sharing raw data. A dynamic hard-negative mining strategy further strengthens inter-class discrimination, while a linearized and delayed GradNorm mechanism stabilizes multi-objective optimization throughout continual updates. Comprehensive experiments across vision, text, and audio modalities show that FLOAM delivers consistently superior performance in accuracy, forgetting mitigation, and cross-client representation consistency over state-of-the-art federated continual learning baselines. These results highlight FLOAM as a scalable and principled solution for federated continual multimedia intelligence.

## Data Partition

Before running the main code, the data allocation program need to be executed.Data allocation methods are categorized into multi-task and class-incremental types, all data allocation code is located in the `./dataset` folder.

## Run Code

The code can be run as follows:

```shell
python main_floam.py --dataset cifar10 --model resnet18 --num_classes 10 --epochs 100 --lr 0.1 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 50 --results_save run0 --wd 0.0 --datasetpath ./dataset/cifar10-dir-0.1-task-10 --task_num 10
```

If you want to run other baseline algorithms, simply replace the main script with the corresponding one.

```shell
python main_floam.py --dataset speechcommands --model speechresnet --num_classes 30 --epochs 100 --lr 0.1 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 50 --results_save run0 --wd 0.0 --datasetpath ./dataset/speechcommands-dir-0.1-task-10 --task_num 10
```
