### Abstract
Federated continual learning (FCL) aims to enable decentralized agents to incrementally acquire knowledge from non-stationary and privacy-sensitive data distributions. However, real-world deployments face three compounding challenges: catastrophic forgetting, representation inconsistency across clients, and unstable local adaptation due to data heterogeneity. To address these issues, we propose FLOAM (Federated Learning with Optimized Anchored Memory), a modular framework that integrates semantic anchor memory, anchor-guided distillation, and meta-adaptive optimization. FLOAM introduces class-level anchor prototypes that serve as shared semantic references across clients, enabling both contrastive feature alignment and privacy-preserving knowledge retention. A dynamic hard negative mining mechanism enhances inter-class discriminability, while meta-learned loss balancing improves client-side stability under continual updates. We evaluate FLOAM on multiple benchmarks (CIFAR-10, CIFAR-100, Tiny-ImageNet) under class-incremental and multitask settings, demonstrating superior performance over strong FCL baselines in terms of accuracy, forgetting, and adaptability. FLOAM provides a scalable and principled solution for continual federated intelligence.

### Split Data
Before running the main code, the data allocation program need to be executed.Data allocation methods are categorized into multi-task and class-incremental types, all data allocation code is located in the `./dataset` folder.

### Run FLOAM
The code can be run as follows:<br>
`python main_floam.py --dataset cifar10 --model resnet18 --num_classes 10 --epochs 100 --lr 0.1 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 50 --results_save run0 --wd 0.0 --datasetpath ./dataset/cifar10-dir-0.1-task-10 --task_num 10`<br>
If you want to run other baseline algorithms, simply replace the main script with the corresponding one.


python main_floam.py --dataset speechcommands --model speechresnet --num_classes 30 --epochs 100 --lr 0.1 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 50 --results_save run0 --wd 0.0 --datasetpath ./dataset/speechcommands-dir-0.1-task-10 --task_num 10