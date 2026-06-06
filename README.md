# Abstract
Federated continual learning over heterogeneous and time-evolving client streams is increasingly required in distributed environments such as camera networks, acoustic sensing systems, and mobile applications. In these settings, raw data cannot be centralized, while client distributions differ across data sources and drift over time. Existing methods often rely on replay, regularization, distillation, or model expansion to mitigate forgetting, but provide limited control over the distortion of class-conditional representations across heterogeneous clients and evolving phases.
We propose FLOAM, an anchor-guided framework that maintains shared class anchors as semantic reference directions through client-balanced aggregation. FLOAM uses these anchors to define an anchor-enhanced optimal transport distillation objective to regularize round-to-round predictive drift and an anchor-driven contrastive objective to improve cross-client representation consistency. A delayed gradient-norm equalization strategy further balances classification, transport distillation, and contrastive alignment during non-IID and temporally evolving local updates.
Experiments on six visual, textual, and acoustic benchmarks show that FLOAM consistently improves final accuracy while maintaining competitive or lower forgetting against representative federated learning, continual learning-enhanced federated learning, and federated continual learning baselines. Spatial and temporal prototype diagnostics further show reduced cross-client misalignment and temporal drift, supporting the effectiveness of anchor-guided representation control for heterogeneous federated continual learning.


# Data Partition
Before running the main code, the data allocation program need to be executed.Data allocation methods are categorized into multi-task and class-incremental types, all data allocation code is located in the `./dataset` folder.

# Run FLOAM
The code can be run as follows:
`python main_floam.py --dataset cifar10 --model resnet18 --num_classes 10 --epochs 100 --lr 0.1 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 50 --results_save run0 --wd 0.0 --datasetpath ./dataset/cifar10-dir-0.1-task-10 --task_num 10`


# Run Other Algorithms
`python main_[algorithms].py --dataset cifar10 --model resnet18 --num_classes 10 --epochs 100 --lr 0.1 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 50 --results_save run0 --wd 0.0 --datasetpath ./dataset/cifar10-dir-0.1-task-10 --task_num 10`
If you want to run other baseline algorithms, simply replace the main script with the corresponding one.

# pretrained checkpoints
We provide the pre-trained checkpoints of FLOAM on various datasets and tasks.

