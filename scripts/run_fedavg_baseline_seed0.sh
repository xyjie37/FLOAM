#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root regardless of where this script is invoked.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

run_dir="./save/dataset/cifar10-dir-0.1-task-10/resnet18_num20_C0.5_le5_bs50_round100_m0.5_lr0.1/fedavg_baseline/fedavg/seed_0"
mkdir -p "${run_dir}"

{
  echo "Verifying the frozen CIFAR-10 data split..."
  (
    cd ./dataset/cifar10-dir-0.1-task-10
    # The manifest was generated on Windows and has CRLF line endings. Strip
    # only the carriage returns in the stream so GNU sha256sum sees the exact
    # relative file names while the original manifest remains unchanged.
    tr -d '\r' < SHA256SUMS.txt | sha256sum --quiet -c -
    sha256sum SHA256SUMS.txt
  )
  echo "Dataset SHA-256 verification passed."

  python -u main_fedavg.py \
    --dataset cifar10 \
    --datasetpath ./dataset/cifar10-dir-0.1-task-10 \
    --model resnet18 \
    --norm batch_norm \
    --num_channels 3 \
    --num_classes 10 \
    --task_num 10 \
    --epochs 100 \
    --num_users 20 \
    --frac 0.5 \
    --local_ep 5 \
    --local_bs 50 \
    --bs 500 \
    --opt SGD \
    --lr 0.1 \
    --lr_decay 1.0 \
    --momentum 0.5 \
    --wd 0.0 \
    --split user \
    --fl_alg FedAvg \
    --seed 0 \
    --gpu 0 \
    --test_freq 10 \
    --results_save fedavg_baseline
} 2>&1 | tee "${run_dir}/train.log"
