#!/usr/bin/env bash
set -euo pipefail

# FedTA on CIFAR-10/100, task-5 & task-10.
# GPU 0: cifar10-t5 -> cifar100-t5
# GPU 1: cifar10-t10 -> cifar100-t10
# CUDA_VISIBLE_DEVICES is set before python so each job binds to the intended card.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/home/jxy/miniconda3/envs/floam/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/fedta}"
mkdir -p "$LOG_ROOT"

run_one() {
  local gpu_id="$1"
  local dataset="$2"
  local num_classes="$3"
  local task_num="$4"
  local datasetpath="$5"
  local tag="$6"
  local log_file="$7"

  echo "[$(date '+%F %T')] [GPU ${gpu_id}] START ${tag}"
  echo "  log -> ${log_file}"

  # Bind the physical GPU before torch initializes; expose it as cuda:0 inside the process.
  CUDA_VISIBLE_DEVICES="${gpu_id}" "$PYTHON" main_fedta.py \
    --dataset "${dataset}" \
    --model resnet18 \
    --num_classes "${num_classes}" \
    --epochs 100 \
    --lr 0.1 \
    --num_users 20 \
    --frac 0.5 \
    --local_ep 5 \
    --local_bs 50 \
    --results_save "fedta_seed0" \
    --wd 0.0 \
    --datasetpath "${datasetpath}" \
    --task_num "${task_num}" \
    --gpu 0 \
    > "${log_file}" 2>&1

  echo "[$(date '+%F %T')] [GPU ${gpu_id}] DONE  ${tag}"
}

run_gpu0() {
  run_one 0 cifar10 10 5 \
    ./dataset/cifar10-dir-0.1-task-5 \
    cifar10_task5 \
    "${LOG_ROOT}/cifar10_task5.log"

  run_one 0 cifar100 100 5 \
    ./dataset/cifar100-dir-0.1-task-5 \
    cifar100_task5 \
    "${LOG_ROOT}/cifar100_task5.log"
}

run_gpu1() {
  run_one 1 cifar10 10 10 \
    ./dataset/cifar10-dir-0.1-task-10 \
    cifar10_task10 \
    "${LOG_ROOT}/cifar10_task10.log"

  run_one 1 cifar100 100 10 \
    ./dataset/cifar100-dir-0.1-task-10 \
    cifar100_task10 \
    "${LOG_ROOT}/cifar100_task10.log"
}

echo "[$(date '+%F %T')] FedTA pipeline start"
echo "  GPU0: cifar10-t5 -> cifar100-t5"
echo "  GPU1: cifar10-t10 -> cifar100-t10"

run_gpu0 > "${LOG_ROOT}/gpu0_master.log" 2>&1 &
PID0=$!
run_gpu1 > "${LOG_ROOT}/gpu1_master.log" 2>&1 &
PID1=$!

echo "  GPU0 PID=${PID0}, log -> ${LOG_ROOT}/gpu0_master.log"
echo "  GPU1 PID=${PID1}, log -> ${LOG_ROOT}/gpu1_master.log"

wait "${PID0}"
echo "[$(date '+%F %T')] GPU0 chain finished."
wait "${PID1}"
echo "[$(date '+%F %T')] GPU1 chain finished."
echo "[$(date '+%F %T')] FedTA pipeline completed."
