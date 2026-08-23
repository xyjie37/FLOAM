#!/usr/bin/env bash
set -euo pipefail

# 1) Split CIFAR-10 into task-5
# 2) Download/split CIFAR-100 in background (CPU/network)
# 3) Run FedAvg task-5 x5 on GPU 1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/home/jxy/miniconda3/envs/floam/bin/python}"
GPU_ID="${GPU_ID:-1}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/fedavg_cifar10_task5}"
CIFAR100_LOG_DIR="${ROOT_DIR}/logs/cifar100_download"
mkdir -p "$LOG_DIR" "$CIFAR100_LOG_DIR" /home/jxy/data/cifar100

SPLIT_OUT="${ROOT_DIR}/dataset/cifar10-dir-0.1-task-5"

echo "[$(date '+%F %T')] Step 1/3: Split CIFAR-10 into task-5 + start CIFAR-100 download in parallel"

# CIFAR-100 download/split runs in parallel (CPU/network, no GPU needed)
(
  cd "${ROOT_DIR}/dataset"
  NUM_TASK=10 DATASET_ROOT=/home/jxy/data/cifar100 \
    "$PYTHON" split_data_cifar100.py
) > "${CIFAR100_LOG_DIR}/split_cifar100.log" 2>&1 &
CIFAR100_PID=$!
echo "  CIFAR-100 job PID=${CIFAR100_PID}, log -> ${CIFAR100_LOG_DIR}/split_cifar100.log"

if [[ -d "${SPLIT_OUT}/train" && -d "${SPLIT_OUT}/test" ]]; then
  echo "  Skip split: ${SPLIT_OUT} already exists"
else
  cd "${ROOT_DIR}/dataset"
  NUM_TASK=5 DATASET_ROOT=/home/jxy/data/cifar10 \
    "$PYTHON" split_data_cifar10.py > "${LOG_DIR}/split_task5.log" 2>&1
  cd "$ROOT_DIR"
  echo "  Split done -> ${SPLIT_OUT}"
fi

echo "[$(date '+%F %T')] Step 2/3: Run FedAvg task-5 (5 runs) on GPU ${GPU_ID}"
GPU_ID="${GPU_ID}" bash "${ROOT_DIR}/scripts/run_fedavg_cifar10_task5_5runs.sh"

echo "[$(date '+%F %T')] FedAvg task-5 runs finished. Waiting for CIFAR-100 job (PID=${CIFAR100_PID})..."
wait "${CIFAR100_PID}" || true
echo "[$(date '+%F %T')] Pipeline completed."
