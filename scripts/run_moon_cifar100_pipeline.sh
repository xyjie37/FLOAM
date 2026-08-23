#!/usr/bin/env bash
set -euo pipefail

# 1) Split CIFAR-100 into task-5 (skip if exists)
# 2) Run MOON task-5 x5 on GPU 0 and task-10 x5 on GPU 1 in parallel

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/home/jxy/miniconda3/envs/floam/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/moon_cifar100}"
mkdir -p "$LOG_DIR"

SPLIT_OUT="${ROOT_DIR}/dataset/cifar100-dir-0.1-task-5"
TASK10_OUT="${ROOT_DIR}/dataset/cifar100-dir-0.1-task-10"

echo "[$(date '+%F %T')] Step 1/2: Split CIFAR-100 task-5 (task-10 already expected at ${TASK10_OUT})"

if [[ -d "${SPLIT_OUT}/train" && -d "${SPLIT_OUT}/test" ]]; then
  echo "  Skip split: ${SPLIT_OUT} already exists"
else
  cd "${ROOT_DIR}/dataset"
  NUM_TASK=5 DATASET_ROOT=/home/jxy/data/cifar100 \
    "$PYTHON" split_data_cifar100.py > "${LOG_DIR}/split_task5.log" 2>&1
  cd "$ROOT_DIR"
  echo "  Split done -> ${SPLIT_OUT}"
fi

if [[ ! -d "${TASK10_OUT}/train" || ! -d "${TASK10_OUT}/test" ]]; then
  echo "ERROR: task-10 dataset not found at ${TASK10_OUT}"
  exit 1
fi

echo "[$(date '+%F %T')] Step 2/2: Run MOON task-5 (GPU 0) and task-10 (GPU 1), 5 runs each"

GPU_ID=0 LOG_DIR="${ROOT_DIR}/logs/moon_cifar100_task5" \
  bash "${ROOT_DIR}/scripts/run_moon_cifar100_task5_5runs.sh" \
  > "${LOG_DIR}/task5_master.log" 2>&1 &
TASK5_PID=$!

GPU_ID=1 LOG_DIR="${ROOT_DIR}/logs/moon_cifar100_task10" \
  bash "${ROOT_DIR}/scripts/run_moon_cifar100_task10_5runs.sh" \
  > "${LOG_DIR}/task10_master.log" 2>&1 &
TASK10_PID=$!

echo "  task-5 MOON PID=${TASK5_PID}, log -> ${LOG_DIR}/task5_master.log"
echo "  task-10 MOON PID=${TASK10_PID}, log -> ${LOG_DIR}/task10_master.log"

wait "${TASK5_PID}"
echo "[$(date '+%F %T')] MOON task-5 runs finished."
wait "${TASK10_PID}"
echo "[$(date '+%F %T')] MOON task-10 runs finished."
echo "[$(date '+%F %T')] Pipeline completed."
