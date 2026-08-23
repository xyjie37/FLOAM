#!/usr/bin/env bash
set -euo pipefail

# MOON on CIFAR-10, GPU 1, 5 sequential runs (nohup-friendly).
# Usage:
#   nohup bash scripts/run_moon_cifar10_5runs.sh > logs/moon_cifar10_5runs.nohup.log 2>&1 &

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/home/jxy/miniconda3/envs/floam/bin/python}"
GPU_ID="${GPU_ID:-1}"
NUM_RUNS="${NUM_RUNS:-5}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/moon_cifar10}"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --dataset cifar10
  --model resnet18
  --num_classes 10
  --epochs 100
  --lr 0.1
  --num_users 20
  --frac 0.5
  --local_ep 5
  --local_bs 50
  --wd 0.0
  --datasetpath ./dataset/cifar10-dir-0.1-task-10
  --task_num 10
  --gpu "${GPU_ID}"
)

echo "[$(date '+%F %T')] Start MOON CIFAR-10: ${NUM_RUNS} runs on GPU ${GPU_ID}"
echo "Project root: ${ROOT_DIR}"
echo "Logs: ${LOG_DIR}"

for RUN_IDX in $(seq 0 $((NUM_RUNS - 1))); do
  RESULTS_TAG="fedavg_seed0_run${RUN_IDX}"
  RUN_LOG="${LOG_DIR}/run${RUN_IDX}.log"

  echo "[$(date '+%F %T')] Run ${RUN_IDX}/${NUM_RUNS}: results_save=${RESULTS_TAG}, seed=${RUN_IDX}"
  echo "  log -> ${RUN_LOG}"

  "$PYTHON" main_moon.py \
    "${COMMON_ARGS[@]}" \
    --seed "${RUN_IDX}" \
    --results_save "${RESULTS_TAG}" \
    > "${RUN_LOG}" 2>&1

  echo "[$(date '+%F %T')] Run ${RUN_IDX} finished."
done

echo "[$(date '+%F %T')] All ${NUM_RUNS} runs completed."
