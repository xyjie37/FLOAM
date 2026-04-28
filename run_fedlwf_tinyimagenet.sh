#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash run_fedlwf_tinyimagenet.sh
# Optional overrides:
#   EPOCHS=100 LR=0.1 NUM_USERS=20 FRAC=0.5 LOCAL_EP=5 LOCAL_BS=50 WD=0.0 MODEL=resnet18 bash run_fedlwf_tinyimagenet.sh

EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.1}"
NUM_USERS="${NUM_USERS:-20}"
FRAC="${FRAC:-0.5}"
LOCAL_EP="${LOCAL_EP:-5}"
LOCAL_BS="${LOCAL_BS:-50}"
WD="${WD:-0.0}"
MODEL="${MODEL:-tinyresnet18}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ALGO="fedlwf"
DATASET="tinyimagenet"
NUM_CLASSES=200
TASKS=(5 10)
SEEDS=(0 1 2)

run_one() {
  local task_num="$1"
  local seed="$2"
  local entry="main_${ALGO}.py"
  local dataset_path="./dataset/${DATASET}-dir-0.1-task-${task_num}"
  local result_name="${ALGO}_${DATASET}_task${task_num}_seed${seed}"

  echo "=================================================="
  echo "Running: ${ALGO} on ${DATASET} | task=${task_num} | seed=${seed}"
  echo "Entry: ${entry}"
  echo "Dataset path: ${dataset_path}"
  echo "Result tag: ${result_name}"
  echo "=================================================="

  "${PYTHON_BIN}" "${entry}" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --num_classes "${NUM_CLASSES}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --num_users "${NUM_USERS}" \
    --frac "${FRAC}" \
    --local_ep "${LOCAL_EP}" \
    --local_bs "${LOCAL_BS}" \
    --results_save "${result_name}" \
    --wd "${WD}" \
    --datasetpath "${dataset_path}" \
    --task_num "${task_num}" \
    --seed "${seed}"
}

for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "${task}" "${seed}"
  done
done

echo "All runs finished."
