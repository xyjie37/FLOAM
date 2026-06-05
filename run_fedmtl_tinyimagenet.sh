#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash run_fedmtl_tinyimagenet.sh
# FedMTL algorithm on tinyimagenet dataset
# Runs with task_num=5 and task_num=10

EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.1}"
NUM_USERS="${NUM_USERS:-20}"
FRAC="${FRAC:-0.5}"
LOCAL_EP="${LOCAL_EP:-5}"
LOCAL_BS="${LOCAL_BS:-50}"
WD="${WD:-0.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="tinyimagenet"
MODEL="tinyresnet18"
NUM_CLASSES=200
TASK_NUMS=(5 10)

run_one() {
  local task_num="$1"

  local entry="main_fedmtl.py"
  local dataset_path="./dataset/${DATASET}-dir-0.1-task-${task_num}"
  local result_name="fedmtl_${DATASET}_t${task_num}"

  echo "=================================================="
  echo "Running: FedMTL on ${DATASET} (task_num=${task_num})"
  echo "Entry: ${entry}"
  echo "Model: ${MODEL}"
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
    --task_num "${task_num}"
}

for task_num in "${TASK_NUMS[@]}"; do
  run_one "${task_num}"
done

echo "All runs finished."
