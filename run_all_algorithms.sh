#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash run_all_algorithms.sh
# Optional overrides:
#   EPOCHS=100 LR=0.1 NUM_USERS=20 FRAC=0.5 LOCAL_EP=5 LOCAL_BS=50 WD=0.0 bash run_all_algorithms.sh

EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.1}"
NUM_USERS="${NUM_USERS:-20}"
FRAC="${FRAC:-0.5}"
LOCAL_EP="${LOCAL_EP:-5}"
LOCAL_BS="${LOCAL_BS:-50}"
WD="${WD:-0.0}"
TASK_NUM="${TASK_NUM:-5}"
MODEL="${MODEL:-resnet18}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ALGORITHMS=("cgofed" "fedavg" "fedclass" "fedewc" "fedknow" "fedlwf" "fedmtl" "fedprox" "fedta" "floam" "refed" "reffil")
DATASETS=("cifar10" "cifar100")

run_one() {
  local algo="$1"
  local dataset="$2"

  local num_classes
  if [[ "$dataset" == "cifar10" ]]; then
    num_classes=10
  elif [[ "$dataset" == "cifar100" ]]; then
    num_classes=100
  else
    echo "Unsupported dataset: $dataset"
    exit 1
  fi

  local entry
  if [[ "$algo" == "reffil" ]]; then
    entry="main_RefFiL.py"
  else
    entry="main_${algo}.py"
  fi
  local dataset_path="./dataset/${dataset}-dir-0.1-task-${TASK_NUM}"
  local result_name="${algo}_${dataset}"

  echo "=================================================="
  echo "Running: ${algo} on ${dataset}"
  echo "Entry: ${entry}"
  echo "Result tag: ${result_name}"
  echo "=================================================="

  "${PYTHON_BIN}" "${entry}" \
    --dataset "${dataset}" \
    --model "${MODEL}" \
    --num_classes "${num_classes}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --num_users "${NUM_USERS}" \
    --frac "${FRAC}" \
    --local_ep "${LOCAL_EP}" \
    --local_bs "${LOCAL_BS}" \
    --results_save "${result_name}" \
    --wd "${WD}" \
    --datasetpath "${dataset_path}" \
    --task_num "${TASK_NUM}"
}

for algo in "${ALGORITHMS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    run_one "${algo}" "${dataset}"
  done
done

echo "All runs finished."
