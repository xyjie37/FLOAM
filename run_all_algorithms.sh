#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash run_all_algorithms.sh
# 默认：Tiny-ImageNet、task_num=5 和 10、num_classes=200、num_users=20、frac=0.5、lr=0.1、model=tinyresnet18；
# 下列算法各运行一次（t5 和 t10 两种任务数配置）。
# Optional overrides:
#   EPOCHS=100 LR=0.1 NUM_USERS=20 FRAC=0.5 LOCAL_EP=5 LOCAL_BS=50 WD=0.0 bash run_all_algorithms.sh

EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.1}"
NUM_USERS="${NUM_USERS:-20}"
FRAC="${FRAC:-0.5}"
LOCAL_EP="${LOCAL_EP:-5}"
LOCAL_BS="${LOCAL_BS:-50}"
WD="${WD:-0.0}"
MODEL="${MODEL:-tinyresnet18}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# RefFiL 对应入口 main_RefFiL.py（algo 名 reffil）
ALGORITHMS=("fedavg" "fedclass" "fedewc" "fedknow" "fedlwf" "fedprox" "refed" "reffil" "floam")
DATASET="tinyimagenet"
NUM_CLASSES=200
TASK_NUMS=(10)

run_one() {
  local algo="$1"
  local dataset="$2"
  local task_num="$3"

  local entry
  if [[ "$algo" == "reffil" ]]; then
    entry="main_RefFiL.py"
  else
    entry="main_${algo}.py"
  fi
  local dataset_path="./dataset/${dataset}-dir-0.1-task-${task_num}"
  local result_name="${algo}_${dataset}_t${task_num}"

  echo "=================================================="
  echo "Running: ${algo} on ${dataset} (task_num=${task_num})"
  echo "Entry: ${entry}"
  echo "Result tag: ${result_name}"
  echo "=================================================="

  "${PYTHON_BIN}" "${entry}" \
    --dataset "${dataset}" \
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
  for algo in "${ALGORITHMS[@]}"; do
    run_one "${algo}" "${DATASET}" "${task_num}"
  done
done

echo "All runs finished."
