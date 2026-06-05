#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash run_floam_multi_datasets.sh
# FLOAM algorithm on yahooanswers, 20newsgroups, and speechcommands datasets
# Each configuration runs 2 times with task_num=5 and task_num=10

EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.1}"
NUM_USERS="${NUM_USERS:-20}"
FRAC="${FRAC:-0.5}"
LOCAL_EP="${LOCAL_EP:-5}"
LOCAL_BS="${LOCAL_BS:-50}"
WD="${WD:-0.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TASK_NUMS=(5 10)
RUNS=2

declare -A DATASETS
DATASETS=(
  ["yahooanswers"]="textcnn 10 0.1 20"
  ["20newsgroup"]="textcnn 20 0.3 5"
  ["speechcommands"]="speechresnet 30 0.1 20"
)

declare -A DATASET_PATHS
DATASET_PATHS=(
  ["yahooanswers_t5"]="/home/jxy/FLOAM-all-from-zzs/FLOAM-MM/dataset/yahooanswers-dir-0.1-task-5"
  ["yahooanswers_t10"]="/home/jxy/FLOAM-all-from-zzs/FLOAM-MM/dataset/yahooanswers-dir-0.1-task-10"
  ["speechcommands_t5"]="/home/jxy/FLOAM-all-from-zzs/FLOAM-MM/dataset/speechcommands-dir-0.1-task-5"
  ["speechcommands_t10"]="/home/jxy/FLOAM-all-from-zzs/FLOAM-MM/dataset/speechcommands-dir-0.1-task-10"
)

run_one() {
  local dataset="$1"
  local model="$2"
  local num_classes="$3"
  local dir="$4"
  local num_users="$5"
  local task_num="$6"
  local run_id="$7"

  local entry="main_floam.py"
  local dataset_path
  local path_key="${dataset}_t${task_num}"
  if [[ -n "${DATASET_PATHS[$path_key]+x}" ]]; then
    dataset_path="${DATASET_PATHS[$path_key]}"
  else
    dataset_path="./dataset/${dataset}-dir-${dir}-task-${task_num}"
  fi
  local result_name="floam_${dataset}_t${task_num}_run${run_id}"

  echo "=================================================="
  echo "Running: FLOAM on ${dataset} (task_num=${task_num}, run=${run_id})"
  echo "Entry: ${entry}"
  echo "Model: ${model}"
  echo "Result tag: ${result_name}"
  echo "=================================================="

  "${PYTHON_BIN}" "${entry}" \
    --dataset "${dataset}" \
    --model "${model}" \
    --num_classes "${num_classes}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --num_users "${num_users}" \
    --frac "${FRAC}" \
    --local_ep "${LOCAL_EP}" \
    --local_bs "${LOCAL_BS}" \
    --results_save "${result_name}" \
    --wd "${WD}" \
    --datasetpath "${dataset_path}" \
    --task_num "${task_num}"
}

for dataset in "${!DATASETS[@]}"; do
  IFS=' ' read -r model num_classes dir num_users <<< "${DATASETS[$dataset]}"
  for task_num in "${TASK_NUMS[@]}"; do
    for ((run=1; run<=RUNS; run++)); do
      run_one "${dataset}" "${model}" "${num_classes}" "${dir}" "${num_users}" "${task_num}" "${run}"
    done
  done
done

echo "All runs finished."
