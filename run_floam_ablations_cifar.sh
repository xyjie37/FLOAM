#!/usr/bin/env bash

set -euo pipefail

# CIFAR-10 / CIFAR-100 ablations only (20newsgroup already completed).
# Usage:
#   bash run_floam_ablations_cifar.sh
# Optional:
#   EPOCHS=100 SEED=1 GPU=0 bash run_floam_ablations_cifar.sh

EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.1}"
FRAC="${FRAC:-0.5}"
LOCAL_EP="${LOCAL_EP:-5}"
LOCAL_BS="${LOCAL_BS:-50}"
WD="${WD:-0.0}"
TASK_NUM="${TASK_NUM:-5}"
SEED="${SEED:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/jxy/FLOAM_ALL/scripts/python_waihai.sh}"
GPU="${GPU:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

declare -A DATASETS
DATASETS=(
  ["cifar10"]="resnet18 10 20 0.1"
  ["cifar100"]="resnet18 100 20 0.1"
)

declare -a VARIANTS=(
  "full:"
  "no_pred_geom:--ot_cost uniform"
  "no_shared_anchor:--contrast_target local_mean"
  "sample_weighted:--anchor_agg sample_weighted"
)

run_one() {
  local dataset="$1"
  local model="$2"
  local num_classes="$3"
  local num_users="$4"
  local dir="$5"
  local variant_name="$6"
  local variant_args="$7"

  local dataset_path="./dataset/${dataset}-dir-${dir}-task-${TASK_NUM}"
  local result_name="floam_abl_${variant_name}_${dataset}_t${TASK_NUM}_seed${SEED}"

  echo "=================================================="
  echo "Running: FLOAM ablation=${variant_name} on ${dataset}"
  echo "Dataset path: ${dataset_path}"
  echo "Result tag: ${result_name}"
  echo "Extra args: ${variant_args}"
  echo "=================================================="

  local -a cmd=(
    "${PYTHON_BIN}" main_floam.py
    --dataset "${dataset}"
    --model "${model}"
    --num_classes "${num_classes}"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --num_users "${num_users}"
    --frac "${FRAC}"
    --local_ep "${LOCAL_EP}"
    --local_bs "${LOCAL_BS}"
    --results_save "${result_name}"
    --wd "${WD}"
    --datasetpath "${dataset_path}"
    --task_num "${TASK_NUM}"
    --seed "${SEED}"
    --gpu "${GPU}"
  )

  if [[ -n "${variant_args}" ]]; then
    # shellcheck disable=SC2206
    cmd+=(${variant_args})
  fi

  if [[ "${SKIP_EVAL}" == "1" ]]; then
    cmd+=(--skip_eval)
  fi

  "${cmd[@]}"
}

for dataset in cifar10 cifar100; do
  IFS=' ' read -r model num_classes num_users dir <<< "${DATASETS[$dataset]}"
  for variant in "${VARIANTS[@]}"; do
    variant_name="${variant%%:*}"
    variant_args="${variant#*:}"
    run_one "${dataset}" "${model}" "${num_classes}" "${num_users}" "${dir}" \
      "${variant_name}" "${variant_args}"
  done
done

echo "CIFAR ablation runs finished."
