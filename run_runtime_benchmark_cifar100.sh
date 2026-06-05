#!/usr/bin/env bash

set -euo pipefail

# Benchmark per-round runtime on CIFAR-100 for FedKNOW, RefFiL, CgoFed, and FLOAM.
# Round time = max(selected client times) + server time (serial client simulation).
#
# Usage:
#   bash run_runtime_benchmark_cifar100.sh
#
# Optional overrides:
#   EPOCHS=10 DATASET_PATH=/path/to/cifar100-dir-0.1-task-10 GPU=0 bash run_runtime_benchmark_cifar100.sh

EPOCHS="${EPOCHS:-10}"
LR="${LR:-0.1}"
NUM_USERS="${NUM_USERS:-20}"
FRAC="${FRAC:-0.5}"
LOCAL_EP="${LOCAL_EP:-5}"
LOCAL_BS="${LOCAL_BS:-50}"
WD="${WD:-0.0}"
MODEL="${MODEL:-resnet18}"
TASK_NUM="${TASK_NUM:-10}"
NUM_CLASSES="${NUM_CLASSES:-100}"
DATASET="${DATASET:-cifar100}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/jxy/miniconda3/envs/waihai/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATASET_PATH="${PROJECT_ROOT}/dataset/${DATASET}-dir-0.1-task-${TASK_NUM}"
FALLBACK_DATASET_PATH="/home/jxy/hys/data/${DATASET}-dir-0.1-task-${TASK_NUM}"
DATASET_PATH="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runtime_stats/cifar100_t${TASK_NUM}}"

if [[ ! -d "${DATASET_PATH}" ]]; then
  if [[ -d "${FALLBACK_DATASET_PATH}" ]]; then
    echo "Dataset not found at ${DATASET_PATH}, using fallback: ${FALLBACK_DATASET_PATH}"
    DATASET_PATH="${FALLBACK_DATASET_PATH}"
  else
    echo "Error: CIFAR-100 partition not found."
    echo "Expected: ${DEFAULT_DATASET_PATH}"
    echo "Or set DATASET_PATH to an existing partition directory."
    exit 1
  fi
fi

mkdir -p "${OUTPUT_DIR}"

declare -A ALGO_ENTRIES=(
  ["fedknow"]="main_fedknow.py"
  ["reffil"]="main_RefFiL.py"
  ["cgofed"]="main_cgofed.py"
  ["floam"]="main_floam.py"
)

COMMON_ARGS=(
  --dataset "${DATASET}"
  --model "${MODEL}"
  --num_classes "${NUM_CLASSES}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --num_users "${NUM_USERS}"
  --frac "${FRAC}"
  --local_ep "${LOCAL_EP}"
  --local_bs "${LOCAL_BS}"
  --wd "${WD}"
  --datasetpath "${DATASET_PATH}"
  --task_num "${TASK_NUM}"
  --gpu "${GPU}"
  --benchmark_runtime
  --skip_eval
)

run_one() {
  local algo="$1"
  local entry="${ALGO_ENTRIES[$algo]}"
  local runtime_csv="${OUTPUT_DIR}/${algo}_runtime.csv"
  local result_name="runtime_${algo}_cifar100_t${TASK_NUM}"

  echo "=================================================="
  echo "Benchmark runtime: ${algo}"
  echo "Entry: ${entry}"
  echo "Output CSV: ${runtime_csv}"
  echo "=================================================="

  "${PYTHON_BIN}" "${PROJECT_ROOT}/${entry}" \
    "${COMMON_ARGS[@]}" \
    --results_save "${result_name}" \
    --runtime_csv "${runtime_csv}"
}

for algo in fedknow reffil cgofed floam; do
  run_one "${algo}"
done

SUMMARY_CSV="${OUTPUT_DIR}/all_algorithms_runtime_summary.csv"
"${PYTHON_BIN}" - <<PY
import glob
import json
import os
import pandas as pd

output_dir = "${OUTPUT_DIR}"
rows = []
for csv_path in sorted(glob.glob(os.path.join(output_dir, "*_runtime.csv"))):
    df = pd.read_csv(csv_path)
    avg_row = df[df["round"].astype(str) == "AVG"]
    if avg_row.empty:
        continue
    algo = os.path.basename(csv_path).replace("_runtime.csv", "")
    detail = {}
    try:
        detail = json.loads(avg_row.iloc[0]["client_times_json"])
    except Exception:
        pass
    rows.append({
        "algorithm": algo,
        "num_rounds": int(avg_row.iloc[0]["num_selected_clients"]),
        "avg_max_client_time_s": float(avg_row.iloc[0]["max_client_time"]),
        "avg_server_time_s": float(avg_row.iloc[0]["server_time"]),
        "avg_round_time_s": float(avg_row.iloc[0]["round_time"]),
        "std_round_time_s": float(detail.get("std_round_time", 0.0)),
        "runtime_csv": csv_path,
    })

summary_df = pd.DataFrame(rows)
summary_path = "${SUMMARY_CSV}"
summary_df.to_csv(summary_path, index=False)
print(f"Combined summary saved to: {summary_path}")
if not summary_df.empty:
    print(summary_df.to_string(index=False))
PY

echo "All runtime benchmarks finished."
echo "Per-algorithm CSV files are under: ${OUTPUT_DIR}"
