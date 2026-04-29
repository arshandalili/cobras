#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash runner.sh [REPEAT]
#
# Examples:
#   uv run bash runner.sh
#   uv run bash runner.sh 5
#   RUN_UF=0 uv run bash runner.sh
#   RUN_TQA=0 uv run bash runner.sh
#   RUN_TQA=0 RUN_UF=0 RUN_MMLU=1 RUN_GSM8k=0 uv run bash runner.sh 1
#   RUN_TQA=0 RUN_UF=0 RUN_MMLU=0 RUN_GSM8k=1 uv run bash runner.sh 1
# Note: You can set the environment variables RUN_TQA, RUN_UF, and RUN_MMLU to control which experiments to run. 
# By default, all experiments will be run.

REPEAT="${1:-3}"
RUN_TQA="${RUN_TQA:-1}"
RUN_UF="${RUN_UF:-1}"
RUN_MMLU="${RUN_MMLU:-1}"
RUN_GSM8K="${RUN_GSM8K:-1}"

declare -A MODEL_TO_LAYER=(
  ["Llama3.1-8B-Base"]=13
  ["Falcon-7B-Base"]=14
  ["Mistral-7B-Base"]=15
  ["Qwen2.5-7B-Base"]=13
)

declare -A MODEL_TO_T_VALUES=(
  ["Llama3.1-8B-Base"]="3 4 5 6"
  ["Falcon-7B-Base"]="20 21 22 23"
  ["Mistral-7B-Base"]="2 3 4 5"
  ["Qwen2.5-7B-Base"]="13 14 15 16"
)

MODELS=(
  "Llama3.1-8B-Base"
  "Falcon-7B-Base"
  "Mistral-7B-Base"
  "Qwen2.5-7B-Base"
)

for model in "${MODELS[@]}"; do
  layer_idx="${MODEL_TO_LAYER[$model]}"
  t_values="${MODEL_TO_T_VALUES[$model]}"

  echo "========================================"
  echo "MODEL=${model}"
  echo "LAYER_IDX=${layer_idx}"
  echo "REPEAT=${REPEAT}"
  echo "T_VALUES=${t_values}"
  echo "========================================"

  for t in ${t_values}; do
    if [[ "${RUN_TQA}" == "1" ]]; then
      echo "===== Running TruthfulQA | MODEL=${model} | LAYER=${layer_idx} | T=${t} ====="
      uv run bash experiments_truthfulqa.sh "${model}" "${layer_idx}" "${REPEAT}" "${t}"
    fi

    if [[ "${RUN_UF}" == "1" ]]; then
      echo "===== Running UltraFeedback | MODEL=${model} | LAYER=${layer_idx} | T=${t} ====="
      uv run bash experiments_ultrafeedback.sh "${model}" "${layer_idx}" "${REPEAT}" "${t}"
    fi

    if [[ "${RUN_MMLU}" == "1" ]]; then
      echo "===== Running MMLU OOD | MODEL=${model} | LAYER=${layer_idx} | T=${t} ====="
      uv run bash experiments_mmlu.sh "${model}" "${layer_idx}" "${REPEAT}" "${t}"
    fi
    if [[ "${RUN_GSM8K}" == "1" ]]; then
      echo "===== Running GSM8K OOD | MODEL=${model} | LAYER=${layer_idx} | T=${t} ====="
      uv run bash experiments_gsm8k.sh "${model}" "${layer_idx}" "${REPEAT}" "${t}"
    fi
  done
done