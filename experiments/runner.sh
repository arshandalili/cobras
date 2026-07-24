#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash experiments/runner.sh [REPEAT]
#
# Examples:
#
#   BASELINES ::: 
# 
#   RUN_UF=0 uv run bash experiments/runner.sh 3
#   RUN_TQA=1 uv run bash experiments/runner.sh 3
#   RUN_TOXICITY=1 uv run bash experiments/runner.sh 3
#
#   OOD EVALS :::
#
#   RUN_MMLU=1 uv run bash experiments/runner.sh 1
#   RUN_GSM8K=1 uv run bashexperiments/runner.sh 1
#   RUN_NQ=1 uv run bash experiments/runner.sh 1
#   RUN_TRIVIAQA=1 uv run bash experiments/runner.sh 1
# Note: You can set the environment variables RUN_TQA, RUN_UF, and RUN_MMLU to control which experiments to run.
# By default, none of the experiments will be run.

REPEAT="${1:-3}"
RUN_TQA="${RUN_TQA:-0}"
RUN_UF="${RUN_UF:-0}"
RUN_MMLU="${RUN_MMLU:-0}"
RUN_GSM8K="${RUN_GSM8K:-0}"
RUN_NQ="${RUN_NQ:-0}"
RUN_TRIVIAQA="${RUN_TRIVIAQA:-0}"
RUN_TOXICITY="${RUN_TOXICITY:-0}"

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
      uv run bash experiments/experiments_truthfulqa.sh "${model}" "${layer_idx}" "${REPEAT}" "${t}"
    fi

    if [[ "${RUN_UF}" == "1" ]]; then
      echo "===== Running UltraFeedback | MODEL=${model} | LAYER=${layer_idx} | T=${t} ====="
      uv run bash experiments/experiments_ultrafeedback.sh "${model}" "${layer_idx}" "${REPEAT}" "${t}"
    fi
  
    if [[ "${RUN_TOXICITY}" == "1" ]]; then
      echo "===== Running TOXICITY | MODEL=${model} | LAYER=${layer_idx} | T=${t} ====="
      uv run bash experiments/experiments_toxicity.sh "${model}" "${layer_idx}" "${REPEAT}" "${t}"
    fi

    if [[ "${RUN_MMLU}" == "1" ]]; then
      echo "===== Running MMLU OOD | MODEL=${model} | LAYER=${layer_idx} | T=${t} ====="
      uv run bash experiments/experiments_mmlu.sh "${model}" "${layer_idx}" "${REPEAT}" "${t}"
    fi

    if [[ "${RUN_GSM8K}" == "1" ]]; then
      echo "===== Running GSM8K OOD | MODEL=${model} | LAYER=${layer_idx} | T=${t} ====="
      uv run bash experiments/experiments_gsm8k.sh "${model}" "${layer_idx}" "${REPEAT}" "${t}"
    fi

    if [[ "${RUN_NQ}" == "1" ]]; then
      echo "===== Running NQ OOD | MODEL=${model} | LAYER=${layer_idx} | T=${t} ====="
      uv run bash experiments/experiments_nq.sh "${model}" "${layer_idx}" "${REPEAT}" "${t}"
    fi

    if [[ "${RUN_TRIVIAQA}" == "1" ]]; then
      echo "===== Running TriviaQA OOD | MODEL=${model} | LAYER=${layer_idx} | T=${t} ====="
      uv run bash experiments/experiments_triviaqa.sh "${model}" "${layer_idx}" "${REPEAT}" "${t}"
    fi
  done
done