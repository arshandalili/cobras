#!/usr/bin/env bash
set -euo pipefail

# Separates the contribution of the abstention gate from that of the steering rule:
# every COBRAS gate/step variant, and every baseline with the identical kNN gate.
#
# Usage:
#   uv run bash experiments/gate_sweep.sh <TASK> <MODEL> [LAYER_IDX] [T] [SEED]
#
# Example:
#   CUDA_VISIBLE_DEVICES=4 uv run bash experiments/gate_sweep.sh truthfulqa Llama3.1-8B-Base 13 4 42

TASK="${1:?Please provide TASK: truthfulqa | gsm8k | mmlu | nq | triviaqa}"
MODEL="${2:?Please provide MODEL, e.g. Llama3.1-8B-Base}"
LAYER_IDX="${3:-13}"
T="${4:-4}"
SEED="${5:-42}"

COBRAS_T="${COBRAS_T:-0.5}"
GSM8K_N="${GSM8K_N:-1000}"
MMLU_N="${MMLU_N:-3000}"
QA_N="${QA_N:-1000}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

# baselines are swept at the model-specific T, COBRAS variants at COBRAS_T
BASELINE_STEERS=(NoSteer CAA GateSweep-CAA-Gate ODESteer GateSweep-ODESteer-Gate SphericalSteer GateSweep-SphericalSteer-Gate)
COBRAS_STEERS=(
  COBRAS GateSweep-COBRAS-Gate GateSweep-COBRAS-Raw GateSweep-COBRAS-NoVMF GateSweep-COBRAS-RawNoVMF
  GateSweep-COBRAS-SBGate-density GateSweep-COBRAS-SBGate-drift
)

echo "===== Config ====="
echo "TASK=${TASK} MODEL=${MODEL} LAYER_IDX=${LAYER_IDX} T=${T} COBRAS_T=${COBRAS_T} SEED=${SEED}"

run_one () {
  local steer="$1" steer_t="$2"
  echo "===== ${TASK} | ${steer} | T=${steer_t} | seed ${SEED} ====="
  case "${TASK}" in
    truthfulqa)
      uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
        model="${MODEL}" layer_idx="${LAYER_IDX}" steer="${steer}" steer.T="${steer_t}" seed="${SEED}"
      ;;
    mmlu)
      uv run python -u scripts/mmlu/mmlu_generate.py \
        model="${MODEL}" layer_idx="${LAYER_IDX}" steer="${steer}" steer.T="${steer_t}" \
        seed="${SEED}" num_examples="${MMLU_N}"
      ;;
    gsm8k)
      if [[ "${NUM_PROCESSES}" == "1" ]]; then
        # the accelerate path resolves a device without an index when run single-process
        uv run python -u scripts/gsm8k/gsm8k_generate.py \
          model="${MODEL}" layer_idx="${LAYER_IDX}" steer="${steer}" steer.T="${steer_t}" \
          seed="${SEED}" num_examples="${GSM8K_N}"
      else
        uv run accelerate launch --num_processes "${NUM_PROCESSES}" scripts/gsm8k/gsm8k_generate_fast.py \
          model="${MODEL}" layer_idx="${LAYER_IDX}" steer="${steer}" steer.T="${steer_t}" \
          seed="${SEED}" num_examples="${GSM8K_N}"
        uv run python -u scripts/gsm8k/gsm8k_generate_merge.py \
          model="${MODEL}" layer_idx="${LAYER_IDX}" steer="${steer}" steer.T="${steer_t}" \
          seed="${SEED}" num_examples="${GSM8K_N}"
      fi
      ;;
    nq|triviaqa)
      uv run python -u "scripts/${TASK}/${TASK}_generate.py" \
        model="${MODEL}" layer_idx="${LAYER_IDX}" steer="${steer}" steer.T="${steer_t}" \
        seed="${SEED}" num_examples="${QA_N}"
      ;;
    *)
      echo "Unknown TASK: ${TASK}" && exit 1
      ;;
  esac
}

for steer in "${BASELINE_STEERS[@]}"; do
  run_one "${steer}" "${T}"
done

for steer in "${COBRAS_STEERS[@]}"; do
  run_one "${steer}" "${COBRAS_T}"
done

echo "===== Evaluating ====="
case "${TASK}" in
  truthfulqa)
    uv run python -u scripts/truthfulqa/truthfulqa_eval.py -m "${MODEL}" -l "${LAYER_IDX}" --seed "${SEED}" -d
    ;;
  gsm8k)
    uv run python -u scripts/gsm8k/gsm8k_eval.py -m "${MODEL}" -l "${LAYER_IDX}" --seed "${SEED}"
    ;;
  mmlu)
    uv run python -u scripts/mmlu/mmlu_eval.py -m "${MODEL}" -l "${LAYER_IDX}" --seed "${SEED}"
    ;;
  nq|triviaqa)
    uv run python -u "scripts/${TASK}/${TASK}_eval.py" -m "${MODEL}" -l "${LAYER_IDX}" --seed "${SEED}"
    ;;
esac
