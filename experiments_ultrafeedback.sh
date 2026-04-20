#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   uv run bash experiments_ultrafeedback.sh <MODEL> [LAYER_IDX] [REPEAT]
#
# Example:
#   uv run bash experiments_ultrafeedback.sh Llama3.1-8B-Base 13 3

MODEL="${1:?Please provide MODEL as the first argument, e.g. Llama3.1-8B-Base}"
LAYER_IDX="${2:-13}"
REPEAT="${3:-3}"

echo "===== Config ====="
echo "MODEL=${MODEL}"
echo "LAYER_IDX=${LAYER_IDX}"
echo "REPEAT=${REPEAT}"

echo "===== Preparing Data ====="
uv run bash data/ultrafeedback.sh

for ((i=0; i<REPEAT; i++)); do
  SEED=$((42 + i))

  echo "===== Seed: ${SEED} | NoSteer ====="
  uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=NoSteer seed="${SEED}"

  echo "===== Seed: ${SEED} | RepE ====="
  uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=RepE steer.T=1.0 seed="${SEED}"

  echo "===== Seed: ${SEED} | ITI ====="
  uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=ITI steer.T=1.0 seed="${SEED}"

  echo "===== Seed: ${SEED} | CAA ====="
  uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=CAA steer.T=1.0 seed="${SEED}"

  echo "===== Seed: ${SEED} | MiMiC ====="
  uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=MiMiC steer.T=1.0 seed="${SEED}"

  echo "===== Seed: ${SEED} | LinAcT ====="
  uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=LinAcT steer.T=1.0 seed="${SEED}"

  echo "===== Seed: ${SEED} | ODESteer ====="
  uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=ODESteer steer.T=5.0 seed="${SEED}"

  echo "===== Seed: ${SEED} | SphericalSteer ====="
  uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=SphericalSteer steer.T=1.0 seed="${SEED}"

  echo "===== Evaluating the generated responses ====="
  uv run python -u scripts/ultrafeedback/ultrafeedback_eval.py \
    -m "${MODEL}" -l "${LAYER_IDX}" --seed "${SEED}" -d
done

