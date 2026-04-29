#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   uv run bash experiments_gsm8k.sh <MODEL> [LAYER_IDX] [REPEAT] [T]
#
# Example:
#   uv run bash experiments_gsm8k.sh Llama3.1-8B-Base 13 3 1.0

MODEL="${1:?Please provide MODEL as the first argument, e.g. Llama3.1-8B-Base}"
LAYER_IDX="${2:-13}"
REPEAT="${3:-1}"
T="${4:-1.0}"

echo "===== Config ====="
echo "MODEL=${MODEL}"
echo "LAYER_IDX=${LAYER_IDX}"
echo "REPEAT=${REPEAT}"
echo "T=${T}"


for ((i=0; i<REPEAT; i++)); do
  SEED=$((42 + i))

  echo "===== Seed: ${SEED} | NoSteer ====="
  uv run accelerate launch --num_processes 3 scripts/gsm8k/gsm8k_generate_fast.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=NoSteer seed="${SEED}"
  uv run python -u scripts/gsm8k/gsm8k_generate_merge.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=NoSteer seed="${SEED}"

  echo "===== Seed: ${SEED} | RepE ====="
  uv run accelerate launch --num_processes 3 scripts/gsm8k/gsm8k_generate_fast.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=RepE steer.T="${T}" seed="${SEED}"
  uv run python -u scripts/gsm8k/gsm8k_generate_merge.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=RepE steer.T="${T}" seed="${SEED}"

  echo "===== Seed: ${SEED} | ITI ====="
  uv run accelerate launch --num_processes 3 scripts/gsm8k/gsm8k_generate_fast.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=ITI steer.T="${T}" seed="${SEED}"
  uv run python -u scripts/gsm8k/gsm8k_generate_merge.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=ITI steer.T="${T}" seed="${SEED}"

  echo "===== Seed: ${SEED} | CAA ====="
  uv run accelerate launch --num_processes 3 scripts/gsm8k/gsm8k_generate_fast.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=CAA steer.T="${T}" seed="${SEED}"
  uv run python -u scripts/gsm8k/gsm8k_generate_merge.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=CAA steer.T="${T}" seed="${SEED}"

  echo "===== Seed: ${SEED} | MiMiC ====="
  uv run accelerate launch --num_processes 3 scripts/gsm8k/gsm8k_generate_fast.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=MiMiC steer.T="${T}" seed="${SEED}"
  uv run python -u scripts/gsm8k/gsm8k_generate_merge.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=MiMiC steer.T="${T}" seed="${SEED}"


  echo "===== Seed: ${SEED} | LinAcT ====="
  uv run accelerate launch --num_processes 3 scripts/gsm8k/gsm8k_generate_fast.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=LinAcT steer.T="${T}" seed="${SEED}"
  uv run python -u scripts/gsm8k/gsm8k_generate_merge.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=LinAcT steer.T="${T}" seed="${SEED}"

  echo "===== Seed: ${SEED} | ODESteer ====="
  uv run accelerate launch --num_processes 3 scripts/gsm8k/gsm8k_generate_fast.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=ODESteer steer.T="${T}" seed="${SEED}"
  uv run python -u scripts/gsm8k/gsm8k_generate_merge.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=ODESteer steer.T="${T}" seed="${SEED}"

  echo "===== Seed: ${SEED} | SphericalSteer ====="
  uv run accelerate launch --num_processes 3 scripts/gsm8k/gsm8k_generate_fast.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=SphericalSteer steer.T="${T}" seed="${SEED}"
  uv run python -u scripts/gsm8k/gsm8k_generate_merge.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=SphericalSteer steer.T="${T}" seed="${SEED}"

  echo "===== Evaluating the generated responses ====="
  uv run python -u scripts/gsm8k/gsm8k_eval.py \
    -m "${MODEL}" -l "${LAYER_IDX}" --seed "${SEED}"
done