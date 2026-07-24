#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   uv run bash experiments_nq.sh <MODEL> [LAYER_IDX] [REPEAT] [T]
#
# Example:
#   uv run bash experiments_nq.sh Llama3.1-8B-Base 13 3 1.0

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
  uv run python -u scripts/nq/nq_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=NoSteer seed="${SEED}"

  echo "===== Seed: ${SEED} | RepE ====="
  uv run python -u scripts/nq/nq_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=RepE steer.T="${T}" seed="${SEED}"

  echo "===== Seed: ${SEED} | ITI ====="
  uv run python -u scripts/nq/nq_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=ITI steer.T="${T}" seed="${SEED}"

  echo "===== Seed: ${SEED} | CAA ====="
  uv run python -u scripts/nq/nq_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=CAA steer.T="${T}" seed="${SEED}"

  echo "===== Seed: ${SEED} | MiMiC ====="
  uv run python -u scripts/nq/nq_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=MiMiC steer.T="${T}" seed="${SEED}"

  echo "===== Seed: ${SEED} | LinAcT ====="
  uv run python -u scripts/nq/nq_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=LinAcT steer.T="${T}" seed="${SEED}"

  echo "===== Seed: ${SEED} | ODESteer ====="
  uv run python -u scripts/nq/nq_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=ODESteer steer.T="${T}" seed="${SEED}"

  echo "===== Seed: ${SEED} | SphericalSteer ====="
  uv run python -u scripts/nq/nq_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=SphericalSteer steer.T="${T}" seed="${SEED}"

  echo "===== Seed: ${SEED} | COBRAS ====="
  uv run python -u scripts/nq/nq_generate.py \
    model="${MODEL}" layer_idx="${LAYER_IDX}" steer=COBRAS steer.T="${T}" seed="${SEED}"

  echo "===== Evaluating the generated responses ====="
  uv run python -u scripts/nq/nq_eval.py \
    -m "${MODEL}" -l "${LAYER_IDX}" --seed "${SEED}"
done
