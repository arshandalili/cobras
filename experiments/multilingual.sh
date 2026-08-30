#!/usr/bin/env bash
set -euo pipefail

# The Chinese (HalluQA / CMMLU) evaluation, end to end.
#
# This pipeline was previously driven by hand, one command at a time, and so was the only
# experiment in the repo with no runner. The order below is the one the reported numbers
# were produced in; each stage reads what the previous one wrote.
#
# Usage:
#   bash experiments/multilingual.sh <MODEL> [LAYER] [JUDGE]
#
# Examples:
#   CUDA_VISIBLE_DEVICES=6 bash experiments/multilingual.sh Qwen2.5-7B-Base 13
#   CUDA_VISIBLE_DEVICES=7 bash experiments/multilingual.sh Qwen/Qwen2.5-7B-Instruct 13
#
# Environment:
#   SEEDS        test seeds, default "42 43 44"
#   METHODS      steer methods, default "NoSteer CAA ITI COBRAS"
#   SWEEP_LAYERS layers to extract and sweep, default "10 .. 26"
#   STAGES       subset of: prep sweep val test judge table cmmlu robustness
#                default: all of them
#
# Stages that need a GPU: prep (activations), sweep, val, test, judge, cmmlu, robustness.

MODEL="${1:?MODEL, e.g. Qwen2.5-7B-Base or Qwen/Qwen2.5-7B-Instruct}"
LAYER="${2:-13}"
JUDGE="${3:-Qwen/Qwen2.5-14B-Instruct}"

SEEDS="${SEEDS:-42 43 44}"
METHODS="${METHODS:-NoSteer CAA ITI COBRAS}"
SWEEP_LAYERS="${SWEEP_LAYERS:-10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26}"
STAGES="${STAGES:-prep sweep val test judge table cmmlu robustness}"

# the generation scripts flatten a hub id into the output file name
NAME="${MODEL//\//-}"
RESULTS="results/q5_halluqa"
DELTA="${RESULTS}/q5_selected_delta_gen_${NAME}.json"

has () { [[ " ${STAGES} " == *" $1 "* ]]; }
say () { printf '\n===== %s =====\n' "$*"; }

# 1. dataset and activations ------------------------------------------------------------
if has prep; then
  say "prep: format HalluQA and extract D+/D- activations"
  uv run python data/q5_halluqa/format_dataset.py
  uv run python data/q5_halluqa/extract_activations_multi.py \
    -m "${MODEL}" -l ${SWEEP_LAYERS}
fi

# 2. choose the steering layer on validation --------------------------------------------
# Layer 13 is inherited from the English setup; Qwen2.5-7B has 28 layers, so it is
# re-selected rather than assumed. Reads the sweep, does not change LAYER automatically.
if has sweep; then
  say "sweep: pick the steering layer on validation"
  uv run python scripts/multilingual/layer_sweep.py \
    -m "${MODEL}" --methods ${METHODS} --seed 42
fi

# 3. validation generations, judged, then one common displacement budget per method -------
if has val; then
  say "val: generate and judge the validation sweep, then select the displacement"
  uv run python scripts/multilingual/halluqa_generate.py \
    -m "${MODEL}" -l "${LAYER}" --mode val --methods ${METHODS} --seed 42
  uv run python scripts/multilingual/halluqa_judge.py \
    --judge "${JUDGE}" --gen_file "${RESULTS}/q5_gen_val_${NAME}_l${LAYER}_seed42.jsonl"
  uv run python scripts/multilingual/select_delta.py \
    --judged "${RESULTS}/q5_gen_val_${NAME}_l${LAYER}_seed42_judged.jsonl" \
    --out "${DELTA}"
fi

# 4. held-out test generations at the selected budget, one file per seed ------------------
if has test; then
  say "test: generate held-out answers at the selected displacement"
  for s in ${SEEDS}; do
    uv run python scripts/multilingual/halluqa_generate.py \
      -m "${MODEL}" -l "${LAYER}" --mode test --methods ${METHODS} \
      --selected_delta "${DELTA}" --seed "${s}"
  done
fi

if has judge; then
  say "judge: score the test generations with the Chinese judge"
  for s in ${SEEDS}; do
    uv run python scripts/multilingual/halluqa_judge.py \
      --judge "${JUDGE}" \
      --gen_file "${RESULTS}/q5_gen_test_${NAME}_l${LAYER}_seed${s}.jsonl"
  done
fi

# 5. the reported table ------------------------------------------------------------------
if has table; then
  say "table: pooled two-fold non-hallucination rate, paired bootstrap vs NoSteer"
  for s in ${SEEDS}; do
    uv run python scripts/multilingual/summarize.py \
      --judged "${RESULTS}/q5_gen_test_${NAME}_l${LAYER}_seed${s}_judged.jsonl"
  done
fi

# 6. the Chinese analogue of the MMLU capability check ------------------------------------
if has cmmlu; then
  say "cmmlu: capability check under the same steer models"
  uv run python scripts/multilingual/cmmlu.py \
    -m "${MODEL}" -l "${LAYER}" --methods ${METHODS} --selected_delta "${DELTA}"
fi

# 7. robustness: does the ranking survive cutting trailing text, and how lenient is the judge?
if has robustness; then
  say "robustness: first-line-only re-judge, and the judge leniency probe"
  for s in ${SEEDS}; do
    uv run python scripts/multilingual/truncate_firstline.py \
      --gen_file "${RESULTS}/q5_gen_test_${NAME}_l${LAYER}_seed${s}.jsonl"
    uv run python scripts/multilingual/halluqa_judge.py \
      --judge "${JUDGE}" \
      --gen_file "${RESULTS}/q5_gen_test_${NAME}_l${LAYER}_seed${s}_firstline.jsonl"
  done
  uv run python scripts/multilingual/judge_probe.py --judge "${JUDGE}" || \
    echo "judge_probe.py needs its own arguments; see its docstring"
fi

say "done: ${MODEL} layer ${LAYER}"
