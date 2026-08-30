#!/usr/bin/env bash
# Q1/Q2, queue F: the TruthfulQA side of the Q2 gate-coverage sweep, so the trade-off curve is
# measured end to end on both axes rather than only on GSM8K. All 817 questions, seed 42.
# Runs after queue E (pid 3752068) exits.
set -uo pipefail
cd /data/arshan/cobras
export CUDA_VISIBLE_DEVICES=4
LOG=results/q1_logs/queue_f.log

while kill -0 3752068 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] queue E finished" >> "$LOG"

run () {
  local steer="$1" T="$2"; shift 2
  echo "[$(date +%H:%M:%S)] START truthfulqa ${steer} T=${T} $*" >> "$LOG"
  uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
    model=Llama3.1-8B-Base layer_idx=13 steer="${steer}" steer.T="${T}" seed=42 "$@" >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  truthfulqa ${steer} T=${T} $*" >> "$LOG"
}

run OOD-COBRAS-Gate       0.65 steer.kwargs.abstain_percentile=0.9
run OOD-COBRAS-Gate       0.65 steer.kwargs.abstain_percentile=0.95
run OOD-COBRAS-Gate       0.65 steer.kwargs.abstain_percentile=0.995
run OOD-COBRAS-Gate   0.65 steer.kwargs.abstain_percentile=0.4
run OOD-COBRAS-Gate   0.65 steer.kwargs.abstain_percentile=0.7
run OOD-COBRAS-Gate   0.65 steer.kwargs.abstain_percentile=0.8
run OOD-COBRAS-Gate   0.65 steer.kwargs.abstain_percentile=0.9
run OOD-COBRAS-GateCov90 0.65 steer.kwargs.abstain_percentile=0.534
run OOD-COBRAS-GateCov90 0.65 steer.kwargs.abstain_percentile=0.8
uv run python -u scripts/analysis/ood_gate/tqa_eval.py -m Llama3.1-8B-Base -l 13 -s 42 >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] ===== QUEUE F COMPLETE =====" >> "$LOG"
