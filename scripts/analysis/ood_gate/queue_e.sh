#!/usr/bin/env bash
# Q1/Q2 gate ablation, Llama3.1-8B-Base layer 13. GPU 4 only.
#
# GSM8K and MMLU run at num_examples=100 on a fixed seed-42 subsample, EVERY row including the
# unsteered reference, so the columns are internally matched. s.e. ~5 points at N=100: these are
# shape, not publishable numbers. The full-size re-runs are in experiments/ood_gate_ablation.sh.
# All N=100 outputs carry the `-n100` name suffix so they cannot block the full-size runs.
# TruthfulQA runs on all 817 questions.
set -uo pipefail
cd /data/arshan/cobras
export CUDA_VISIBLE_DEVICES=4
LOG=results/q1_logs/queue_e.log
N=100

run () {
  local task="$1" steer="$2" T="$3" seed="${4:-42}"
  shift 3; [ $# -gt 0 ] && shift
  echo "[$(date +%H:%M:%S)] START ${task} ${steer} T=${T} seed=${seed} $*" >> "$LOG"
  case "$task" in
    mmlu)
      uv run python -u scripts/analysis/ood_gate/mmlu_tf32.py \
        model=Llama3.1-8B-Base layer_idx=13 steer="${steer}" steer.T="${T}" seed="${seed}" \
        steer.suffix=-n${N} num_examples="${N}" "$@" >> "$LOG" 2>&1 ;;
    gsm8k)
      uv run python -u scripts/gsm8k/gsm8k_generate.py \
        model=Llama3.1-8B-Base layer_idx=13 steer="${steer}" steer.T="${T}" seed="${seed}" \
        steer.suffix=-n${N} num_examples="${N}" "$@" >> "$LOG" 2>&1 ;;
    truthfulqa)
      uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
        model=Llama3.1-8B-Base layer_idx=13 steer="${steer}" steer.T="${T}" seed="${seed}" \
        "$@" >> "$LOG" 2>&1 ;;
  esac
  echo "[$(date +%H:%M:%S)] DONE  ${task} ${steer} T=${T} seed=${seed} $*" >> "$LOG"
}

score () { uv run python -u scripts/analysis/ood_gate/eval.py -t "$1" -m Llama3.1-8B-Base -l 13 >> "$LOG" 2>&1; }
judge () { uv run python -u scripts/analysis/ood_gate/tqa_eval.py -m Llama3.1-8B-Base -l 13 -s "$1" >> "$LOG" 2>&1; }

# ---- stage 1: MMLU, N = 100, all ten rows
for spec in "OOD-NoSteer 1.0" "OOD-CAA 4" "OOD-ODESteer 4" \
            "OOD-SphericalSteer 4" "OOD-COBRAS-NoGate 0.65" \
            "OOD-COBRAS-Gate 0.65" "OOD-COBRAS-Gate 0.65"; do
  run mmlu $spec
done
score mmlu
echo "[$(date +%H:%M:%S)] ===== STAGE 1 MMLU DONE =====" >> "$LOG"

# ---- stage 2: GSM8K, N = 100, all ten rows
for spec in "OOD-NoSteer 1.0" "OOD-COBRAS-Gate 0.65" "OOD-COBRAS-NoGate 0.65" "OOD-COBRAS-Gate 0.65" \
            "OOD-SphericalSteer 4" "OOD-CAA 4" \
            "OOD-ODESteer 4"; do
  run gsm8k $spec
done
score gsm8k
echo "[$(date +%H:%M:%S)] ===== STAGE 2 GSM8K DONE =====" >> "$LOG"

# ---- stage 3: TruthfulQA, all 817, seed 42
for spec in "OOD-COBRAS-Gate 0.65" "OOD-COBRAS-NoGate 0.65" \
            "OOD-COBRAS-Gate 0.65"; do
  run truthfulqa $spec
done
judge 42
echo "[$(date +%H:%M:%S)] ===== STAGE 3 TQA SEED 42 JUDGED =====" >> "$LOG"

# ---- stage 4: GSM8K gate-coverage sweep, N = 100
run gsm8k OOD-COBRAS-Gate      0.65 42 steer.kwargs.abstain_percentile=0.9
run gsm8k OOD-COBRAS-Gate      0.65 42 steer.kwargs.abstain_percentile=0.95
run gsm8k OOD-COBRAS-Gate      0.65 42 steer.kwargs.abstain_percentile=0.995
run gsm8k OOD-COBRAS-Gate  0.65 42 steer.kwargs.abstain_percentile=0.4
run gsm8k OOD-COBRAS-Gate  0.65 42 steer.kwargs.abstain_percentile=0.7
run gsm8k OOD-COBRAS-Gate  0.65 42 steer.kwargs.abstain_percentile=0.8
run gsm8k OOD-COBRAS-Gate  0.65 42 steer.kwargs.abstain_percentile=0.9
run gsm8k OOD-COBRAS-GateCov90 0.65 42 steer.kwargs.abstain_percentile=0.534
run gsm8k OOD-COBRAS-GateCov90 0.65 42 steer.kwargs.abstain_percentile=0.8
score gsm8k
echo "[$(date +%H:%M:%S)] ===== STAGE 4 GSM8K COVERAGE SWEEP DONE =====" >> "$LOG"

# ---- stage 5: TruthfulQA seeds 43 and 44
for seed in 43 44; do
  for spec in "OOD-COBRAS-Gate 0.65" "OOD-COBRAS-NoGate 0.65" \
              "OOD-COBRAS-Gate 0.65"; do
    run truthfulqa $spec $seed
  done
  judge $seed
  echo "[$(date +%H:%M:%S)] ===== STAGE 5 TQA SEED ${seed} JUDGED =====" >> "$LOG"
done

# ---- stage 6: the ungated baselines on TruthfulQA at seeds 43 and 44, to match the grid
for seed in 43 44; do
  for spec in "OOD-CAA 4" "OOD-ODESteer 4" "OOD-SphericalSteer 4" "OOD-NoSteer 1.0"; do
    run truthfulqa $spec $seed
  done
  judge $seed
done
echo "[$(date +%H:%M:%S)] ===== QUEUE E COMPLETE =====" >> "$LOG"
