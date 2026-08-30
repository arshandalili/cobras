#!/usr/bin/env bash
#
# Repairs the MMLU panel of Fig. 2 by running COBRAS at the configuration Table 1 and the GSM8K
# panel use, for all four models.
#
# The cached MMLU results run COBRAS at T = 0.5, abstention percentile 0.995, sharpness 50, for
# every model, while the GSM8K results run it at T = 0.65, percentile 0.98, sharpness 200. The two
# OOD panels therefore do not plot the same method. This script produces the missing MMLU column
# at the GSM8K configuration, which is what confs/steer/OOD-COBRAS-Paper.yaml encodes.
#
#   bash experiments/ood_paper_config.sh
#   CUDA_VISIBLE_DEVICES=3 MODELS="Llama3.1-8B-Base:13" bash experiments/ood_paper_config.sh
#
# TF32 is on: full MMLU costs about 18 minutes per model instead of 92, at 99.8% prediction
# agreement, and leaves the COBRAS update unchanged (cosine 1.000000 against strict fp32).
# The unsteered reference for each model is already cached at full size in results/mmlu/*.csv.
#
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-Llama3.1-8B-Base:13 Mistral-7B-Base:15 Falcon-7B-Base:14 Qwen2.5-7B-Base:13}"
MMLU_N="${MMLU_N:-14042}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
LOG="${LOG:-results/q0_logs/q0_fullsize.log}"
mkdir -p "$(dirname "$LOG")"

say () { echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a "$LOG"; }

say "MODELS='$MODELS' MMLU_N=$MMLU_N SEED=$SEED DRY_RUN=$DRY_RUN"
say "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset, will use all>}"

status=0
for spec in $MODELS; do
  model="${spec%%:*}"; layer="${spec##*:}"
  name="q0-COBRAS-paper-p0.98-k32-sh200.0-T0.65"
  out="results/mmlu/raw_outputs/${model}/${model}-l${layer}-${name}-MMLU-seed${SEED}.jsonl"

  if [ -f "$out" ]; then
    n=$(wc -l < "$out")
    if [ "$n" -ge "$MMLU_N" ]; then say "skip $model, $out already has $n records"; continue; fi
    say "removing undersized $out ($n records)"
    [ "$DRY_RUN" = "0" ] && rm -f "$out"
  fi

  say "MMLU | $model layer $layer | $name"
  cmd=(uv run python -u scripts/analysis/ood_gate/mmlu_tf32.py
       model="$model" layer_idx="$layer" steer=OOD-COBRAS-Paper steer.T=0.65
       seed="$SEED" num_examples="$MMLU_N")
  if [ "$DRY_RUN" = "1" ]; then
    say "DRY_RUN  ${cmd[*]}"
  else
    "${cmd[@]}" >> "$LOG" 2>&1 || { say "FAILED $model"; status=1; }
  fi
done

if [ "$DRY_RUN" = "0" ]; then
  for spec in $MODELS; do
    model="${spec%%:*}"; layer="${spec##*:}"
    uv run python -u scripts/mmlu/mmlu_eval.py -m "$model" -l "$layer" --seed "$SEED" >> "$LOG" 2>&1 || true
  done
fi

say "q0_fullsize done (exit $status)"
exit "$status"
