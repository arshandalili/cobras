#!/usr/bin/env bash
#
# Runs every Reviewer-4oei full-size re-run: GSM8K at the full 1319-question test split and MMLU at
# the full 14042-question test split, for all four question sections.
#
# The interactive pass on the development machine ran GSM8K and MMLU at num_examples=100, which has
# a standard error of about 5 points and was only ever meant to fix the shape of each table. Every
# number those drafts mark as N=100 is superseded by whatever this script produces.
#
#   bash experiments/run_all_fullsize.sh                 # one GPU, sequential
#   GPUS="0 1 2 3" bash experiments/run_all_fullsize.sh  # one section per GPU, in parallel
#   SECTIONS="q1 q6" GPUS="0 1" bash experiments/run_all_fullsize.sh
#
# Each section script is self-contained: it purges its own stale outputs, generates, and evaluates.
# They write to disjoint filenames (the q1-, q4-, q5-, q6- steer-name prefixes), so running them
# concurrently on separate GPUs is safe.
#
set -uo pipefail
cd "$(dirname "$0")/.."

# q0 repairs the MMLU panel of Fig. 2, which was run at a different COBRAS configuration from the
# GSM8K panel. q5 stayed on TruthfulQA at full size, so it has no full-size script and is skipped.
#
# To verify afterwards that nothing was left at N=100:
#   for q in q0 q1 q4 q6; do DRY_RUN=1 ARCHIVE_N100=0 bash experiments/${q}_fullsize.sh > /tmp/dry_$q.log 2>&1; done
#   uv run python scripts/analysis/coverage_audit.py /tmp/dry_q1.log /tmp/dry_q4.log /tmp/dry_q6.log
#
# The Q1 TruthfulQA grid ran at seeds 42 43 44. The one comparison that did not reach significance
# there was COBRAS + gate against SphericalSteer + gate, +1.96 with sd 1.77. To settle it:
#   TQA_SEEDS="42 43 44 45 46" TASKS=truthfulqa ROWS=grid ARCHIVE_N100=0 bash experiments/ood_gate_ablation.sh

GPUS="${GPUS:-0}"
SECTIONS="${SECTIONS:-q0 q1 q4 q5 q6}"
LOGDIR="${LOGDIR:-results/fullsize_logs}"
mkdir -p "$LOGDIR"

present=()
for s in $SECTIONS; do
  if [ -f "experiments/${s}_fullsize.sh" ]; then
    present+=("$s")
  else
    echo "skipping ${s}: experiments/${s}_fullsize.sh not present"
  fi
done
[ ${#present[@]} -eq 0 ] && { echo "nothing to run"; exit 1; }

read -r -a gpu_arr <<< "$GPUS"
pids=()
names=()

i=0
for s in "${present[@]}"; do
  gpu="${gpu_arr[$(( i % ${#gpu_arr[@]} ))]}"
  log="${LOGDIR}/${s}_fullsize.log"
  echo "[$(date +%F' '%H:%M:%S)] ${s} -> GPU ${gpu}, log ${log}"
  if [ "${#gpu_arr[@]}" -eq 1 ]; then
    CUDA="$gpu" CUDA_VISIBLE_DEVICES="$gpu" bash "experiments/${s}_fullsize.sh" 2>&1 | tee "$log"
  else
    CUDA="$gpu" CUDA_VISIBLE_DEVICES="$gpu" nohup bash "experiments/${s}_fullsize.sh" > "$log" 2>&1 &
    pids+=("$!"); names+=("$s")
  fi
  i=$(( i + 1 ))
done

# wait on explicit PIDs, never on a process-name match
status=0
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then
    echo "[$(date +%F' '%H:%M:%S)] ${names[$k]} finished"
  else
    echo "[$(date +%F' '%H:%M:%S)] ${names[$k]} FAILED, see ${LOGDIR}/${names[$k]}_fullsize.log"
    status=1
  fi
done

echo "[$(date +%F' '%H:%M:%S)] full-size re-runs done (exit ${status})"
exit "$status"
