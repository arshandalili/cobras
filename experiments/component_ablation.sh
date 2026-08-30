#!/usr/bin/env bash
# =============================================================================
# Q4 (Reviewer 4oei) leave-one-out component ablation: full-split OOD re-run.
#
# WHAT THIS IS FOR
#   The GSM8K and MMLU columns of the ablation table in 4oei_Q4.md were produced with
#   num_examples=100 so that all ten rows could be covered on one GPU in an hour. At N = 100 the
#   standard error is about 5 points per cell, which resolves the 50-point gaps but not the
#   1-point ones. This script repeats every row on the full test splits, GSM8K N = 1319 and MMLU
#   N = 14042, and writes finished CSVs. The TruthfulQA column is already at full size (all 817
#   questions, seeds 42/43/44) and is NOT re-run here.
#
#   Numbers in 4oei_Q4.md that this script supersedes: the whole GSM8K column and the whole MMLU
#   column of the Part 1 table, and the same two columns in the Part 2 per-row table. Nothing in
#   the TruthfulQA column, and none of the mechanism measurements in Part 2 Sec. 3, changes.
#
# USAGE (from the repository root, no arguments needed)
#   CUDA_VISIBLE_DEVICES=<gpu> bash experiments/component_ablation.sh
#
#   CUDA_VISIBLE_DEVICES is taken from the environment and never set here. If it is unset the
#   script warns and uses whatever the driver exposes.
#
# ENVIRONMENT VARIABLES
#   MODEL        default Llama3.1-8B-Base
#   LAYER_IDX    default 13
#   SEED         default 42     GSM8K is greedy and MMLU is argmax scoring, so both are
#                               deterministic and one seed is enough.
#   TASKS        default "gsm8k mmlu"     restrict to one task to split the work over machines
#   MMLU_TF32    default 1      TF32 matmuls for MMLU only. Measured at 99.8% prediction
#                               agreement with strict fp32, with the COBRAS update itself
#                               unchanged (cosine 1.000000), and a 5x speedup: about 18 min per
#                               row instead of 92. Set 0 for strict fp32. Whichever is chosen
#                               applies to every row including the unsteered reference.
#   KEEP_SMALL   default 0      1 moves the N = 100 raw outputs aside instead of deleting them.
#   DRY_RUN      default 0      1 prints what would be deleted and what would be run, then exits
#                               without touching anything. Use this first after a move.
#
# EXPECTED COST, from timings measured on the original machine (1x H100, fp32)
#   GSM8K about 31 min per row, 10 rows, about 5.2 h
#   MMLU  about 18 min per row with TF32 (92 min without), 10 rows, about 3.0 h (or 15.3 h)
#
# FILES THIS SCRIPT DEPENDS ON. All of these must come along in a move.
#   configs (steering variants, one per table row)
#     confs/steer/Ablation-NoSteer.yaml         unsteered reference        name: q4-nosteer
#     confs/steer/Ablation-Full.yaml            COBRAS as shipped          name: q4-full
#     confs/steer/Ablation-NoSphere.yaml        - spherical projection     name: q4-nosphere
#     confs/steer/Ablation-NoSinkhorn.yaml      - Sinkhorn solve           name: q4-nosinkhorn
#     confs/steer/Ablation-OneStep.yaml         - multi-step, K=1          name: q4-onestep
#     confs/steer/Ablation-RawStep.yaml         - direction normalization  name: q4-rawstep
#     confs/steer/Ablation-NoVMF.yaml           - vMF strength gate        name: q4-novmf
#     confs/steer/Ablation-NoVMF-Tmatch.yaml    - vMF gate, step matched   name: q4-novmf-tmatch
#     confs/steer/Ablation-NoAbstain.yaml       - kNN abstention gate      name: q4-noabstain
#     confs/steer/Ablation-UniformW.yaml        - Eq. 18 query weights     name: q4-uniformw
#   task configs read by the generation scripts
#     confs/gsm8k.yaml, confs/mmlu.yaml
#   analysis helpers written for this question
#     scripts/analysis/ablation/eval.py         evaluates only q4-* outputs, into results/analysis/q4x
#     scripts/analysis/ablation/run_tf32.py     runs a generation script with TF32 matmuls on
#     scripts/analysis/ablation/table.py        assembles the table (not called here)
#     scripts/analysis/ablation/field_diag.py   mechanism diagnostics (not called here)
#     scripts/analysis/ablation/sinkhorn_check.py, scripts/analysis/ablation/weight_entropy.py (not called)
#   the Euclidean control, which the Ablation-NoSphere row resolves through hydra
#     src/cobras/steer/_euclidean_cobras.py      defines EuclideanCOBRAS
#   generation and steering code, unmodified
#     scripts/gsm8k/gsm8k_generate.py, scripts/mmlu/mmlu_generate.py, src/cobras/**
#   fitted activations, which must be present or regenerated on the new machine
#     data/truthfulqa/activations/<MODEL>/{pos,neg}_{0,1}_activations_layer<LAYER_IDX>.pt
#     data/query_activations/<MODEL>/truthfulqa_layer<LAYER_IDX>.pt  (only read when
#         abstain_on_queries is true, which none of the Q4 configs sets)
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

MODEL="${MODEL:-Llama3.1-8B-Base}"
LAYER_IDX="${LAYER_IDX:-13}"
SEED="${SEED:-42}"
TASKS="${TASKS:-gsm8k mmlu}"
MMLU_TF32="${MMLU_TF32:-1}"
KEEP_SMALL="${KEEP_SMALL:-0}"
DRY_RUN="${DRY_RUN:-0}"

GSM8K_N=1319     # full GSM8K test split
MMLU_N=14042     # full MMLU test split

# Row order of the ablation table. Parallel arrays: CFGS[i] is the hydra config and NAMES[i] is
# its `name:` field, which is the string the generation scripts put in the output filename.
CFGS=(Ablation-NoSteer Ablation-Full Ablation-NoSphere Ablation-NoSinkhorn Ablation-OneStep Ablation-RawStep Ablation-NoVMF
      Ablation-NoVMF-Tmatch Ablation-NoAbstain Ablation-UniformW)
NAMES=(q4-nosteer q4-full q4-nosphere q4-nosinkhorn q4-onestep q4-rawstep q4-novmf
       q4-novmf-tmatch q4-noabstain q4-uniformw)

say () { echo "[q4-fullsize] $*"; }

# --- preflight ---------------------------------------------------------------
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  say "WARNING: CUDA_VISIBLE_DEVICES is not set, the run will use whatever the driver exposes."
fi
say "repo root       $(pwd)"
say "model           ${MODEL}  layer ${LAYER_IDX}  seed ${SEED}"
say "tasks           ${TASKS}"
say "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  MMLU_TF32=${MMLU_TF32}  DRY_RUN=${DRY_RUN}"

missing=0
for cfg in "${CFGS[@]}"; do
  [[ -f "confs/steer/${cfg}.yaml" ]] || { say "MISSING config confs/steer/${cfg}.yaml"; missing=1; }
done
for f in scripts/analysis/ablation/eval.py scripts/analysis/ablation/run_tf32.py \
         src/cobras/steer/_euclidean_cobras.py \
         scripts/gsm8k/gsm8k_generate.py scripts/mmlu/mmlu_generate.py \
         confs/gsm8k.yaml confs/mmlu.yaml; do
  [[ -f "${f}" ]] || { say "MISSING dependency ${f}"; missing=1; }
done
act="data/truthfulqa/activations/${MODEL}"
for s in 0 1; do
  for p in pos neg; do
    [[ -f "${act}/${p}_${s}_activations_layer${LAYER_IDX}.pt" ]] || {
      say "MISSING activations ${act}/${p}_${s}_activations_layer${LAYER_IDX}.pt"; missing=1; }
  done
done
if [[ "${missing}" != "0" ]]; then
  say "preflight failed, nothing was run"
  exit 1
fi
say "preflight ok, all ${#CFGS[@]} configs and all dependencies present"

# --- clear the N=100 outputs so the generation scripts do not skip -----------
# The generation scripts skip when the output file exists, and the filename encodes the model,
# layer, steer name, task and seed but NOT num_examples. A stale N = 100 file would therefore be
# silently accepted as the full-split result.
clear_task () {
  local task_dir="$1" tag="$2"
  local csv="results/analysis/q4x/${task_dir}-${MODEL}-l${LAYER_IDX}-seed${SEED}.csv"
  if [[ -f "${csv}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      say "DRY RUN would copy ${csv} to ${csv%.csv}-n100.csv, then remove the original"
    else
      cp "${csv}" "${csv%.csv}-n100.csv" && rm -f "${csv}"
      say "kept the small-N table as ${csv%.csv}-n100.csv"
    fi
  fi
  local f n
  for name in "${NAMES[@]}"; do
    f="results/${task_dir}/raw_outputs/${MODEL}/${MODEL}-l${LAYER_IDX}-${name}-${tag}-seed${SEED}.jsonl"
    [[ -f "${f}" ]] || continue
    n="$(wc -l < "${f}")"
    if [[ "${DRY_RUN}" == "1" ]]; then
      if [[ "${KEEP_SMALL}" == "1" ]]; then
        say "DRY RUN would move aside ${f} (${n} records)"
      else
        say "DRY RUN would delete ${f} (${n} records)"
      fi
    elif [[ "${KEEP_SMALL}" == "1" ]]; then
      mkdir -p "results/analysis/q4x/raw_smallN/${task_dir}"
      mv "${f}" "results/analysis/q4x/raw_smallN/${task_dir}/" && say "moved aside ${f} (${n} records)"
    else
      rm -f "${f}" && say "deleted ${f} (${n} records)"
    fi
  done
}

# --- run ---------------------------------------------------------------------
run_task () {
  local task="$1" n="$2" cfg
  say "===== ${task}, full test split N=${n} ====="
  for cfg in "${CFGS[@]}"; do
    say "----- $(date +%H:%M:%S) ${task} ${cfg} -----"
    local cmd
    if [[ "${task}" == "mmlu" && "${MMLU_TF32}" == "1" ]]; then
      cmd=(uv run python -u scripts/analysis/ablation/run_tf32.py scripts/mmlu/mmlu_generate.py)
    else
      cmd=(uv run python -u "scripts/${task}/${task}_generate.py")
    fi
    cmd+=(model="${MODEL}" layer_idx="${LAYER_IDX}" steer="${cfg}" seed="${SEED}" num_examples="${n}")
    if [[ "${DRY_RUN}" == "1" ]]; then
      say "DRY RUN would run: ${cmd[*]}"
    else
      "${cmd[@]}"
    fi
  done
  if [[ "${DRY_RUN}" == "1" ]]; then
    say "DRY RUN would run: uv run python -u scripts/analysis/ablation/eval.py --task ${task} -m ${MODEL} -l ${LAYER_IDX} -s ${SEED}"
  else
    uv run python -u scripts/analysis/ablation/eval.py \
      --task "${task}" -m "${MODEL}" -l "${LAYER_IDX}" -s "${SEED}"
  fi
}

for task in ${TASKS}; do
  case "${task}" in
    gsm8k) clear_task gsm8k GSM8K; run_task gsm8k "${GSM8K_N}" ;;
    mmlu)  clear_task mmlu  MMLU;  run_task mmlu  "${MMLU_N}"  ;;
    *) say "unknown task ${task}"; exit 1 ;;
  esac
done

say "done. Finished CSVs are in results/analysis/q4x/:"
say "  gsm8k-${MODEL}-l${LAYER_IDX}-seed${SEED}.csv"
say "  mmlu-${MODEL}-l${LAYER_IDX}-seed${SEED}.csv"
say "Rebuild the table with: uv run python scripts/analysis/ablation/table.py"
