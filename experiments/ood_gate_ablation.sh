#!/usr/bin/env bash
# =================================================================================================
# Full-size re-run of the Q1/Q2 gate-ablation experiments for Reviewer 4oei, submission 32612.
#
#   bash experiments/ood_gate_ablation.sh                     # everything, from the repo root
#
# It reproduces every row of every table in 4oei_Q1Q2.md at full size:
#   GSM8K  num_examples=1319   (the whole test split, greedy, deterministic)
#   MMLU   num_examples=14042  (the whole test split, argmax scoring, deterministic)
#   TQA    all 817 questions, seeds 42 43 44 (generation samples at temperature 0.7)
# and runs the eval steps, so this machine produces finished CSVs and not just raw generations.
#
# -------------------------------------------------------------------------------------------------
# WHICH NUMBERS IN 4oei_Q1Q2.md THIS SUPERSEDES
# -------------------------------------------------------------------------------------------------
# Part 1 table, GSM8K and MMLU columns .......... every cell, currently N = 100, s.e. ~5 points
# Part 2 section 2c, GSM8K and MMLU columns ..... every cell, currently N = 100
# Part 2 section 5e, "GSM8K N = 100" column ..... every cell of the coverage sweep
# Part 2 section 5e, the mixture regression ..... refit once the above land
#
# NOT superseded and NOT corrected here. These are population statistics over 1000 prompt
# activations per task, not N-dependent, and the ANALYSIS stage below recomputes them for
# provenance only:
#   sections 3, 4, 5a-5d, 5f  (AUROC, Spearman, gate values, radius quantiles, Table 2 regression)
#   section 2a                (intervention magnitudes, from results/analysis/*-intervention.json)
# The Part 1 TruthfulQA column and section 2c's TruthfulQA columns are ALREADY full size at three
# seeds, so re-running them here is a reproduction and not a correction. To do only the columns
# that are currently provisional:  TASKS="gsm8k mmlu" bash experiments/ood_gate_ablation.sh
#
# -------------------------------------------------------------------------------------------------
# FILES THIS SCRIPT DEPENDS ON  (move all of them together)
# -------------------------------------------------------------------------------------------------
# steering configs, all authored for this workstream:
#   confs/steer/OOD-NoSteer.yaml            confs/steer/OOD-CAA.yaml
#   confs/steer/OOD-CAA-Gate.yaml           confs/steer/OOD-ODESteer.yaml
#   confs/steer/OOD-ODESteer-Gate.yaml      confs/steer/OOD-SphericalSteer.yaml
#   confs/steer/OOD-SphericalSteer-Gate.yaml
#   confs/steer/OOD-COBRAS-NoGate.yaml      confs/steer/OOD-COBRAS-Gate.yaml
#   confs/steer/OOD-COBRAS-GateMarginal.yaml    confs/steer/OOD-COBRAS-GateQuantile.yaml
#
# helpers, all authored for this workstream:
#   scripts/analysis/ood_gate/mmlu_tf32.py          TF32 wrapper around the unmodified MMLU scorer
#   scripts/analysis/ood_gate/eval.py               GSM8K / MMLU accuracy for q1-* rows only
#   scripts/analysis/ood_gate/tqa_eval.py           TruthfulQA True x Info for q1-* rows only
#   scripts/analysis/ood_gate/collect.py            assembles the tables
#   scripts/analysis/ood_gate/implied_marginal.py   sections 3a and 4
#   scripts/analysis/ood_gate/gate_calibration.py   sections 5a-5d
#   scripts/analysis/ood_gate/table2_regression.py  sections 5c-ter and 5f
#   scripts/analysis/ood_gate/sb_marginal_gate.py      section 3b   (pre-existing)
#   scripts/prepare/extract_query_activations.py  fills data/query_activations/ (pre-existing)
#
# unmodified repo code these call into:
#   scripts/gsm8k/gsm8k_generate.py  scripts/gsm8k/gsm8k_eval.py
#   scripts/mmlu/mmlu_generate.py    scripts/truthfulqa/truthfulqa_generate.py
#   src/cobras/steer/_cobras.py      src/cobras/steer/_gated_steer.py
#
# data that must exist before the first run:
#   the contrastive TruthfulQA activations every experiment in this repo fits on, and
#   data/query_activations/<MODEL>/{truthfulqa,truthfulqa_split0,truthfulqa_split1,gsm8k,mmlu,
#                                   nq,triviaqa}_layer<LAYER>.pt
#   The generate scripts load the latter unconditionally and the recalibrated gates use it as
#   their in-distribution reference. PREFLIGHT regenerates any that are missing.
#   The two TruthfulQA judges (allenai/truthfulqa-{truth,info}-judge-llama2-7B, ~13 GB each) are
#   downloaded on first use by scripts/analysis/ood_gate/tqa_eval.py.
#
# -------------------------------------------------------------------------------------------------
# ENVIRONMENT
# -------------------------------------------------------------------------------------------------
#   CUDA_VISIBLE_DEVICES  inherited from the environment, never set here. Export it yourself.
#   MODEL=Llama3.1-8B-Base   LAYER=13
#   TASKS="truthfulqa gsm8k mmlu"    subset to split the work across machines
#   GSM8K_N=1319   MMLU_N=14042   TQA_SEEDS="42 43 44"
#   ROWS="grid sweep"       "grid" is the Part 1 / 2c table, "sweep" is the Q2 coverage sweep
#   TF32=1                  MMLU only. 0 runs strict fp32, about five times slower.
#   PURGE=1                 move a full-size-named output aside before regenerating it, and drop
#                           its row from the eval csv so the new generation is actually scored
#   ARCHIVE_N100=1          move the interactive N = 100 raw outputs aside, once
#   DRY_RUN=0               1 prints every command and every file move and runs nothing
#   ANALYSIS=1              re-run the N-independent analysis scripts at the end
#
# NOTHING IS EVER DELETED. Superseded raw outputs are moved to results/q1_superseded/<timestamp>/
# and the N = 100 outputs to results/q1_n100_archive/<timestamp>/, so an interrupted or mistaken
# run is always recoverable by moving them back. Start with DRY_RUN=1 to see exactly what will
# move and what will run.
#
# Rough wall clock on one A100 80GB:
#   GSM8K ~60 min/row x 21 rows     MMLU ~18 min/row x 21 rows (TF32)
#   TQA   ~12 min/row x 21 rows x 3 seeds, plus ~15 min judging per seed
# Split with e.g.  CUDA_VISIBLE_DEVICES=0 TASKS=gsm8k bash experiments/ood_gate_ablation.sh &
#                  CUDA_VISIBLE_DEVICES=1 TASKS=mmlu  bash experiments/ood_gate_ablation.sh &
# =================================================================================================
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || { echo "cannot find repo root"; exit 1; }

MODEL="${MODEL:-Llama3.1-8B-Base}"
LAYER="${LAYER:-13}"
TASKS="${TASKS:-truthfulqa gsm8k mmlu}"
ROWS="${ROWS:-grid sweep}"
GSM8K_N="${GSM8K_N:-1319}"
MMLU_N="${MMLU_N:-14042}"
TQA_SEEDS="${TQA_SEEDS:-42 43 44}"
TF32="${TF32:-1}"
PURGE="${PURGE:-1}"
ARCHIVE_N100="${ARCHIVE_N100:-1}"
DRY_RUN="${DRY_RUN:-0}"
ANALYSIS="${ANALYSIS:-1}"
LOG="${LOG:-results/q1_logs/q1_fullsize.log}"
N100_SUFFIX="-n100"

mkdir -p "$(dirname "$LOG")"
say () { echo "[$(date +'%F %H:%M:%S')] $*" | tee -a "$LOG"; }

MMLU_SCRIPT="scripts/analysis/ood_gate/mmlu_tf32.py"
[ "$TF32" = "0" ] && MMLU_SCRIPT="scripts/mmlu/mmlu_generate.py"

# -------------------------------------------------------------------------------------------------
# the rows.  "<config> <T> [hydra overrides...]"
# -------------------------------------------------------------------------------------------------
GRID=(
  "OOD-NoSteer 1.0"                                  # unsteered reference, run like every other row
  "OOD-CAA 4"
  "OOD-CAA-Gate 4"
  "OOD-ODESteer 4"
  "OOD-ODESteer-Gate 4"
  "OOD-SphericalSteer 4"
  "OOD-SphericalSteer-Gate 4"
  "OOD-COBRAS-NoGate 0.65"
  "OOD-COBRAS-Gate 0.65"                             # the shipped Table 1 / Fig. 2 configuration
  "OOD-COBRAS-GateMarginal 0.65"                         # bridge marginal, matched in-distribution cost
)

SWEEP=(
  "OOD-COBRAS-Gate 0.65 steer.kwargs.abstain_percentile=0.9"
  "OOD-COBRAS-Gate 0.65 steer.kwargs.abstain_percentile=0.95"
  "OOD-COBRAS-Gate 0.65 steer.kwargs.abstain_percentile=0.995"
  "OOD-COBRAS-Gate 0.65 steer.kwargs.abstain_k=128"    # Table 2's K ablation, percentile stays 0.98
  "OOD-COBRAS-Gate 0.65 steer.kwargs.abstain_k=256"
  "OOD-COBRAS-GateMarginal 0.65 steer.kwargs.abstain_percentile=0.4"
  "OOD-COBRAS-GateMarginal 0.65 steer.kwargs.abstain_percentile=0.7"
  "OOD-COBRAS-GateMarginal 0.65 steer.kwargs.abstain_percentile=0.8"
  "OOD-COBRAS-GateMarginal 0.65 steer.kwargs.abstain_percentile=0.9"
  "OOD-COBRAS-GateQuantile 0.65 steer.kwargs.abstain_percentile=0.534"
  "OOD-COBRAS-GateQuantile 0.65 steer.kwargs.abstain_percentile=0.8"
)

ALL=()
for r in $ROWS; do
  case "$r" in
    grid)  ALL+=("${GRID[@]}")  ;;
    sweep) ALL+=("${SWEEP[@]}") ;;
    *) echo "unknown ROWS entry '$r' (expected 'grid' and/or 'sweep')"; exit 1 ;;
  esac
done

# -------------------------------------------------------------------------------------------------
# PREFLIGHT
# -------------------------------------------------------------------------------------------------
say "===== PREFLIGHT ====="
say "repo root $(pwd)"
say "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset, all visible GPUs>}"
say "MODEL=$MODEL LAYER=$LAYER TASKS='$TASKS' ROWS='$ROWS' GSM8K_N=$GSM8K_N MMLU_N=$MMLU_N"
say "TQA_SEEDS='$TQA_SEEDS' TF32=$TF32 PURGE=$PURGE ARCHIVE_N100=$ARCHIVE_N100 DRY_RUN=$DRY_RUN"

command -v uv >/dev/null 2>&1 || { say "FATAL: uv not on PATH"; exit 1; }

missing=0
for f in scripts/analysis/ood_gate/mmlu_tf32.py scripts/analysis/ood_gate/eval.py scripts/analysis/ood_gate/tqa_eval.py \
         scripts/analysis/ood_gate/collect.py scripts/prepare/extract_query_activations.py \
         scripts/gsm8k/gsm8k_generate.py scripts/mmlu/mmlu_generate.py \
         scripts/truthfulqa/truthfulqa_generate.py; do
  [ -f "$f" ] || { say "FATAL: missing $f"; missing=1; }
done
for spec in "${ALL[@]}"; do
  set -- $spec
  [ -f "confs/steer/$1.yaml" ] || { say "FATAL: missing confs/steer/$1.yaml"; missing=1; }
done
[ "$missing" = "1" ] && exit 1

ACT_DIR="data/query_activations/${MODEL}"
need_act=0
for t in truthfulqa truthfulqa_split0 truthfulqa_split1 gsm8k mmlu nq triviaqa; do
  [ -f "${ACT_DIR}/${t}_layer${LAYER}.pt" ] || need_act=1
done
if [ "$need_act" = "1" ]; then
  say "query activations incomplete under ${ACT_DIR}, extracting"
  if [ "$DRY_RUN" = "1" ]; then
    say "DRY_RUN would run: uv run python -u scripts/prepare/extract_query_activations.py -m $MODEL -l $LAYER"
  else
    uv run python -u scripts/prepare/extract_query_activations.py -m "$MODEL" -l "$LAYER" >> "$LOG" 2>&1 \
      || { say "FATAL: extract_query_activations failed, see $LOG"; exit 1; }
  fi
else
  say "query activations present under ${ACT_DIR}"
fi

# -------------------------------------------------------------------------------------------------
# resolve every steer name once, so the filenames below are exactly what hydra will produce
# -------------------------------------------------------------------------------------------------
declare -A NAME
resolve_all () {
  local out n=0
  # the program arrives on stdin via the heredoc, the row specs arrive as argv
  out="$(uv run python - "${ALL[@]}" <<'PY'
import sys
from omegaconf import OmegaConf
for line in sys.argv[1:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    cfg, T, overrides = parts[0], parts[1], parts[2:]
    root = OmegaConf.create({"steer": OmegaConf.load(f"confs/steer/{cfg}.yaml")})
    root.steer.T = T
    for ov in overrides:
        k, v = ov.split("=", 1)
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass
        OmegaConf.update(root, k, v)
    print(f"{line}\t{OmegaConf.select(root, 'steer.name')}")
PY
)" || return 1
  while IFS=$'\t' read -r spec name; do
    [ -z "${name:-}" ] && continue
    NAME["$spec"]="$name"
    n=$((n + 1))
  done <<< "$out"
  [ "$n" -eq "${#ALL[@]}" ] || { say "FATAL: resolved $n names for ${#ALL[@]} rows"; return 1; }
  return 0
}
resolve_all || exit 1
say "resolved ${#ALL[@]} steer names:"
for spec in "${ALL[@]}"; do say "  ${NAME[$spec]}"; done

raw_path () {  # raw_path <task> <steer name> <seed>
  local tag
  case "$1" in gsm8k) tag=GSM8K ;; mmlu) tag=MMLU ;; *) tag=TruthfulQA ;; esac
  echo "results/$1/raw_outputs/${MODEL}/${MODEL}-l${LAYER}-$2-${tag}-seed$3.jsonl"
}

csv_path () {  # csv_path <task> <seed>
  local tag
  case "$1" in gsm8k) tag=GSM8K ;; mmlu) tag=MMLU ;; *) tag=TruthfulQA ;; esac
  echo "results/$1/eval_results/stat_results/q1-${MODEL}-l${LAYER}-${tag}-seed$2.csv"
}

# The eval scripts skip a steer method that is already in their csv. If we regenerate a row we
# must drop its old row too, or the new generation is never scored.
drop_csv_row () {  # drop_csv_row <task> <steer name> <seed>
  local csv; csv="$(csv_path "$1" "$3")"
  [ -f "$csv" ] || return 0
  if [ "$DRY_RUN" = "1" ]; then
    uv run python - "$csv" "$2" <<'PY'
import sys, pandas as pd
csv, name = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv)
if (d["Steering Method"] == name).any():
    print(f"DRY_RUN would drop stale eval row '{name}' from {csv}")
PY
    return 0
  fi
  uv run python - "$csv" "$2" <<'PY'
import sys, pandas as pd
csv, name = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv)
keep = d[d["Steering Method"] != name]
if len(keep) != len(d):
    keep.to_csv(csv, index=False)
    print(f"dropped stale eval row '{name}' from {csv}")
PY
}

# -------------------------------------------------------------------------------------------------
# ARCHIVE the interactive N = 100 outputs, once, so nothing at N = 100 can be mistaken for a
# full-size result. They carry a `-n100` suffix and therefore cannot collide with the files below,
# but they are moved anyway so the results tree holds one N per task.
# -------------------------------------------------------------------------------------------------
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
STALE_DIR="results/q1_superseded/${RUN_STAMP}"

if [ "$ARCHIVE_N100" = "1" ]; then
  say "===== ARCHIVE N=100 OUTPUTS ====="
  ARCH="results/q1_n100_archive/${RUN_STAMP}"
  moved=0
  for task in gsm8k mmlu truthfulqa; do
    d="results/${task}/raw_outputs/${MODEL}"
    [ -d "$d" ] || continue
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      if [ "$DRY_RUN" = "1" ]; then
        say "DRY_RUN would archive $f -> ${ARCH}/${task}/"
      else
        mkdir -p "${ARCH}/${task}"
        mv "$f" "${ARCH}/${task}/"
      fi
      moved=$((moved + 1))
    done < <(find "$d" -maxdepth 1 -name "*${N100_SUFFIX}-*.jsonl" 2>/dev/null)
  done
  say "archived ${moved} N=100 raw output file(s)"
  say "their accuracies remain in the q1-*.csv eval files under the -n100 steer name, and"
  say "scripts/analysis/ood_gate/collect.py prefers the full-size row whenever one exists."
fi

# -------------------------------------------------------------------------------------------------
# run
# -------------------------------------------------------------------------------------------------
FAILED=()

run_one () {  # run_one <task> <seed> <spec>
  local task="$1" seed="$2" spec="$3"
  local name="${NAME[$spec]}"
  set -- $spec
  local cfg="$1" T="$2"; shift 2
  local out; out="$(raw_path "$task" "$name" "$seed")"

  # The generate scripts skip when the output file exists, and the filename does not encode
  # num_examples, so a stale file of the wrong size would be silently reused. Move it aside
  # (never rm, so an earlier run is always recoverable) and drop its eval row.
  if [ -f "$out" ]; then
    if [ "$PURGE" = "1" ]; then
      if [ "$DRY_RUN" = "1" ]; then
        say "DRY_RUN would move stale $out -> ${STALE_DIR}/${task}/"
      else
        mkdir -p "${STALE_DIR}/${task}"
        mv "$out" "${STALE_DIR}/${task}/"
        say "moved stale $out -> ${STALE_DIR}/${task}/"
      fi
      drop_csv_row "$task" "$name" "$seed" | tee -a "$LOG"
    else
      say "PURGE=0 and $out exists, the generate script will skip this row"
      return 0
    fi
  fi

  local cmd
  case "$task" in
    gsm8k) cmd=(uv run python -u scripts/gsm8k/gsm8k_generate.py model="$MODEL" layer_idx="$LAYER"
                steer="$cfg" steer.T="$T" seed="$seed" num_examples="$GSM8K_N" "$@") ;;
    mmlu)  cmd=(uv run python -u "$MMLU_SCRIPT" model="$MODEL" layer_idx="$LAYER"
                steer="$cfg" steer.T="$T" seed="$seed" num_examples="$MMLU_N" "$@") ;;
    truthfulqa) cmd=(uv run python -u scripts/truthfulqa/truthfulqa_generate.py model="$MODEL"
                layer_idx="$LAYER" steer="$cfg" steer.T="$T" seed="$seed" "$@") ;;
  esac

  say "RUN ${task} | ${name} | seed ${seed}"
  if [ "$DRY_RUN" = "1" ]; then
    say "DRY_RUN  ${cmd[*]}"
    return 0
  fi
  "${cmd[@]}" >> "$LOG" 2>&1
  if [ ! -f "$out" ]; then
    say "FAILED ${task} | ${name} | seed ${seed}  (no output file, see $LOG)"
    FAILED+=("${task}|${name}|seed${seed}")
  fi
}

score () {  # score <task> [seed]
  local task="$1" seed="${2:-42}"
  local cmd
  case "$task" in
    truthfulqa) cmd=(uv run python -u scripts/analysis/ood_gate/tqa_eval.py -m "$MODEL" -l "$LAYER" -s "$seed") ;;
    *)          cmd=(uv run python -u scripts/analysis/ood_gate/eval.py -t "$task" -m "$MODEL" -l "$LAYER") ;;
  esac
  say "EVAL ${task} seed ${seed}"
  if [ "$DRY_RUN" = "1" ]; then say "DRY_RUN  ${cmd[*]}"; return 0; fi
  "${cmd[@]}" 2>&1 | tee -a "$LOG"
}

for task in $TASKS; do
  case "$task" in
    gsm8k)
      say "===== gsm8k, ${#ALL[@]} rows at ${GSM8K_N} examples ====="
      for spec in "${ALL[@]}"; do run_one gsm8k 42 "$spec"; done
      score gsm8k
      ;;
    mmlu)
      say "===== mmlu, ${#ALL[@]} rows at ${MMLU_N} examples, TF32=${TF32} ====="
      for spec in "${ALL[@]}"; do run_one mmlu 42 "$spec"; done
      score mmlu
      ;;
    truthfulqa)
      for seed in $TQA_SEEDS; do
        say "===== truthfulqa, ${#ALL[@]} rows, all 817 questions, seed ${seed} ====="
        for spec in "${ALL[@]}"; do run_one truthfulqa "$seed" "$spec"; done
        score truthfulqa "$seed"
      done
      ;;
    *) say "unknown task '$task', skipping" ;;
  esac
done

# -------------------------------------------------------------------------------------------------
# N-independent analysis, re-run so the new machine reproduces the whole deliverable
# -------------------------------------------------------------------------------------------------
if [ "$ANALYSIS" = "1" ]; then
  say "===== ANALYSIS (population statistics, not N-dependent) ====="
  for a in scripts/analysis/ood_gate/implied_marginal.py \
           scripts/analysis/ood_gate/gate_calibration.py \
           scripts/analysis/ood_gate/table2_regression.py \
           scripts/analysis/ood_gate/sb_marginal_gate.py; do
    if [ ! -f "$a" ]; then say "skipping missing $a"; continue; fi
    say "ANALYSIS $a"
    if [ "$DRY_RUN" = "1" ]; then
      say "DRY_RUN  uv run python -u $a -m $MODEL -l $LAYER"
    else
      uv run python -u "$a" -m "$MODEL" -l "$LAYER" 2>&1 | tee -a "$LOG"
    fi
  done
fi

say "===== COLLECT ====="
if [ "$DRY_RUN" = "1" ]; then
  say "DRY_RUN  uv run python -u scripts/analysis/ood_gate/collect.py -m $MODEL -l $LAYER --seeds $TQA_SEEDS"
else
  uv run python -u scripts/analysis/ood_gate/collect.py -m "$MODEL" -l "$LAYER" --seeds $TQA_SEEDS 2>&1 | tee -a "$LOG"
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  say "===== ${#FAILED[@]} ROW(S) FAILED ====="
  for f in "${FAILED[@]}"; do say "  $f"; done
  say "re-running this script retries them: PURGE deletes and regenerates whatever is present"
  exit 1
fi
say "q1_fullsize complete, log at $LOG"
