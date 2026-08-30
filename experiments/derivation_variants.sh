#!/bin/bash
# =============================================================================================
# Reviewer 4oei, weakness 6: "strictly derived" versus the implemented update.
# Full-size rerun of everything behind /data/arshan/cobras/4oei_Q6.md.
#
#   bash experiments/derivation_variants.sh                  # uses whatever CUDA_VISIBLE_DEVICES is set
#   CUDA_VISIBLE_DEVICES=3 bash experiments/derivation_variants.sh
#   DRY_RUN=1 bash experiments/derivation_variants.sh        # print every command, move nothing, run nothing
#
# Run it from anywhere: it cd's to the repository root itself. It takes no arguments. Every step
# is idempotent, because the generation scripts skip when their output file already exists and the
# two scorers skip rows already present in their CSVs, so it can be interrupted and restarted.
#
# ---------------------------------------------------------------------------------------------
# WHAT IS AT N=100 TODAY AND WILL BE SUPERSEDED BY THIS SCRIPT
# ---------------------------------------------------------------------------------------------
# In 4oei_Q6.md, every GSM8K and MMLU number is currently at num_examples=100, seed 42. That is
# the whole of the GSM8K and MMLU columns of the ladder table in Part 1 and of tables 4 and 6 in
# Part 2, including the unsteered reference (GSM8K 57.0, MMLU 66.0). At N=100 the standard error
# is about 5 points, so only the large effects there carry weight: the ungated rungs scoring 0.0
# on GSM8K against an unsteered 57.0, and the gated rungs recovering the unsteered MMLU exactly.
# This script replaces those two columns with GSM8K N=1319 and MMLU N=14042.
#
# Nothing else in 4oei_Q6.md is at a reduced size. TruthfulQA is always the full 817 questions,
# the autograd tables are over 128 queries and do not depend on any run size, and the step
# magnitude and endpoint tables use 817 in-distribution and 1000 out-of-distribution cached query
# activations.
#
# For reference, the fp32 numbers already in this tree are GSM8K NoSteer 49.28 and shipped COBRAS
# 47.54 at N=1319, and MMLU NoSteer 61.58 at N=14042. Those predate this script and were produced
# without TF32; see the TF32 note below before comparing against them.
#
# ---------------------------------------------------------------------------------------------
# FILES THIS SCRIPT DEPENDS ON (move all of these with it)
# ---------------------------------------------------------------------------------------------
# Steering configurations, all under confs/steer/, all with a `name:` beginning `q6-`:
#   confs/steer/Deriv-NoSteer.yaml        unsteered reference
#   confs/steer/Deriv-Strict.yaml         rung 1: plain potentials, fixed bandwidth, raw step,
#                                      drift=gradient, no vMF gate, no abstention gate
#   confs/steer/Deriv-Unit.yaml           rung 2: rung 1 plus the unit-normalized step
#   confs/steer/Deriv-UnitVMF.yaml        rung 3: rung 2 plus the vMF strength gate
#   confs/steer/Deriv-Exact.yaml          rung 4: rung 3 plus the abstention gate
#   confs/steer/Deriv-Shipped.yaml        rung 5: the paper's Table 1 / Fig. 2 configuration
#   confs/steer/Deriv-ShippedNoGate.yaml  rung 5 with the abstention gate off
#   confs/steer/Deriv-ExtFixed.yaml       extended potentials with a constant bandwidth, ungated
# Analysis helpers, all under scripts/analysis/:
#   scripts/analysis/derivation/run_tf32.py       enables TF32 and then runs a generation script
#   scripts/analysis/derivation/score.py          scores this ladder's own GSM8K / MMLU raw outputs
#   scripts/analysis/derivation/tqa_eval.py       TruthfulQA True x Info for the q6-* files only
#   scripts/analysis/derivation/tables.py         assembles the ladder and T-sweep tables
#   scripts/analysis/derivation/exact_gradient.py the autograd comparison (imports exact_gradient.py)
#   scripts/analysis/derivation/step_magnitude.py per-rung intervention magnitude, ID versus OOD
#   scripts/analysis/derivation/endpoint_gap.py   distance between the exact and the shipped endpoints
#   scripts/analysis/derivation/freezing_error.py    pre-existing, imported by derivation/exact_gradient.py, unedited
#   scripts/gsm8k/gsm8k_eval.py           pre-existing, imported by derivation/score.py for its answer
#                                         extraction, unedited
# Read-only generation scripts, used exactly as they are:
#   scripts/truthfulqa/truthfulqa_generate.py, scripts/gsm8k/gsm8k_generate.py,
#   scripts/mmlu/mmlu_generate.py
# Cached inputs that must already exist on the target machine:
#   data/query_activations/<model>/{truthfulqa,truthfulqa_split0,truthfulqa_split1,gsm8k,mmlu,
#                                   nq,triviaqa}_layer<L>.pt
#   and the contrastive activations read by cobras.utils.data.load_tqa_gen_data_all_splits.
#
# ---------------------------------------------------------------------------------------------
# NOTES
# ---------------------------------------------------------------------------------------------
# TF32. Every GSM8K and MMLU row here goes through scripts/analysis/derivation/run_tf32.py, including the
# unsteered reference, so the whole table is produced under one setting. Full MMLU took 92 min in
# strict fp32 and 18 min with TF32, at 99.8% prediction agreement, with the COBRAS steering update
# itself unchanged (cosine 1.000000). TruthfulQA is deliberately NOT run under TF32, because the
# TruthfulQA numbers already in results/truthfulqa are fp32 and the two must not be mixed. The
# autograd comparison is fp32 as well, since its claim is about machine-precision agreement.
#
# Seeds. GSM8K decodes greedily and MMLU is argmax scoring, so both are deterministic and seed 42
# alone is enough. TruthfulQA samples, so every row that carries a claimed difference is run at
# seeds 42, 43 and 44.
#
# Stale files. The generation scripts skip when the output file exists, and the filename does not
# encode num_examples, so the N=100 outputs must be moved out of the way or every full-size run
# below is silently skipped. STEP 0 does that, moving rather than deleting, and only for files
# whose steering name begins with q6-.
# =============================================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
REPO=$(pwd)

MODEL=${MODEL:-Llama3.1-8B-Base}
LAYER=${LAYER:-13}
ALT_MODEL=${ALT_MODEL:-Mistral-7B-Base}
ALT_LAYER=${ALT_LAYER:-15}
SEED=42
GSM_N=1319          # full GSM8K test split
MMLU_N=14042        # full MMLU test split
DRY_RUN=${DRY_RUN:-0}
GPU=${CUDA_VISIBLE_DEVICES:-0}

PY="uv run python -u"
BACKUP="$REPO/results/analysis/q6/n100_raw_backup"

say () { echo; echo "=== $(date '+%F %T')  $*"; }
run () { if [ "$DRY_RUN" = "1" ]; then echo "[dry-run] $*"; else "$@"; fi; }

say "repo $REPO   model $MODEL   layer $LAYER   CUDA_VISIBLE_DEVICES=$GPU   DRY_RUN=$DRY_RUN"

# ---------------------------------------------------------------------------------------------
# STEP 0. Move the N=100 GSM8K / MMLU outputs aside. They are moved outside results/*/raw_outputs
# so that no glob in the scorers can pick them up again, and only q6-* files are touched, so no
# other rebuttal thread's outputs are affected.
# ---------------------------------------------------------------------------------------------
say "STEP 0  moving stale N=100 outputs aside"
for ds in gsm8k mmlu; do
  case $ds in gsm8k) TAG=GSM8K;; mmlu) TAG=MMLU;; esac
  src="$REPO/results/$ds/raw_outputs/$MODEL"
  dst="$BACKUP/$ds"
  n=0
  if [ -d "$src" ]; then
    for f in "$src"/${MODEL}-l${LAYER}-q6-*-${TAG}-seed${SEED}.jsonl; do
      [ -e "$f" ] || continue
      if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] mv $f -> $dst/$(basename "$f")"
      else
        mkdir -p "$dst"
        mv "$f" "$dst/$(basename "$f")"
      fi
      n=$((n + 1))
    done
  fi
  echo "  $ds: $n file(s)"
done

# ---------------------------------------------------------------------------------------------
# STEP 1. TruthfulQA, 817 questions per run, fp32, no TF32. The ladder at the paper's T = 0.65 at
# three seeds, then each rung's own steering-strength sweep at seed 42.
# ---------------------------------------------------------------------------------------------
say "STEP 1  TruthfulQA generation (817 questions per run)"
tqa () {  # $1 config, $2 T, $3 seed
  run env CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/truthfulqa/truthfulqa_generate.py \
    model=$MODEL layer_idx=$LAYER steer=$1 steer.T=$2 seed=$3
}
for s in 42 43 44; do
  for c in Deriv-Strict Deriv-Unit Deriv-UnitVMF Deriv-Exact Deriv-Shipped Deriv-ShippedNoGate Deriv-ExtFixed; do
    tqa $c 0.65 $s
  done
  tqa Deriv-Exact 0.6 $s
done
for t in 0.5 0.35 0.2;         do tqa Deriv-Strict  $t $SEED; done
for t in 0.5 0.35;             do tqa Deriv-Unit    $t $SEED; done
for t in 0.5;                  do tqa Deriv-UnitVMF $t $SEED; done
for t in 0.35 0.5 0.8 1.0 1.3; do tqa Deriv-Exact   $t $SEED; done
for t in 0.5 0.8 1.0;          do tqa Deriv-Shipped $t $SEED; done

# ---------------------------------------------------------------------------------------------
# STEP 2. GSM8K, full 1319, greedy, TF32, seed 42. Every rung, the unsteered reference, and the
# two ungated rungs at the smaller steering strengths where they are best in distribution.
# ---------------------------------------------------------------------------------------------
say "STEP 2  GSM8K generation (N=$GSM_N)"
gsm () {  # $1 config, $2 T
  run env CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/analysis/derivation/run_tf32.py \
    scripts/gsm8k/gsm8k_generate.py \
    model=$MODEL layer_idx=$LAYER steer=$1 steer.T=$2 seed=$SEED num_examples=$GSM_N
}
gsm Deriv-NoSteer 0.65
for c in Deriv-Strict Deriv-Unit Deriv-UnitVMF Deriv-Exact Deriv-Shipped; do gsm $c 0.65; done
gsm Deriv-Strict 0.35
gsm Deriv-Strict 0.2
gsm Deriv-Unit   0.35

# ---------------------------------------------------------------------------------------------
# STEP 3. MMLU, full 14042, argmax scoring, TF32, seed 42. The same rows.
# ---------------------------------------------------------------------------------------------
say "STEP 3  MMLU generation (N=$MMLU_N)"
mml () {  # $1 config, $2 T
  run env CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/analysis/derivation/run_tf32.py \
    scripts/mmlu/mmlu_generate.py \
    model=$MODEL layer_idx=$LAYER steer=$1 steer.T=$2 seed=$SEED num_examples=$MMLU_N
}
mml Deriv-NoSteer 0.65
for c in Deriv-Strict Deriv-Unit Deriv-UnitVMF Deriv-Exact Deriv-Shipped; do mml $c 0.65; done
mml Deriv-Strict 0.35
mml Deriv-Strict 0.2
mml Deriv-Unit   0.35

# ---------------------------------------------------------------------------------------------
# STEP 4. Scoring. All three write only into results/analysis/q6/, never into the shared eval
# CSVs that the other rebuttal threads append to. derivation/tqa_eval.py loads the two allenai TruthfulQA
# judges and wants the GPU to itself, which is why it goes last.
# ---------------------------------------------------------------------------------------------
say "STEP 4  scoring"
run env CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/analysis/derivation/score.py -d gsm8k -m $MODEL -l $LAYER
run env CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/analysis/derivation/score.py -d mmlu  -m $MODEL -l $LAYER
run env CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/analysis/derivation/tqa_eval.py -m $MODEL -l $LAYER -b 32

# ---------------------------------------------------------------------------------------------
# STEP 5. The analyses that do not depend on any run size, re-run so that the whole section comes
# out of this one script. The autograd comparison stays in full fp32 and must not be given TF32.
# ---------------------------------------------------------------------------------------------
say "STEP 5  size-independent analyses"
run env CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/analysis/derivation/exact_gradient.py -m $MODEL -l $LAYER -n 128 --chunk 8
run env CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/analysis/derivation/exact_gradient.py -m $ALT_MODEL -l $ALT_LAYER -n 128 --chunk 8
run env CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/analysis/derivation/step_magnitude.py -m $MODEL -l $LAYER -T 0.65 --chunk 32
run env CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/analysis/derivation/endpoint_gap.py   -m $MODEL -l $LAYER -T 0.65 --chunk 16

say "STEP 6  tables"
run $PY scripts/analysis/derivation/tables.py

say "done. everything is under results/analysis/q6/"
