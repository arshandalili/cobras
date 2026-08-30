#!/usr/bin/env bash
#
# Packs everything the Reviewer-4oei full-size re-runs need into one tarball, so the whole set can
# be moved to a faster machine and run there unattended.
#
#   bash experiments/bundle_for_remote.sh                  # writes /tmp/cobras_4oei_bundle.tar.gz
#   OUT=/path/to/bundle.tar.gz bash experiments/bundle_for_remote.sh
#   MODELS="Llama3.1-8B-Base Mistral-7B-Base" bash experiments/bundle_for_remote.sh
#
# What goes in: the package, the runner scripts, the Q1/Q4/Q5/Q6 configs and analysis helpers, and
# the cached contrastive and query activations for the models being re-run. What stays out: the
# 12 GB of activations for layers and models this rebuttal does not touch, the results tree, and
# the virtualenv.
#
# On the target machine:
#   tar xzf cobras_4oei_bundle.tar.gz && cd cobras
#   uv sync                                    # or pip install -e .
#   bash experiments/run_all_fullsize.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-/tmp/cobras_4oei_bundle.tar.gz}"
MODELS="${MODELS:-Llama3.1-8B-Base}"
LAYERS="${LAYERS:-13}"

MANIFEST="$(mktemp)"
trap 'rm -f "$MANIFEST"' EXIT

add () { for p in "$@"; do [ -e "$p" ] && echo "$p" >> "$MANIFEST"; done; return 0; }

# --- code and packaging -------------------------------------------------------------------------
add pyproject.toml uv.lock README.md
add src
add scripts/truthfulqa scripts/gsm8k scripts/mmlu
add scripts/aggregate_stats.py

# --- runner scripts and the configs they name ----------------------------------------------------
add experiments/run_all_fullsize.sh
add experiments/ood_paper_config.sh experiments/ood_gate_ablation.sh
add experiments/component_ablation.sh experiments/derivation_variants.sh
add scripts/analysis/coverage_audit.py
add confs/gsm8k.yaml confs/mmlu.yaml confs/truthfulqa.yaml
add confs/steer
# every analysis helper the section runners call; EuclideanCOBRAS now lives in src/, which
# is already bundled above
add scripts/analysis/ood_gate scripts/analysis/ablation
add scripts/analysis/sensitivity scripts/analysis/derivation
add scripts/prepare

# --- cached activations, only for the models and layers being re-run ------------------------------
for m in $MODELS; do
  for l in $LAYERS; do
    while IFS= read -r f; do add "$f"; done < <(
      find "data/truthfulqa/activations/${m}" -name "*layer${l}.pt" 2>/dev/null
      find "data/query_activations/${m}"      -name "*layer${l}.pt" 2>/dev/null
    )
  done
done
add data/truthfulqa/texts data/truthfulqa/extract_activations.py data/truthfulqa/format_dataset.py

# --- the drafts, so whoever runs this can see which numbers the runs supersede ---------------------
add 4oei.md 4oei_plan.md
for q in Q1Q2 Q4 Q5 Q6; do add "4oei_${q}.md"; done

sort -u "$MANIFEST" -o "$MANIFEST"
echo "files: $(wc -l < "$MANIFEST")"
echo "size:  $(du -ch $(tr '\n' ' ' < "$MANIFEST") 2>/dev/null | tail -1 | cut -f1)"

tar czf "$OUT" --transform 's,^,cobras/,' -T "$MANIFEST"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
