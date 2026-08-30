"""Assemble the Q1/Q2 gate-ablation grid from whatever has landed so far.

GSM8K and MMLU rows are looked up first under the `-n100` name suffix used by the interactive
pass and then, failing that, under the plain full-size name, so the same command works before and
after `experiments/ood_gate_ablation.sh` has run. TruthfulQA has no suffix.

    uv run python -u scripts/analysis/ood_gate/collect.py
    uv run python -u scripts/analysis/ood_gate/collect.py --seeds 42 43 44
"""

import argparse

import numpy as np
import pandas as pd

from cobras.utils import get_project_dir

ROOT = get_project_dir()

# (row label, truthfulqa steer name, gsm8k/mmlu steer name)
ROWS = [
    ("Unsteered", "q1-NoSteer|NoSteer", "q1-NoSteer"),
    ("CAA (T=4)", "q1-CAA-T4|CAA-T4", "q1-CAA-T4"),
    ("CAA + gate (T=4)", "q1-CAA-gate0.98-k32-sh200.0-T4", "q1-CAA-gate0.98-k32-sh200.0-T4"),
    ("ODESteer (T=4)", "q1-ODESteer-T4|ODESteer-euler-steps10-nc8000-degree2-gamma0.1-coef01.0-lr-T4",
     "q1-ODESteer-T4"),
    ("ODESteer + gate (T=4)", "q1-ODESteer-gate0.98-k32-sh200.0-T4", "q1-ODESteer-gate0.98-k32-sh200.0-T4"),
    ("SphericalSteer (T=4)", "q1-SphericalSteer-T4|SphericalSteer-kappa20.0-alpha0.7-beta-0.15-T4",
     "q1-SphericalSteer-T4"),
    ("SphericalSteer + gate (T=4)", "q1-SphericalSteer-gate0.98-k32-sh200.0-T4",
     "q1-SphericalSteer-gate0.98-k32-sh200.0-T4"),
    ("COBRAS, no gate (T=0.65)", "q1-cobras-nogate-vmf20-T0.65", "q1-cobras-nogate-vmf20-T0.65"),
    ("COBRAS + k-NN gate, paper (T=0.65)", "q1-cobras-knn-ratio-p0.98-k32-sh200.0-T0.65",
     "q1-cobras-knn-ratio-p0.98-k32-sh200.0-T0.65"),
    ("COBRAS + bridge-marginal gate (T=0.65)", "q1-cobras-marginal-b0.00390625-cov0.534-T0.65",
     "q1-cobras-marginal-b0.00390625-cov0.534-T0.65"),
]

SWEEP = [
    ("shipped k-NN gate, percentile 0.90", "q1-cobras-knn-ratio-p0.9-k32-sh200.0-T0.65"),
    ("shipped k-NN gate, percentile 0.95", "q1-cobras-knn-ratio-p0.95-k32-sh200.0-T0.65"),
    ("shipped k-NN gate, percentile 0.98", "q1-cobras-knn-ratio-p0.98-k32-sh200.0-T0.65"),
    ("shipped k-NN gate, percentile 0.995", "q1-cobras-knn-ratio-p0.995-k32-sh200.0-T0.65"),
    ("shipped k-NN gate, K = 128", "q1-cobras-knn-ratio-p0.98-k128-sh200.0-T0.65"),
    ("shipped k-NN gate, K = 256", "q1-cobras-knn-ratio-p0.98-k256-sh200.0-T0.65"),
    ("recalibrated k-NN, coverage 0.534", "q1-cobras-knn-quantile-k32-cov0.534-T0.65"),
    ("recalibrated k-NN, coverage 0.80", "q1-cobras-knn-quantile-k32-cov0.8-T0.65"),
    ("recalibrated marginal, coverage 0.40", "q1-cobras-marginal-b0.00390625-cov0.4-T0.65"),
    ("recalibrated marginal, coverage 0.534", "q1-cobras-marginal-b0.00390625-cov0.534-T0.65"),
    ("recalibrated marginal, coverage 0.70", "q1-cobras-marginal-b0.00390625-cov0.7-T0.65"),
    ("recalibrated marginal, coverage 0.80", "q1-cobras-marginal-b0.00390625-cov0.8-T0.65"),
    ("recalibrated marginal, coverage 0.90", "q1-cobras-marginal-b0.00390625-cov0.9-T0.65"),
]


def load(path, val, n_col="N"):
    if not path.exists():
        return {}
    d = pd.read_csv(path)
    return {r["Steering Method"]: (float(r[val]) * 100.0, int(r[n_col])) for _, r in d.iterrows()}


def get(store, key, suffix=""):
    """`key` may be a "|"-separated preference list. Full-size rows win over the `-n100`
    interactive rows, so the same command is correct before and after
    experiments/ood_gate_ablation.sh has run."""
    names = key.split("|")
    for n in names:
        if n in store:
            return store[n]
    for n in names:
        if n + suffix in store:
            return store[n + suffix]
    return None


def binom_se(v):
    p, n = v[0] / 100.0, v[1]
    return 100.0 * float(np.sqrt(max(p * (1 - p), 1e-9) / n))


def cell(v):
    return f"{'--':>18}" if v is None else f"{v[0]:.1f}+-{binom_se(v):.1f} (N={v[1]})".rjust(18)


def tqa_cell(vals):
    if not vals:
        return f"{'--':>18}"
    mean = float(np.mean([v[0] for v in vals]))
    if len(vals) == 1:
        return f"{mean:.2f} (1 seed)".rjust(18)
    sd = float(np.std([v[0] for v in vals], ddof=1))
    return f"{mean:.2f}+-{sd:.2f} ({len(vals)}s)".rjust(18)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--suffix", default="-n100", help="fallback name suffix for GSM8K / MMLU")
    args = ap.parse_args()
    m, l, sfx = args.model, args.layer_idx, args.suffix

    tqa = {}
    for s in args.seeds:
        d = load(ROOT / "results/truthfulqa/eval_results/stat_results" / f"{m}-l{l}-TruthfulQA-seed{s}.csv",
                 "True * Info", "Truthfulness")
        d = {k: (v[0], 817) for k, v in d.items()}
        d.update(load(ROOT / "results/truthfulqa/eval_results/stat_results"
                      / f"q1-{m}-l{l}-TruthfulQA-seed{s}.csv", "True * Info"))
        tqa[s] = d

    gsm = load(ROOT / "results/gsm8k/eval_results/stat_results" / f"q1-{m}-l{l}-GSM8K-seed42.csv", "Accuracy")
    mml = load(ROOT / "results/mmlu/eval_results/stat_results" / f"q1-{m}-l{l}-MMLU-seed42.csv", "Accuracy")
    full_gsm = load(ROOT / "results/gsm8k/stat_results" / f"{m}-l{l}-GSM8K-seed42.csv", "Accuracy")
    full_mml = load(ROOT / "results/mmlu" / f"{m}-l{l}-MMLU-seed42.csv", "Accuracy")

    def block(title, rows):
        print(f"\n{title:<40}{'TQA TxI':>18}{'GSM8K':>18}{'MMLU':>18}")
        for label, a, b in rows:
            vals = [v for v in (get(tqa[s], a) for s in args.seeds) if v]
            print(f"{label:<40}{tqa_cell(vals)}{cell(get(gsm, b, sfx))}{cell(get(mml, b, sfx))}")

    block("Q1 grid", ROWS)
    block("Q2 gate-coverage sweep", [(lab, k, k) for lab, k in SWEEP])

    print("\nfull-size anchors on disk (different N, do not mix into the tables above):")
    keep = {"NoSteer", "CAA-T4", "SphericalSteer-kappa20.0-alpha0.7-beta-0.15-T4",
            "ODESteer-euler-steps10-nc8000-degree2-gamma0.1-coef01.0-lr-T4"}
    for k, v in sorted(full_gsm.items()):
        if k in keep or k.startswith("COBRAS-k_bw5"):
            print(f"  GSM8K N={v[1]:<6}{v[0]:6.2f}  {k}")
    for k, v in sorted(full_mml.items()):
        if k in keep:
            print(f"  MMLU  N={v[1]:<6}{v[0]:6.2f}  {k}")
