"""TruthfulQA True x Info for the Q1/Q2 gate-ablation rows only.

Identical judging to `scripts/truthfulqa/truthfulqa_eval.py` (the two allenai TruthfulQA
truth / info judges of App. B.3), restricted to files whose steer name matches `--pattern` and
written to its own csv so that concurrent rebuttal workstreams cannot race on the shared one.
Perplexity and distinct-n are skipped: only True x Info is needed here.

    uv run python -u scripts/analysis/ood_gate/tqa_eval.py -m Llama3.1-8B-Base -l 13 --pattern 'q1-*'
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cobras.utils import get_project_dir
from cobras.utils.metric import TruthfulQAJudge

COLS = ["Model", "Steering Method", "True * Info", "Truthfulness", "Informativeness", "N"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("-s", "--seed", type=int, default=42)
    ap.add_argument("-b", "--batch_size", type=int, default=10)
    ap.add_argument("--pattern", default="q1-*")
    args = ap.parse_args()

    root = get_project_dir() / "results" / "truthfulqa"
    raw_dir = root / "raw_outputs" / args.model
    out_path = root / "eval_results" / "stat_results" / f"q1-{args.model}-l{args.layer_idx}-TruthfulQA-seed{args.seed}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=COLS)
    prefix = f"{args.model}-l{args.layer_idx}-"
    suffix = f"-TruthfulQA-seed{args.seed}"
    files = sorted(raw_dir.glob(f"{prefix}{args.pattern}{suffix}.jsonl"))
    todo = [f for f in files if f.stem[len(prefix):-len(suffix)] not in set(df_out["Steering Method"])]
    print(f"{len(files)} matching files, {len(todo)} to judge")
    if not todo:
        print(df_out.to_string(index=False))
        raise SystemExit(0)

    judge = TruthfulQAJudge(display=True)
    for f in todo:
        steer = f.stem[len(prefix):-len(suffix)]
        d = pd.read_json(f, orient="records", lines=True)
        ti, tr, inf = judge.batch_evaluate(d.prompt.tolist(), d.output.tolist(), args.batch_size)
        df_out.loc[len(df_out)] = [args.model, steer, float(np.nanmean(ti)),
                                   float(np.nanmean(tr)), float(np.nanmean(inf)), len(d)]
        print(f"{steer}: TxI = {np.nanmean(ti) * 100:.2f}  True = {np.nanmean(tr) * 100:.2f} "
              f"Info = {np.nanmean(inf) * 100:.2f}  (N = {len(d)})")
        df_out.to_csv(out_path, index=False)

    print(f"\nsaved {out_path}")
    print(df_out.to_string(index=False))
