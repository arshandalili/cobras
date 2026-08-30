"""Q4 (Reviewer EJdD): score the q4 TruthfulQA generations.

Same judges and metric definition as scripts/truthfulqa/truthfulqa_eval.py
(allenai/truthfulqa-truth-judge-llama2-7B and .../info-judge), but reads only
results/analysis/q4/gen/ and writes only under results/analysis/q4/, so it does
not touch the shared eval CSV.

CUDA_VISIBLE_DEVICES=6 ./.venv/bin/python scripts/analysis/cost/eval.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cobras.utils import get_project_dir
from cobras.utils.metric import QualityEvaluator, TruthfulQAJudge

COLS = ["Variant", "True * Info", "Truthfulness", "Informativeness",
        "Perplexity", "Dist-1", "Dist-2", "Dist-3", "N"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama3.1-8B-Base")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=10)
    args = ap.parse_args()

    gen_dir = get_project_dir() / "results/analysis/q4/gen"
    out = get_project_dir() / f"results/analysis/q4/q4_truthfulqa_eval-{args.model}-l{args.layer}.csv"
    df_out = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=COLS)

    files = sorted(gen_dir.glob(f"q4-{args.model}-l{args.layer}-*-TruthfulQA-seed{args.seed}.jsonl"))
    todo = []
    for f in files:
        variant = f.stem.split(f"-l{args.layer}-")[1].rsplit("-TruthfulQA", 1)[0]
        if variant not in set(df_out["Variant"].astype(str)):
            todo.append((variant, f))
    if not todo:
        print("nothing to evaluate"); print(df_out.to_string(index=False)); return
    print("evaluating:", [v for v, _ in todo])

    judge = TruthfulQAJudge(display=False)
    quality = QualityEvaluator(device="auto")

    for variant, path in todo:
        df = pd.read_json(path, orient="records", lines=True)
        ti, tr, info = judge.batch_evaluate(df.prompt.tolist(), df.output.tolist())
        ppl, d1, d2, d3 = quality.batch_evaluate(df.output.tolist(), args.batch_size)
        row = [variant, float(np.nanmean(ti)), float(np.nanmean(tr)), float(np.nanmean(info)),
               float(np.nanmean(ppl)), float(np.nanmean(d1)), float(np.nanmean(d2)),
               float(np.nanmean(d3)), len(df)]
        df_out.loc[len(df_out)] = row
        print(f"{variant:>10}  TxI={row[1]*100:.2f}  True={row[2]*100:.2f}  "
              f"Info={row[3]*100:.2f}  PPL={row[4]:.2f}  n={len(df)}")
        df_out.to_csv(out, index=False)

    print(df_out.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
