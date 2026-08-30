"""TruthfulQA True x Info for the Q6 ladder only.

The shared truthfulqa_eval.py globs every raw output in the tree and appends to one CSV, which
several rebuttal jobs are writing at once. This evaluates only the q6-* files, loads the two
allenai judges once, and writes its own CSV. Truthfulness / informativeness / True x Info are
computed exactly as TqaGenEvaluator does; the perplexity and distinct-n columns are skipped.

    uv run python -u scripts/analysis/derivation/tqa_eval.py -m Llama3.1-8B-Base -l 13
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cobras.utils import get_project_dir
from cobras.utils.metric import TruthfulQAJudge

COLS = ["Model", "Layer", "Steering Method", "Seed", "True * Info", "Truthfulness",
        "Informativeness", "N", "SE"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("-b", "--batch_size", type=int, default=32)
    ap.add_argument("--pattern", default="q6-*")
    args = ap.parse_args()

    raw_dir = get_project_dir() / "results" / "truthfulqa" / "raw_outputs" / args.model
    out_path = get_project_dir() / "results" / "analysis" / "q6" / f"q6-{args.model}-l{args.layer_idx}-tqa.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=COLS)

    prefix = f"{args.model}-l{args.layer_idx}-"
    files = sorted(raw_dir.glob(f"{prefix}{args.pattern}-TruthfulQA-seed*.jsonl"))
    todo = []
    for f in files:
        stem = f.stem[len(prefix):]
        method, seed = stem.rsplit("-TruthfulQA-seed", 1)
        if ((df_out["Steering Method"] == method) & (df_out["Seed"] == int(seed))).any():
            continue
        todo.append((f, method, int(seed)))
    print(f"{len(files)} q6 files, {len(todo)} to evaluate")
    if not todo:
        return

    judge = TruthfulQAJudge(display=False)
    for f, method, seed in todo:
        df = pd.read_json(f, orient="records", lines=True)
        prompts, outputs = df.prompt.tolist(), df.output.tolist()
        ti, tr, inf = judge.batch_evaluate(prompts, outputs, batch_size=args.batch_size)
        ti = np.asarray(ti, dtype=float)
        m = float(np.nanmean(ti))
        row = [args.model, args.layer_idx, method, seed, m,
               float(np.nanmean(tr)), float(np.nanmean(inf)), len(ti),
               float(np.sqrt(m * (1 - m) / len(ti)))]
        df_out.loc[len(df_out)] = row
        df_out.to_csv(out_path, index=False)
        print(f"{method:<28} seed{seed}  TxI {100 * m:.2f}  (true {100 * np.nanmean(tr):.2f}, "
              f"info {100 * np.nanmean(inf):.2f}, N={len(ti)})", flush=True)

    print(df_out.sort_values(["Steering Method", "Seed"]).to_string(index=False))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
