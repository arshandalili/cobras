"""Q3 (Reviewer EJdD): evaluate the layer-sweep generations with the paper's TruthfulQA judges.

Same judges and metric definition as scripts/truthfulqa/truthfulqa_eval.py
(allenai/truthfulqa-truth-judge-llama2-7B and allenai/truthfulqa-info-judge-llama2-7B,
True x Info = mean over prompts of truthful AND informative), but reads and writes only
under results/analysis/q3_layer_sweep so it never touches the shared eval CSVs.

    CUDA_VISIBLE_DEVICES=5 ./.venv/bin/python -u scripts/analysis/geometry/eval_sweep.py
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from cobras.utils import get_project_dir
from cobras.utils.metric import TruthfulQAJudge

SWEEP = get_project_dir() / "results" / "analysis" / "q3_layer_sweep"
PAT = re.compile(r"^(?P<model>.+)-l(?P<layer>\d+)-(?P<method>[a-z]+)-TruthfulQA-seed(?P<seed>\d+)$")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch_size", type=int, default=16)
    args = ap.parse_args()

    out_csv = SWEEP / "q3_layer_sweep_eval.csv"
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.model, prev.layer, prev.method, prev.seed))
    else:
        prev = pd.DataFrame()

    files = sorted((SWEEP / "raw_outputs").glob("*-TruthfulQA-seed*.jsonl"))
    todo = []
    for f in files:
        m = PAT.match(f.stem)
        assert m is not None, f
        key = (m["model"], int(m["layer"]), m["method"], int(m["seed"]))
        if key not in done:
            todo.append((f, key))
    print(f"{len(files)} files, {len(todo)} to evaluate", flush=True)
    if not todo:
        raise SystemExit(0)

    judge = TruthfulQAJudge(display=True)
    rows = []
    for f, key in todo:
        df = pd.read_json(f, orient="records", lines=True)
        ti, tr, inf = judge.batch_evaluate(df.prompt.tolist(), df.output.tolist(), args.batch_size)
        row = dict(model=key[0], layer=key[1], method=key[2], seed=key[3],
                   n=len(df), true_info=float(np.nanmean(ti)),
                   truthfulness=float(np.nanmean(tr)),
                   informativeness=float(np.nanmean(inf)))
        rows.append(row)
        print(row, flush=True)
        pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).sort_values(
            ["model", "method", "layer"]).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
