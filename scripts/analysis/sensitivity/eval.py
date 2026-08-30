"""Q5 (reviewer 4oei): judge the sensitivity-sweep generations.

Runs only the TruthfulQA truth and info judges (no perplexity / Dist-n), over the
jsonl files written by sensitivity/sweep.py, and appends to a private CSV so that the
shared results/truthfulqa/eval_results tree is untouched.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cobras.utils import get_project_dir
from cobras.utils.metric import TruthfulQAJudge


COLS = ["Model", "Variant", "Seed", "N", "True * Info", "Truthfulness", "Informativeness"]
NAME_RE = re.compile(r"^(?P<model>.+?)-l(?P<layer>\d+)-(?P<variant>q5-.+?)-TruthfulQA-seed(?P<seed>\d+)$")


def pending(gen_dir: Path, df: pd.DataFrame) -> list[tuple[Path, tuple]]:
    done = set(zip(df["Model"], df["Variant"], df["Seed"])) if len(df) else set()
    todo = []
    for p in sorted(gen_dir.glob("*-TruthfulQA-seed*.jsonl")):
        m = NAME_RE.match(p.stem)
        if m is None:
            continue
        key = (f"{m['model']}-l{m['layer']}", m["variant"], int(m["seed"]))
        if key not in done:
            todo.append((p, key))
    return todo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch_size", type=int, default=10)
    ap.add_argument(
        "--expect",
        type=int,
        default=0,
        help="keep polling for new generation files until this many rows exist",
    )
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()

    gen_dir = get_project_dir() / "results" / "analysis" / "q5x" / "gen"
    out_csv = get_project_dir() / "results" / "analysis" / "q5x" / "q5x_sensitivity.csv"

    df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=COLS)

    if not pending(gen_dir, df) and args.expect <= len(df):
        print("nothing to evaluate")
        return

    judge = TruthfulQAJudge(display=True)

    while True:
        todo = pending(gen_dir, df)
        if not todo:
            if len(df) >= args.expect:
                break
            print(f"= {len(df)}/{args.expect} judged, waiting for more generations")
            time.sleep(args.poll)
            continue

        for p, key in todo:
            gen = pd.read_json(p, orient="records", lines=True)
            prompts, outputs = gen.prompt.tolist(), gen.output.tolist()
            ti, tr, info = judge.batch_evaluate(prompts, outputs, batch_size=args.batch_size)
            row = [
                key[0],
                key[1],
                key[2],
                len(prompts),
                float(np.nanmean(ti)),
                float(np.nanmean(tr)),
                float(np.nanmean(info)),
            ]
            df.loc[len(df)] = row
            df.to_csv(out_csv, index=False)
            print(
                f"[{len(df)}/{args.expect}] {key[1]:<14} seed{key[2]}  N={len(prompts)}  "
                f"TxI={row[4] * 100:.2f}  True={row[5] * 100:.2f}  Info={row[6] * 100:.2f}",
                flush=True,
            )

    print(f"\n+ wrote {out_csv}")


if __name__ == "__main__":
    main()
