"""Q4 (Reviewer 4oei): evaluate only the q4-* ablation runs, into a private results file.

The shipped eval scripts glob every raw output in the tree and append to a shared CSV, which
is not safe while other rebuttal runs are in flight. This one matches `*-q4-*-<TASK>-seed*`
only and writes to results/analysis/q4x/. The GSM8K answer extraction and the MMLU scoring
are copied verbatim from scripts/gsm8k/gsm8k_eval.py and scripts/mmlu/mmlu_eval.py so the
numbers are directly comparable with the cached full-size results.

Usage:
  uv run python scripts/analysis/ablation/eval.py --task gsm8k|mmlu           (CPU)
  CUDA_VISIBLE_DEVICES=5 uv run python scripts/analysis/ablation/eval.py --task truthfulqa   (GPU)
"""

from __future__ import annotations

import argparse
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

from cobras.utils import get_project_dir

OUT_DIR = get_project_dir() / "results" / "analysis" / "q4x"


def extract_answer(output: str) -> str | None:
    output = output.split("\nQuestion:")[0]
    if "####" in output:
        after = output.split("####")[1].strip()
        m = re.search(r"-?[\d,]+(?:\.\d+)?", after)
        if m:
            return m.group(0).replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", output)
    return nums[-1].replace(",", "") if nums else None


def normalize(ans: str) -> str:
    ans = ans.replace(",", "").strip()
    try:
        v = Decimal(ans)
        return str(int(v)) if v == v.to_integral_value() else format(v.normalize(), "f")
    except (InvalidOperation, ValueError):
        return ans


def _files(task_dir: Path, model: str, layer: int, tag: str, seed: int) -> list[Path]:
    prefix = f"{model}-l{layer}-"
    suffix = f"-{tag}-seed{seed}"
    return sorted(task_dir.glob(f"{prefix}q4-*{suffix}.jsonl"))


def eval_gsm8k(args) -> pd.DataFrame:
    raw = get_project_dir() / "results" / "gsm8k" / "raw_outputs" / args.model
    rows = []
    for fp in _files(raw, args.model, args.layer_idx, "GSM8K", args.seed):
        method = fp.stem[len(f"{args.model}-l{args.layer_idx}-") : -len(f"-GSM8K-seed{args.seed}")]
        df = pd.read_json(fp, orient="records", lines=True)
        correct = parsed = 0
        for _, r in df.iterrows():
            pred = extract_answer(str(r["output"]))
            if pred is not None:
                parsed += 1
                correct += normalize(pred) == normalize(str(r["correct_answer"]))
        rows.append(dict(Model=args.model, Method=method, Seed=args.seed,
                         Accuracy=correct / len(df), N=len(df), N_parsed=parsed))
        print(f"{method:16s} acc={rows[-1]['Accuracy']:.4f} ({correct}/{len(df)}) parsed={parsed}")
    return pd.DataFrame(rows)


def eval_mmlu(args) -> pd.DataFrame:
    raw = get_project_dir() / "results" / "mmlu" / "raw_outputs" / args.model
    rows = []
    for fp in _files(raw, args.model, args.layer_idx, "MMLU", args.seed):
        method = fp.stem[len(f"{args.model}-l{args.layer_idx}-") : -len(f"-MMLU-seed{args.seed}")]
        df = pd.read_json(fp, orient="records", lines=True)
        correct = int((df["output"] == df["correct"]).sum())
        rows.append(dict(Model=args.model, Method=method, Seed=args.seed,
                         Accuracy=correct / len(df), N=len(df)))
        print(f"{method:16s} acc={rows[-1]['Accuracy']:.4f} ({correct}/{len(df)})")
    return pd.DataFrame(rows)


def eval_truthfulqa(args) -> pd.DataFrame:
    from cobras.utils.metric import TruthfulQAJudge

    raw = get_project_dir() / "results" / "truthfulqa" / "raw_outputs" / args.model
    files = _files(raw, args.model, args.layer_idx, "TruthfulQA", args.seed)
    if args.include:
        files += sorted(raw.glob(
            f"{args.model}-l{args.layer_idx}-{args.include}-TruthfulQA-seed{args.seed}.jsonl"))
    if args.skip_done:
        done = OUT_DIR / f"truthfulqa-{args.model}-l{args.layer_idx}-seed{args.seed}.csv"
        if done.exists():
            have = set(pd.read_csv(done)["Method"])
            files = [f for f in files if f.stem.split(f"-l{args.layer_idx}-")[1].rsplit(
                f"-TruthfulQA-seed{args.seed}", 1)[0] not in have]
    if not files:
        print("no files")
        return pd.DataFrame()
    judge = TruthfulQAJudge(display=True)
    rows = []
    for fp in files:
        method = fp.stem[len(f"{args.model}-l{args.layer_idx}-") : -len(f"-TruthfulQA-seed{args.seed}")]
        df = pd.read_json(fp, orient="records", lines=True)
        ti, tr, info = judge.batch_evaluate(df.prompt.tolist(), df.output.tolist())
        rows.append(dict(Model=args.model, Method=method, Seed=args.seed,
                         TrueTimesInfo=float(np.nanmean(ti)), True_=float(np.nanmean(tr)),
                         Info=float(np.nanmean(info)), N=len(df)))
        print(f"{method:16s} TxI={rows[-1]['TrueTimesInfo']:.4f} "
              f"True={rows[-1]['True_']:.4f} Info={rows[-1]['Info']:.4f} N={len(df)}")
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["gsm8k", "mmlu", "truthfulqa"])
    p.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    p.add_argument("-l", "--layer_idx", type=int, default=13)
    p.add_argument("-s", "--seed", type=int, default=42)
    p.add_argument("--include", default=None,
                   help="extra glob (without model/layer/task/seed) to evaluate as well")
    p.add_argument("--skip-done", action="store_true",
                   help="truthfulqa only: skip methods already in the output CSV")
    args = p.parse_args()

    df = {"gsm8k": eval_gsm8k, "mmlu": eval_mmlu, "truthfulqa": eval_truthfulqa}[args.task](args)
    if df.empty:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.task}-{args.model}-l{args.layer_idx}-seed{args.seed}.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old["Method"].isin(df["Method"])], df], ignore_index=True)
    df = df.sort_values("Method")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
