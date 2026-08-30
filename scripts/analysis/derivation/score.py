"""Score the Q6 ladder's own MMLU / GSM8K raw outputs without touching the shared eval CSVs.

Uses the same answer extraction as scripts/gsm8k/gsm8k_eval.py, imported rather than copied.

    uv run python scripts/analysis/derivation/score.py -d gsm8k -m Llama3.1-8B-Base -l 13
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

from cobras.utils import get_project_dir

_spec = importlib.util.spec_from_file_location(
    "_gsm8k_eval", str(get_project_dir() / "scripts" / "gsm8k" / "gsm8k_eval.py"))
_gsm = importlib.util.module_from_spec(_spec)
sys.modules["_gsm8k_eval"] = _gsm
_spec.loader.exec_module(_gsm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", choices=["gsm8k", "mmlu"], required=True)
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("--pattern", default="q6-*")
    args = ap.parse_args()

    tag = {"gsm8k": "GSM8K", "mmlu": "MMLU"}[args.dataset]
    raw_dir = get_project_dir() / "results" / args.dataset / "raw_outputs" / args.model
    prefix = f"{args.model}-l{args.layer_idx}-"
    rows = []
    for f in sorted(raw_dir.glob(f"{prefix}{args.pattern}-{tag}-seed*.jsonl")):
        stem = f.stem[len(prefix):]
        method, seed = stem.rsplit(f"-{tag}-seed", 1)
        df = pd.read_json(f, orient="records", lines=True)
        if args.dataset == "mmlu":
            n_parsed = len(df)
            correct = int((df["output"] == df["correct"]).sum())
        else:
            correct = n_parsed = 0
            for out, gold in zip(df["output"], df["correct_answer"]):
                pred = _gsm.extract_answer(str(out))
                if pred is None:
                    continue
                n_parsed += 1
                if _gsm.normalize(pred) == _gsm.normalize(str(gold)):
                    correct += 1
        rows.append(dict(method=method, seed=int(seed), acc=100.0 * correct / len(df),
                         N=len(df), N_parsed=n_parsed))
    out = pd.DataFrame(rows).sort_values(["method", "seed"])
    print(out.to_string(index=False))
    p = get_project_dir() / "results" / "analysis" / "q6" / f"q6-{args.model}-l{args.layer_idx}-{args.dataset}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
