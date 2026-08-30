"""GSM8K / MMLU accuracy for the Q1/Q2 gate-ablation rows only.

The shipped eval scripts append to a csv shared with the other rebuttal workstreams, so this one
restricts itself to files whose steer name matches `--pattern` and writes its own csv. GSM8K
answer extraction is imported from `scripts/gsm8k/gsm8k_eval.py` rather than reimplemented, so
the parsing is identical to the cached paper rows.

    uv run python -u scripts/analysis/ood_gate/eval.py -t gsm8k -m Llama3.1-8B-Base -l 13
    uv run python -u scripts/analysis/ood_gate/eval.py -t mmlu  -m Llama3.1-8B-Base -l 13
"""

import argparse
import importlib.util

import pandas as pd

from cobras.utils import get_project_dir

ROOT = get_project_dir()


def _gsm8k_fns():
    spec = importlib.util.spec_from_file_location("_gsm8k_eval", ROOT / "scripts" / "gsm8k" / "gsm8k_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_answer, mod.normalize


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--task", choices=["gsm8k", "mmlu"], required=True)
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("-s", "--seed", type=int, default=42)
    ap.add_argument("--pattern", default="q1-*")
    args = ap.parse_args()

    tag = "GSM8K" if args.task == "gsm8k" else "MMLU"
    raw_dir = ROOT / "results" / args.task / "raw_outputs" / args.model
    out_path = (ROOT / "results" / args.task / "eval_results" / "stat_results"
                / f"q1-{args.model}-l{args.layer_idx}-{tag}-seed{args.seed}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = ["Model", "Steering Method", "Accuracy", "N", "N_parsed"]
    df_out = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=cols)
    prefix, suffix = f"{args.model}-l{args.layer_idx}-", f"-{tag}-seed{args.seed}"
    done = set(df_out["Steering Method"])
    files = sorted(raw_dir.glob(f"{prefix}{args.pattern}{suffix}.jsonl"))
    todo = [f for f in files if f.stem[len(prefix):-len(suffix)] not in done]
    print(f"{len(files)} matching files, {len(todo)} to score")

    extract = normalize = None
    if args.task == "gsm8k":
        extract, normalize = _gsm8k_fns()

    for f in todo:
        steer = f.stem[len(prefix):-len(suffix)]
        d = pd.read_json(f, orient="records", lines=True)
        if args.task == "gsm8k":
            correct = parsed = 0
            for _, r in d.iterrows():
                p = extract(str(r["output"]))
                if p is not None:
                    parsed += 1
                    correct += normalize(p) == normalize(str(r["correct_answer"]))
            acc, n_parsed = correct / len(d), parsed
        else:
            acc = float((d["output"] == d["correct"]).mean())
            n_parsed = len(d)
        df_out.loc[len(df_out)] = [args.model, steer, acc, len(d), n_parsed]
        print(f"{steer}: {acc * 100:.2f}  (N = {len(d)}, parsed {n_parsed})")
        df_out.to_csv(out_path, index=False)

    print(f"\nsaved {out_path}")
    if len(df_out):
        print(df_out.sort_values("Steering Method").to_string(index=False))
