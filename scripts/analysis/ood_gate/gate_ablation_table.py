"""Collect the gate ablation into one in-distribution vs OOD table.

    uv run python -u scripts/analysis/ood_gate/gate_ablation_table.py -m Llama3.1-8B-Base -l 13
"""

import argparse
import re

import pandas as pd

from cobras.utils import get_project_dir


_OOD_TASKS = {"gsm8k": "GSM8k", "mmlu": "MMLU", "nq": "NQ", "triviaqa": "TriviaQA"}

# how each run maps onto the two axes the reviewer cares about: is a gate active, and
# is the steering direction query-adaptive
_VARIANTS = [
    ("NoSteer", "Original", "-", "-"),
    ("CAA-T", "CAA", "no", "fixed"),
    ("CAA-gate", "CAA + kNN gate", "kNN", "fixed"),
    ("ODESteer-euler", "ODESteer", "no", "fixed"),
    ("ODESteer-gate", "ODESteer + kNN gate", "kNN", "fixed"),
    ("SphericalSteer-kappa", "SphericalSteer", "no", "fixed"),
    ("SphericalSteer-gate", "SphericalSteer + kNN gate", "kNN", "fixed"),
    ("COBRAS-k_bw", "COBRAS (no gate)", "no", "query-adaptive"),
    ("COBRAS-gate0.98", "COBRAS + kNN gate (paper)", "kNN", "query-adaptive"),
    ("COBRAS-gate1.0-k32-sh50.0-vmf20-raw", "COBRAS, raw drift step", "no", "query-adaptive"),
    ("COBRAS-gate1.0-k32-sh50.0-vmfNone-unit", "COBRAS, no vMF strength", "no", "query-adaptive"),
    ("COBRAS-gate1.0-k32-sh50.0-vmfNone-raw", "COBRAS, raw step, no vMF", "no", "query-adaptive"),
    ("COBRAS-sbgatedensity", "COBRAS + SB density gate", "psi*phi", "query-adaptive"),
    ("COBRAS-sbgatedrift", "COBRAS + SB drift gate", "|grad|", "query-adaptive"),
]


def _match(method: str) -> tuple[str, str, str] | None:
    for prefix, label, gate, direction in _VARIANTS:
        if method.startswith(prefix):
            return label, gate, direction
    return None


def _read(path, metric: str) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for method, value in zip(df["Steering Method"], df[metric]):
        matched = _match(str(method))
        if matched is not None:
            out[matched[0]] = float(value)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="Llama3.1-8B-Base")
    parser.add_argument("-l", "--layer_idx", type=int, default=13)
    parser.add_argument("-s", "--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = get_project_dir() / "results"
    tag = f"{args.model}-l{args.layer_idx}"

    columns = {
        "TQA T*Info": _read(
            results_dir / "truthfulqa" / "eval_results" / "stat_results"
            / f"{tag}-TruthfulQA-seed{args.seed}.csv", "True * Info",
        )
    }
    for task, name in _OOD_TASKS.items():
        sub = "eval_results/stat_results" if task == "gsm8k" else "eval_results"
        columns[name] = _read(
            results_dir / task / sub / f"{tag}-{name.upper() if task == 'mmlu' else name}-seed{args.seed}.csv",
            "Accuracy",
        )

    rows = []
    for _, label, gate, direction in _VARIANTS:
        row = {"Method": label, "Gate": gate, "Direction": direction}
        row.update({col: vals.get(label) for col, vals in columns.items()})
        if any(row[c] is not None for c in columns):
            rows.append(row)

    table = pd.DataFrame(rows)
    for col in _OOD_TASKS.values():
        if col in table and table[col].notna().any():
            table[col] = (table[col] * 100).round(1)
    out_path = results_dir / "analysis" / f"{tag}-gate-ablation-seed{args.seed}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False, na_rep="--"))
    print(f"\n✓ Saved {out_path}")
