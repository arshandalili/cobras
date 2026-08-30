"""Q4 (Reviewer 4oei): assemble the leave-one-out ablation table from the q4x result files."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cobras.utils import get_project_dir

Q4X = get_project_dir() / "results" / "analysis" / "q4x"
MODEL, LAYER = "Llama3.1-8B-Base", 13

ORDER = [
    ("q4-nosteer", "unsteered"),
    ("q4-full", "COBRAS (shipped)"),
    ("q4-nosphere", "- spherical projection"),
    ("q4-nosinkhorn", "- Sinkhorn solve"),
    ("q4-onestep", "- multi-step (K=1)"),
    ("q4-rawstep", "- direction normalization"),
    ("q4-novmf", "- vMF strength gate"),
    ("q4-novmf-tmatch", "- vMF gate, T matched"),
    ("q4-noabstain", "- kNN abstention gate"),
    ("q4-uniformw", "- Eq. 18 query weights"),
]


def load(task: str, seeds: list[int]) -> pd.DataFrame:
    frames = []
    for s in seeds:
        p = Q4X / f"{task}-{MODEL}-l{LAYER}-seed{s}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    tqa = load("truthfulqa", [42, 43, 44])
    gsm = load("gsm8k", [42])
    mml = load("mmlu", [42])

    rows = []
    for key, label in ORDER:
        r = {"variant": label, "key": key}
        t = tqa[tqa.Method == key] if len(tqa) else tqa
        if len(t):
            v = 100 * t.TrueTimesInfo.values
            r["tqa"] = f"{v.mean():.1f}"
            r["tqa_sd"] = f"{v.std(ddof=1):.1f}" if len(v) > 1 else "-"
            r["tqa_n"] = len(v)
            r["tqa_true"] = f"{100 * t.True_.values.mean():.1f}"
            r["tqa_info"] = f"{100 * t.Info.values.mean():.1f}"
        for name, df in (("gsm8k", gsm), ("mmlu", mml)):
            d = df[df.Method == key] if len(df) else df
            r[name] = f"{100 * d.Accuracy.values[0]:.2f}" if len(d) else "-"
        rows.append(r)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print()
    print("| Variant | TQA TxI | GSM8K | MMLU |")
    print("|---|---|---|---|")
    for _, r in out.iterrows():
        print(f"| {r['variant']} | {r.get('tqa', '-')} | {r.get('gsm8k', '-')} | {r.get('mmlu', '-')} |")

    print("\nper-seed TruthfulQA True x Info (%), N = 817 each\n")
    print("| Variant | seed 42 | seed 43 | seed 44 | mean | sd | True | Info |")
    print("|---|---|---|---|---|---|---|---|")
    for key, label in ORDER:
        t = tqa[tqa.Method == key] if len(tqa) else tqa
        if not len(t):
            continue
        per = {int(s): 100 * v for s, v in zip(t.Seed, t.TrueTimesInfo)}
        v = np.array(list(per.values()))
        cells = " | ".join(f"{per[s]:.2f}" if s in per else "-" for s in (42, 43, 44))
        sd = f"{v.std(ddof=1):.2f}" if len(v) > 1 else "-"
        print(f"| {label} | {cells} | {v.mean():.2f} | {sd} | "
              f"{100 * t.True_.mean():.1f} | {100 * t.Info.mean():.1f} |")


if __name__ == "__main__":
    main()
