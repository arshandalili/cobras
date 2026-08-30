"""Assemble the Q6 ladder and T-sweep tables from the q6-only result CSVs."""

import re

import numpy as np
import pandas as pd

from cobras.utils import get_project_dir

D = get_project_dir() / "results" / "analysis" / "q6"
MODEL, LAYER = "Llama3.1-8B-Base", 13

ORDER = ["q6-strict", "q6-unit", "q6-unitvmf", "q6-exact", "q6-shipped", "q6-nosteer"]
LABEL = {
    "q6-strict": "1 strict Eq.13 (exact grad, raw step, no vMF, no gate)",
    "q6-unit": "2 + unit-normalized step",
    "q6-unitvmf": "3 + vMF strength gate",
    "q6-exact": "4 + abstention gate = COBRAS w/ exact gradient",
    "q6-shipped": "5 shipped COBRAS (extended potentials, adaptive bw)",
    "q6-nosteer": "0 unsteered",
}


def split(name):
    m = re.match(r"(q6-[a-z]+)(?:-T([0-9.]+))?$", name)
    return (m.group(1), float(m.group(2)) if m.group(2) else None) if m else (name, None)


def main():
    tqa = pd.read_csv(D / f"q6-{MODEL}-l{LAYER}-tqa.csv")
    tqa[["cfg", "T"]] = tqa["Steering Method"].apply(lambda s: pd.Series(split(s)))
    tqa["TxI"] = 100 * tqa["True * Info"]
    g = (tqa.groupby(["cfg", "T"])["TxI"]
         .agg(mean="mean", sd=lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
              n="count").reset_index())

    print("\n=== TruthfulQA True x Info, N=817 per seed, mean over seeds (sd over seeds)")
    print(f"{'config':<12}{'T':>6}{'mean':>9}{'sd':>8}{'seeds':>7}")
    for _, r in g.sort_values(["cfg", "T"]).iterrows():
        print(f"{r['cfg']:<12}{r['T']:>6}{r['mean']:>9.2f}"
              f"{(r['sd'] if r['sd'] == r['sd'] else float('nan')):>8.2f}{int(r['n']):>7}")

    ood = {}
    for d in ("gsm8k", "mmlu"):
        p = D / f"q6-{MODEL}-l{LAYER}-{d}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df[["cfg", "T"]] = df["method"].apply(lambda s: pd.Series(split(s)))
            ood[d] = df.set_index("cfg")

    print("\n=== ladder at T = 0.65")
    print(f"{'rung':<54}{'TQA TxI':>12}{'GSM8K':>9}{'MMLU':>9}")
    for c in ORDER:
        row = g[(g.cfg == c) & ((g["T"] == 0.65) | (g["T"].isna()))]
        t = (f"{row['mean'].iloc[0]:.2f} +- {row['sd'].iloc[0]:.2f}"
             if len(row) and row["sd"].iloc[0] == row["sd"].iloc[0]
             else (f"{row['mean'].iloc[0]:.2f}" if len(row) else "-"))
        cells = []
        for d in ("gsm8k", "mmlu"):
            v = ood.get(d)
            cells.append(f"{v.loc[c, 'acc']:.1f}" if v is not None and c in v.index else "-")
        print(f"{LABEL.get(c, c):<54}{t:>12}{cells[0]:>9}{cells[1]:>9}")

    print("\n=== T sweep")
    piv = g.pivot_table(index="T", columns="cfg", values=["mean", "sd", "n"])
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
