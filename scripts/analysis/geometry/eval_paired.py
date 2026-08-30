"""Q3 (Reviewer EJdD): per-prompt judgements for the layer sweep, and a paired test of
COBRAS against its Euclidean control at each layer.

geometry/eval_sweep.py stores only the per-file means, which cannot say whether a 3-5 point gap
between the two variants at one layer is real. The judges are deterministic
(`generate(do_sample=False)`), so re-running them reproduces those means exactly and
additionally retains the per-prompt labels needed for a paired (McNemar) test on the same
817 questions.

    CUDA_VISIBLE_DEVICES=5 ./.venv/bin/python -u scripts/analysis/geometry/eval_paired.py

Writes, under results/analysis/q3_layer_sweep/:
    q3_paired_labels.csv   one row per (layer, method, question): true, info, true_info
    q3_paired_stats.csv    one row per layer: means, McNemar counts, exact binomial p
Re-running with the labels file already present skips the GPU work and recomputes the stats.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from cobras.utils import get_project_dir

SWEEP = get_project_dir() / "results" / "analysis" / "q3_layer_sweep"
PAT = re.compile(r"^(?P<model>.+)-l(?P<layer>\d+)-(?P<method>[a-z]+)-TruthfulQA-seed(?P<seed>\d+)$")
LABELS = SWEEP / "q3_paired_labels.csv"
STATS = SWEEP / "q3_paired_stats.csv"


def judge_all(batch_size: int) -> pd.DataFrame:
    from cobras.utils.metric import TruthfulQAJudge

    judge = TruthfulQAJudge(display=True)
    frames = []
    for f in sorted((SWEEP / "raw_outputs").glob("*-TruthfulQA-seed*.jsonl")):
        m = PAT.match(f.stem)
        assert m is not None, f
        df = pd.read_json(f, orient="records", lines=True)
        ti, tr, inf = judge.batch_evaluate(df.prompt.tolist(), df.output.tolist(), batch_size)
        frames.append(pd.DataFrame({
            "model": m["model"], "layer": int(m["layer"]), "method": m["method"],
            "seed": int(m["seed"]), "qidx": np.arange(len(df)), "prompt": df.prompt,
            "true": np.asarray(tr, dtype=int), "info": np.asarray(inf, dtype=int),
            "true_info": np.asarray(ti, dtype=int),
        }))
        print(f"[judged] {f.stem}  true_info={np.mean(ti):.6f}", flush=True)
    return pd.concat(frames, ignore_index=True)


def stats(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, layer, seed), g in lab.groupby(["model", "layer", "seed"]):
        a = g[g.method == "cobras"].sort_values("qidx")
        b = g[g.method == "euclid"].sort_values("qidx")
        if len(a) == 0 or len(b) == 0:
            continue
        assert list(a.prompt) == list(b.prompt), (model, layer)
        x, y = a.true_info.to_numpy(), b.true_info.to_numpy()
        n10 = int(((x == 1) & (y == 0)).sum())   # COBRAS only
        n01 = int(((x == 0) & (y == 1)).sum())   # Euclidean only
        n11 = int(((x == 1) & (y == 1)).sum())
        n00 = int(((x == 0) & (y == 0)).sum())
        disc = n10 + n01
        p = binomtest(n10, disc, 0.5).pvalue if disc > 0 else 1.0
        rows.append(dict(
            model=model, layer=layer, seed=seed, n=len(x),
            cobras=float(x.mean()), euclid=float(y.mean()), delta=float(x.mean() - y.mean()),
            both=n11, neither=n00, cobras_only=n10, euclid_only=n01,
            discordant=disc, mcnemar_p=float(p),
            agree_frac=float((x == y).mean()),
        ))
    return pd.DataFrame(rows).sort_values(["model", "layer"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch_size", type=int, default=16)
    ap.add_argument("--stats_only", action="store_true")
    args = ap.parse_args()

    if LABELS.exists() and args.stats_only:
        lab = pd.read_csv(LABELS)
    else:
        lab = judge_all(args.batch_size)
        SWEEP.mkdir(parents=True, exist_ok=True)
        lab.to_csv(LABELS, index=False)
        print(f"wrote {LABELS}  rows={len(lab)}", flush=True)

    # Cross-check: the per-prompt labels must average to the means geometry/eval_sweep.py wrote.
    ev = pd.read_csv(SWEEP / "q3_layer_sweep_eval.csv")
    agg = lab.groupby(["model", "layer", "method", "seed"]).agg(
        ti=("true_info", "mean"), tr=("true", "mean"), inf=("info", "mean")).reset_index()
    m = ev.merge(agg, on=["model", "layer", "method", "seed"])
    dmax = max((m.true_info - m.ti).abs().max(),
               (m.truthfulness - m.tr).abs().max(),
               (m.informativeness - m["inf"]).abs().max())
    print(f"cross-check against q3_layer_sweep_eval.csv: {len(m)} cells, "
          f"max abs difference {dmax:.3e}")

    st = stats(lab)
    st.to_csv(STATS, index=False)
    print(st.to_string(index=False))
    print(f"wrote {STATS}")
