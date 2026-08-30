"""Does the gate's one knob do what it says?

The abstention gate reads the bridge's own time marginal and calibrates it as a quantile over
held-out in-distribution *prompt* activations, so `abstain_percentile` is a nominal
in-distribution coverage. Two things have to hold for that to be a knob rather than a cliff:

1. The nominal coverage has to control the realized one. Under the reference the gate is
   min(1, (1 - rank) / (1 - p)) with rank uniform, so the mean in-distribution gate is exactly
   (1 + p) / 2. This checks the identity holds on real activations, and reports how far each
   out-of-distribution task falls below it.

2. The score has to separate in-distribution from out-of-distribution at all, which is what the
   AUROC column measures.

The reference is prompt activations rather than the contrastive pairs on purpose: the pairs are
question+answer activations at a different token position, so a threshold read off them does not
control coverage over the queries the gate is actually applied to.

    uv run python -u scripts/analysis/ood_gate/gate_calibration.py -m Llama3.1-8B-Base -l 13
"""

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cobras.steer import COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data_all_splits

ID_TASK = "truthfulqa"
_COBRAS_KWARGS = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
                      vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0)
_PCTS = [0.3, 0.534, 0.8, 0.9, 0.95]


def summarize(gate: np.ndarray) -> tuple[float, float]:
    return float(gate.mean()), float((gate > 0.5).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--b_scale", type=float, default=1 / 256,
                    help="abstention bandwidth as a multiple of sigma^2 (shipped: 1/256)")
    args = ap.parse_args()

    dev = args.device
    pos, neg = load_tqa_gen_data_all_splits(args.model, args.layer_idx)
    m = COBRAS(**_COBRAS_KWARGS, abstain_bandwidth_scale=args.b_scale)
    m = m.fit(pos.float().to(dev), neg.float().to(dev))
    R, s2 = m.R, m.sigma2

    act_dir = get_project_dir() / "data" / "query_activations" / args.model
    Q = {p.stem.replace(f"_layer{args.layer_idx}", ""):
         torch.load(p, weights_only=True).float().to(dev)
         for p in sorted(act_dir.glob(f"*_layer{args.layer_idx}.pt"))}
    Q = {k: v for k, v in Q.items() if not k.startswith("truthfulqa_split")}
    assert ID_TASK in Q, f"missing {ID_TASK} query activations in {act_dir}"
    tasks = [ID_TASK] + [t for t in Q if t != ID_TASK]

    score = {}
    with torch.no_grad():
        for t in tasks:
            x = Q[t]
            q = x * (R / x.norm(dim=-1, keepdim=True))
            score[t] = m._abstain_score(q).cpu().numpy()

    res = {"model": args.model, "layer_idx": args.layer_idx, "b_scale": args.b_scale,
           "R": R, "sigma": s2 ** 0.5}

    # --- 1. where the score puts each task ------------------------------------------------
    print(f"\n{'=' * 92}\n1. The abstention score, -log p_0 at b = sigma^2 x {args.b_scale:g} "
          f"(larger = further out)\n{'=' * 92}")
    print(f"{'query set':<34}{'median':>10}{'p10':>10}{'p90':>10}{'AUROC':>10}")
    res["score_quantiles"], res["auroc"] = {}, {}
    for t in tasks:
        v = score[t]
        if t == ID_TASK:
            auc = float("nan")
        else:
            y = np.r_[np.ones(len(score[ID_TASK])), np.zeros(len(v))]
            auc = roc_auc_score(y, -np.r_[score[ID_TASK], v])
            res["auroc"][t] = float(auc)
        res["score_quantiles"][t] = {str(p): float(np.quantile(v, p)) for p in (0.1, 0.5, 0.9)}
        tag = "  <- in-distribution" if t == ID_TASK else ""
        auc_s = "     --" if t == ID_TASK else f"{auc:>10.3f}"
        print(f"{t + tag:<34}{np.median(v):>10.3f}{np.quantile(v, 0.1):>10.3f}"
              f"{np.quantile(v, 0.9):>10.3f}{auc_s}")
    if res["auroc"]:
        res["auroc"]["mean"] = float(np.mean(list(res["auroc"].values())))
        print(f"\nmean OOD AUROC {res['auroc']['mean']:.3f}")

    # --- 2. does the nominal coverage control the realized one? ----------------------------
    print(f"\n{'=' * 92}\n2. Realized coverage under the shipped quantile calibration"
          f"\n{'=' * 92}")
    print(f"{'nominal coverage p':<22}{'(1+p)/2':>9}" + "".join(f"{t[:9]:>20}" for t in tasks))
    res["coverage"] = {}
    ref_sorted = np.sort(score[ID_TASK])
    for p in _PCTS:
        row = {}
        for t in tasks:
            rank = np.searchsorted(ref_sorted, score[t]) / len(ref_sorted)
            row[t] = summarize(np.clip((1.0 - rank) / (1.0 - p), 0.0, 1.0))
        res["coverage"][str(p)] = {"predicted_id_mean": (1 + p) / 2, **{t: row[t] for t in tasks}}
        print(f"{p:<22g}{(1 + p) / 2:>9.3f}"
              + "".join(f"{row[t][0]:>12.3f} ({row[t][1]:.2f})" for t in tasks))
    print("cell = mean gate (fraction left unattenuated, gate > 0.5). 1 = steer fully, 0 = abstain.")
    print("column 2 is the closed form; the in-distribution column should reproduce it.")

    out = get_project_dir() / "results" / "analysis" / f"q1-{args.model}-l{args.layer_idx}-gate-calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nsaved {out}")
