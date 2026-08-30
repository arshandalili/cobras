"""Table 2 is one mis-calibrated threshold swept twice, not two sensitive hyperparameters.

Both of Table 2's ablations - the abstention percentile and the neighbourhood size K - move the
same quantity: where `rho_ref` sits inside the distribution of query k-NN radii. This measures
that quantity for each row of Table 2 (the mean gate value on the evaluation set in question) and
regresses the reported numbers on it, one slope and one intercept per column.

The fit having two free parameters, `r` alone would prove nothing. What makes it evidence is that
the fitted intercepts are not given to the regression and come back at the measured unsteered
scores, and the gate = 1 ends come back at the ungated scores.

    uv run python -u scripts/analysis/ood_gate/table2_regression.py -m Llama3.1-8B-Base -l 13
    uv run python -u scripts/analysis/ood_gate/table2_regression.py -m Mistral-7B-Base  -l 15
"""

import argparse
import json

import numpy as np
import torch

from cobras.steer import COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data_all_splits
from cobras.utils.sphere import knn_geodesic_dist

_KW = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
           vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0)

# Table 2 of the paper, verbatim. (label, percentile, K, TQA TxI, GSM8K acc)
TABLE2 = {
    "Llama3.1-8B-Base": [
        ("percentile 90.0", 0.90, 32, 44.67, 48.67),
        ("percentile 95.0", 0.95, 32, 54.83, 48.90),
        ("percentile 98.0", 0.98, 32, 65.48, 47.53),
        ("percentile 99.5", 0.995, 32, 69.15, 11.59),
        ("K = 128", 0.98, 128, 68.91, 25.92),
        ("K = 256", 0.98, 256, 69.40, 15.08),
    ],
    "Mistral-7B-Base": [
        ("percentile 90.0", 0.90, 32, 36.23, 36.89),
        ("percentile 95.0", 0.95, 32, 37.08, 36.31),
        ("percentile 98.0", 0.98, 32, 42.71, 36.23),
        ("percentile 99.5", 0.995, 32, 59.48, 35.53),
        ("K = 128", 0.98, 128, 49.81, 36.51),
        ("K = 256", 0.98, 256, 52.50, 36.90),
    ],
}
# unsteered reference, for checking the fitted intercept: (TQA TxI, GSM8K acc)
UNSTEERED = {"Llama3.1-8B-Base": (41.98, 49.28), "Mistral-7B-Base": (36.9, None)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sharpness", type=float, default=200.0)
    args = ap.parse_args()

    pos, neg = load_tqa_gen_data_all_splits(args.model, args.layer_idx)
    m = COBRAS(**_KW).fit(pos.float().to(args.device), neg.float().to(args.device))
    R = m.R
    act = get_project_dir() / "data" / "query_activations" / args.model
    Q = {t: torch.load(act / f"{t}_layer{args.layer_idx}.pt", weights_only=True).float().to(args.device)
         for t in ("truthfulqa", "gsm8k")}
    Q = {t: v * (R / v.norm(dim=-1, keepdim=True)) for t, v in Q.items()}

    rows = TABLE2[args.model]
    g_id, g_ood = [], []
    for _, p, k, _, _ in rows:
        m.abstain_k = k
        ref = float(np.quantile(knn_geodesic_dist(m.h_neg, R, k=min(k, m.h_neg.size(0) - 1)).cpu().numpy(), p))
        for tag, acc in (("truthfulqa", g_id), ("gsm8k", g_ood)):
            r = m._abstain_score(Q[tag]).cpu().numpy()
            acc.append(float((1.0 / (1.0 + (r / ref) ** args.sharpness)).mean()))
    g_id, g_ood = np.array(g_id), np.array(g_ood)
    tqa = np.array([r[3] for r in rows])
    gsm = np.array([r[4] for r in rows])

    print(f"\n{args.model} l{args.layer_idx}, gate sharpness {args.sharpness:g}\n"
          f"{'Table 2 row':<18}{'ID gate':>9}{'GSM8K gate':>12}{'TxI rep':>9}{'TxI fit':>9}"
          f"{'GSM rep':>9}{'GSM fit':>9}")
    out = {"model": args.model, "layer_idx": args.layer_idx, "rows": []}
    fits = {}
    for name, g, y in (("TxI", g_id, tqa), ("GSM8K", g_ood, gsm)):
        A = np.vstack([g, np.ones_like(g)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        fits[name] = (coef, A @ coef, float(np.corrcoef(g, y)[0, 1]))
    for i, (label, p, k, t, s) in enumerate(rows):
        print(f"{label:<18}{g_id[i]:>9.3f}{g_ood[i]:>12.3f}{t:>9.2f}{fits['TxI'][1][i]:>9.2f}"
              f"{s:>9.2f}{fits['GSM8K'][1][i]:>9.2f}")
        out["rows"].append(dict(label=label, percentile=p, k=k, id_gate=g_id[i], ood_gate=g_ood[i],
                                tqa_reported=t, gsm8k_reported=s,
                                tqa_fit=float(fits["TxI"][1][i]), gsm8k_fit=float(fits["GSM8K"][1][i])))
    for name, unst in (("TxI", UNSTEERED[args.model][0]), ("GSM8K", UNSTEERED[args.model][1])):
        (slope, icpt), pred, r = fits[name]
        res = (tqa if name == "TxI" else gsm) - pred
        print(f"\n{name}: slope {slope:+.2f}  intercept {icpt:.2f}"
              + (f"  (measured unsteered {unst})" if unst is not None else "  (unsteered not measured)")
              + f"\n     gate = 1 extrapolates to {slope + icpt:.1f}"
              + f"\n     r = {r:+.4f}, rms residual {np.sqrt((res ** 2).mean()):.2f}, max {np.abs(res).max():.2f}")
        out[name] = dict(slope=float(slope), intercept=float(icpt), r=r,
                         gate1=float(slope + icpt), rms=float(np.sqrt((res ** 2).mean())))

    path = get_project_dir() / "results" / "analysis" / f"q1-{args.model}-l{args.layer_idx}-table2-regression.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved {path}")
