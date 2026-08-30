"""Why Table 2's abstention percentile is so sensitive, and what makes it stable.

Two diagnoses, both measurable without generating a single token.

1. The threshold rho_ref of Alg. 1 is a percentile of the k-NN radius over the *contrastive*
   activations, which are question+answer pairs.  At inference the queries are prompts, at a
   different token position.  The two radius distributions do not coincide, so the nominal
   percentile does not control the realized in-distribution coverage: it is absorbing a
   distribution shift it was never meant to model.

2. The map from score to gate is 1 / (1 + (rho/rho_ref)^s) with s = 200, applied to a ratio
   whose in-distribution / out-of-distribution spread is a few percent in 4096 dimensions.  A
   sigmoid that steep on a variable that concentrated is a step function whose location is the
   only thing that matters, which is what makes it a cliff.

The fix uses machinery already in `_cobras.py`: calibrate on held-out in-distribution *prompt*
activations (`abstain_on_queries: true`, `ref_X`) and read the gate off the score's quantile
rank under that reference (`abstain_calibration: quantile`), so the one knob is a nominal
in-distribution coverage.

    uv run python -u scripts/analysis/ood_gate/gate_calibration.py -m Llama3.1-8B-Base -l 13
"""

import argparse
import json

import numpy as np
import torch

from cobras.steer import COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data_all_splits
from cobras.utils.sphere import knn_geodesic_dist

ID_TASK = "truthfulqa"
_COBRAS_KWARGS = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
                      vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0)
_PCTS = [0.8, 0.9, 0.95, 0.98, 0.995]


def summarize(gate: np.ndarray) -> tuple[float, float]:
    return float(gate.mean()), float((gate > 0.5).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("-k", "--abstain_k", type=int, default=32)
    ap.add_argument("--b_scale", type=float, default=1 / 256)
    args = ap.parse_args()

    dev = args.device
    pos, neg = load_tqa_gen_data_all_splits(args.model, args.layer_idx)
    m = COBRAS(**_COBRAS_KWARGS).fit(pos.float().to(dev), neg.float().to(dev))
    m.abstain_k = args.abstain_k
    R, s2 = m.R, m.sigma2

    act_dir = get_project_dir() / "data" / "query_activations" / args.model
    Q = {p.stem.replace(f"_layer{args.layer_idx}", ""):
         torch.load(p, weights_only=True).float().to(dev)
         for p in sorted(act_dir.glob(f"*_layer{args.layer_idx}.pt"))}
    Q = {k: v for k, v in Q.items() if not k.startswith("truthfulqa_split")}
    tasks = [ID_TASK] + [t for t in Q if t != ID_TASK]

    # radius of the contrastive negatives (what Alg. 1 calibrates on) and of every query set
    rho_train = knn_geodesic_dist(m.h_neg, R, k=min(args.abstain_k, m.h_neg.size(0) - 1)).cpu().numpy()
    rho, marg = {}, {}
    with torch.no_grad():
        for t in tasks:
            x = Q[t]
            q = x * (R / x.norm(dim=-1, keepdim=True))
            rho[t] = m._abstain_score(q).cpu().numpy()
            b = args.b_scale * s2
            d2p, d2n = m._geo_sq(q, m.h_pos), m._geo_sq(q, m.h_neg)
            marg[t] = -(torch.logsumexp(m.log_psi.unsqueeze(0) - d2p / (2 * b), -1)
                        + torch.logsumexp(m.log_phi.unsqueeze(0) - d2n / (2 * b), -1)).cpu().numpy()

    res = {"model": args.model, "layer_idx": args.layer_idx, "abstain_k": args.abstain_k,
           "R": R, "sigma": s2 ** 0.5}

    print(f"\n{'=' * 92}\n1. The two populations the threshold sits between (k = {args.abstain_k} geodesic radius)"
          f"\n{'=' * 92}")
    print(f"{'population':<34}{'median':>10}{'p90':>10}{'p98':>10}{'p99.5':>10}{'max':>10}")
    def qline(name, v):
        print(f"{name:<34}" + "".join(f"{np.quantile(v, p):>10.4f}" for p in (0.5, 0.9, 0.98, 0.995))
              + f"{v.max():>10.4f}")
    qline("contrastive negatives (Alg. 1)", rho_train)
    for t in tasks:
        qline(f"{t} prompts" + ("  <- in-distribution" if t == ID_TASK else ""), rho[t])
    res["radius_quantiles"] = {
        "train_neg": {str(p): float(np.quantile(rho_train, p)) for p in (0.5, 0.9, 0.98, 0.995)},
        **{t: {str(p): float(np.quantile(rho[t], p)) for p in (0.5, 0.9, 0.98, 0.995)} for t in tasks},
    }
    lo, hi = np.median(rho[ID_TASK]), np.median(rho["gsm8k"])
    res["id_ood_relative_gap"] = float((hi - lo) / lo)
    print(f"\nmedian in-distribution prompt radius {lo:.4f} vs GSM8K {hi:.4f}: "
          f"a relative gap of {(hi - lo) / lo * 100:.1f}% in {pos.shape[1]} dimensions.")
    print(f"the shipped threshold is read off the top row, the queries all come from the rows below it.")

    # --- realized coverage of the shipped rule
    print(f"\n{'=' * 92}\n2. Shipped rule: rho_ref = percentile of the contrastive radius, "
          f"gate = 1/(1+(rho/rho_ref)^200)\n{'=' * 92}")
    print(f"{'nominal percentile':<22}{'rho_ref':>9}" + "".join(f"{t[:9]:>20}" for t in tasks))
    res["shipped"] = {}
    for p in _PCTS:
        ref = float(np.quantile(rho_train, p))
        row = {}
        for t in tasks:
            g = 1.0 / (1.0 + (rho[t] / ref) ** 200.0)
            row[t] = summarize(g)
        res["shipped"][str(p)] = {"rho_ref": ref, **{t: row[t] for t in tasks}}
        print(f"{p:<22g}{ref:>9.4f}" + "".join(f"{row[t][0]:>12.3f} ({row[t][1]:.2f})" for t in tasks))
    print("cell = mean gate (fraction of queries left unattenuated, gate > 0.5). "
          "gate 1 = steer fully, 0 = abstain.")

    # --- realized coverage of the recalibrated rule
    print(f"\n{'=' * 92}\n3. Recalibrated: threshold from held-out in-distribution *prompt* activations, "
          f"gate from the quantile rank\n{'=' * 92}")
    for tag, score in (("k-NN radius, k = %d" % args.abstain_k, rho),
                       ("bridge marginal -log p_0 at b = sigma^2/256", marg)):
        print(f"\nstatistic: {tag}")
        print(f"{'nominal ID coverage':<22}{'':>9}" + "".join(f"{t[:9]:>20}" for t in tasks))
        key = "recal_knn" if score is rho else "recal_marginal"
        res[key] = {}
        ref_sorted = np.sort(score[ID_TASK])
        for p in _PCTS:
            row = {}
            for t in tasks:
                rank = np.searchsorted(ref_sorted, score[t]) / len(ref_sorted)
                g = np.clip((1.0 - rank) / (1.0 - p), 0.0, 1.0)
                row[t] = summarize(g)
            res[key][str(p)] = {t: row[t] for t in tasks}
            print(f"{p:<22g}{'':>9}" + "".join(f"{row[t][0]:>12.3f} ({row[t][1]:.2f})" for t in tasks))

    out = get_project_dir() / "results" / "analysis" / f"q1-{args.model}-l{args.layer_idx}-gate-calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nsaved {out}")
