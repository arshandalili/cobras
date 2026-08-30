"""Is the abstention rule intrinsic to the bridge, and at what bandwidth?

The probability-flow drift is the *ratio* of the two Schroedinger potentials, in which their
common kernel decay in the distance to the data cancels exactly. The decay survives only in the
*product*, which is the bridge's own time marginal p_0 = phi_hat * psi_hat -- the one factor of
the SB solution the drift never reads. This script asks whether thresholding that product
recovers the k-NN abstention rule of Alg. 1, and shows that the answer depends entirely on the
bandwidth at which the marginal is evaluated:

    s_b(x) := -2b log phi_hat_b(x)   is   mean_j d(x,h_j^-)^2 + const   at b = sigma^2
                                     and  min_j  d(x,h_j^-)^2 + const   as b -> 0.

So the shipped statistic is not external to the formulation: it is the small-bandwidth end of a
one-parameter family the bridge itself defines. The drift uses the other end.

    uv run python -u scripts/analysis/ood_gate/sb_marginal_gate.py -m Llama3.1-8B-Base -l 13
"""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import torch

from cobras.steer import COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data_all_splits

_EPS = 1e-7
ID_TASK = "truthfulqa"
_COBRAS_KWARGS = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
                      vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0)
_SCALES = [1.0, 1 / 4, 1 / 16, 1 / 64, 1 / 256, 1 / 1024]
_KS = [1, 2, 4, 8, 32, 128]


def auroc_row(sig: dict[str, np.ndarray], higher_is_id: bool) -> dict:
    """AUROC separating the in-distribution set from each OOD set."""
    out = {}
    for task, v in sig.items():
        if task == ID_TASK:
            continue
        y = np.r_[np.ones(len(sig[ID_TASK])), np.zeros(len(v))]
        s = np.r_[sig[ID_TASK], v]
        out[task] = float(roc_auc_score(y, s if higher_is_id else -s))
    out["mean"] = float(np.mean([v for k, v in out.items()]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=128)
    args = ap.parse_args()

    pos, neg = load_tqa_gen_data_all_splits(args.model, args.layer_idx)
    m = COBRAS(**_COBRAS_KWARGS).fit(pos.float().to(args.device), neg.float().to(args.device))
    R, s2, Nn = m.R, m.sigma2, m.h_neg.shape[0]
    log_phi_range = float(m.log_phi.max() - m.log_phi.min())

    act_dir = get_project_dir() / "data" / "query_activations" / args.model
    Q = {p.stem.replace(f"_layer{args.layer_idx}", ""):
         torch.load(p, weights_only=True).float().to(args.device)
         for p in sorted(act_dir.glob(f"*_layer{args.layer_idx}.pt"))}
    Q = {k: v for k, v in Q.items() if not k.startswith("truthfulqa_split")}
    assert ID_TASK in Q, f"missing {ID_TASK} query activations in {act_dir}"
    tasks = [ID_TASK] + [t for t in Q if t != ID_TASK]
    print(f"{args.model} l{args.layer_idx}: R={R:.3f} sigma={s2**0.5:.3f} N-={Nn}; "
          + ", ".join(f"{t}({len(Q[t])})" for t in tasks))

    # --- per-query statistics, computed once
    stats = {}
    with torch.no_grad():
        for t in tasks:
            X = Q[t]
            acc = {}
            for i in range(0, len(X), args.chunk):
                x = X[i:i + args.chunk]
                q = x * (R / x.norm(dim=-1, keepdim=True))
                d2n, d2p = m._geo_sq(q, m.h_neg), m._geo_sq(q, m.h_pos)
                cur = {f"knn{k}": d2n.topk(k, dim=1, largest=False).values[:, -1].sqrt() for k in _KS}
                cur["min_d2"] = d2n.min(1).values
                cur["mean_d2"] = d2n.mean(1)
                for c in _SCALES:
                    b = c * s2
                    lphi = torch.logsumexp(m.log_phi.unsqueeze(0) - d2n / (2 * b), -1)
                    lpsi = torch.logsumexp(m.log_psi.unsqueeze(0) - d2p / (2 * b), -1)
                    cur[f"s_plain@{c:g}"] = lphi + lpsi
                    cur[f"logphi@{c:g}"] = lphi
                # the Eq. (15)-(16) extension. `abstain_signal="density"` forces the fixed
                # bandwidth; the drift itself runs on the adaptive one, so record both.
                keep = m.bandwidth
                for bw, tag in (("fixed", "s_ext_fixed"), ("adaptive", "s_ext_adaptive")):
                    m.bandwidth = bw
                    cur[tag] = (m._weighted_centroid(q, m.h_neg, m._query_log_phi(q))[1]
                                + m._weighted_centroid(q, m.h_pos, m._query_log_psi(q))[1])
                m.bandwidth = keep
                for k, v in cur.items():
                    acc.setdefault(k, []).append(v.cpu())
            stats[t] = {k: torch.cat(v).numpy() for k, v in acc.items()}

    res = {"model": args.model, "layer_idx": args.layer_idx, "R": R, "sigma": s2 ** 0.5,
           "N_neg": int(Nn), "log_phi_range": log_phi_range, "auroc": {}}
    oods = [t for t in tasks if t != ID_TASK]

    print(f"\n{'='*100}\nAUROC, in-distribution vs OOD. The marginal is a one-parameter family; b = sigma^2 is "
          f"what the drift uses.\n{'='*100}")
    print(f"{'gate statistic':<28}" + "".join(f"{t[:9]:>11}" for t in oods) + f"{'mean':>9}")
    rows = [(f"s_plain@{c:g}", f"log p_0 at b = sigma^2 x {c:g}", True) for c in _SCALES]
    rows += [("s_ext_fixed", "log p_0, Eq.(15-16) ext, b=sigma^2", True),
             ("s_ext_adaptive", "log p_0, Eq.(15-16) ext, adaptive b", True)]
    rows += [(f"knn{k}", f"k-NN distance to D-, k = {k}", False) for k in _KS]
    for key, label, hi in rows:
        r = auroc_row({t: stats[t][key] for t in tasks}, hi)
        res["auroc"][key] = r
        print(f"{label:<28}" + "".join(f"{r[t]:>11.3f}" for t in oods) + f"{r['mean']:>9.3f}")

    # --- rank agreement between the shipped statistic and the marginal
    pool = {k: np.concatenate([stats[t][k] for t in tasks]) for k in stats[ID_TASK]}
    res["spearman_vs_knn32"] = {c: float(spearmanr(pool["knn32"], -pool[f"s_plain@{c:g}"]).statistic)
                                for c in _SCALES}
    for t in ("s_ext_fixed", "s_ext_adaptive"):
        res["spearman_vs_knn32"][t] = float(spearmanr(pool["knn32"], -pool[t]).statistic)
    print(f"\nSpearman(k-NN_32 radius, -log p_0) over all {len(pool['knn32'])} pooled activations:")
    print("  " + "  ".join(f"b=s2x{c:g}: {res['spearman_vs_knn32'][c]:+.3f}" for c in _SCALES)
          + f"  | ext(fixed): {res['spearman_vs_knn32']['s_ext_fixed']:+.3f}"
          + f"  ext(adaptive): {res['spearman_vs_knn32']['s_ext_adaptive']:+.3f}")

    # --- what -2b log phi_hat actually equals, as a function of b
    print(f"\n{'='*100}\nWhat -2b log phi_hat_b(x) equals. Bracket = 2b(log N- + range log phi), the width of the\n"
          f"two-sided logsumexp bound around min_j d^2; it is only informative once it is small\n"
          f"relative to sd(min_j d^2) = {pool['min_d2'].std():.2f}.\n{'='*100}")
    print(f"{'b / sigma^2':>12}{'b':>10}{'bracket':>11}{'sd(gap to min)':>16}{'sd(gap to mean)':>17}"
          f"{'r(.,min)':>10}{'r(.,mean)':>11}")
    res["lemma"] = {}
    for c in _SCALES:
        b = c * s2
        lhs = -2 * b * pool[f"logphi@{c:g}"]
        e = dict(bracket=float(2 * b * (np.log(Nn) + log_phi_range)),
                 sd_gap_min=float((lhs - pool["min_d2"]).std()),
                 sd_gap_mean=float((lhs - pool["mean_d2"]).std()),
                 r_min=float(np.corrcoef(lhs, pool["min_d2"])[0, 1]),
                 r_mean=float(np.corrcoef(lhs, pool["mean_d2"])[0, 1]))
        res["lemma"][f"{c:g}"] = e
        print(f"{c:>12g}{b:>10.3f}{e['bracket']:>11.1f}{e['sd_gap_min']:>16.3f}"
              f"{e['sd_gap_mean']:>17.3f}{e['r_min']:>10.4f}{e['r_mean']:>11.4f}")
    res["sd_min_d2"] = float(pool["min_d2"].std())
    res["sd_mean_d2"] = float(pool["mean_d2"].std())

    out = get_project_dir() / "results" / "analysis" / f"{args.model}-l{args.layer_idx}-sb-marginal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\n saved {out}")
