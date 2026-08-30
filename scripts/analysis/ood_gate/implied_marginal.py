"""Every steering method posits a pair of activation densities, so every one has an implied
marginal p_+ p_-.  This measures how well each of them detects that a query lies off the
contrastive support, so that the claim in the Q1 reply is a measurement rather than an
assertion.

    CAA / RepE / ITI   v = mu_+ - mu_-      <=>  N(mu_+, I) and N(mu_-, I)
                       implied marginal     log p_+ + log p_- = -(|x-mu_+|^2 + |x-mu_-|^2)/2 + c
    SphericalSteer     rotate toward mu     <=>  vMF(mu, kappa) against a uniform reference
                       implied marginal     kappa * cos(x, mu) + c
    ODESteer           fits log(p_+/p_-)    directly with a classifier: no marginal exists
    COBRAS             psi_hat * phi_hat    the Sinkhorn-weighted KDE product of Eqs. (10)-(11)

Also re-checks, at the paper's configuration, that the drift's ratio log psi_hat - log phi_hat
is constant across in- and out-of-distribution query sets while the product separates them.

    uv run python -u scripts/analysis/ood_gate/implied_marginal.py -m Llama3.1-8B-Base -l 13
"""

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

import torch

from cobras.steer import COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data_all_splits

ID_TASK = "truthfulqa"
_COBRAS_KWARGS = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
                      vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0)


def auroc_row(sig: dict[str, np.ndarray], oods: list[str]) -> dict:
    """AUROC separating the in-distribution set from each OOD set, higher score = more ID."""
    out = {}
    for t in oods:
        y = np.r_[np.ones(len(sig[ID_TASK])), np.zeros(len(sig[t]))]
        out[t] = float(roc_auc_score(y, np.r_[sig[ID_TASK], sig[t]]))
    out["mean"] = float(np.mean([out[t] for t in oods]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--b_scale", type=float, default=1 / 256)
    args = ap.parse_args()

    dev = args.device
    pos, neg = load_tqa_gen_data_all_splits(args.model, args.layer_idx)
    pos, neg = pos.float().to(dev), neg.float().to(dev)
    m = COBRAS(**_COBRAS_KWARGS).fit(pos, neg)
    R, s2 = m.R, m.sigma2

    # --- the density models the baselines posit, fit on the same contrastive data
    mu_pos_raw, mu_neg_raw = pos.mean(0), neg.mean(0)                  # CAA / RepE / ITI
    mu_sph = (m.h_pos.mean(0) - m.h_neg.mean(0))                       # SphericalSteer target
    mu_sph = mu_sph / mu_sph.norm()
    mu_pos_sph = m.h_pos.mean(0)
    mu_pos_sph = mu_pos_sph / mu_pos_sph.norm()

    act_dir = get_project_dir() / "data" / "query_activations" / args.model
    Q = {p.stem.replace(f"_layer{args.layer_idx}", ""):
         torch.load(p, weights_only=True).float().to(dev)
         for p in sorted(act_dir.glob(f"*_layer{args.layer_idx}.pt"))}
    Q = {k: v for k, v in Q.items() if not k.startswith("truthfulqa_split")}
    tasks = [ID_TASK] + [t for t in Q if t != ID_TASK]
    oods = [t for t in tasks if t != ID_TASK]
    print(f"{args.model} l{args.layer_idx}: R={R:.3f} sigma={s2 ** 0.5:.3f}; "
          + ", ".join(f"{t}({len(Q[t])})" for t in tasks))

    stats: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for t in tasks:
            X, acc = Q[t], {}
            for i in range(0, len(X), args.chunk):
                x = X[i:i + args.chunk]
                q = x * (R / x.norm(dim=-1, keepdim=True))
                cur = {}
                # CAA / RepE / ITI: product of two unit-covariance Gaussians
                cur["gauss"] = -0.5 * ((x - mu_pos_raw).pow(2).sum(-1)
                                       + (x - mu_neg_raw).pow(2).sum(-1))
                # the same, on the sphere COBRAS works on, so the comparison is geometry-matched
                cur["gauss_sph"] = -0.5 * ((q - mu_pos_raw).pow(2).sum(-1)
                                           + (q - mu_neg_raw).pow(2).sum(-1))
                # SphericalSteer: vMF(mu, kappa) x uniform -> rank-1 in cos(x, mu)
                u = x / x.norm(dim=-1, keepdim=True)
                cur["vmf_diff"] = 20.0 * (u @ mu_sph)
                cur["vmf_pos"] = 20.0 * (u @ mu_pos_sph)
                # COBRAS: the Sinkhorn-weighted KDE product of Eqs. (10)-(11)
                d2p, d2n = m._geo_sq(q, m.h_pos), m._geo_sq(q, m.h_neg)
                for tag, b in (("cobras_b1", s2), (f"cobras_b{args.b_scale:g}", args.b_scale * s2)):
                    lpsi = torch.logsumexp(m.log_psi.unsqueeze(0) - d2p / (2 * b), -1)
                    lphi = torch.logsumexp(m.log_phi.unsqueeze(0) - d2n / (2 * b), -1)
                    cur[tag] = lpsi + lphi
                    cur[tag + "_ratio"] = lpsi - lphi
                for k, v in cur.items():
                    acc.setdefault(k, []).append(v.cpu())
            stats[t] = {k: torch.cat(v).numpy() for k, v in acc.items()}

    res = {"model": args.model, "layer_idx": args.layer_idx, "R": R, "sigma": s2 ** 0.5,
           "n_per_task": {t: int(len(Q[t])) for t in tasks}, "auroc": {}}

    print(f"\n{'=' * 96}\nAUROC of each method's own implied marginal, in-distribution vs OOD"
          f"\n{'=' * 96}")
    print(f"{'implied density model':<46}" + "".join(f"{t[:9]:>11}" for t in oods) + f"{'mean':>9}")
    rows = [
        ("gauss", "CAA/RepE/ITI  N(mu+,I) N(mu-,I), raw acts"),
        ("gauss_sph", "CAA/RepE/ITI  same, on the sphere"),
        ("vmf_pos", "SphericalSteer  vMF(mu+, 20) x Unif"),
        ("vmf_diff", "SphericalSteer  vMF(mu+-mu-, 20) x Unif"),
        ("cobras_b1", "COBRAS  psi_hat phi_hat at b = sigma^2"),
        (f"cobras_b{args.b_scale:g}", f"COBRAS  psi_hat phi_hat at b = sigma^2 x {args.b_scale:g}"),
    ]
    for key, label in rows:
        r = auroc_row({t: stats[t][key] for t in tasks}, oods)
        res["auroc"][key] = r
        print(f"{label:<46}" + "".join(f"{r[t]:>11.3f}" for t in oods) + f"{r['mean']:>9.3f}")
    print("ODESteer  parameterises log(p+/p-) only            "
          + "".join(f"{'--':>11}" for t in oods) + f"{'--':>9}")

    # --- the ratio the drift reads against the product it never reads
    print(f"\n{'=' * 96}\nPer-task means at b = sigma^2: the drift is the gradient of the ratio"
          f"\n{'=' * 96}")
    print(f"{'task':<14}{'mean log psi - log phi':>26}{'sd':>10}{'mean log psi + log phi':>26}{'sd':>10}")
    res["ratio_vs_product"] = {}
    for t in tasks:
        ra, pr = stats[t]["cobras_b1_ratio"], stats[t]["cobras_b1"]
        res["ratio_vs_product"][t] = dict(ratio_mean=float(ra.mean()), ratio_sd=float(ra.std()),
                                          product_mean=float(pr.mean()), product_sd=float(pr.std()))
        print(f"{t:<14}{ra.mean():>26.5f}{ra.std():>10.5f}{pr.mean():>26.3f}{pr.std():>10.3f}")
    allr = np.concatenate([stats[t]["cobras_b1_ratio"] for t in tasks])
    means = np.array([stats[t]["cobras_b1_ratio"].mean() for t in tasks])
    res["ratio_spread_across_tasks"] = float(means.max() - means.min())
    res["ratio_sd_pooled"] = float(allr.std())
    print(f"\nspread of the per-task mean ratio across all {len(tasks)} task sets: "
          f"{means.max() - means.min():.2e}  (pooled sd {allr.std():.2e})")
    r = auroc_row({t: stats[t]["cobras_b1_ratio"] for t in tasks}, oods)
    res["auroc"]["cobras_b1_ratio"] = r
    print("AUROC of the ratio: " + "  ".join(f"{t}={r[t]:.3f}" for t in oods) + f"  mean={r['mean']:.3f}")

    out = get_project_dir() / "results" / "analysis" / f"q1-{args.model}-l{args.layer_idx}-implied-marginal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nsaved {out}")
