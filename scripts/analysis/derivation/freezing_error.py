"""How much do Eq. (27) and Eq. (31) lose by freezing sigma^2(x) and the extended potentials?

App. A.5 differentiates log psi_hat while holding two x-dependent objects constant: the extended
potentials psi_tilde_i(x) of Eq. (16) (stated at line 467) and, implicitly, the bandwidth --
Eqs. (29)-(30) write sigma^2(x) but Eqs. (27) and (31) pull 1/sigma^2 out of the sum as a
constant. Both are approximations; this measures them by comparing the implemented field against
the exact Riemannian gradient of the same scalar, obtained by autograd.

    uv run python -u scripts/analysis/derivation/freezing_error.py -m Llama3.1-8B-Base -l 13
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

# the ablation switches this script sweeps (potentials=, drift=, bandwidth=, step_mode=,
# uniform_weights=, abstain_signal=) live on AblationCOBRAS; with defaults it is COBRAS
from cobras.steer import AblationCOBRAS as COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data_all_splits

_EPS = 1e-7
_COBRAS_KWARGS = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
                      vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0)


def log_ratio_plain(m: COBRAS, x: torch.Tensor, bandwidth: str, freeze_bandwidth: bool) -> torch.Tensor:
    """log(psi_hat/phi_hat) in the plain form of Eqs. (10)-(11), where the Sinkhorn potentials
    really are constants in x, so no analogue of line 467's approximation is needed."""
    def geo2(H):
        cos_t = ((x @ H.T) / (m.R ** 2)).clamp(-1.0 + _EPS, 1.0 - _EPS)
        return (m.R * torch.acos(cos_t)).pow(2)

    d2p, d2n = geo2(m.h_pos), geo2(m.h_neg)

    def bw(d2):
        if bandwidth == "fixed":
            return torch.as_tensor(m.sigma2, device=x.device, dtype=x.dtype)
        b = d2.max(dim=-1, keepdim=True).values.clamp(min=(m.R * 1e-3) ** 2)
        return b.detach() if freeze_bandwidth else b

    return (torch.logsumexp(m.log_psi[None] - d2p / (2 * bw(d2p)), -1)
            - torch.logsumexp(m.log_phi[None] - d2n / (2 * bw(d2n)), -1))


def log_ratio(m: COBRAS, x: torch.Tensor, bandwidth: str, freeze_potentials: bool,
              freeze_bandwidth: bool) -> torch.Tensor:
    """log psi_hat(x) - log phi_hat(x), with each x-dependence optionally detached so that
    autograd reproduces exactly the approximation made in the appendix."""
    def geo2(H):
        cos_t = ((x @ H.T) / (m.R ** 2)).clamp(-1.0 + _EPS, 1.0 - _EPS)
        return (m.R * torch.acos(cos_t)).pow(2)

    d2p, d2n = geo2(m.h_pos), geo2(m.h_neg)

    def bw(d2):
        if bandwidth == "fixed":
            return torch.as_tensor(m.sigma2, device=x.device, dtype=x.dtype)
        b = d2.max(dim=-1, keepdim=True).values.clamp(min=(m.R * 1e-3) ** 2)
        return b.detach() if freeze_bandwidth else b

    bp, bn = bw(d2p), bw(d2n)
    # Eqs. (15)-(16): the extension of the potentials to the query
    lpsi_t = torch.logsumexp((m.log_phi[None] - d2n / (2 * bn)).unsqueeze(2) - m.cost[None], dim=1)
    lphi_t = torch.logsumexp((m.log_psi[None] - d2p / (2 * bp)).unsqueeze(1) - m.cost[None], dim=2)
    if freeze_potentials:
        lpsi_t, lphi_t = lpsi_t.detach(), lphi_t.detach()
    return (torch.logsumexp(lpsi_t - d2p / (2 * bp), -1)
            - torch.logsumexp(lphi_t - d2n / (2 * bn), -1))


def riemannian_grad(m: COBRAS, X: torch.Tensor, fn=log_ratio, **kw) -> torch.Tensor:
    x = X.clone().requires_grad_(True)
    g = torch.autograd.grad(fn(m, x, **kw).sum(), x)[0]
    return g - (g * x).sum(-1, keepdim=True) / (m.R ** 2) * x   # project onto T_x S


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("-n", "--n_queries", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    pos, neg = load_tqa_gen_data_all_splits(args.model, args.layer_idx)
    m = COBRAS(**_COBRAS_KWARGS).fit(pos.float().to(args.device), neg.float().to(args.device))

    X = torch.load(get_project_dir() / "data" / "query_activations" / args.model
                   / f"truthfulqa_layer{args.layer_idx}.pt", weights_only=True).float().to(args.device)
    X = X[:args.n_queries]
    q0 = X * (m.R / X.norm(dim=-1, keepdim=True))

    res = {"model": args.model, "layer_idx": args.layer_idx, "n_queries": int(len(q0))}
    for bandwidth in ("adaptive", "fixed"):
        m.bandwidth, m.drift = bandwidth, "gradient"
        impl = torch.cat([m._field(q0[i:i + args.chunk])[0] for i in range(0, len(q0), args.chunk)])

        variants = {}
        for tag, fp, fb in (("exact", False, False),
                            ("freeze potentials only", True, False),
                            ("freeze bandwidth only", False, True),
                            ("freeze both (App. A.5)", True, True)):
            g = torch.cat([riemannian_grad(m, q0[i:i + args.chunk], bandwidth=bandwidth,
                                           freeze_potentials=fp, freeze_bandwidth=fb)
                           for i in range(0, len(q0), args.chunk)])
            variants[tag] = g
        for tag, fb in (("PLAIN Eqs.(10-11), exact", False), ("PLAIN, freeze bandwidth", True)):
            variants[tag] = torch.cat([
                riemannian_grad(m, q0[i:i + args.chunk], fn=log_ratio_plain,
                                bandwidth=bandwidth, freeze_bandwidth=fb)
                for i in range(0, len(q0), args.chunk)])

        variants["implemented _field (Eq. 31)"] = impl
        print(f"\n=== bandwidth = {bandwidth} " + "=" * 60)
        keys = list(variants)
        short = {k: k[:26] for k in keys}
        print("pairwise mean cosine between the candidate drifts")
        print(f"{'':<28}" + "".join(f"{i:>9}" for i in range(len(keys))))
        for a, ka in enumerate(keys):
            row = "".join(f"{float(F.cosine_similarity(variants[ka], variants[kb], dim=-1).mean()):>9.4f}"
                          for kb in keys)
            print(f"{a}. {short[ka]:<25}" + row)
        print("relative magnitude |g| (median over queries):")
        for ka in keys:
            print(f"    {short[ka]:<28}{float(variants[ka].norm(dim=-1).median()):>12.6f}")

        ref = variants["exact"]
        print(f"{'gradient of log(psi_hat/phi_hat)':<28}{'cos to exact':>14}{'angle':>9}{'|g|/|exact|':>13}")
        rec = {}
        for tag, g in variants.items():
            c = F.cosine_similarity(g, ref, dim=-1)
            rec[tag] = dict(cos=float(c.mean()),
                            angle_deg=float(np.degrees(np.arccos(min(1.0, float(c.mean()))))),
                            norm_ratio=float((g.norm(dim=-1) / ref.norm(dim=-1)).mean()))
            print(f"{tag:<28}{rec[tag]['cos']:>14.5f}{rec[tag]['angle_deg']:>9.2f}"
                  f"{rec[tag]['norm_ratio']:>13.4f}")
        c = F.cosine_similarity(impl, ref, dim=-1)
        rec["implemented _field"] = dict(cos=float(c.mean()),
                                         angle_deg=float(np.degrees(np.arccos(min(1.0, float(c.mean()))))),
                                         norm_ratio=float((impl.norm(dim=-1) / ref.norm(dim=-1)).mean()))
        print(f"{'implemented _field (Eq. 31)':<28}{rec['implemented _field']['cos']:>14.5f}"
              f"{rec['implemented _field']['angle_deg']:>9.2f}{rec['implemented _field']['norm_ratio']:>13.4f}")
        # how much of the steering step would change if the exact gradient were used
        cs = F.cosine_similarity(impl, ref, dim=-1)
        print(f"  per-query cos(implemented, exact): min {cs.min():.5f}  median {cs.median():.5f}  "
              f"max {cs.max():.5f}")
        res[bandwidth] = rec
    m.bandwidth, m.drift = "adaptive", "centroid"

    out = get_project_dir() / "results" / "analysis" / f"{args.model}-l{args.layer_idx}-exact-gradient.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\n saved {out}")
