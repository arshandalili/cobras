"""Q3 (Reviewer EJdD): isolate norm dispersion as the cause of the spherical/Euclidean gap.

The layer sweep varies the concentration by changing layer, which also changes everything else
about the representation. This is the controlled version: take one layer, keep every direction
fixed, and dial the norm dispersion continuously by

    ||h||  ->  R + alpha * (||h|| - R),      alpha in [0, 1]

so alpha = 0 gives CoV exactly 0 (the assumption holds perfectly) and alpha = 1 gives the real
data. Refit COBRAS and EuclideanCOBRAS at each alpha and measure how far apart their updates
are. If norm dispersion is what makes the two geometries disagree, the disagreement must grow
with alpha.

    ./.venv/bin/python -u scripts/analysis/geometry/dispersion_sweep.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cobras.steer import COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data

from cobras.steer import EuclideanCOBRAS

KW = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1.0e-3, epsilon=0.0, max_iters=10,
          vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0, abstain_k=32,
          abstain_sharpness=50.0)
T_STEER = 0.5
OUT = get_project_dir() / "results" / "analysis" / "q3_geometry"


def retune(X: torch.Tensor, R: float, alpha: float) -> torch.Tensor:
    """Keep directions, shrink the norm spread toward R by (1 - alpha)."""
    n = X.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return X * ((R + alpha * (n - R)) / n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer", type=int, default=13)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_query", type=int, default=64)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    args = ap.parse_args()

    dev = args.device
    pos, neg = load_tqa_gen_data(args.model, args.layer, 0)
    te_pos, te_neg = load_tqa_gen_data(args.model, args.layer, 1)
    pos, neg = pos.to(torch.float32).to(dev), neg.to(torch.float32).to(dev)
    Q = torch.cat([te_pos, te_neg], 0).to(torch.float32).to(dev)
    g = torch.Generator().manual_seed(0)
    Q = Q[torch.randperm(Q.size(0), generator=g)[: args.n_query]]

    R0 = float(torch.cat([pos, neg], 0).norm(dim=-1).mean().item())
    print(f"{args.model} L{args.layer}  R={R0:.4f}  N+={len(pos)} N-={len(neg)} "
          f"n_query={len(Q)}  device={dev}", flush=True)

    rows = []
    for a in args.alphas:
        P, N, X = retune(pos, R0, a), retune(neg, R0, a), retune(Q, R0, a)
        nn = torch.cat([P, N], 0).norm(dim=-1)
        cov = float((nn.std() / nn.mean()).item())

        s_sph = COBRAS(**KW).fit(P, N)
        s_euc = EuclideanCOBRAS(**KW).fit(P, N)
        Y_sph, Y_euc = s_sph.steer(X, T=T_STEER), s_euc.steer(X, T=T_STEER)
        d_sph, d_euc = Y_sph - X, Y_euc - X
        xn = X.norm(dim=-1)

        row = {
            "model": args.model, "layer": args.layer, "alpha": a, "cov": cov,
            "cos_updates": float(torch.nn.functional.cosine_similarity(
                d_sph, d_euc, dim=-1).mean().item()),
            "cos_updates_p05": float(np.percentile(
                torch.nn.functional.cosine_similarity(d_sph, d_euc, dim=-1).cpu().numpy(), 5)),
            "rel_disp_sph": float((d_sph.norm(dim=-1) / xn).mean().item()),
            "rel_disp_euc": float((d_euc.norm(dim=-1) / xn).mean().item()),
            "mu_T_cos": float((s_sph.mu_T * s_euc.mu_T).sum().item()),
        }
        rows.append(row)
        print(f"  alpha={a:<5.2f} CoV={cov:.5f}  cos(updates)={row['cos_updates']:.5f} "
              f"(p05 {row['cos_updates_p05']:.5f})  reldisp sph/euc="
              f"{row['rel_disp_sph']:.4f}/{row['rel_disp_euc']:.4f}  "
              f"mu_T cos={row['mu_T_cos']:.5f}", flush=True)
        del s_sph, s_euc, P, N, X, Y_sph, Y_euc

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"q3_dispersion_sweep_{args.model}_L{args.layer}.json", "w") as f:
        json.dump(rows, f, indent=2)
    c = np.array([r["cov"] for r in rows])
    u = np.array([r["cos_updates"] for r in rows])
    print(f"\nPearson(CoV, cos_updates) = {np.corrcoef(c, u)[0, 1]:+.4f}")
    print(f"wrote {OUT / f'q3_dispersion_sweep_{args.model}_L{args.layer}.json'}")
