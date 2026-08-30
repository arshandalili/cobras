"""Q3 (Reviewer EJdD): COBRAS depends on the activations only through their directions.

Reading the implementation, every step of COBRAS factors through h/||h||:

  fit    R = mean_i ||h_i||, then h <- R h/||h||          (_cobras.py:89-92)
  steer  p0 = R X/||X||, ... , return q_K * ||X||/R       (_cobras.py:302, 336)

so COBRAS(X) = ||X|| * f( X/||X|| ; {h_i/||h_i||} ), and the norms enter only as a
pass-through scale. Combined with the scale invariance in geometry/scale_check.py (the value of R
does not matter either), this means the *distribution of the norms cannot affect the output at
all*. This script tests that claim the hard way: replace every norm in the contrastive set and
in the queries by an arbitrary random value and check the steered output is unchanged.

    ./.venv/bin/python -u scripts/analysis/geometry/radial_invariance.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cobras.steer import COBRAS
from cobras.utils.data import load_tqa_gen_data

from cobras.steer import EuclideanCOBRAS

KW = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1.0e-3, epsilon=0.0, max_iters=10,
          vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0, abstain_k=32,
          abstain_sharpness=50.0)


def randomise_norms(X, lo, hi, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = (torch.rand(X.size(0), 1, generator=g) * (hi - lo) + lo).to(X.device)
    return X / X.norm(dim=-1, keepdim=True).clamp(min=1e-8) * r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer", type=int, default=13)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_query", type=int, default=64)
    args = ap.parse_args()

    pos, neg = load_tqa_gen_data(args.model, args.layer, 0)
    te_pos, te_neg = load_tqa_gen_data(args.model, args.layer, 1)
    pos, neg = pos.float().to(args.device), neg.float().to(args.device)
    Q = torch.cat([te_pos, te_neg], 0).float().to(args.device)
    Q = Q[torch.randperm(Q.size(0), generator=torch.Generator().manual_seed(0))[: args.n_query]]

    base = COBRAS(**KW).fit(pos, neg)
    Y0 = base.steer(Q, T=0.5)
    n0 = torch.cat([pos, neg], 0).norm(dim=-1)
    print(f"{args.model} L{args.layer}: original CoV of ||h|| = {n0.std()/n0.mean():.5f}, "
          f"R = {base.R:.4f}")

    for name, lo, hi in [("uniform in [0.5R, 2R]", 0.5, 2.0),
                         ("uniform in [0.1R, 10R]", 0.1, 10.0),
                         ("all norms equal to R", 1.0, 1.0)]:
        P = randomise_norms(pos, lo * base.R, hi * base.R, 1)
        N = randomise_norms(neg, lo * base.R, hi * base.R, 2)
        X = randomise_norms(Q, lo * base.R, hi * base.R, 3)
        nn = torch.cat([P, N], 0).norm(dim=-1)
        m = COBRAS(**KW).fit(P, N)
        Y = m.steer(X, T=0.5)
        # compare directions; the output norm is by construction the input norm
        cos = torch.nn.functional.cosine_similarity(Y, Y0, dim=-1)
        print(f"  COBRAS  {name:24s} CoV={nn.std()/nn.mean():.5f} R={m.R:8.4f}  "
              f"min cos(steered dir, original) = {cos.min():.8f}")

        me = EuclideanCOBRAS(**KW).fit(P, N)
        Ye = me.steer(X, T=0.5)
        base_e = EuclideanCOBRAS(**KW).fit(pos, neg)
        Ye0 = base_e.steer(Q, T=0.5)
        cose = torch.nn.functional.cosine_similarity(Ye, Ye0, dim=-1)
        print(f"  Euclid  {name:24s} {'':30s}min cos = {cose.min():.8f}")
