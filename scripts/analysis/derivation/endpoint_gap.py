"""How far apart do the exact-gradient update and the shipped update leave a query?

The autograd table says the two step *directions* agree to about 5 degrees, and the step-magnitude
table says the two travel the same arc. This closes the loop by integrating both flows for the
full K = 10 steps and measuring the geodesic distance between the two endpoints, as a fraction of
the distance either one travelled. Query activations only, no language model.

    uv run python -u scripts/analysis/derivation/endpoint_gap.py -m Llama3.1-8B-Base -l 13 -T 0.65
"""

import argparse
import json

import torch

# the ablation switches this script sweeps (potentials=, drift=, bandwidth=, step_mode=,
# uniform_weights=, abstain_signal=) live on AblationCOBRAS; with defaults it is COBRAS
from cobras.steer import AblationCOBRAS as COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data_all_splits

_EPS = 1e-7
_BASE = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
             vmf_kappa=20, abstain_percentile=0.534)
EXACT = dict(potentials="plain", bandwidth="fixed", step_mode="unit", drift="gradient")
SHIPPED = dict(potentials="extended", bandwidth="adaptive", step_mode="unit", drift="centroid")


def arc(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return torch.acos((a * b).sum(-1).clamp(-1 + _EPS, 1 - _EPS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("-T", "--T", type=float, default=0.65)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    qdir = get_project_dir() / "data" / "query_activations" / args.model
    sets = ["truthfulqa", "gsm8k", "mmlu", "nq", "triviaqa"]
    X = {s: torch.load(qdir / f"{s}_layer{args.layer_idx}.pt", weights_only=True).float().to(args.device)
         for s in sets}
    pos, neg = load_tqa_gen_data_all_splits(args.model, args.layer_idx)
    pos, neg = pos.float().to(args.device), neg.float().to(args.device)

    me = COBRAS(**_BASE, **EXACT).fit(pos, neg, ref_X=X["truthfulqa"])
    ms = COBRAS(**_BASE, **SHIPPED).fit(pos, neg, ref_X=X["truthfulqa"])

    res = {"model": args.model, "layer_idx": args.layer_idx, "T": args.T}
    print(f"\n=== {args.model} l{args.layer_idx}  T={args.T}   exact-gradient vs shipped endpoints")
    print(f"{'set':<12}{'arc travelled':>15}{'endpoint gap':>15}{'gap/arc':>10}")
    for s in sets:
        trav, gap = [], []
        for i in range(0, len(X[s]), args.chunk):
            xi = X[s][i:i + args.chunk]
            me.reset_gate(); ms.reset_gate()
            ye, ys = me.steer(xi, T=args.T), ms.steer(xi, T=args.T)
            trav.append(arc(xi, ye))
            gap.append(arc(ye, ys))
        trav, gap = torch.cat(trav), torch.cat(gap)
        res[s] = dict(n=len(trav), arc=float(trav.median()), gap=float(gap.median()),
                      ratio=float((gap / trav.clamp(min=1e-9)).median()))
        print(f"{s:<12}{res[s]['arc']:>15.4f}{res[s]['gap']:>15.4f}{res[s]['ratio']:>10.4f}")

    out = (get_project_dir() / "results" / "analysis" / "q6"
           / f"q6-{args.model}-l{args.layer_idx}-endpoint-gap-T{args.T}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nsaved {out}")
