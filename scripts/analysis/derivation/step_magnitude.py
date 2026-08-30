"""How far does each rung of the Q6 ladder actually move a query, in distribution and out?

The point of the abstention gate is that the drift of Eq. (13) is a ratio of the two potentials,
in which the kernel decay in the distance to the data cancels, so the strictly derived update has
no way to know it is off support. This measures that directly: the geodesic arc travelled from
q_0 to q_K, in radians, on TruthfulQA queries (in distribution) and on four OOD query sets, for
each rung. No language model is involved -- only the cached query activations at the steering site.

    uv run python -u scripts/analysis/derivation/step_magnitude.py -m Llama3.1-8B-Base -l 13 -T 0.65
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

_BASE = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10)

RUNGS = {
    "1 strict Eq.13 (plain, fixed, raw, no gates)":
        dict(potentials="plain", bandwidth="fixed", step_mode="raw", drift="gradient",
             vmf_kappa=None, abstain_percentile=1.0),
    "2 + unit-normalized step":
        dict(potentials="plain", bandwidth="fixed", step_mode="unit", drift="gradient",
             vmf_kappa=None, abstain_percentile=1.0),
    "3 + vMF strength gate":
        dict(potentials="plain", bandwidth="fixed", step_mode="unit", drift="gradient",
             vmf_kappa=20, abstain_percentile=1.0),
    "4 + abstention gate (exact-gradient COBRAS)":
        dict(potentials="plain", bandwidth="fixed", step_mode="unit", drift="gradient",
             vmf_kappa=20, abstain_percentile=0.534),
    "5 shipped COBRAS (extended, adaptive)":
        dict(potentials="extended", bandwidth="adaptive", step_mode="unit", drift="centroid",
             vmf_kappa=20, abstain_percentile=0.534),
}


def arc(a: torch.Tensor, b: torch.Tensor, R: float) -> torch.Tensor:
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return torch.acos((a * b).sum(-1).clamp(-1 + _EPS, 1 - _EPS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("-T", "--T", type=float, default=0.65)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    qdir = get_project_dir() / "data" / "query_activations" / args.model
    sets = ["truthfulqa", "gsm8k", "mmlu", "nq", "triviaqa"]
    X = {s: torch.load(qdir / f"{s}_layer{args.layer_idx}.pt", weights_only=True).float().to(args.device)
         for s in sets}
    pos, neg = load_tqa_gen_data_all_splits(args.model, args.layer_idx)
    pos, neg = pos.float().to(args.device), neg.float().to(args.device)

    res = {"model": args.model, "layer_idx": args.layer_idx, "T": args.T,
           "n": {s: int(len(v)) for s, v in X.items()}}
    print(f"\n=== {args.model} l{args.layer_idx}  T={args.T}  "
          + "  ".join(f"{s}:{len(v)}" for s, v in X.items()))
    hdr = f"{'rung':<46}" + "".join(f"{s[:9]:>11}" for s in sets) + f"{'OOD/ID':>9}"
    print(hdr)
    for name, kw in RUNGS.items():
        m = COBRAS(**_BASE, **kw).fit(pos, neg, ref_X=X["truthfulqa"])
        med = {}
        for s in sets:
            a = []
            for i in range(0, len(X[s]), args.chunk):  # the extended potentials build [B,N-,N+]
                m.reset_gate()
                xi = X[s][i:i + args.chunk]
                a.append(arc(xi, m.steer(xi, T=args.T), m.R))
            med[s] = float(torch.cat(a).median())
        ood = sum(med[s] for s in sets[1:]) / 4.0
        res[name] = dict(median_arc_rad=med, ood_over_id=ood / max(med["truthfulqa"], 1e-12))
        print(f"{name:<46}" + "".join(f"{med[s]:>11.4f}" for s in sets)
              + f"{ood / max(med['truthfulqa'], 1e-12):>9.3f}")

    out_p = (get_project_dir() / "results" / "analysis" / "q6"
             / f"q6-{args.model}-l{args.layer_idx}-step-magnitude-T{args.T}.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nsaved {out_p}")
