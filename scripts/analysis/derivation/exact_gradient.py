"""Q6: is the implemented update the exact Riemannian gradient of the SB drift?

`scripts/analysis/derivation/freezing_error.py` prints a pairwise cosine matrix but only saves each
variant's agreement with one reference. The rows this asks for are the *matched* ones: for each
choice of potentials (extended Eqs. (15)-(16) vs plain Eqs. (10)-(11)) and bandwidth (the shipped
sigma^2(x) = max_i d^2 vs a constant sigma^2), compare the field COBRAS actually integrates
against the exact Riemannian gradient of the very scalar that choice defines, with nothing
detached. Imports the gradient definitions from exact_gradient.py so the two agree by construction.

    uv run python -u scripts/analysis/derivation/exact_gradient.py -m Llama3.1-8B-Base -l 13
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freezing_error import log_ratio, log_ratio_plain, riemannian_grad  # noqa: E402

# the ablation switches this script sweeps (potentials=, drift=, bandwidth=, step_mode=,
# uniform_weights=, abstain_signal=) live on AblationCOBRAS; with defaults it is COBRAS
from cobras.steer import AblationCOBRAS as COBRAS
from cobras.steer import COBRAS  # noqa: E402
from cobras.utils import get_project_dir  # noqa: E402
from cobras.utils.data import load_tqa_gen_data_all_splits  # noqa: E402

_COBRAS_KWARGS = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
                      vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0)


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Agreement of field `a` with reference field `b`, per query then averaged."""
    c = F.cosine_similarity(a, b, dim=-1)
    r = a.norm(dim=-1) / b.norm(dim=-1)
    return dict(cos=float(c.mean()), cos_min=float(c.min()), cos_max=float(c.max()),
                angle_deg=float(np.degrees(np.arccos(np.clip(float(c.mean()), -1.0, 1.0)))),
                ratio=float(r.mean()), ratio_median=float(r.median()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("-n", "--n_queries", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()

    # full fp32: the claim here is about machine-precision agreement
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    pos, neg = load_tqa_gen_data_all_splits(args.model, args.layer_idx)
    m = COBRAS(**_COBRAS_KWARGS).fit(pos.float().to(args.device), neg.float().to(args.device))

    X = torch.load(get_project_dir() / "data" / "query_activations" / args.model
                   / f"truthfulqa_layer{args.layer_idx}.pt", weights_only=True).float().to(args.device)
    q0 = X[:args.n_queries]
    q0 = q0 * (m.R / q0.norm(dim=-1, keepdim=True))
    n = len(q0)

    def field(potentials, bandwidth, drift):
        m.potentials, m.bandwidth, m.drift = potentials, bandwidth, drift
        return torch.cat([m._field(q0[i:i + args.chunk])[0] for i in range(0, n, args.chunk)])

    def grad(fn, bandwidth, **kw):
        return torch.cat([riemannian_grad(m, q0[i:i + args.chunk], fn=fn, bandwidth=bandwidth, **kw)
                          for i in range(0, n, args.chunk)])

    # exact Riemannian gradients of log(psi_hat/phi_hat), nothing detached
    g_plain_fixed = grad(log_ratio_plain, "fixed", freeze_bandwidth=False)
    g_plain_adapt = grad(log_ratio_plain, "adaptive", freeze_bandwidth=False)
    g_ext_fixed = grad(log_ratio, "fixed", freeze_potentials=False, freeze_bandwidth=False)
    g_ext_adapt = grad(log_ratio, "adaptive", freeze_potentials=False, freeze_bandwidth=False)

    rows = {
        # (label): (implemented field, exact gradient of the scalar that choice defines)
        "plain potentials, fixed bandwidth":
            (field("plain", "fixed", "gradient"), g_plain_fixed),
        "plain potentials, adaptive bandwidth (shipped sigma^2(x))":
            (field("plain", "adaptive", "gradient"), g_plain_adapt),
        "extended potentials, fixed bandwidth":
            (field("extended", "fixed", "gradient"), g_ext_fixed),
        "extended potentials, adaptive bandwidth (the paper's code path)":
            (field("extended", "adaptive", "gradient"), g_ext_adapt),
    }
    # cross-check: the extended field against the gradient of the *plain* scalar
    rows["extended field vs exact gradient of PLAIN, fixed bandwidth"] = (
        field("extended", "fixed", "gradient"), g_plain_fixed)
    # is the code doing exactly what App. A.5 says, i.e. freezing what line 467 says it freezes?
    rows["extended, fixed bw vs gradient with potentials frozen (line 467)"] = (
        field("extended", "fixed", "gradient"),
        grad(log_ratio, "fixed", freeze_potentials=True, freeze_bandwidth=False))
    rows["extended, adaptive bw vs gradient with potentials AND sigma^2(x) frozen (App. A.5)"] = (
        field("extended", "adaptive", "gradient"),
        grad(log_ratio, "adaptive", freeze_potentials=True, freeze_bandwidth=True))

    res = {"model": args.model, "layer_idx": args.layer_idx, "n_queries": n,
           "sigma2": float(m.sigma2), "R": float(m.R),
           "alpha_sigma": float(m.alpha_sigma),
           "one_over_1_plus_alpha": 1.0 / (1.0 + m.alpha_sigma)}
    print(f"\n=== {args.model} l{args.layer_idx}  n={n}  sigma^2={m.sigma2:.4f}  R={m.R:.4f}")
    print(f"{'implemented _field vs exact Riemannian gradient':<62}{'cos':>10}{'angle':>8}{'|impl|/|exact|':>16}")
    for k, (a, b) in rows.items():
        res[k] = stats(a, b)
        print(f"{k:<62}{res[k]['cos']:>10.6f}{res[k]['angle_deg']:>8.2f}{res[k]['ratio']:>16.4f}")

    # direction of the shipped step vs direction of the exact-gradient step
    shipped = field("extended", "adaptive", "centroid")
    exact_impl = field("plain", "fixed", "gradient")
    res["shipped step direction vs exact-gradient step direction"] = stats(shipped, exact_impl)
    d = res["shipped step direction vs exact-gradient step direction"]
    print(f"\n{'shipped step dir vs exact-gradient step dir':<62}{d['cos']:>10.6f}{d['angle_deg']:>8.2f}")
    print(f"  per-query cos: min {d['cos_min']:.4f}  max {d['cos_max']:.4f}")

    m.potentials, m.bandwidth, m.drift = "extended", "adaptive", "centroid"
    out = get_project_dir() / "results" / "analysis" / f"q6-{args.model}-l{args.layer_idx}-exact-gradient.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nsaved {out}")
