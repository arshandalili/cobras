"""Q4 (Reviewer 4oei): how selective the Eq. (18) weights actually are.

The `uniform_weights` ablation replaces the Eq. (18) softmax by a flat distribution over the
contrastive samples. To read that row we need to know how far from flat the real weights are.
This reports the normalised entropy H/log N and the effective sample size ESS/N of the Eq. (18)
weights at the true query activations, under the shipped adaptive bandwidth
sigma^2(x) = max_i d(x, h_i)^2 and under the fixed bandwidth sigma^2 for contrast.

Run on CPU.
"""

from __future__ import annotations

import json

import torch

# the ablation switches this script sweeps (potentials=, drift=, bandwidth=, step_mode=,
# uniform_weights=, abstain_signal=) live on AblationCOBRAS; with defaults it is COBRAS
from cobras.steer import AblationCOBRAS as COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data_all_splits, load_query_activations

MODEL, LAYER = "Llama3.1-8B-Base", 13
N_Q, CHUNK = 128, 8
BASE = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
            vmf_kappa=20, abstain_percentile=0.534)


def stats(w: torch.Tensor) -> dict:
    n = w.shape[-1]
    h = -(w.clamp(min=1e-30) * w.clamp(min=1e-30).log()).sum(-1) / torch.log(torch.tensor(float(n)))
    ess = 1.0 / (w.pow(2).sum(-1)) / n
    return dict(n=n, H_norm=float(h.mean()), ESS_frac=float(ess.mean()),
                w_max_over_uniform=float((w.max(-1).values * n).mean()))


def main() -> None:
    pos, neg = load_tqa_gen_data_all_splits(MODEL, LAYER)
    ref = load_query_activations(MODEL, LAYER, "truthfulqa")
    out = {}
    for bw, scale in (("adaptive", 1.0), ("fixed", 1.0), ("fixed", 1.0 / 16)):
        m = COBRAS(**BASE, bandwidth=bw, bandwidth_scale=scale).fit(pos, neg, ref_X=ref)
        wp, wn = [], []
        q = ref.to(torch.float32)[:N_Q]
        for i in range(0, q.size(0), CHUNK):
            a, b = m.attribution(q[i : i + CHUNK])
            wp.append(a)
            wn.append(b)
        key = f"{bw}-scale{scale:g}"
        out[key] = dict(pos=stats(torch.cat(wp)), neg=stats(torch.cat(wn)),
                        sigma2=m.sigma2, R=m.R)
        print(key, json.dumps(out[key]), flush=True)

    d = get_project_dir() / "results" / "analysis" / "q4x"
    d.mkdir(parents=True, exist_ok=True)
    (d / "weight_entropy.json").write_text(json.dumps(out, indent=2))
    print("wrote", d / "weight_entropy.json")


if __name__ == "__main__":
    main()
