"""Q4 (Reviewer 4oei): what `n_sinkhorn = 0` actually removes.

The claim to check is that setting `n_sinkhorn = 0` leaves the Schroedinger potentials at
their uniform initialisation, so the drift becomes the plain KDE density-ratio drift that
Prop. 1 describes in the c_max -> 0 limit.

Two things are separable here:
  (a) the Sinkhorn *solve*, i.e. whether log psi / log phi are non-uniform;
  (b) the cost matrix, which still appears in the Eq. (15)-(16) extension even when the
      potentials are uniform.
The exact c_max -> 0 limit kills both. `n_sinkhorn = 0` kills only (a). This script measures
how far apart the two are at the actual steering site.

Run on CPU; nothing here needs a GPU.
"""

from __future__ import annotations

import torch

# the ablation switches this script sweeps (potentials=, drift=, bandwidth=, step_mode=,
# uniform_weights=, abstain_signal=) live on AblationCOBRAS; with defaults it is COBRAS
from cobras.steer import AblationCOBRAS as COBRAS
from cobras.utils.data import load_tqa_gen_data_all_splits, load_query_activations

MODEL, LAYER = "Llama3.1-8B-Base", 13
KW = dict(k_bw=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10, vmf_kappa=20,
          abstain_percentile=0.98, abstain_k=32, abstain_sharpness=200.0)


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def main() -> None:
    pos, neg = load_tqa_gen_data_all_splits(MODEL, LAYER)
    print(f"pos {tuple(pos.shape)}  neg {tuple(neg.shape)}")

    fits = {}
    for tag, kw in [
        ("sink5-extended", dict(n_sinkhorn=5, potentials="extended")),
        ("sink0-extended", dict(n_sinkhorn=0, potentials="extended")),
        ("sink0-plain", dict(n_sinkhorn=0, potentials="plain")),
        ("sink5-plain", dict(n_sinkhorn=5, potentials="plain")),
    ]:
        m = COBRAS(**KW, **kw).fit(pos, neg)
        fits[tag] = m
        print(f"{tag:16s} log_psi: max|.|={m.log_psi.abs().max():.4e} "
              f"std={m.log_psi.std():.4e} | log_phi: max|.|={m.log_phi.abs().max():.4e} "
              f"std={m.log_phi.std():.4e} | cost max={m.cost.max():.3f} "
              f"mean={m.cost.mean():.3f} sigma2={m.sigma2:.3f}")

    q = load_query_activations(MODEL, LAYER, "truthfulqa")
    if q is None:
        q = torch.cat([pos, neg], 0)
        print("no cached query activations; falling back to the contrastive samples")
    q = q.to(torch.float32)[:512]
    print(f"queries {tuple(q.shape)}")

    V = {}
    for tag, m in fits.items():
        V[tag] = m.vector_field(q)

    ref = "sink0-plain"  # the exact c_max -> 0 limit: uniform potentials and no cost
    for tag in fits:
        c = cos(V[tag], V[ref])
        print(f"cos(field[{tag:16s}], field[{ref}])  median={c.median():.6f}  "
              f"mean={c.mean():.6f}  min={c.min():.6f}")

    c = cos(V["sink5-extended"], V["sink0-extended"])
    print(f"cos(full, n_sinkhorn=0)  median={c.median():.6f} mean={c.mean():.6f} min={c.min():.6f}")


if __name__ == "__main__":
    main()
