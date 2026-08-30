"""Q4 (Reviewer EJdD): how much of the exact field each approximation keeps.

Two things, on real Llama3.1-8B-Base layer-13 TruthfulQA activations:

  (a) how concentrated the Eq. (18) weights actually are, as the perplexity
      exp(H(w)) of the weight distribution over the N contrastive samples.
      This says whether a top-k neighbourhood can carry the softmax mass.

  (b) cosine agreement between the exact Eq. (19) field and the field of each
      approximation, at real query activations:
        - topk<k>   : per-step truncation to the k nearest samples
        - sub<pct>  : bridge refitted on a random <pct>% of each set
                      (5 seeds, mean and std)

CUDA_VISIBLE_DEVICES=6 ./.venv/bin/python scripts/analysis/cost/approx_error.py
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from cobras.steer import COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data, load_query_activations
from variants import TopKCOBRAS, subsample

KW = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
          vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0, abstain_k=32,
          abstain_sharpness=50.0)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    pos, neg = load_tqa_gen_data("Llama3.1-8B-Base", 13, 0)
    q = load_query_activations("Llama3.1-8B-Base", 13, "truthfulqa_split1").to(dev)
    pos, neg = pos.to(dev), neg.to(dev)
    N_pos, N_neg = pos.size(0), neg.size(0)
    print(f"N+={N_pos} N-={N_neg} queries={q.size(0)}")

    ref = COBRAS(**KW).fit(pos, neg)
    V_ref = ref.vector_field(q)

    # (a) concentration of the Eq. (18) weights at the query
    w_pos, w_neg = ref.attribution(q)
    res = {}
    for tag, w, N in (("pos", w_pos, N_pos), ("neg", w_neg, N_neg)):
        ent = -(w.clamp(min=1e-30).log() * w).sum(-1)
        res[f"eff_support_{tag}"] = dict(
            N=int(N),
            perplexity_mean=round(ent.exp().mean().item(), 2),
            perplexity_frac_of_N=round((ent.exp().mean() / N).item(), 4),
            top64_mass_mean=round(w.topk(64, -1).values.sum(-1).mean().item(), 4),
            top256_mass_mean=round(w.topk(256, -1).values.sum(-1).mean().item(), 4),
            max_weight_mean=round(w.max(-1).values.mean().item(), 6),
            uniform_weight=round(1.0 / N, 6),
        )
    print(json.dumps(res, indent=2))

    rows = []
    for k in (1024, 512, 256, 128, 64, 32, 16):
        m = TopKCOBRAS(topk=k, **KW).fit(pos, neg)
        c = F.cosine_similarity(m.vector_field(q), V_ref, dim=-1)
        rows.append(dict(approx=f"topk{k}", param=k, cos_mean=round(c.mean().item(), 4),
                         cos_std=round(c.std().item(), 4), cos_min=round(c.min().item(), 4)))
        print(json.dumps(rows[-1]))
        del m

    for pct in (50, 25, 10, 5, 2, 1):
        cs = []
        for s in range(5):
            m = COBRAS(**KW).fit(subsample(pos, pct / 100, seed=100 * s),
                                 subsample(neg, pct / 100, seed=100 * s + 1))
            cs.append(F.cosine_similarity(m.vector_field(q), V_ref, dim=-1).mean().item())
            del m
        t = torch.tensor(cs)
        rows.append(dict(approx=f"sub{pct}", param=pct, seeds=5,
                         cos_mean=round(t.mean().item(), 4),
                         cos_std_over_seeds=round(t.std().item(), 4),
                         N_pos_used=int(round(pct / 100 * N_pos)),
                         N_neg_used=int(round(pct / 100 * N_neg))))
        print(json.dumps(rows[-1]))

    out = get_project_dir() / "results/analysis/q4/q4_approx_error.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=dict(
        model="Llama3.1-8B-Base", layer=13, fit_split=0, query_split=1,
        N_pos=int(N_pos), N_neg=int(N_neg), n_queries=int(q.size(0)),
        dtype="float32", device="cuda"), weights=res, rows=rows), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
