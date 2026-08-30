"""Q4: sanity check that TopKCOBRAS with topk >= N reproduces COBRAS exactly,
and report the truncation error of the field direction as topk shrinks.

CUDA_VISIBLE_DEVICES=6 ./.venv/bin/python scripts/analysis/cost/check_topk.py
"""
from __future__ import annotations

import json

import torch

from cobras.steer import COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data, load_query_activations
from variants import TopKCOBRAS

KW = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
          vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0, abstain_k=32,
          abstain_sharpness=50.0)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    pos, neg = load_tqa_gen_data("Llama3.1-8B-Base", 13, 0)
    q = load_query_activations("Llama3.1-8B-Base", 13, "truthfulqa_split1")[:128]
    pos, neg, q = pos.to(dev), neg.to(dev), q.to(dev)
    print(f"N+={pos.size(0)} N-={neg.size(0)} queries={q.size(0)}")

    ref = COBRAS(**KW).fit(pos, neg)
    V_ref = ref.vector_field(q)
    S_ref = ref.steer(q, T=0.5)

    rows = []
    for k in [None, 4096, 1024, 512, 256, 128, 64, 32, 16, 8, 4]:
        m = TopKCOBRAS(topk=k, **KW).fit(pos, neg)
        V = m.vector_field(q)
        S = m.steer(q, T=0.5)
        cos_field = torch.nn.functional.cosine_similarity(V, V_ref, dim=-1)
        # angle between the steered activation and the reference steered activation,
        # relative to the angle the reference moved
        d_ref = torch.acos(((S_ref / S_ref.norm(dim=-1, keepdim=True)) *
                            (q / q.norm(dim=-1, keepdim=True))).sum(-1).clamp(-1, 1))
        d_err = torch.acos(((S / S.norm(dim=-1, keepdim=True)) *
                            (S_ref / S_ref.norm(dim=-1, keepdim=True))).sum(-1).clamp(-1, 1))
        row = dict(topk=k,
                   cos_field_mean=round(cos_field.mean().item(), 6),
                   cos_field_min=round(cos_field.min().item(), 6),
                   endpoint_err_over_travel=round((d_err / d_ref.clamp(min=1e-9)).mean().item(), 6))
        print(json.dumps(row))
        rows.append(row)

    out = get_project_dir() / "results/analysis/q4/q4_topk_error.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        meta=dict(model="Llama3.1-8B-Base", layer=13, split_fit=0, split_query=1,
                  N_pos=int(pos.size(0)), N_neg=int(neg.size(0)), T=0.5,
                  dtype="float32", device="cuda"), rows=rows), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
