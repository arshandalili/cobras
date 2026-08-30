"""Q4 (Reviewer 4oei): what each removed component does to the intervention itself.

For every leave-one-out variant we take the cached last-token query activations of the three
evaluation tasks, apply `steer` at the shipped T, and report

  * the mean geodesic displacement ||q_K - q_0|| / R, i.e. how far the method actually moves
    a query, separately in distribution (TruthfulQA) and out of distribution (GSM8K, MMLU);
  * the cosine between the variant's displacement and the shipped method's displacement,
    which says whether the component changes the *direction* or only the *size* of the step.

Run on CPU. Chunked because the Eq. (15)-(16) marginalisation materialises [B, N_-, N_+].
"""

from __future__ import annotations

import json

import torch

from cobras.steer import get_steer_model
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data_all_splits, load_query_activations

MODEL, LAYER, T = "Llama3.1-8B-Base", 13, 0.65
N_Q, CHUNK = 128, 8

BASE = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
            vmf_kappa=20, vmf_beta=0.0, abstain_percentile=0.98, abstain_k=32,
            abstain_sharpness=200.0, step_mode="unit", drift="centroid",
            uniform_weights=False)

# every row builds AblationCOBRAS: the ablated ones need its switches, and with default
# kwargs it is the shipped COBRAS, so the q4-full baseline stays the real method
VARIANTS = {
    "q4-full": ("AblationCOBRAS", {}),
    "q4-nosphere": ("EuclideanCOBRAS", {}),
    "q4-nosinkhorn": ("AblationCOBRAS", dict(n_sinkhorn=0)),
    "q4-onestep": ("AblationCOBRAS", dict(max_iters=1)),
    "q4-rawstep": ("AblationCOBRAS", dict(step_mode="raw", drift="gradient")),
    "q4-novmf": ("AblationCOBRAS", dict(vmf_kappa=None)),
    "q4-noabstain": ("AblationCOBRAS", dict(abstain_percentile=1.0)),
    "q4-uniformw": ("AblationCOBRAS", dict(uniform_weights=True)),
}


def steer_chunked(m, q: torch.Tensor) -> torch.Tensor:
    out = []
    for i in range(0, q.size(0), CHUNK):
        m.reset_gate()
        out.append(m.steer(q[i : i + CHUNK], T=T))
    return torch.cat(out)


def main() -> None:
    pos, neg = load_tqa_gen_data_all_splits(MODEL, LAYER)
    ref = load_query_activations(MODEL, LAYER, "truthfulqa")
    tasks = {t: load_query_activations(MODEL, LAYER, t).to(torch.float32)[:N_Q]
             for t in ("truthfulqa", "gsm8k", "mmlu")}

    disp, res = {}, {}
    for name, (typ, over) in VARIANTS.items():
        kw = dict(BASE)
        kw.update(over)
        m = get_steer_model(typ, **kw).fit(pos, neg, ref_X=ref)
        R = m.R
        res[name] = {}
        for task, q in tasks.items():
            out = steer_chunked(m, q)
            d = (out - q)
            disp[(name, task)] = d
            rel = (d.norm(dim=-1) / R)
            gate = None
            if m.abstain_percentile is not None and m.rho_ref is not None:
                m.reset_gate()
                p0 = q * (R / q.norm(dim=-1, keepdim=True))
                gate = float(m._abstain_gate(p0).mean())
            res[name][task] = dict(disp_mean=float(rel.mean()), disp_med=float(rel.median()),
                                   gate=gate)
        print(name, {k: round(v["disp_mean"], 4) for k, v in res[name].items()},
              "gate", {k: (None if v["gate"] is None else round(v["gate"], 3))
                       for k, v in res[name].items()}, flush=True)

    for name in VARIANTS:
        for task in tasks:
            a, b = disp[(name, task)], disp[("q4-full", task)]
            c = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            res[name][task]["cos_to_full"] = float(c.median())
        print(name, {k: round(v["cos_to_full"], 4) for k, v in res[name].items()}, flush=True)

    out_dir = get_project_dir() / "results" / "analysis" / "q4x"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "field_diag.json").write_text(json.dumps(res, indent=2))
    print("wrote", out_dir / "field_diag.json")


if __name__ == "__main__":
    main()
