"""Is COBRAS invariant to the value of R? Refit with R forced to other values and compare
the steered outputs on the same queries."""
import torch
from cobras.steer import COBRAS
from cobras.utils.data import load_tqa_gen_data, load_query_activations

KW = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1.0e-3, epsilon=0.0, max_iters=10,
          vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0)

pos, neg = load_tqa_gen_data("Llama3.1-8B-Base", 13, 0)
q = load_query_activations("Llama3.1-8B-Base", 13, "truthfulqa_split1")[:200].cuda().float()
pos, neg = pos.cuda().float(), neg.cuda().float()

base = COBRAS(**KW).fit(pos, neg)
Y0 = base.steer(q, T=0.5)
print(f"R_fit = {base.R:.4f}, sigma2 = {base.sigma2:.4f}, query mean norm = {q.norm(dim=-1).mean():.4f}")

for name, scale in [("R x 0.5", 0.5), ("R x 2", 2.0), ("R = query radius", float(q.norm(dim=-1).mean()) / base.R)]:
    m = COBRAS(**KW).fit(pos, neg)
    # rescale the fitted sphere: R, the projected samples, and sigma^2 together
    m.R = base.R * scale
    m.h_pos = base.h_pos * scale
    m.h_neg = base.h_neg * scale
    m.sigma2 = base.sigma2 * scale ** 2
    m.cost = base.cost.clone()
    m.log_psi, m.log_phi = base.log_psi.clone(), base.log_phi.clone()
    Y = m.steer(q, T=0.5)
    rel = ((Y - Y0).norm(dim=-1) / Y0.norm(dim=-1)).max().item()
    cos = torch.nn.functional.cosine_similarity(Y, Y0, dim=-1).min().item()
    print(f"  {name:18s} (R={m.R:8.4f}): max rel diff of steered output = {rel:.3e}, min cosine = {cos:.8f}")
