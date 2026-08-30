import time, sys, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cobras.steer import COBRAS
from cobras.utils.data import load_tqa_gen_data, load_query_activations
from cobras.steer import EuclideanCOBRAS, _sq_dists

KW = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1.0e-3, epsilon=0.0, max_iters=10,
          vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0)
pos, neg = load_tqa_gen_data("Llama3.1-8B-Base", 13, 0)
pos, neg = pos.cuda().float(), neg.cuda().float()
q = load_query_activations("Llama3.1-8B-Base", 13, "truthfulqa_split1")[:10].cuda().float()

for cls in (COBRAS, EuclideanCOBRAS):
    m = cls(**KW).fit(pos, neg)
    for _ in range(2): m.steer(q, T=0.5)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(10): m.steer(q, T=0.5)
    torch.cuda.synchronize()
    print(f"{cls.__name__:18s} {(time.time()-t)/10*1000:.1f} ms per steer(B=10)")

# does the matmul expansion match cdist?
A, B = q, neg
d1 = torch.cdist(A, B).pow(2)
d2 = (A.pow(2).sum(-1, keepdim=True) + B.pow(2).sum(-1) - 2.0 * (A @ B.T)).clamp(min=0)
print(f"cdist vs expansion: max abs err {(d1-d2).abs().max():.3e}, max rel err {((d1-d2).abs()/d1.clamp(min=1e-6)).max():.3e}")
torch.cuda.synchronize(); t=time.time()
for _ in range(50): torch.cdist(A, B).pow(2)
torch.cuda.synchronize(); print(f"cdist    {(time.time()-t)/50*1000:.2f} ms")
t=time.time()
for _ in range(50): (A.pow(2).sum(-1,keepdim=True) + B.pow(2).sum(-1) - 2.0*(A@B.T)).clamp(min=0)
torch.cuda.synchronize(); print(f"expansion {(time.time()-t)/50*1000:.2f} ms")
