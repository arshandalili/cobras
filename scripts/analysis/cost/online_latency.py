"""Q4 (Reviewer EJdD): online cost of the steering hook, per generated token.

The hook fires once per generated token on a batch of B activations, and
COBRAS evaluates its field at each of K geodesic Euler steps. This script times
one `steer()` call (= the whole per-token overhead) in isolation, with CUDA
synchronisation and warm-up iterations discarded, for

  * COBRAS as shipped, over N and over K
  * COBRAS with the Eq. (15)-(16) marginalisation dropped (potentials="plain")
  * TopKCOBRAS, over the neighbourhood size k
  * CAA (vector add), SphericalSteer (one rotation), ODESteer (K sketched-
    classifier gradient steps)

Real TruthfulQA activations where N fits, synthetic directions at the same
radius beyond that (timing only).

CUDA_VISIBLE_DEVICES=6 ./.venv/bin/python scripts/analysis/cost/online_latency.py
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import torch

# the ablation switches this script sweeps (potentials=, drift=, bandwidth=, step_mode=,
# uniform_weights=, abstain_signal=) live on AblationCOBRAS; with defaults it is COBRAS
from cobras.steer import AblationCOBRAS as COBRAS
from cobras.steer import CAA, ODESteer, SphericalSteer
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data_all_splits, load_query_activations
from variants import TopKCOBRAS

KW = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0,
          vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0)


def timeit(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts = torch.tensor(ts)
    return dict(median_ms=round(ts.median().item() * 1e3, 4),
                mean_ms=round(ts.mean().item() * 1e3, 4),
                p90_ms=round(ts.quantile(0.9).item() * 1e3, 4))


def peak_mem(fn):
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn(); torch.cuda.synchronize()
    return round((torch.cuda.max_memory_allocated() - base) / 2**20, 2)


def grow(X, n, gen, dev):
    if n <= X.size(0):
        return X[:n].to(dev)
    R = X.norm(dim=-1).mean()
    Y = torch.randn(n, X.size(1), generator=gen, dtype=torch.float32)
    Y = Y / Y.norm(dim=-1, keepdim=True) * R
    Y[: X.size(0)] = X
    return Y.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama3.1-8B-Base")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--batch", type=int, default=10)  # confs/truthfulqa.yaml batch_size
    ap.add_argument("--out", default="results/analysis/q4/q4_online_latency.json")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    B = args.batch
    pos_r, neg_r = load_tqa_gen_data_all_splits(args.model, args.layer)
    d = pos_r.size(1)
    q_all = load_query_activations(args.model, args.layer, "truthfulqa").to(dev)
    q = q_all[:B].contiguous()
    gen = torch.Generator().manual_seed(0)
    rows = []

    def add(row):
        print(json.dumps(row)); rows.append(row)

    # ---- baselines at the real N -------------------------------------------
    pos, neg = pos_r.to(dev), neg_r.to(dev)
    caa = CAA(); caa.fit(pos, neg)
    add(dict(method="CAA", N_pos=pos.size(0), N_neg=neg.size(0), K=1, B=B,
             mem_MB=peak_mem(lambda: caa.steer(q, T=4.0)),
             **timeit(lambda: caa.steer(q, T=4.0))))

    sph = SphericalSteer(kappa=20.0, alpha=0.7, beta=-0.15); sph.fit(pos, neg)
    add(dict(method="SphericalSteer", N_pos=pos.size(0), N_neg=neg.size(0), K=1, B=B,
             mem_MB=peak_mem(lambda: sph.steer(q, T=4.0)),
             **timeit(lambda: sph.steer(q, T=4.0))))

    ode = ODESteer(solver="euler", steps=10, n_components=8000, degree=2,
                   gamma=0.1, coef0=1.0, lin_clf_type="lr")
    ode.fit(pos.cpu(), neg.cpu())
    ode.clf.to(dev)
    for K in (1, 10):
        ode.steps = K
        add(dict(method="ODESteer", N_pos=pos.size(0), N_neg=neg.size(0), K=K, B=B,
                 mem_MB=peak_mem(lambda: ode.steer(q, T=4.0)),
                 **timeit(lambda: ode.steer(q, T=4.0))))
    del ode
    gc.collect(); torch.cuda.empty_cache()

    # ---- COBRAS: scaling in N (K = 10, the paper default) -------------------
    for N in (1300, 2600, 5000, 10000, 20000, 40000):
        try:
            p, n_ = grow(pos_r, N, gen, dev), grow(neg_r, N, gen, dev)
            for name, cls, kw in (
                ("COBRAS", COBRAS, {}),
                ("COBRAS-plain", COBRAS, dict(potentials="plain")),
                ("COBRAS-topk64", TopKCOBRAS, dict(topk=64)),
            ):
                m = cls(max_iters=10, **KW, **kw).fit(p, n_)
                try:
                    add(dict(method=name, N_pos=N, N_neg=N, K=10, B=B,
                             mem_MB=peak_mem(lambda: m.steer(q, T=0.5)),
                             **timeit(lambda: m.steer(q, T=0.5), warmup=3, iters=15)))
                except torch.cuda.OutOfMemoryError:
                    add(dict(method=name, N_pos=N, N_neg=N, K=10, B=B, oom=True))
                del m
                gc.collect(); torch.cuda.empty_cache()
            del p, n_
        except torch.cuda.OutOfMemoryError:
            add(dict(method="fit", N_pos=N, N_neg=N, oom=True))
        gc.collect(); torch.cuda.empty_cache()

    # ---- COBRAS: scaling in K, and in the top-k neighbourhood, at real N ----
    for K in (1, 2, 5, 10, 20):
        m = COBRAS(max_iters=K, **KW).fit(pos, neg)
        add(dict(method="COBRAS", N_pos=pos.size(0), N_neg=neg.size(0), K=K, B=B,
                 mem_MB=peak_mem(lambda: m.steer(q, T=0.5)),
                 **timeit(lambda: m.steer(q, T=0.5))))
        del m; gc.collect(); torch.cuda.empty_cache()

    for k in (512, 256, 128, 64, 32, 16):
        m = TopKCOBRAS(topk=k, max_iters=10, **KW).fit(pos, neg)
        add(dict(method=f"COBRAS-topk{k}", N_pos=pos.size(0), N_neg=neg.size(0),
                 K=10, B=B, topk=k, mem_MB=peak_mem(lambda: m.steer(q, T=0.5)),
                 **timeit(lambda: m.steer(q, T=0.5))))
        del m; gc.collect(); torch.cuda.empty_cache()

    m = COBRAS(max_iters=10, potentials="plain", **KW).fit(pos, neg)
    add(dict(method="COBRAS-plain", N_pos=pos.size(0), N_neg=neg.size(0), K=10, B=B,
             mem_MB=peak_mem(lambda: m.steer(q, T=0.5)),
             **timeit(lambda: m.steer(q, T=0.5))))
    del m

    # ---- batch-size sweep at the real N ------------------------------------
    for bs in (1, 4, 10, 32):
        qb = q_all[:bs].contiguous()
        m = COBRAS(max_iters=10, **KW).fit(pos, neg)
        add(dict(method="COBRAS", N_pos=pos.size(0), N_neg=neg.size(0), K=10, B=bs,
                 mem_MB=peak_mem(lambda: m.steer(qb, T=0.5)),
                 **timeit(lambda: m.steer(qb, T=0.5))))
        del m; gc.collect(); torch.cuda.empty_cache()

    out = get_project_dir() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=dict(
        gpu=torch.cuda.get_device_name(0), dtype="float32", device="cuda",
        torch=torch.__version__, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        d=d, model=args.model, layer=args.layer, batch=B,
        real_N_pos=int(pos_r.size(0)), real_N_neg=int(neg_r.size(0)),
    ), rows=rows), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
