"""Q4 (Reviewer EJdD): offline cost of solving the static Schroedinger bridge.

Measures wall-clock and peak GPU memory of COBRAS.fit() as a function of the
contrastive-set size N (= N_+ = N_-), with a phase breakdown:

  project   -- radius estimate + spherical projection            O((N_+ + N_-) d)
  bandwidth -- sigma^2 from the k-NN geodesic distance           O((N_+ + N_-)^2 d)   <- materialises (2N)^2
  cost      -- c_ji = d_S(h_j^-, h_i^+)^2 / (2 sigma^2)          O(N_+ N_- d)         <- materialises N_- x N_+
  sinkhorn  -- n_sinkhorn log-domain iterations                  O(n_it N_+ N_-)

Also measures a memory-lean *blocked* fit that computes exactly the same
potentials but never materialises the (2N)^2 kNN matrix and builds the cost
matrix in row blocks, to separate "the algorithm does not scale" from
"the reference implementation does not scale".

Real TruthfulQA activations are used where N is small enough; beyond that the
tensors are synthetic (random directions at the measured radius). Timing is
data-independent here (matmul / acos / topk / logsumexp), quality experiments
use real data only.

Usage:
  CUDA_VISIBLE_DEVICES=6 ./.venv/bin/python scripts/analysis/cost/offline_cost.py
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from cobras.steer import COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data_all_splits
from cobras.utils.sphere import knn_geodesic_dist

_EPS = 1e-7
_SAME_POINT_THR = 1e-8


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.marks = {}
        self._t = None

    def start(self):
        sync()
        self._t = time.perf_counter()

    def stop(self, name):
        sync()
        self.marks[name] = time.perf_counter() - self._t
        self._t = time.perf_counter()


@torch.no_grad()
def fit_reference(pos, neg, k_bw=5, n_sinkhorn=5):
    """Line-for-line the phases of COBRAS.fit (src/cobras/steer/_cobras.py:83)."""
    t = Timer()
    t.start()
    R = torch.cat([pos, neg], 0).norm(dim=-1).mean().item()
    h_pos = pos * (R / pos.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR))
    h_neg = neg * (R / neg.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR))
    H_all = torch.cat([h_pos, h_neg], 0)
    t.stop("project")

    k = min(k_bw, H_all.size(0) - 1)
    sigma2 = knn_geodesic_dist(H_all, R, k=k).median().item() ** 2
    sigma2 = float(max(sigma2, (R * 1e-3) ** 2))
    del H_all
    t.stop("bandwidth")

    cos_np = (h_neg @ h_pos.T) / (R ** 2)
    cos_np = cos_np.clamp(-1.0 + _EPS, 1.0 - _EPS)
    cost = (R * torch.acos(cos_np)).pow(2) / (2.0 * sigma2)
    del cos_np
    t.stop("cost")

    log_psi, log_phi = COBRAS._sinkhorn(cost, n_sinkhorn)
    t.stop("sinkhorn")
    return t.marks, (log_psi, log_phi, cost)


@torch.no_grad()
def fit_blocked(pos, neg, k_bw=5, n_sinkhorn=5, block=4096):
    """Same potentials, blocked. kNN bandwidth in row blocks (no (2N)^2 tensor);
    cost built row block by row block (one temp block instead of 3 full copies)."""
    t = Timer()
    t.start()
    R = torch.cat([pos, neg], 0).norm(dim=-1).mean().item()
    h_pos = pos * (R / pos.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR))
    h_neg = neg * (R / neg.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR))
    H_all = torch.cat([h_pos, h_neg], 0)
    t.stop("project")

    N = H_all.size(0)
    k = min(k_bw, N - 1)
    kth = torch.empty(N, device=H_all.device, dtype=H_all.dtype)
    for i in range(0, N, block):
        Hi = H_all[i : i + block]
        cos_i = ((Hi @ H_all.T) / (R ** 2)).clamp(-1.0 + _EPS, 1.0 - _EPS)
        d_i = R * torch.acos(cos_i)
        rows = torch.arange(Hi.size(0), device=d_i.device)
        d_i[rows, rows + i] = float("inf")
        kth[i : i + block] = d_i.topk(k, dim=1, largest=False).values[:, -1]
        del cos_i, d_i
    sigma2 = float(max(kth.median().item() ** 2, (R * 1e-3) ** 2))
    del H_all, kth
    t.stop("bandwidth")

    Nn, Np = h_neg.size(0), h_pos.size(0)
    cost = torch.empty(Nn, Np, device=h_pos.device, dtype=h_pos.dtype)
    for j in range(0, Nn, block):
        cj = ((h_neg[j : j + block] @ h_pos.T) / (R ** 2)).clamp(-1.0 + _EPS, 1.0 - _EPS)
        torch.acos(cj, out=cj)
        cj.mul_(R).pow_(2).div_(2.0 * sigma2)
        cost[j : j + block] = cj
        del cj
    t.stop("cost")

    log_psi, log_phi = COBRAS._sinkhorn(cost, n_sinkhorn)
    t.stop("sinkhorn")
    return t.marks, (log_psi, log_phi, cost)


def make_data(n_pos, n_neg, d, device, real_pos, real_neg, gen):
    """Real activations when N fits, otherwise synthetic directions at the same radius."""
    src = []
    for n, real in ((n_pos, real_pos), (n_neg, real_neg)):
        if n <= real.size(0):
            X = real[:n].clone()
            synth = False
        else:
            R = real.norm(dim=-1).mean()
            X = torch.randn(n, d, generator=gen, dtype=torch.float32)
            X = X / X.norm(dim=-1, keepdim=True) * R
            X[: real.size(0)] = real
            synth = True
        src.append((X.to(device), synth))
    return src[0][0], src[1][0], (src[0][1] or src[1][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama3.1-8B-Base")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/analysis/q4/q4_offline_cost.json")
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False  # repo default: true fp32
    real_pos, real_neg = load_tqa_gen_data_all_splits(args.model, args.layer)
    d = real_pos.size(1)
    print(f"real TruthfulQA (both splits): N+={real_pos.size(0)} N-={real_neg.size(0)} d={d}")

    gen = torch.Generator().manual_seed(0)
    Ns = [1300, 2600, 5000, 10000, 20000, 30000, 40000, 60000, 100000]
    rows = []
    for N in Ns:
        for variant, fn in (("reference", fit_reference), ("blocked", fit_blocked)):
            gc.collect(); torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                pos, neg, synth = make_data(N, N, d, device, real_pos, real_neg, gen)
                base_mem = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                reps = args.repeats if N <= 20000 else 1
                times = []
                for r in range(reps):
                    marks, out = fn(pos, neg)
                    times.append(marks)
                    if r < reps - 1:
                        del out, marks
                        gc.collect(); torch.cuda.empty_cache()
                peak = torch.cuda.max_memory_allocated()
                best = min(times, key=lambda m: sum(m.values()))
                row = dict(
                    N=N, variant=variant, synthetic=bool(synth), reps=reps,
                    peak_mem_GB=round((peak) / 2**30, 3),
                    data_mem_GB=round(base_mem / 2**30, 3),
                    total_s=round(sum(best.values()), 4),
                    **{f"{k}_s": round(v, 4) for k, v in best.items()},
                )
                del out, pos, neg
            except torch.cuda.OutOfMemoryError as e:
                row = dict(N=N, variant=variant, oom=True,
                           msg=str(e).split("\n")[0][:200])
                try:
                    del pos, neg
                except Exception:
                    pass
            gc.collect(); torch.cuda.empty_cache()
            print(json.dumps(row))
            rows.append(row)

    out_path = get_project_dir() / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(
        gpu=torch.cuda.get_device_name(0),
        dtype="float32", device="cuda", torch=torch.__version__,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        d=d, model=args.model, layer=args.layer,
        real_N_pos=int(real_pos.size(0)), real_N_neg=int(real_neg.size(0)),
        k_bw=5, n_sinkhorn=5,
    )
    out_path.write_text(json.dumps(dict(meta=meta, rows=rows), indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
