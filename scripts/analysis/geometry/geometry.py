"""Q3 (Reviewer EJdD): how stable is the spherical geometry across layers and architectures?

Part A: per-layer norm concentration of the TruthfulQA contrastive activations that COBRAS
        actually fits on, for the four 7-8B models of the paper (all layers).
Part B: per-layer norm concentration for four architecturally different models, from the
        precomputed per-token residual-norm dumps (Gemma-2-2b, GPT-OSS-20b (MoE),
        Llama-3.2-1B, Qwen2.5-1.5B), plus the pre-/post-RMSNorm comparison.

Quantities, with R = mean_i ||h_i|| exactly as COBRAS.fit sets it (src/cobras/steer/_cobras.py:89):
  cov      = std(||h||) / mean(||h||)                     (the paper's Figure 1 statistic)
  reldisp  = || h - R h/||h|| || / ||h|| = |1 - R/||h|||   (relative displacement of the projection)
  varfrac  = E[(||h|| - R)^2] / E[|| h - mean(h) ||^2]     (share of activation variance the
                                                            projection removes)
  aucs     = AUC of a logistic probe for pos-vs-neg, fit on one split and tested on the other,
             on the raw activations and on the projected ones (Part A only)

    ./.venv/bin/python -u scripts/analysis/geometry/geometry.py --part A --probe
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data


PAPER_MODELS = {
    "Llama3.1-8B-Base": 13,
    "Mistral-7B-Base": 15,
    "Falcon-7B-Base": 14,
    "Qwen2.5-7B-Base": 13,
}
STATS_DIR = "/data/arshan/hallucination/nullsteer/norm_concentration/model_stats"
OUT_DIR = get_project_dir() / "results" / "analysis" / "q3_geometry"


def norm_stats(H: np.ndarray) -> dict:
    """H: [N, d] float64."""
    n = np.linalg.norm(H, axis=-1)
    R = float(n.mean())
    cov = float(n.std() / n.mean())
    reldisp = np.abs(1.0 - R / n)
    centred = H - H.mean(0, keepdims=True)
    tot_var = float((centred ** 2).sum(-1).mean())
    rad_var = float(((n - R) ** 2).mean())
    return {
        "N": int(H.shape[0]),
        "d": int(H.shape[1]),
        "R": R,
        "cov": cov,
        "reldisp_mean": float(reldisp.mean()),
        "reldisp_p95": float(np.percentile(reldisp, 95)),
        "reldisp_max": float(reldisp.max()),
        "varfrac": rad_var / tot_var if tot_var > 0 else float("nan"),
        "norm_p05": float(np.percentile(n, 5)),
        "norm_p95": float(np.percentile(n, 95)),
    }


def norm_stats_1d(n: np.ndarray) -> dict:
    """Same, but only the norms are available (no vectors -> no varfrac)."""
    n = n.astype(np.float64)
    R = float(n.mean())
    reldisp = np.abs(1.0 - R / np.clip(n, 1e-12, None))
    return {
        "N": int(n.size),
        "R": R,
        "cov": float(n.std() / n.mean()),
        "reldisp_mean": float(reldisp.mean()),
        "reldisp_p95": float(np.percentile(reldisp, 95)),
        "reldisp_max": float(reldisp.max()),
        "norm_p05": float(np.percentile(n, 5)),
        "norm_p95": float(np.percentile(n, 95)),
    }


def probe_auc(Xtr, ytr, Xte, yte) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    scale = float(np.linalg.norm(Xtr, axis=-1).mean())
    clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
    clf.fit(Xtr / scale, ytr)
    return float(roc_auc_score(yte, clf.decision_function(Xte / scale)))


def project(H: np.ndarray, R: float) -> np.ndarray:
    return H * (R / np.clip(np.linalg.norm(H, axis=-1, keepdims=True), 1e-12, None))


def run_part_a(with_probe: bool) -> None:
    rows = []
    for model, steer_layer in PAPER_MODELS.items():
        act_dir = get_project_dir() / "data" / "truthfulqa" / "activations" / model
        layers = sorted(
            int(os.path.basename(p).split("layer")[1].split(".")[0])
            for p in glob.glob(str(act_dir / "pos_0_activations_layer*.pt"))
        )
        for l in layers:
            t0 = time.time()
            pos0, neg0 = load_tqa_gen_data(model, l, 0)
            pos1, neg1 = load_tqa_gen_data(model, l, 1)
            P0, N0 = pos0.to(torch.float64).numpy(), neg0.to(torch.float64).numpy()
            P1, N1 = pos1.to(torch.float64).numpy(), neg1.to(torch.float64).numpy()
            H = np.concatenate([P0, N0, P1, N1], 0)
            row = {"model": model, "layer": l, "is_steer_layer": l == steer_layer}
            row.update(norm_stats(H))
            if with_probe:
                # fit on split 0, test on split 1, and the reverse; average.
                aucs_raw, aucs_prj = [], []
                for (Ptr, Ntr, Pte, Nte) in ((P0, N0, P1, N1), (P1, N1, P0, N0)):
                    Xtr = np.concatenate([Ptr, Ntr], 0)
                    ytr = np.concatenate([np.ones(len(Ptr)), np.zeros(len(Ntr))])
                    Xte = np.concatenate([Pte, Nte], 0)
                    yte = np.concatenate([np.ones(len(Pte)), np.zeros(len(Nte))])
                    R_tr = float(np.linalg.norm(Xtr, axis=-1).mean())
                    aucs_raw.append(probe_auc(Xtr, ytr, Xte, yte))
                    aucs_prj.append(
                        probe_auc(project(Xtr, R_tr), ytr, project(Xte, R_tr), yte)
                    )
                row["auc_raw"] = float(np.mean(aucs_raw))
                row["auc_sphere"] = float(np.mean(aucs_prj))
            rows.append(row)
            print(
                f"{model:20s} L{l:<3d} CoV={row['cov']:.4f} "
                f"reldisp={row['reldisp_mean']:.4f} varfrac={row['varfrac']:.5f} "
                + (f"auc {row.get('auc_raw', float('nan')):.4f}/"
                   f"{row.get('auc_sphere', float('nan')):.4f} " if with_probe else "")
                + f"({time.time() - t0:.1f}s)",
                flush=True,
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "q3_part_a_paper_models.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'q3_part_a_paper_models.json'}")


def run_part_b() -> None:
    rows, rms_rows = [], []
    for path in sorted(glob.glob(os.path.join(STATS_DIR, "*_norms.npz"))):
        model = os.path.basename(path).replace("_norms.npz", "")
        d = np.load(path)
        L, dim = int(d["L"]), int(d["d"])
        layers = sorted(int(k) for k in d.files if k not in ("L", "d"))
        for l in layers:
            n = d[str(l)].astype(np.float64)
            row = {"model": model, "layer": l, "L": L, "d": dim}
            row.update(norm_stats_1d(n))
            # The dumps cover every token position. A small, fixed set of positions carries
            # a residual norm one to three orders of magnitude above the rest: the
            # attention-sink / "massive activation" tokens (BOS and a few delimiters).
            # There are exactly 500 such tokens in each 500-sequence dump, i.e. one per
            # sequence. Flag them by a scale-free rule and report the two regimes apart.
            med = np.median(n)
            sink = n > 5.0 * med
            row["sink_frac"] = float(sink.mean())
            row["sink_max_over_median"] = float(n.max() / med)
            row.update({f"ns_{k}": v for k, v in norm_stats_1d(n[~sink]).items()})
            rows.append(row)
            print(
                f"{model:16s} L{l:<3d} CoV={row['cov']:.4f} (nosink {row['ns_cov']:.4f}) "
                f"reldisp={row['reldisp_mean']:.4f} (nosink {row['ns_reldisp_mean']:.4f}) "
                f"sinkfrac={row['sink_frac']:.5f}",
                flush=True,
            )
        rms_path = path.replace("_norms.npz", "_post_rmsnorm.npz")
        if os.path.exists(rms_path):
            r = np.load(rms_path)
            for l in sorted(int(k.split("_")[1]) for k in r.files if k.startswith("pre_")):
                pre, post = r[f"pre_{l}"].astype(np.float64), r[f"post_{l}"].astype(np.float64)
                rms_rows.append({
                    "model": model, "layer": l,
                    "pre_cov": float(pre.std() / pre.mean()),
                    "post_cov": float(post.std() / post.mean()),
                    "pre_reldisp_mean": float(np.abs(1.0 - pre.mean() / pre).mean()),
                    "post_reldisp_mean": float(np.abs(1.0 - post.mean() / post).mean()),
                    "N": int(pre.size),
                })
            print(f"{model:16s} rmsnorm: {len(rms_rows)} rows so far", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "q3_part_b_other_archs.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(OUT_DIR / "q3_part_b_rmsnorm.json", "w") as f:
        json.dump(rms_rows, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'q3_part_b_other_archs.json'} and q3_part_b_rmsnorm.json")


def run_part_c() -> None:
    """Does the radius R fitted on the contrastive set transfer to the query distributions?

    COBRAS sets R from D+ u D- and then maps every query to R q/||q||. If an OOD prompt
    population sits at a different radius, that map is a larger displacement than it is
    in distribution. Uses the prompt activations at each model's steering layer.
    """
    from cobras.utils.data import load_query_activations, load_tqa_gen_data_all_splits

    rows = []
    for model, layer in PAPER_MODELS.items():
        pos, neg = load_tqa_gen_data_all_splits(model, layer)
        n_fit = torch.cat([pos, neg], 0).to(torch.float64).norm(dim=-1).numpy()
        R_fit = float(n_fit.mean())
        for task in ("truthfulqa", "gsm8k", "mmlu", "triviaqa", "nq"):
            q = load_query_activations(model, layer, task)
            if q is None:
                continue
            nq = q.to(torch.float64).norm(dim=-1).numpy()
            rows.append({
                "model": model, "layer": layer, "task": task, "N": int(nq.size),
                "R_fit": R_fit, "R_task": float(nq.mean()),
                "ratio": float(nq.mean()) / R_fit,
                "cov_task": float(nq.std() / nq.mean()),
                # displacement of COBRAS's own projection q -> R_fit q/||q||
                "reldisp_at_R_fit": float(np.abs(1.0 - R_fit / nq).mean()),
                "reldisp_at_R_task": float(np.abs(1.0 - nq.mean() / nq).mean()),
            })
            print(f"{model:20s} L{layer:<3d} {task:12s} N={nq.size:5d} R_fit={R_fit:8.3f} "
                  f"R_task={nq.mean():8.3f} ratio={rows[-1]['ratio']:.4f} "
                  f"CoV={rows[-1]['cov_task']:.4f} "
                  f"reldisp@R_fit={rows[-1]['reldisp_at_R_fit']:.4f} "
                  f"reldisp@R_task={rows[-1]['reldisp_at_R_task']:.4f}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "q3_part_c_query_radius.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'q3_part_c_query_radius.json'}")


def run_part_d(n_sample: int = 2000, k: int = 32, seed: int = 0) -> None:
    """How much does the projection distort the geometry COBRAS's weights actually read?

    Every quantity in Eqs. (15)-(19) is a softmax over distances to the contrastive samples,
    so what matters is whether d_S(R x/||x||, R y/||y||) ranks pairs the way ||x - y|| does.
    If all norms were equal the two orders would agree exactly, so the disagreement is the
    price of the assumption. Reported per layer as (i) Spearman rho over all sampled pairs
    and (ii) the mean Jaccard overlap of the k nearest neighbours under the two metrics.
    """
    from scipy.stats import spearmanr

    rows = []
    for model, steer_layer in PAPER_MODELS.items():
        act_dir = get_project_dir() / "data" / "truthfulqa" / "activations" / model
        layers = sorted(
            int(os.path.basename(p).split("layer")[1].split(".")[0])
            for p in glob.glob(str(act_dir / "pos_0_activations_layer*.pt"))
        )
        for l in layers:
            pos, neg = load_tqa_gen_data(model, l, 0)
            H = torch.cat([pos, neg], 0).to(torch.float32)
            idx = torch.randperm(H.size(0), generator=torch.Generator().manual_seed(seed))[:n_sample]
            H = H[idx].cuda()
            R = H.norm(dim=-1).mean()
            P = H * (R / H.norm(dim=-1, keepdim=True))

            d_euc = torch.cdist(H, H)
            cos = ((P @ P.T) / (R ** 2)).clamp(-1 + 1e-7, 1 - 1e-7)
            d_geo = R * torch.acos(cos)
            iu = torch.triu_indices(H.size(0), H.size(0), offset=1, device=H.device)
            a, b = d_euc[iu[0], iu[1]].cpu().numpy(), d_geo[iu[0], iu[1]].cpu().numpy()
            sub = np.random.default_rng(seed).choice(a.size, size=min(200_000, a.size), replace=False)
            rho = float(spearmanr(a[sub], b[sub]).statistic)

            d_euc.fill_diagonal_(float("inf"))
            d_geo.fill_diagonal_(float("inf"))
            nn_e = d_euc.topk(k, dim=1, largest=False).indices
            nn_g = d_geo.topk(k, dim=1, largest=False).indices
            hit = torch.zeros(H.size(0), H.size(0), dtype=torch.bool, device=H.device)
            hit.scatter_(1, nn_e, True)
            overlap = float(hit.gather(1, nn_g).float().sum(1).mean().item() / k)

            rows.append({"model": model, "layer": l, "is_steer_layer": l == steer_layer,
                         "n_sample": int(H.size(0)), "k": k,
                         "spearman_euc_vs_geo": rho, "knn_overlap": overlap})
            print(f"{model:20s} L{l:<3d} spearman={rho:.5f} knn{k}_overlap={overlap:.4f}", flush=True)
            del H, P, d_euc, d_geo, hit
            torch.cuda.empty_cache()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "q3_part_d_metric_distortion.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'q3_part_d_metric_distortion.json'}")


def run_part_e() -> None:
    """How much of the sphere does the data actually occupy?

    If the activations lie in a narrow angular cap, the sphere is locally almost flat there and
    the spherical and Euclidean geometries can barely differ, whatever the norms do. The mean
    resultant length rbar = || mean_i h_i/||h_i|| || is 1 for a point mass and 0 for an
    isotropic cloud; the mean pairwise angle says the same thing in degrees.
    """
    rows = []
    for model, steer_layer in PAPER_MODELS.items():
        act_dir = get_project_dir() / "data" / "truthfulqa" / "activations" / model
        layers = sorted(
            int(os.path.basename(p).split("layer")[1].split(".")[0])
            for p in glob.glob(str(act_dir / "pos_0_activations_layer*.pt"))
        )
        for l in layers:
            pos, neg = load_tqa_gen_data(model, l, 0)
            H = torch.cat([pos, neg], 0).to(torch.float64)
            U = H / H.norm(dim=-1, keepdim=True)
            rbar = float(U.mean(0).norm().item())
            C = (U @ U.T).clamp(-1.0, 1.0)
            iu = torch.triu_indices(U.size(0), U.size(0), offset=1)
            ang = torch.rad2deg(torch.acos(C[iu[0], iu[1]]))
            rows.append({
                "model": model, "layer": l, "is_steer_layer": l == steer_layer,
                "N": int(U.size(0)), "rbar": rbar,
                "mean_pairwise_angle_deg": float(ang.mean().item()),
                "p95_pairwise_angle_deg": float(torch.quantile(ang.float(), 0.95).item()),
                "max_pairwise_angle_deg": float(ang.max().item()),
            })
            print(f"{model:20s} L{l:<3d} rbar={rbar:.4f} "
                  f"mean_angle={rows[-1]['mean_pairwise_angle_deg']:6.2f} deg  "
                  f"p95={rows[-1]['p95_pairwise_angle_deg']:6.2f}  "
                  f"max={rows[-1]['max_pairwise_angle_deg']:6.2f}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "q3_part_e_angular_spread.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'q3_part_e_angular_spread.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "C", "D", "E", "both"], default="both")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.part in ("B", "both"):
        run_part_b()
    if args.part in ("C", "both"):
        run_part_c()
    if args.part == "D":
        run_part_d()
    if args.part == "E":
        run_part_e()
    if args.part in ("A", "both"):
        run_part_a(args.probe)
