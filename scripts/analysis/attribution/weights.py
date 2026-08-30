"""What does the Eq. (18)-(19) decomposition expose about a single query?

COBRAS's step is a convex combination over the contrastive samples, so for every query one
can ask which training examples it leans on. This script measures (i) how concentrated that
distribution is, (ii) whether it tracks TruthfulQA's own semantic categories, (iii) whether
it is causal for the step, and (iv) how much the resulting direction actually varies across
queries. It also writes the worked examples used in the appendix.

    uv run python -u scripts/analysis/attribution/weights.py -m Llama3.1-8B-Base -l 13
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import torch
import torch.nn.functional as F

# the ablation switches this script sweeps (potentials=, drift=, bandwidth=, step_mode=,
# uniform_weights=, abstain_signal=) live on AblationCOBRAS; with defaults it is COBRAS
from cobras.steer import AblationCOBRAS as COBRAS
from cobras.steer import CAA
from cobras.utils import get_project_dir
from cobras.utils.data import (
    load_tqa_gen_data,
    load_query_activations,
    load_tqa_gen_questions,
)

_EPS = 1e-7

# the Table 1 configuration; the two gates rescale the step and leave Eq. (18) untouched
_COBRAS_KWARGS = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
                      vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0)

# sigma^2 multipliers for the bandwidth ladder, on top of the derived fixed bandwidth
_BW_LADDER = [1.0, 1 / 4, 1 / 16, 1 / 64]


def tqa_categories() -> dict:
    """TruthfulQA's own 38 category labels, aligned to the D+/D- row order on disk."""
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    ds = ds.train_test_split(test_size=0.5, seed=42)  # the split used by data/truthfulqa
    texts = get_project_dir() / "data" / "truthfulqa" / "texts"
    out = {}
    for split, key in enumerate(["train", "test"]):
        q_cat = ds[key].to_pandas()["category"].to_numpy()
        rec = {"q_cat": q_cat}
        for tag in ("pos", "neg"):
            df = pd.read_json(texts / f"{tag}_{split}.jsonl", lines=True)
            idx = df["idx"].to_numpy()
            rec[tag] = dict(q_idx=idx, cat=q_cat[idx],
                            question=df["question"].tolist(), answer=df["answer"].tolist())
        out[split] = rec
    return out


def fit_cobras(pos: torch.Tensor, neg: torch.Tensor, **kw) -> COBRAS:
    return COBRAS(**{**_COBRAS_KWARGS, **kw}).fit(pos, neg)


def steering_step(model: COBRAS, q_dir: torch.Tensor) -> torch.Tensor:
    """Eq. (19) at the initial iterate, for queries given as unit directions."""
    return model._field(q_dir * model.R)[0]


def step_with_weights(m: COBRAS, q: torch.Tensor, w_pos: torch.Tensor, w_neg: torch.Tensor) -> torch.Tensor:
    """Eq. (19) with the weights replaced by hand, to price what the weighting is worth."""
    def side(H, w):
        cos_t = ((q @ H.T) / (m.R ** 2)).clamp(-1.0 + _EPS, 1.0 - _EPS)
        th = torch.acos(cos_t)
        coeff = torch.where(th < 1e-8, torch.zeros_like(th), th / torch.sin(th).clamp(min=1e-8))
        wc = w * coeff
        return (wc @ H - (wc * cos_t).sum(-1, keepdim=True) * q) / (1.0 + m.alpha_sigma)
    V = side(m.h_pos, w_pos) - side(m.h_neg, w_neg)
    return V - (V * q).sum(-1, keepdim=True) / (m.R ** 2) * q


# --------------------------------------------------------------------------- concentration
def concentration(w: torch.Tensor, k: int) -> dict:
    N = w.shape[1]
    return dict(
        N=int(N),
        ess_frac=float((1.0 / (w ** 2).sum(-1)).mean() / N),
        topk_mass=float(w.topk(k, dim=1).values.sum(-1).mean()),
        topk_mass_uniform=k / N,
        max_over_uniform=float(w.max(-1).values.mean() * N),
        n_distinct_top1=int(len(set(w.argmax(-1).tolist()))),
    )


# ------------------------------------------------------------------------------- semantics
def category_share(score: torch.Tensor, ex_cat: np.ndarray, q_cat: np.ndarray, k: int) -> dict:
    """Share of the top-k ranked examples drawn from the query's own category, against the
    frequency-matched baseline (that category's share of D+/D-, which is far from 1/38)."""
    idx = score.topk(k, dim=1).indices.cpu().numpy()
    freq = pd.Series(ex_cat).value_counts(normalize=True)
    hit = np.array([(ex_cat[idx[b]] == q_cat[b]).mean() for b in range(len(idx))])
    chance = np.array([float(freq.get(q_cat[b], 0.0)) for b in range(len(idx))])
    rng = np.random.default_rng(0)
    d = hit - chance
    draws = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(1)
    return dict(share=float(hit.mean()), chance=float(chance.mean()),
                lift=float(hit.mean() / chance.mean()), excess=float(d.mean()),
                ci=[float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))])


# ------------------------------------------------------------------------------- adaptivity
def karcher_mean(P: torch.Tensor, R: float, iters: int = 30) -> torch.Tensor:
    m = P.mean(0)
    m = m * (R / m.norm())
    for _ in range(iters):
        cos = ((P @ m) / (R ** 2)).clamp(-1 + _EPS, 1 - _EPS)
        th = torch.acos(cos).unsqueeze(-1)
        tng = torch.where(th.abs() < 1e-8, torch.zeros_like(P),
                          (th / torch.sin(th).clamp(min=1e-8)) * (P - cos.unsqueeze(-1) * m))
        s = tng.mean(0)
        if s.norm() < 1e-6 * R:
            break
        n = s.norm().clamp(min=1e-12)
        m = torch.cos(n / R) * m + torch.sin(n / R) * R * (s / n)
    return m


def parallel_transport(V: torch.Tensor, P: torch.Tensor, mu: torch.Tensor, R: float) -> torch.Tensor:
    """Transport each V_b in T_{P_b}S to T_mu S along the minimising geodesic. Steering
    directions at different queries live in different tangent spaces, so they are not
    comparable until they are brought to a common base point."""
    p, mh = P / R, mu / R
    cos = (p @ mh).clamp(-1 + _EPS, 1 - _EPS).unsqueeze(-1)
    th = torch.acos(cos)
    u = (mh.unsqueeze(0) - cos * p) / torch.sin(th).clamp(min=1e-8)
    vu = (V * u).sum(-1, keepdim=True)
    return V + vu * ((torch.cos(th) - 1.0) * u - torch.sin(th) * p)


def direction_spread(V: torch.Tensor, P: torch.Tensor, R: float) -> dict:
    Vt = parallel_transport(V, P, karcher_mean(P, R), R)
    Vn = Vt / Vt.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    n = len(Vn)
    G = Vn @ Vn.T
    mean_cos = float((G.sum() - G.diag().sum()) / (n * (n - 1)))
    d = Vn.mean(0)
    d = d / d.norm()
    resid = Vn - (Vn @ d).unsqueeze(-1) * d.unsqueeze(0)
    s = torch.linalg.svdvals(resid.double()) ** 2
    return dict(mean_pairwise_cos=mean_cos,
                mean_pairwise_angle_deg=float(np.degrees(np.arccos(min(1.0, mean_cos)))),
                energy_off_mean_direction=float((resid.norm(dim=-1) ** 2).mean()),
                eff_rank_of_residual=float((s.sum() ** 2 / (s ** 2).sum()).item()))


def distribution_spread(w: torch.Tensor, q_cat: np.ndarray, k: int, n_pairs: int = 20_000) -> dict:
    """Top-k overlap and Jensen-Shannon between the attribution distributions of query pairs.
    Basis-free, unlike a cosine between tangent vectors at different base points."""
    rng = np.random.default_rng(0)
    n = len(w)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    top = w.topk(k, dim=1).indices.cpu().numpy()
    ov, js = [], []
    for i in range(0, len(a), 512):
        ia, ib = a[i:i + 512], b[i:i + 512]
        ov.extend([len(set(top[x]) & set(top[y])) / k for x, y in zip(ia, ib)])
        P, Q = w[ia], w[ib]
        Mx = 0.5 * (P + Q)
        kl = lambda X, Y: (X * (X.clamp(min=1e-30).log() - Y.clamp(min=1e-30).log())).sum(-1)
        js.extend((0.5 * kl(P, Mx) + 0.5 * kl(Q, Mx)).cpu().tolist())
    ov, js = np.array(ov), np.array(js)
    same = np.array([q_cat[x] == q_cat[y] for x, y in zip(a, b)])
    return dict(topk_overlap=float(ov.mean()), jsd_nats=float(js.mean()),
                topk_overlap_same_cat=float(ov[same].mean()), topk_overlap_diff_cat=float(ov[~same].mean()),
                jsd_same_cat=float(js[same].mean()), jsd_diff_cat=float(js[~same].mean()),
                n_same=int(same.sum()), n_diff=int((~same).sum()))


# -------------------------------------------------------------------------------- leave-out
def leave_out(pos, neg, q_dir, V_full, wp, wn, cats, meta_train, q_cat, k, n_queries, seed):
    """Remove examples from D+/D-, re-solve the bridge, and measure how far the step at the
    *same* query turns. Both vectors sit in T_{q}S, so the cosine is well defined."""
    rng = np.random.default_rng(seed)
    cp, cn = meta_train["pos"]["cat"], meta_train["neg"]["cat"]
    ip, iN = meta_train["pos"]["q_idx"], meta_train["neg"]["q_idx"]
    Np, Nn = wp.shape[1], wn.shape[1]
    wp_c, wn_c = wp - wp.mean(0, keepdim=True), wn - wn.mean(0, keepdim=True)
    sel = rng.choice(len(q_dir), size=min(n_queries, len(q_dir)), replace=False)
    rows = []
    for b in sel:
        c = q_cat[b]
        own_p, own_n = np.where(cp == c)[0], np.where(cn == c)[0]
        alt = min([x for x in cats if x != c], key=lambda x: abs((cp == x).sum() - len(own_p)))
        qids, acc, tot = rng.permutation(np.unique(ip)), [], 0
        for qi in qids:
            if meta_train["q_cat"][qi] == c:
                continue
            acc.append(qi)
            tot += int((ip == qi).sum())
            if tot >= len(own_p):
                break
        acc = np.array(acc)
        conds = {
            "top-k by weight": (wp[b].topk(k).indices.cpu().numpy(), wn[b].topk(k).indices.cpu().numpy()),
            "top-k by query-specific weight": (wp_c[b].topk(k).indices.cpu().numpy(),
                                               wn_c[b].topk(k).indices.cpu().numpy()),
            "k at random": (rng.choice(Np, k, replace=False), rng.choice(Nn, k, replace=False)),
            "bottom-k by weight": ((-wp[b]).topk(k).indices.cpu().numpy(),
                                   (-wn[b]).topk(k).indices.cpu().numpy()),
            "the query's own category": (own_p, own_n),
            "another category, size-matched": (np.where(cp == alt)[0], np.where(cn == alt)[0]),
            "random questions, size-matched": (np.where(np.isin(ip, acc))[0], np.where(np.isin(iN, acc))[0]),
        }
        for name, (rp, rn) in conds.items():
            mp = np.ones(Np, bool); mp[rp] = False
            mn = np.ones(Nn, bool); mn[rn] = False
            m_ab = fit_cobras(pos[torch.as_tensor(mp, device=pos.device)],
                              neg[torch.as_tensor(mn, device=neg.device)])
            cos = F.cosine_similarity(V_full[b:b + 1], steering_step(m_ab, q_dir[b:b + 1]), dim=-1).item()
            rows.append(dict(q=int(b), cond=name, cos=cos,
                             angle=float(np.degrees(np.arccos(min(1.0, max(-1.0, cos))))),
                             n_pos_removed=int(len(rp)), n_neg_removed=int(len(rn))))
    return rows


# ---------------------------------------------------------------------------------- report
_GEN_RUNS = {
    "unsteered": "NoSteer",
    "COBRAS": "COBRAS-k_bw5-sink5-alpha0.001-eps0.0-iters10-kappa20-abst1.0-k32-sh50.0-T0.5",
}


def load_generations(model: str, layer_idx: int, questions: list[str], seed: int = 42) -> dict:
    """Generations already on disk for these questions, keyed by run, plus the judge labels
    when the detailed evaluation has been run."""
    base = get_project_dir() / "results" / "truthfulqa"
    out = {}
    for tag, run in _GEN_RUNS.items():
        stem = f"{model}-l{layer_idx}-{run}-TruthfulQA-seed{seed}.jsonl"
        det = base / "eval_results" / "detailed_eval_results" / f"{model}-l{layer_idx}" / stem
        raw = base / "raw_outputs" / model / stem
        path = det if det.exists() else raw
        if not path.exists():
            continue
        df = pd.read_json(path, lines=True).set_index("prompt")
        out[tag] = [df.loc[q, "output"] for q in questions]
        if "true" in df.columns:
            out[f"{tag}_truthful"] = [bool(df.loc[q, "true"]) for q in questions]
    return out


def worked_examples(wp, wn, meta_train, questions, outputs, q_cat, n, top, seed):
    """Top-`top` attributed answers per side, ranked by the query-specific part of the weight."""
    rng = np.random.default_rng(seed)
    wp_c, wn_c = wp - wp.mean(0, keepdim=True), wn - wn.mean(0, keepdim=True)
    picks = rng.choice(len(questions), size=min(n, len(questions)), replace=False)
    out = []
    for b in picks:
        rec = dict(question=questions[b], category=str(q_cat[b]),
                   **{k: v[b] for k, v in outputs.items()})
        for tag, W, Wc in (("positive", wp, wp_c), ("negative", wn, wn_c)):
            m = meta_train[tag[:3]]
            idx = Wc[b].topk(top).indices.cpu().tolist()
            rec[f"top_{tag}"] = [dict(
                weight=float(W[b, i]), uniform=1.0 / W.shape[1],
                category=str(m["cat"][i]), source_question=m["question"][i], answer=m["answer"][i],
            ) for i in idx]
        out.append(rec)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("-k", "--topk", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_leaveout", type=int, default=100, help="queries per fold for the causal test")
    ap.add_argument("--n_examples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bandwidth", choices=("adaptive", "fixed"), default=None,
                    help="override the query-time KDE bandwidth rule for the whole analysis")
    ap.add_argument("--bandwidth_scale", type=float, default=None)
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()

    if args.bandwidth is not None:
        _COBRAS_KWARGS["bandwidth"] = args.bandwidth
    if args.bandwidth_scale is not None:
        _COBRAS_KWARGS["bandwidth_scale"] = args.bandwidth_scale

    meta = tqa_categories()
    res = {"model": args.model, "layer_idx": args.layer_idx, "topk": args.topk,
           "cobras_kwargs": dict(_COBRAS_KWARGS), "folds": {}}
    leave_rows = []
    examples = []

    for test_s in (0, 1):
        train_s = 1 - test_s
        pos, neg = load_tqa_gen_data(args.model, args.layer_idx, train_s)
        pos, neg = pos.float().to(args.device), neg.float().to(args.device)
        X = load_query_activations(args.model, args.layer_idx, f"truthfulqa_split{test_s}")
        assert X is not None, "run scripts/prepare/extract_query_activations.py first"
        X = X.float().to(args.device)
        q_dir = X / X.norm(dim=-1, keepdim=True)
        q_cat = meta[test_s]["q_cat"]
        ex_cat_p, ex_cat_n = meta[train_s]["pos"]["cat"], meta[train_s]["neg"]["cat"]

        m = fit_cobras(pos, neg)
        wp, wn = m.attribution(X)
        V = steering_step(m, q_dir)
        fold = dict(n_queries=int(len(X)), R=m.R, sigma=m.sigma2 ** 0.5)

        fold["concentration"] = {"D+": concentration(wp, args.topk), "D-": concentration(wn, args.topk)}
        fold["log_weight_variance_query_independent"] = float(
            wp.log().mean(0).var(unbiased=False) / wp.log().var(unbiased=False))

        fold["semantics"] = {}
        for side, W, ec in (("D+", wp, ex_cat_p), ("D-", wn, ex_cat_n)):
            Wc = W - W.mean(0, keepdim=True)
            fold["semantics"][side] = dict(
                by_weight=category_share(W, ec, q_cat, args.topk),
                by_query_specific_weight=category_share(Wc, ec, q_cat, args.topk),
            )

        q0 = q_dir * m.R
        caa = CAA().fit(pos, neg).steer_vec
        caa_t = caa.unsqueeze(0) - (caa.unsqueeze(0) * q0).sum(-1, keepdim=True) / (m.R ** 2) * q0
        # w = 1/N everywhere: the step becomes a plain difference of Riemannian centroids
        V_unif = step_with_weights(m, q0, torch.full_like(wp, 1.0 / wp.shape[1]),
                                   torch.full_like(wn, 1.0 / wn.shape[1]))
        # Prop. 1's limit: uniform Sinkhorn potentials, so the weights are the kernel alone
        V_kde = step_with_weights(m, q0,
                                  m._kernel_weights(q0, m.h_pos, torch.zeros(wp.shape[1], device=q0.device))[0],
                                  m._kernel_weights(q0, m.h_neg, torch.zeros(wn.shape[1], device=q0.device))[0])
        fold["adaptivity"] = dict(
            COBRAS=direction_spread(V, q0, m.R),
            CAA_projected=direction_spread(caa_t, q0, m.R),
            cos_to_uniform_weights=float(F.cosine_similarity(V, V_unif, dim=-1).mean()),
            cos_to_kde_weights_prop1=float(F.cosine_similarity(V, V_kde, dim=-1).mean()),
            cos_to_CAA_projected=float(F.cosine_similarity(V, caa_t, dim=-1).mean()),
            weights_Dplus=distribution_spread(wp, q_cat, args.topk),
        )

        fold["bandwidth_ladder"] = []
        for s in _BW_LADDER:
            mb = fit_cobras(pos, neg, bandwidth="fixed", bandwidth_scale=s)
            bp, bn = mb.attribution(X)
            Vb = steering_step(mb, q_dir)
            Wc = bp - bp.mean(0, keepdim=True)
            fold["bandwidth_ladder"].append(dict(
                sigma2_scale=s, sigma_scale=s ** 0.5,
                ess_frac=concentration(bp, args.topk)["ess_frac"],
                topk_mass=concentration(bp, args.topk)["topk_mass"],
                cat_by_weight=category_share(bp, ex_cat_p, q_cat, args.topk)["lift"],
                cat_by_query_specific=category_share(Wc, ex_cat_p, q_cat, args.topk)["lift"],
                cos_to_shipped=float(F.cosine_similarity(Vb, V, dim=-1).mean()),
                cos_to_CAA_projected=float(F.cosine_similarity(Vb, caa_t, dim=-1).mean()),
                mean_pairwise_cos=direction_spread(Vb, q_dir * mb.R, mb.R)["mean_pairwise_cos"],
            ))

        leave_rows += [dict(fold=test_s, **r) for r in leave_out(
            pos, neg, q_dir, V, wp, wn, pd.unique(ex_cat_p), meta[train_s], q_cat,
            args.topk, args.n_leaveout, args.seed)]

        questions = load_tqa_gen_questions(test_s)
        gens = load_generations(args.model, args.layer_idx, questions)
        examples += worked_examples(wp, wn, meta[train_s], questions, gens, q_cat,
                                    args.n_examples, 3, args.seed + test_s)
        res["folds"][f"fold{test_s}"] = fold
        print(f"fold{test_s}: ESS(D+) = {fold['concentration']['D+']['ess_frac']:.4f}N, "
              f"category lift {fold['semantics']['D+']['by_weight']['lift']:.2f}x (raw) / "
              f"{fold['semantics']['D+']['by_query_specific_weight']['lift']:.2f}x (query-specific)")

    lv = pd.DataFrame(leave_rows)
    out_dir = get_project_dir() / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}-l{args.layer_idx}{args.tag}"
    lv.to_csv(out_dir / f"{tag}-attribution-leaveout.csv", index=False)

    g = lv.groupby("cond").agg(n=("angle", "size"), mean_angle=("angle", "mean"),
                               median_angle=("angle", "median"), mean_cos=("cos", "mean"),
                               n_pos=("n_pos_removed", "mean"), n_neg=("n_neg_removed", "mean"))
    piv = lv.pivot_table(index=["fold", "q"], columns="cond", values="angle")
    pairs = [("top-k by weight", "k at random"),
             ("top-k by query-specific weight", "k at random"),
             ("the query's own category", "another category, size-matched"),
             ("the query's own category", "random questions, size-matched")]
    res["leave_out"] = dict(
        summary=json.loads(g.to_json(orient="index")),
        contrasts={f"{a} vs {b}": dict(
            delta_deg=float((piv[a] - piv[b]).mean()),
            median_deg=float((piv[a] - piv[b]).median()),
            p_wilcoxon=float(wilcoxon((piv[a] - piv[b]).dropna()).pvalue),
            frac_greater=float(((piv[a] - piv[b]) > 0).mean()),
        ) for a, b in pairs},
    )
    res["examples"] = examples
    (out_dir / f"{tag}-attribution.json").write_text(json.dumps(res, indent=2, default=float))

    print(f"\n{g.round(4).to_string()}\n")
    for k, v in res["leave_out"]["contrasts"].items():
        print(f"  {k:<62} {v['delta_deg']:+.3f} deg  p={v['p_wilcoxon']:.3g}  "
              f"frac>{0}={v['frac_greater']:.3f}")
    print(f"\n saved {out_dir / f'{tag}-attribution.json'}")
