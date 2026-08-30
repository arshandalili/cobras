"""Do the per-query quantities COBRAS reads at the steering site predict what the model does?

Every quantity here is available before a single token is generated. We ask whether it
anticipates the unsteered model's error on that question, and whether the steering budget it
sets is spent where the error is.

    uv run python -u scripts/analysis/attribution/behaviour.py -m Llama3.1-8B-Base -l 13
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch

# the ablation switches this script sweeps (potentials=, drift=, bandwidth=, step_mode=,
# uniform_weights=, abstain_signal=) live on AblationCOBRAS; with defaults it is COBRAS
from cobras.steer import AblationCOBRAS as COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data, load_query_activations, load_tqa_gen_questions

_COBRAS_KWARGS = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
                      vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0)

# runs compared per stratum; the key is the label used in the report
_RUNS = {
    "unsteered": "NoSteer",
    "CAA": "CAA-T4",
    "ODESteer": "ODESteer-euler-steps10-nc8000-degree2-gamma0.1-coef01.0-lr-T4",
    "SphericalSteer": "SphericalSteer-kappa20.0-alpha0.7-beta-0.15-T4",
    "COBRAS": "COBRAS-k_bw5-sink5-alpha0.001-eps0.0-iters10-kappa20-abst1.0-k32-sh50.0-T0.5",
}


def load_judged(model: str, layer_idx: int, seed: int, questions: list[str]) -> dict:
    d = get_project_dir() / "results" / "truthfulqa" / "eval_results" / "detailed_eval_results" / f"{model}-l{layer_idx}"
    out = {}
    for tag, run in _RUNS.items():
        p = d / f"{model}-l{layer_idx}-{run}-TruthfulQA-seed{seed}.jsonl"
        if not p.exists():
            print(f"  (missing {tag}: {p.name})")
            continue
        df = pd.read_json(p, lines=True)
        assert df.prompt.tolist() == questions, f"{tag}: prompt order does not match the splits"
        out[tag] = df
    return out


def query_signals(model: str, layer_idx: int, device: str) -> dict[str, np.ndarray]:
    """Everything COBRAS can read off h_q, in the same 2-fold protocol as generation."""
    sig = {}
    for test_s in (0, 1):
        pos, neg = load_tqa_gen_data(model, layer_idx, 1 - test_s)
        m = COBRAS(**_COBRAS_KWARGS).fit(pos.float().to(device), neg.float().to(device))
        X = load_query_activations(model, layer_idx, f"truthfulqa_split{test_s}")
        assert X is not None, "run scripts/prepare/extract_query_activations.py first"
        X = X.float().to(device)
        st = m.query_stats(X)
        q0 = X * (m.R / X.norm(dim=-1, keepdim=True))
        cur = dict(
            vmf_strength=st["vmf_strength"],
            cos_to_contrastive_axis=((q0 / m.R) * m.mu_T).sum(-1),
            drift_norm=m._field(q0)[0].norm(dim=-1),
            log_bridge_marginal=st["log_product_fixed"],
            log_potential_ratio=st["log_ratio_fixed"],
            abstain_score=st["abstain_score"],
        )
        for k, v in cur.items():
            sig.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in sig.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Llama3.1-8B-Base")
    ap.add_argument("-l", "--layer_idx", type=int, default=13)
    ap.add_argument("-s", "--seed", type=int, default=42)
    ap.add_argument("-q", "--n_strata", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    questions = load_tqa_gen_questions(0) + load_tqa_gen_questions(1)
    judged = load_judged(args.model, args.layer_idx, args.seed, questions)
    assert "unsteered" in judged, "need the NoSteer detailed evaluation; run truthfulqa_eval.py"
    sig = query_signals(args.model, args.layer_idx, args.device)

    truth = {k: v["true"].to_numpy() for k, v in judged.items()}
    y = 1 - truth["unsteered"]
    res = {"model": args.model, "layer_idx": args.layer_idx, "n": int(len(y)),
           "n_untruthful_unsteered": int(y.sum())}

    rng = np.random.default_rng(0)
    res["auroc_predicting_unsteered_error"] = {}
    print(f"\nn={len(y)}, unsteered model is untruthful on {int(y.sum())}")
    print(f"\n{'signal read at the steering site':<34}{'AUROC':>9}{'permutation 95%':>22}{'p':>10}")
    for k, v in sig.items():
        a = roc_auc_score(y, v)
        null = np.array([roc_auc_score(rng.permutation(y), v) for _ in range(2000)])
        p = float(np.mean(np.abs(null - .5) >= abs(a - .5)))
        res["auroc_predicting_unsteered_error"][k] = dict(auroc=float(a), p_permutation=p)
        print(f"{k:<34}{a:>9.4f}   [{np.quantile(null,.025):.3f},{np.quantile(null,.975):.3f}]"
              f"{p:>16.4f}")

    s = sig["vmf_strength"]
    lab = [f"Q{i+1}" for i in range(args.n_strata)]
    tab = pd.DataFrame({"s": s, **truth})
    tab["stratum"] = pd.qcut(tab.s, args.n_strata, labels=lab)
    agg = tab.groupby("stratum", observed=True).agg(
        n=("s", "size"), mean_strength=("s", "mean"),
        **{k: (k, "mean") for k in truth})
    res["by_vmf_strength"] = json.loads(agg.to_json(orient="index"))

    print(f"\ntruthfulness (%) by quartile of the vMF strength gate (Alg. 1 line 6)")
    show = agg.copy()
    for k in truth:
        show[k] = (100 * show[k]).round(1)
    print(show.round(3).to_string())

    reg = {}
    was_right = tab[tab.unsteered == 1]
    for k in truth:
        if k == "unsteered":
            continue
        reg[k] = dict(
            by_stratum=(100 * (1 - was_right.groupby("stratum", observed=True)[k].mean())).round(1).to_dict(),
            overall=float(100 * (1 - was_right[k].mean())),
        )
    res["regressions_of_correct_answers"] = reg
    print(f"\ncorrect answers lost to steering (% of the questions the unsteered model got right)")
    print(f"{'':<18}" + "".join(f"{l:>9}" for l in lab) + f"{'overall':>10}")
    for k, v in reg.items():
        print(f"{k:<18}" + "".join(f"{v['by_stratum'].get(l, float('nan')):>9.1f}" for l in lab)
              + f"{v['overall']:>10.1f}")

    out = get_project_dir() / "results" / "analysis" / f"{args.model}-l{args.layer_idx}-attribution-behaviour.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\n saved {out}")
