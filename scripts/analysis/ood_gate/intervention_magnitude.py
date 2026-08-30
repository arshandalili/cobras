"""How much does each steering method intervene on in-distribution vs OOD queries?

Fits every steer model on TruthfulQA (as the OOD evaluations do) and measures the
relative displacement it applies at the steering site, plus the internal quantities
COBRAS exposes (drift norm, Schroedinger potentials, kNN radius).

    uv run python -u scripts/analysis/ood_gate/intervention_magnitude.py -m Llama3.1-8B-Base -l 13
"""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import torch

from cobras.steer import get_steer_model
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data_all_splits


ID_TASK = "truthfulqa"

_STEER_KWARGS = {
    "COBRAS": dict(
        k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
        vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0,
    ),
    "ODESteer": dict(
        solver="euler", steps=10, n_components=8000, degree=2,
        gamma=0.1, coef0=1.0, lin_clf_type="lr",
    ),
    "SphericalSteer": dict(kappa=20.0, alpha=0.7, beta=-0.15),
}

# the reported label stays "COBRAS"; the class is AblationCOBRAS because this script reads
# query_stats and sweeps abstain_signal / abstain_calibration, which only it exposes
_STEER_CLS = {"COBRAS": "AblationCOBRAS"}

_GATES = {
    "knn-ratio(p98,paper)": dict(abstain_percentile=0.98, abstain_signal="knn", abstain_calibration="ratio"),
    "knn-quantile(p90)": dict(abstain_percentile=0.90, abstain_signal="knn", abstain_calibration="quantile"),
    "drift-quantile(p90)": dict(abstain_percentile=0.90, abstain_signal="drift", abstain_calibration="quantile"),
}

_MODEL_T = {
    "Llama3.1-8B-Base": 4.0,
    "Falcon-7B-Base": 21.0,
    "Mistral-7B-Base": 3.0,
    "Qwen2.5-7B-Base": 14.0,
}


def _chunked(fn, X: torch.Tensor, chunk: int):
    return [fn(X[i : i + chunk]) for i in range(0, X.size(0), chunk)]


def relative_displacement(steer_model, X: torch.Tensor, T: float, chunk: int) -> torch.Tensor:
    def step(x):
        if hasattr(steer_model, "reset_gate"):
            steer_model.reset_gate()
        return (steer_model.steer(x, T=T) - x).norm(dim=-1) / x.norm(dim=-1)
    return torch.cat(_chunked(step, X, chunk))


def query_stats(steer_model, X: torch.Tensor, chunk: int) -> dict[str, torch.Tensor]:
    parts = _chunked(steer_model.query_stats, X, chunk)
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def summarize(signal: dict[str, np.ndarray], higher_is_id: bool) -> dict:
    """Mean per task plus AUROC of the signal separating ID from each OOD task."""
    id_vals = signal[ID_TASK]
    out = {"mean": {k: float(v.mean()) for k, v in signal.items()}}
    out["ood_id_ratio"] = {
        k: float(v.mean() / id_vals.mean()) for k, v in signal.items() if k != ID_TASK
    }
    out["auroc"] = {}
    for task, vals in signal.items():
        if task == ID_TASK:
            continue
        y = np.concatenate([np.ones_like(id_vals), np.zeros_like(vals)])
        score = np.concatenate([id_vals, vals])
        out["auroc"][task] = float(roc_auc_score(y, score if higher_is_id else -score))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="Llama3.1-8B-Base")
    parser.add_argument("-l", "--layer_idx", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--chunk", type=int, default=64)
    args = parser.parse_args()

    act_dir = get_project_dir() / "data" / "query_activations" / args.model
    queries = {
        p.stem.replace(f"_layer{args.layer_idx}", ""): torch.load(p, weights_only=True).float().to(args.device)
        for p in sorted(act_dir.glob(f"*_layer{args.layer_idx}.pt"))
    }
    assert ID_TASK in queries, f"missing {ID_TASK} activations in {act_dir}"
    print(f"→ Query sets: " + ", ".join(f"{k} ({len(v)})" for k, v in queries.items()))

    # fit on CPU tensors, as the generation scripts do (sklearn-backed baselines need them)
    pos_X, neg_X = load_tqa_gen_data_all_splits(args.model, args.layer_idx)
    pos_X, neg_X = pos_X.float(), neg_X.float()
    T = _MODEL_T.get(args.model, 1.0)

    results = {"model": args.model, "layer_idx": args.layer_idx, "displacement": {}}

    for name in ("RepE", "ITI", "CAA", "MiMiC", "LinAcT", "ODESteer", "SphericalSteer", "COBRAS"):
        steer_model = get_steer_model(_STEER_CLS.get(name, name), **_STEER_KWARGS.get(name, {}))
        steer_model.fit(pos_X, neg_X)
        steer_T = 0.5 if name == "COBRAS" else T
        disp = {
            task: relative_displacement(steer_model, X, steer_T, args.chunk).cpu().numpy()
            for task, X in queries.items()
        }
        results["displacement"][name] = summarize(disp, higher_is_id=True)
        print(f"✓ {name:<15} T={steer_T:<5} " + "  ".join(
            f"{t}={v.mean():.4f}" for t, v in disp.items()
        ))

        if name == "COBRAS":
            stats = {task: query_stats(steer_model, X, args.chunk) for task, X in queries.items()}
            results["cobras_signals"] = {}
            for key in stats[ID_TASK]:
                signal = {t: s[key].cpu().numpy() for t, s in stats.items()}
                # a query is "in distribution" when density/drift is high, radius low
                higher_is_id = not key.endswith("radius") and key != "nn_dist"
                results["cobras_signals"][key] = summarize(signal, higher_is_id)

            rho = np.concatenate([s["knn_radius"].cpu().numpy() for s in stats.values()])
            log_p = np.concatenate([s["log_product_fixed"].cpu().numpy() for s in stats.values()])
            results["knn_vs_density_spearman"] = float(spearmanr(rho, -log_p).statistic)

            # how much steering each abstention rule leaves in place, per task
            results["gate_value"] = {}
            for tag, gate_kwargs in _GATES.items():
                gated = get_steer_model("AblationCOBRAS", **{**_STEER_KWARGS["COBRAS"], **gate_kwargs})
                gated.fit(pos_X, neg_X)
                results["gate_value"][tag] = {}
                for task, X in queries.items():
                    gated._to(X.device, X.dtype)
                    p0 = X * (gated.R / X.norm(dim=-1, keepdim=True))
                    g = torch.cat(_chunked(gated._abstain_gate, p0, args.chunk)).cpu().numpy()
                    results["gate_value"][tag][task] = {
                        "mean": float(g.mean()), "frac_below_0.5": float((g < 0.5).mean()),
                    }

    out_path = get_project_dir() / "results" / "analysis" / f"{args.model}-l{args.layer_idx}-intervention.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved {out_path}")

    print(f"\n{'signal':<26}" + "".join(f"{t[:9]:>11}" for t in queries) + "   AUROC(ID vs OOD)")
    rows = [(f"disp:{k}", v) for k, v in results["displacement"].items()]
    rows += [(f"cobras:{k}", v) for k, v in results["cobras_signals"].items()]
    for label, entry in rows:
        means = "".join(f"{entry['mean'][t]:>11.4f}" for t in queries)
        aurocs = " ".join(f"{t[:4]}={v:.3f}" for t, v in entry["auroc"].items())
        print(f"{label:<26}{means}   {aurocs}")
    print("\nmean gate value (fraction with gate < 0.5)")
    for tag, per_task in results["gate_value"].items():
        print(f"  {tag:<22}" + "  ".join(
            f"{t[:9]}={v['mean']:.3f}({v['frac_below_0.5']:.2f})" for t, v in per_task.items()
        ))
    print(f"spearman(kNN radius, -log density) = {results['knn_vs_density_spearman']:.3f}")
