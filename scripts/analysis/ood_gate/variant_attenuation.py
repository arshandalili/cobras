"""How much of its in-distribution intervention does each COBRAS variant keep on OOD queries?

Measured directly on cached activations, so it predicts the OOD damage of a variant without
running generation: a variant that keeps ~100% of its step on OOD inputs cannot preserve OOD
behaviour, whatever the rest of the method does.

    uv run python -u scripts/analysis/ood_gate/variant_attenuation.py -m Llama3.1-8B-Base -l 13
"""

import argparse
import json

import torch

# the ablation switches this script sweeps (potentials=, drift=, bandwidth=, step_mode=,
# uniform_weights=, abstain_signal=) live on AblationCOBRAS; with defaults it is COBRAS
from cobras.steer import AblationCOBRAS as COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data_all_splits


ID_TASK = "truthfulqa"

_BASE = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10, vmf_beta=0.0)

_VARIANTS = {
    "no gate, unit step (Table 1 config)": dict(vmf_kappa=20, abstain_percentile=1.0),
    "+ kNN gate p98 (Fig. 2 config)": dict(vmf_kappa=20, abstain_percentile=0.534),
    "no gate, raw step, centroid drift": dict(
        vmf_kappa=20, abstain_percentile=1.0, step_mode="raw", drift="centroid"),
    "no gate, raw step, sigma restored": dict(
        vmf_kappa=20, abstain_percentile=1.0, step_mode="raw", drift="gradient"),
    "no gate, raw step, sigma restored, no vMF": dict(
        vmf_kappa=None, abstain_percentile=1.0, step_mode="raw", drift="gradient"),
    "+ SB density gate (p0 = psi*phi)": dict(
        vmf_kappa=20, abstain_percentile=0.9, abstain_signal="density",
        abstain_on_queries=True),
    "+ SB drift gate": dict(
        vmf_kappa=20, abstain_percentile=0.9, abstain_signal="drift",
        abstain_on_queries=True),
}


def _sweep_variants() -> dict[str, dict]:
    """Same coverage levels for the paper's gate and the SB-marginal gate, to compare how
    sharply each reacts to its threshold (Table 2 of the paper varies only the former)."""
    out = {}
    for p in (0.80, 0.90, 0.95, 0.98, 0.995):
        out[f"kNN gate, ratio, p={p}"] = dict(vmf_kappa=20, abstain_percentile=p)
        out[f"SB density gate, quantile, p={p}"] = dict(
            vmf_kappa=20, abstain_percentile=p, abstain_signal="density",
            abstain_on_queries=True)
    return out


def relative_displacement(model: COBRAS, X: torch.Tensor, T: float, chunk: int) -> torch.Tensor:
    out = []
    for i in range(0, X.size(0), chunk):
        x = X[i : i + chunk]
        model.reset_gate()
        out.append((model.steer(x, T=T) - x).norm(dim=-1) / x.norm(dim=-1))
    return torch.cat(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="Llama3.1-8B-Base")
    parser.add_argument("-l", "--layer_idx", type=int, default=13)
    parser.add_argument("-T", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    act_dir = get_project_dir() / "data" / "query_activations" / args.model

    def _load(name: str):
        path = act_dir / f"{name}_layer{args.layer_idx}.pt"
        return torch.load(path, weights_only=True).float() if path.exists() else None

    # calibrate the gate on one TruthfulQA fold and score in-distribution on the other, so the
    # reported ID retention is out-of-sample
    ref_X, id_X = _load("truthfulqa_split0"), _load("truthfulqa_split1")
    if ref_X is None or id_X is None:
        ref_X = id_X = _load("truthfulqa")
    queries = {ID_TASK: id_X.to(args.device)}
    for path in sorted(act_dir.glob(f"*_layer{args.layer_idx}.pt")):
        name = path.stem.replace(f"_layer{args.layer_idx}", "")
        if not name.startswith("truthfulqa"):
            queries[name] = torch.load(path, weights_only=True).float().to(args.device)

    pos_X, neg_X = load_tqa_gen_data_all_splits(args.model, args.layer_idx)

    tasks = [ID_TASK] + [t for t in queries if t != ID_TASK]
    results = {}
    variants = {**_VARIANTS, **(_sweep_variants() if args.sweep else {})}
    print(f"{'variant':<44}" + "".join(f"{t[:9]:>11}" for t in tasks))
    for label, kwargs in variants.items():
        model = COBRAS(**_BASE, **kwargs).fit(pos_X.float(), neg_X.float(), ref_X=ref_X)
        disp = {t: relative_displacement(model, queries[t], args.T, args.chunk).mean().item() for t in tasks}
        kept = {t: disp[t] / max(disp[ID_TASK], 1e-9) for t in tasks if t != ID_TASK}
        results[label] = {"displacement": disp, "kept_vs_id": kept}
        print(f"{label:<44}" + "".join(f"{disp[t]:>11.4f}" for t in tasks))

    print(f"\nfraction of the in-distribution step kept on OOD queries")
    print(f"{'variant':<44}" + "".join(f"{t[:9]:>11}" for t in tasks if t != ID_TASK))
    for label, entry in results.items():
        print(f"{label:<44}" + "".join(f"{v:>11.2f}" for v in entry["kept_vs_id"].values()))

    out_path = get_project_dir() / "results" / "analysis" / f"{args.model}-l{args.layer_idx}-variant-attenuation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved {out_path}")
