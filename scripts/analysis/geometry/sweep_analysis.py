"""Q3 (Reviewer EJdD): join the layer-sweep scores to the per-layer geometry statistics.

Reads only files already written by geometry/layer_sweep.py, geometry/eval_sweep.py and geometry/geometry.py,
and answers one question: does the COBRAS-minus-Euclidean gap track any measure of how well
the spherical approximation holds at that layer?

    ./.venv/bin/python -u scripts/analysis/geometry/sweep_analysis.py

Writes results/analysis/q3_layer_sweep/q3_sweep_summary.csv and
results/analysis/q3_layer_sweep/q3_sweep_correlations.csv
"""

from __future__ import annotations

import json

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from cobras.utils import get_project_dir

ROOT = get_project_dir()
SWEEP = ROOT / "results" / "analysis" / "q3_layer_sweep"
GEOM = ROOT / "results" / "analysis" / "q3_geometry"
MODEL = "Llama3.1-8B-Base"
# results/truthfulqa/eval_results/stat_results/Llama3.1-8B-Base-l13-TruthfulQA-seed42.csv
NOSTEER = 0.419828641370869


def load_json_map(path, keys):
    rows = json.load(open(path))
    return {r["layer"]: {k: r[k] for k in keys if k in r} for r in rows if r["model"] == MODEL}


if __name__ == "__main__":
    ev = pd.read_csv(SWEEP / "q3_layer_sweep_eval.csv")
    ev = ev[ev.model == MODEL]
    wide = ev.pivot(index="layer", columns="method",
                    values=["true_info", "truthfulness", "informativeness"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["delta_true_info"] = wide.true_info_cobras - wide.true_info_euclid

    part_a = load_json_map(GEOM / "q3_part_a_paper_models.json",
                           ["cov", "reldisp_mean", "varfrac", "auc_raw", "auc_sphere", "R"])
    part_d = load_json_map(GEOM / "q3_part_d_metric_distortion.json",
                           ["knn_overlap", "spearman_euc_vs_geo"])
    upd = json.load(open(SWEEP / f"q3_update_geometry_{MODEL}.json"))
    upd = {r["layer"]: r for r in upd}

    for col in ["cov", "reldisp_mean", "varfrac", "auc_raw", "auc_sphere"]:
        wide[col] = [part_a[l][col] for l in wide.layer]
    for col in ["knn_overlap", "spearman_euc_vs_geo"]:
        wide[col] = [part_d[l][col] for l in wide.layer]
    for col in ["R", "mu_T_cos", "cos_updates", "rel_disp_sph", "rel_disp_euc",
                "norm_change_sph", "norm_change_euc"]:
        wide[col] = [
            (upd[l][f"fold0_{col}"] + upd[l][f"fold1_{col}"]) / 2 if l in upd else float("nan")
            for l in wide.layer
        ]
    wide["nosteer"] = NOSTEER
    wide = wide.sort_values("layer")
    wide.to_csv(SWEEP / "q3_sweep_summary.csv", index=False)

    rows = []
    for col in ["layer", "cov", "reldisp_mean", "varfrac", "knn_overlap",
                "spearman_euc_vs_geo", "auc_raw", "cos_updates", "norm_change_euc", "R"]:
        rho, p_s = spearmanr(wide[col], wide.delta_true_info)
        r, p_p = pearsonr(wide[col], wide.delta_true_info)
        rows.append(dict(predictor=col, n_layers=len(wide), spearman=rho, spearman_p=p_s,
                         pearson=r, pearson_p=p_p))
    # Is depth separable from the geometry statistics? Only in a model whose kNN overlap is
    # not itself monotone in depth. Checked over all layers of all four paper models.
    dpart = json.load(open(GEOM / "q3_part_d_metric_distortion.json"))
    dd = pd.DataFrame(dpart)
    for model, g in dd.groupby("model"):
        g = g.sort_values("layer")
        rho, p_s = spearmanr(g.layer, g.knn_overlap)
        rows.append(dict(predictor=f"[all layers of {model}] layer vs knn_overlap",
                         n_layers=len(g), spearman=rho, spearman_p=p_s,
                         pearson=float("nan"), pearson_p=float("nan")))

    corr = pd.DataFrame(rows).sort_values("spearman", key=abs, ascending=False)
    corr.to_csv(SWEEP / "q3_sweep_correlations.csv", index=False)

    pd.set_option("display.width", 200)
    print(wide[["layer", "true_info_cobras", "true_info_euclid", "delta_true_info",
                "cov", "knn_overlap", "cos_updates", "norm_change_euc",
                "auc_raw"]].to_string(index=False))
    print()
    print(corr.to_string(index=False))
    print(f"\nwrote {SWEEP / 'q3_sweep_summary.csv'} and {SWEEP / 'q3_sweep_correlations.csv'}")
