"""Q3 (Reviewer EJdD): render the tables used in the rebuttal from the saved JSON."""

from __future__ import annotations

import collections
import glob
import json
import os

import numpy as np
import pandas as pd

from cobras.utils import get_project_dir

GEO = get_project_dir() / "results" / "analysis" / "q3_geometry"
SWEEP = get_project_dir() / "results" / "analysis" / "q3_layer_sweep"
STEER_LAYER = {"Llama3.1-8B-Base": 13, "Mistral-7B-Base": 15,
               "Falcon-7B-Base": 14, "Qwen2.5-7B-Base": 13}


def table_a() -> None:
    rows = json.load(open(GEO / "q3_part_a_paper_models.json"))
    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("TABLE A1: per-layer geometry, TruthfulQA contrastive activations (last token), N=5918")
    print("=" * 100)
    for m, g in df.groupby("model", sort=False):
        g = g.sort_values("layer")
        L = int(g.layer.max())
        mid = g[(g.layer >= 1) & (g.layer <= L - 1)]
        sl = g[g.layer == STEER_LAYER[m]].iloc[0]
        print(f"\n{m}  (d={int(g.d.iloc[0])}, layers 0-{L}, steering layer {STEER_LAYER[m]})")
        print(f"  layer 0            CoV={g[g.layer==0]["cov"].iloc[0]:.4f}  "
              f"reldisp={g[g.layer==0].reldisp_mean.iloc[0]:.4f}  "
              f"varfrac={g[g.layer==0].varfrac.iloc[0]:.5f}")
        print(f"  layers 1..{L-1:<2d} range CoV=[{mid["cov"].min():.4f}, {mid["cov"].max():.4f}]  "
              f"reldisp=[{mid.reldisp_mean.min():.4f}, {mid.reldisp_mean.max():.4f}]  "
              f"varfrac=[{mid.varfrac.min():.5f}, {mid.varfrac.max():.5f}]")
        print(f"  layer {L:<2d}           CoV={g[g.layer==L]["cov"].iloc[0]:.4f}  "
              f"reldisp={g[g.layer==L].reldisp_mean.iloc[0]:.4f}  "
              f"varfrac={g[g.layer==L].varfrac.iloc[0]:.5f}")
        print(f"  steering layer {STEER_LAYER[m]:<2d}  CoV={sl["cov"]:.4f}  reldisp={sl.reldisp_mean:.4f}  "
              f"p95={sl.reldisp_p95:.4f}  varfrac={sl.varfrac:.5f}"
              + (f"  AUC raw/sphere={sl.auc_raw:.4f}/{sl.auc_sphere:.4f}" if "auc_raw" in df else ""))

    if "auc_raw" in df:
        print("\n" + "=" * 100)
        print("TABLE A2: contrastive separability, raw vs projected (logistic probe, 2-fold, held-out AUC)")
        print("=" * 100)
        for m, g in df.groupby("model", sort=False):
            g = g.sort_values("layer")
            L = int(g.layer.max())
            d = (g.auc_sphere - g.auc_raw)
            sl = g[g.layer == STEER_LAYER[m]].iloc[0]
            print(f"{m:20s} steer L{STEER_LAYER[m]:<2d}: raw={sl.auc_raw:.4f} sphere={sl.auc_sphere:.4f} "
                  f"delta={sl.auc_sphere - sl.auc_raw:+.4f} | over all layers delta in "
                  f"[{d.min():+.4f}, {d.max():+.4f}], mean {d.mean():+.4f}, "
                  f"worst layer {int(g.layer.iloc[int(d.values.argmin())])}")

    print("\nfull per-layer dump (model, layer, CoV, reldisp_mean, reldisp_p95, varfrac"
          + (", auc_raw, auc_sphere)" if "auc_raw" in df else ")"))
    cols = ["model", "layer", "cov", "reldisp_mean", "reldisp_p95", "varfrac"]
    if "auc_raw" in df:
        cols += ["auc_raw", "auc_sphere"]
    print(df[cols].to_string(index=False,
          float_format=lambda x: f"{x:.5f}"))


def table_b() -> None:
    rows = json.load(open(GEO / "q3_part_b_other_archs.json"))
    df = pd.DataFrame(rows)
    rms = pd.DataFrame(json.load(open(GEO / "q3_part_b_rmsnorm.json")))
    print("\n" + "=" * 100)
    print("TABLE B1: per-layer residual-norm geometry on pile-10k (500 texts, ~110k tokens),")
    print("          four architecturally different models. 'sink' = tokens with ||h|| > 5x median.")
    print("=" * 100)
    for m, g in df.groupby("model", sort=False):
        g = g.sort_values("layer")
        L = int(g.L.iloc[0])
        r = rms[rms.model == m]
        n_sink = int(round(g.sink_frac.max() * g.N.iloc[0]))
        print(f"\n{m}  (L={L}, d={int(g.d.iloc[0])}, N={int(g.N.iloc[0])} tokens from 500 sequences)")
        print(f"  all tokens         CoV in [{g["cov"].min():.3f}, {g["cov"].max():.3f}]   "
              f"median {g["cov"].median():.3f}")
        print(f"  non-sink tokens    CoV in [{g.ns_cov.min():.4f}, {g.ns_cov.max():.4f}]  "
              f"median {g.ns_cov.median():.4f}   "
              f"reldisp median {g.ns_reldisp_mean.median():.4f}")
        print(f"  sink tokens        present in layers "
              f"{sorted(g[g.sink_frac > 0].layer.tolist())}, "
              f"{n_sink} tokens ({100 * g.sink_frac.max():.2f}%), "
              f"max ||h||/median = {g.sink_max_over_median.max():.0f}")
        if len(r):
            print(f"  post-RMSNorm       CoV in [{r.pre_cov.min():.4f}, {r.pre_cov.max():.4f}]  "
                  f"median {r.pre_cov.median():.4f}  (attn input, ALL tokens incl. sinks)")
            print(f"                     CoV in [{r.post_cov.min():.4f}, {r.post_cov.max():.4f}]  "
                  f"median {r.post_cov.median():.4f}  (MLP input, ALL tokens incl. sinks)")
    print("\nfull per-layer dump")
    print(df[["model", "layer", "cov", "ns_cov", "reldisp_mean", "ns_reldisp_mean",
              "sink_frac", "sink_max_over_median"]].to_string(
        index=False, float_format=lambda x: f"{x:.5f}"))
    print("\npost-RMSNorm per-layer dump")
    print(rms.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


def table_d() -> None:
    """Does the concentration statistic predict how much the projection reorders the
    neighbourhoods that Eqs. (15)-(19) softmax over?"""
    d = pd.DataFrame(json.load(open(GEO / "q3_part_d_metric_distortion.json")))
    a = pd.DataFrame(json.load(open(GEO / "q3_part_a_paper_models.json")))
    m = d.merge(a[["model", "layer", "cov", "reldisp_mean", "varfrac"]], on=["model", "layer"])
    print("\n" + "=" * 100)
    print("TABLE D: distortion of the metric the KDE weights read, contrastive samples, n=2000/layer")
    print("  spearman = rank corr. between ||x-y|| and d_S(Rx/||x||, Ry/||y||) over all pairs")
    print("  knn_overlap = mean fraction of the 32 Euclidean nearest neighbours retained "
          "under d_S")
    print("=" * 100)
    for mod, g in m.groupby("model", sort=False):
        g = g.sort_values("layer")
        L = int(g.layer.max())
        sl = g[g.layer == STEER_LAYER[mod]].iloc[0]
        w = g.loc[g.knn_overlap.idxmin()]
        print(f"{mod:20s} steer L{STEER_LAYER[mod]:<2d}: spearman={sl.spearman_euc_vs_geo:.3f} "
              f"knn_overlap={sl.knn_overlap:.3f} (CoV {sl["cov"]:.4f}) | "
              f"worst layer {int(w.layer)}: spearman={w.spearman_euc_vs_geo:.3f} "
              f"overlap={w.knn_overlap:.3f} (CoV {w["cov"]:.4f}) | "
              f"best layer {int(g.loc[g.knn_overlap.idxmax()].layer)}: "
              f"overlap={g.knn_overlap.max():.3f}")
    print("\nacross all 123 (model, layer) cells pooled:")
    print(f"  Spearman(CoV, knn_overlap)          = {m["cov"].corr(m.knn_overlap, method='spearman'):+.3f}")
    print(f"  Spearman(CoV, pairwise-rank rho)    = "
          f"{m["cov"].corr(m.spearman_euc_vs_geo, method='spearman'):+.3f}")
    print(f"  Spearman(reldisp, knn_overlap)      = "
          f"{m.reldisp_mean.corr(m.knn_overlap, method='spearman'):+.3f}")
    print("\nwithin each model:")
    for mod, g in m.groupby("model", sort=False):
        print(f"  {mod:20s} Spearman(CoV, knn_overlap) = "
              f"{g["cov"].corr(g.knn_overlap, method='spearman'):+.3f}  (n={len(g)} layers)")
    print("\nfull dump")
    print(m[["model", "layer", "cov", "spearman_euc_vs_geo", "knn_overlap"]].to_string(
        index=False, float_format=lambda x: f"{x:.5f}"))


def table_e() -> None:
    rows = json.load(open(GEO / "q3_part_c_query_radius.json"))
    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("TABLE E: does the radius R fitted on D+ u D- transfer to the query populations?")
    print("  R_fit  = mean ||h|| over the contrastive set (COBRAS.fit, _cobras.py:89)")
    print("  R_task = mean ||q|| over that task's prompt activations at the same layer")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def table_c() -> None:
    p = SWEEP / "q3_layer_sweep_eval.csv"
    if not os.path.exists(p):
        print("\n[table C] no eval csv yet")
        return
    ev = pd.read_csv(p)
    geo = pd.DataFrame(json.load(open(GEO / "q3_part_a_paper_models.json")))
    for extra, cols in ((GEO / "q3_part_d_metric_distortion.json", ["knn_overlap"]),
                        (GEO / "q3_part_e_angular_spread.json", ["mean_pairwise_angle_deg"])):
        if os.path.exists(extra):
            geo = geo.merge(
                pd.DataFrame(json.load(open(extra)))[["model", "layer"] + cols],
                on=["model", "layer"], how="left")
    print("\n" + "=" * 100)
    print("TABLE C: TruthfulQA True x Info (%) by steering layer, COBRAS vs Euclidean control")
    print("=" * 100)
    for m, g in ev.groupby("model", sort=False):
        piv = g.pivot_table(index="layer", columns="method", values="true_info")
        gg = geo[geo.model == m].set_index("layer")
        keep = [c for c in ["cov", "reldisp_mean", "varfrac", "auc_raw",
                            "knn_overlap", "mean_pairwise_angle_deg"] if c in gg.columns]
        piv = piv.join(gg[keep])
        piv["gap"] = 100 * (piv.get("cobras", np.nan) - piv.get("euclid", np.nan))
        piv["cobras"] = 100 * piv["cobras"]
        piv["euclid"] = 100 * piv["euclid"]
        print(f"\n{m}")
        print(piv.to_string(float_format=lambda x: f"{x:.4f}"))
        sub = piv.dropna(subset=["gap"])
        if len(sub) > 2:
            print(f"  Spearman(CoV, COBRAS TxI)   = "
                  f"{sub['cov'].corr(sub['cobras'], method='spearman'):+.3f}")
            print(f"  Spearman(CoV, gap)          = "
                  f"{sub['cov'].corr(sub['gap'], method='spearman'):+.3f}")
    print("\nfull eval csv")
    print(ev.to_string(index=False))

    d = SWEEP / "q3_update_geometry_Llama3.1-8B-Base.json"
    if os.path.exists(d):
        print("\nupdate geometry (per fold, held-out activations)")
        print(pd.DataFrame(json.load(open(d))).to_string(
            index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "abc"
    if "a" in which and os.path.exists(GEO / "q3_part_a_paper_models.json"):
        table_a()
    if "b" in which:
        table_b()
    if "d" in which and os.path.exists(GEO / "q3_part_d_metric_distortion.json"):
        table_d()
    if "e" in which and os.path.exists(GEO / "q3_part_c_query_radius.json"):
        table_e()
    if "c" in which:
        table_c()
