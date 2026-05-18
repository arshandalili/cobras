#!/usr/bin/env python3
"""
    Run it using 
    uv run python -u scripts/plots/ood_pareto.py \
    --save-path results/plots/tqa_vs_ood_curves_absolute.png \

"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from cobras.utils import get_project_dir


TRUTHFULQA_METRICS = [
    "True * Info",
    "Truthfulness",
    "Informativeness",
]

ULTRAFEEDBACK_METRICS = [
    "RM Win-Rate vs NoSteer",
    "RM Mean",
    "RM P90",
]

MMLU_METRICS = [
    "Accuracy"
]

METHOD_ORDER = [
    "Original",
    "RepE",
    "ITI",
    "CAA",
    "MiMiC",
    "HPR",
    "RE-Control",
    "Linear-AcT",
    "TruthFlow",
    "ODESteer",
    "SphericalSteer",
    "COBRAS (Ours)",
]

METHOD_MAP = {
    "NoSteer": "Original",
    "RepE": "RepE",
    "ITI": "ITI",
    "CAA": "CAA",
    "MiMiC": "MiMiC",
    "HPR": "HPR",
    "RE-Control": "RE-Control",
    "REControl": "RE-Control",
    "LinAcT": "Linear-AcT",
    "Linear-AcT": "Linear-AcT",
    "TruthFlow": "TruthFlow",
    "ODESteer": "ODESteer",
    "SphericalSteer": "SphericalSteer",
    "COBRAS": "COBRAS (Ours)",
}

MODEL_DISPLAY_MAP = {
    "Llama3.1-8B-Base-l13": "LLaMA3.1-8B",
    "Falcon-7B-Base-l14": "Falcon-7B",
    "Mistral-7B-Base-l15": "Mistral-7B",
    "Qwen2.5-7B-Base-l13": "Qwen2.5-7B",

    "Llama3.1-8B-Base": "LLaMA3.1-8B",
    "Falcon-7B-Base": "Falcon-7B",
    "Mistral-7B-Base": "Mistral-7B",
    "Qwen2.5-7B-Base": "Qwen2.5-7B",
}

MODEL_ORDER = [
    "LLaMA3.1-8B",
    "Falcon-7B",
    "Mistral-7B",
    "Qwen2.5-7B",
]

SHORTLISTED_T_MAP = {
    "LLaMA3.1-8B" : [3, 4, 5, 6],
    "Falcon-7B" : [20, 21, 22, 23],
    "Mistral-7B" : [2, 3, 4, 5],
    "Qwen2.5-7B" : [13, 14, 15, 16],
}

T_REGEX = re.compile(r"-T(?P<T>\d+(?:\.\d+)?)$")


def format_mean_std(
    mean: float,
    std: float,
    mean_precision: int = 1,
    std_precision: int = 3,
) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{mean_precision}f} {{\\tiny $\\pm$ {std:.{std_precision}f}}}"


def normalize_t(raw: str) -> str | None:
    match = T_REGEX.search(raw)
    if match is None:
        return None

    t = float(match.group("T"))

    if t.is_integer():
        return str(int(t))
    return str(t)



def normalize_method(raw: str) -> str:
    for prefix, mapped in METHOD_MAP.items():
        if raw.startswith(prefix):
            return mapped
    return raw


def normalize_model(raw: str) -> str:
    return MODEL_DISPLAY_MAP.get(raw, raw)


def load_results(task_name: str) -> pd.DataFrame:
    eval_dir = get_project_dir() / "results" / task_name / "eval_results"
    candidate_dirs = [eval_dir / "stat_results", eval_dir]

    for base_dir in candidate_dirs:
        if base_dir.exists():
            csv_files = sorted(base_dir.glob("*.csv"))
            if csv_files:
                dfs = []
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    df["__file__"] = csv_file.name
                    dfs.append(df)
                return pd.concat(dfs, ignore_index=True)

    raise FileNotFoundError(
        f"No CSV files found in any of: {candidate_dirs}"
    )


def expand_original_across_t_values(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """
    NoSteer has no T suffix, but we want Original to appear in every T block.
    This duplicates Original rows across all T values present for the same model.
    """
    df = df.copy()

    non_original = df[df["Method"] != "Original"].copy()
    original = df[df["Method"] == "Original"].copy()

    if original.empty:
        return non_original

    t_values_by_model = (
        non_original[["ModelDisplay", "T"]]
        .dropna()
        .drop_duplicates()
    )

    if t_values_by_model.empty:
        original["T"] = "NA"
        return original

    original = original.drop(columns=["T"], errors="ignore")

    expanded_original = original.merge(
        t_values_by_model,
        on="ModelDisplay",
        how="inner",
    )

    return pd.concat([non_original, expanded_original], ignore_index=True)


def prepare_common_columns(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["Method"] = df["Steering Method"].apply(normalize_method)
    df["ModelDisplay"] = df["Model"].apply(normalize_model)
    df["T"] = df["Steering Method"].apply(normalize_t)

    df = expand_original_across_t_values(df, metric_cols)
    return df

def aggregate_mmlu(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_common_columns(df, MMLU_METRICS)
    df[MMLU_METRICS] = df[MMLU_METRICS] * 100.0

    grouped = (
        df.groupby(["T", "ModelDisplay", "Method"], as_index=False)[MMLU_METRICS]
        .agg(["mean", "std", "count"])
    )

    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    std_cols = [f"{metric}_std" for metric in MMLU_METRICS if f"{metric}_std" in grouped.columns]
    grouped[std_cols] = grouped[std_cols].fillna(0.0)

    return grouped

def aggregate_truthfulqa(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_common_columns(df, TRUTHFULQA_METRICS)
    df[TRUTHFULQA_METRICS] = df[TRUTHFULQA_METRICS] * 100.0

    grouped = (
        df.groupby(["T", "ModelDisplay", "Method"], as_index=False)[TRUTHFULQA_METRICS]
        .agg(["mean", "std", "count"])
    )

    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    return grouped


def aggregate_ultrafeedback(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_common_columns(df, ULTRAFEEDBACK_METRICS)
    df["RM Win-Rate vs NoSteer"] = df["RM Win-Rate vs NoSteer"] * 100.0

    grouped = (
        df.groupby(["T", "ModelDisplay", "Method"], as_index=False)[ULTRAFEEDBACK_METRICS]
        .agg(["mean", "std", "count"])
    )

    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    return grouped

def select_best_t_and_filter_target(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    metric: str,
    higher_is_better: bool = True,
    group_cols: list[str] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select the best T from reference_df for each group, then filter target_df
    to only those T values.
    """
    if group_cols is None:
        group_cols = ["ModelDisplay", "Method"]

    metric_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if metric_col not in reference_df.columns:
        raise KeyError(f"Missing column in reference_df: {metric_col}")
    if "T" not in reference_df.columns:
        raise KeyError("Missing column in reference_df: T")
    if "T" not in target_df.columns:
        raise KeyError("Missing column in target_df: T")

    ref = reference_df.copy()
    tgt = target_df.copy()

    # Normalize join keys
    ref["ModelDisplay"] = ref["ModelDisplay"].astype(str).str.strip()
    tgt["ModelDisplay"] = tgt["ModelDisplay"].astype(str).str.strip()
    ref["Method"] = ref["Method"].astype(str).str.strip()
    tgt["Method"] = tgt["Method"].astype(str).str.strip()

    ref["T_num"] = pd.to_numeric(ref["T"], errors="coerce")
    tgt["T_num"] = pd.to_numeric(tgt["T"], errors="coerce")

    ref = ref.dropna(subset=["T_num", metric_col])
    tgt = tgt.dropna(subset=["T_num"])

    if verbose:
        print("\n[DEBUG] Reference models:", sorted(ref["ModelDisplay"].unique()))
        print("[DEBUG] Target models:", sorted(tgt["ModelDisplay"].unique()))
        print("[DEBUG] Reference methods:", sorted(ref["Method"].unique()))
        print("[DEBUG] Target methods:", sorted(tgt["Method"].unique()))
        print("[DEBUG] Reference T:", sorted(ref["T_num"].unique()))
        print("[DEBUG] Target T:", sorted(tgt["T_num"].unique()))

    # best row first within each group
    ref = ref.sort_values(
        group_cols + [metric_col, "T_num"],
        ascending=[True] * len(group_cols) + [not higher_is_better, True],
    )

    keep_cols = group_cols + ["T", "T_num", metric_col]
    if std_col in ref.columns:
        keep_cols.append(std_col)

    best_t_df = (
        ref.groupby(group_cols, as_index=False)
        .first()[keep_cols]
        .rename(columns={"T": "BestT", "T_num": "BestT_num", metric_col: f"Best_{metric}_mean"})
    )

    if std_col in best_t_df.columns:
        best_t_df = best_t_df.rename(columns={std_col: f"Best_{metric}_std"})

    filtered_target_df = tgt.merge(
        best_t_df[group_cols + ["BestT", "BestT_num"]],
        left_on=group_cols + ["T_num"],
        right_on=group_cols + ["BestT_num"],
        how="inner",
    )

    if verbose:
        print("\n[DEBUG] Best-T rows:", len(best_t_df))
        print("[DEBUG] Filtered target rows:", len(filtered_target_df))

        ref_pairs = set(
            zip(best_t_df["ModelDisplay"], best_t_df["Method"], best_t_df["BestT_num"])
        )
        tgt_pairs = set(
            zip(tgt["ModelDisplay"], tgt["Method"], tgt["T_num"])
        )
        overlap = ref_pairs & tgt_pairs
        print("[DEBUG] Overlapping (model, method, T):", len(overlap))

        if len(overlap) == 0:
            print("[DEBUG] Example best_t_df rows:")
            print(best_t_df[[*group_cols, "BestT", "BestT_num"]].head(20).to_string(index=False))
            print("[DEBUG] Example target_df rows:")
            print(tgt[[*group_cols, "T", "T_num"]].head(20).to_string(index=False))

    return best_t_df, filtered_target_df

def format_best_metric(df: pd.DataFrame, metric: str, decimals: int = 2) -> pd.Series:
    mean_col = f"Best_{metric}_mean"
    std_col = f"Best_{metric}_std"
    if std_col in df.columns:
        return (
            df[mean_col].map(lambda x: f"{x:.{decimals}f}")
            + " ± "
            + df[std_col].map(lambda x: f"{x:.{decimals}f}")
        )
    return df[mean_col].map(lambda x: f"{x:.{decimals}f}")

def filter_to_shortlisted_t(
    df: pd.DataFrame,
    shortlisted_t_map: dict[str, list[int | float]],
) -> pd.DataFrame:
    out = df.copy()
    out["T_num"] = pd.to_numeric(out["T"], errors="coerce")

    keep_mask = pd.Series(False, index=out.index)
    for model, allowed_ts in shortlisted_t_map.items():
        keep_mask |= (
            (out["ModelDisplay"] == model)
            & (out["T_num"].isin([float(x) for x in allowed_ts]))
        )

    return out.loc[keep_mask].copy()


### plots 
### plots

def build_ood_curve_df(
    tqa_agg: pd.DataFrame,
    target_agg: pd.DataFrame,
    task_name: str,
    tqa_metric: str = "True * Info",
    target_metric: str = "Accuracy",
) -> pd.DataFrame:
    """
    Build one row per (T, model, method):

      x = TruthfulQA metric at T
      y = OOD metric at the same T

    This keeps the full T sweep instead of selecting best T.
    """
    tqa = tqa_agg.copy()
    tgt = target_agg.copy()

    tqa["T_num"] = pd.to_numeric(tqa["T"], errors="coerce")
    tgt["T_num"] = pd.to_numeric(tgt["T"], errors="coerce")

    tqa = tqa.dropna(subset=["T_num"])
    tgt = tgt.dropna(subset=["T_num"])

    group_cols = ["T", "T_num", "ModelDisplay", "Method"]

    tqa_cols = group_cols + [f"{tqa_metric}_mean"]
    if f"{tqa_metric}_std" in tqa.columns:
        tqa_cols.append(f"{tqa_metric}_std")

    tgt_cols = group_cols + [f"{target_metric}_mean"]
    if f"{target_metric}_std" in tgt.columns:
        tgt_cols.append(f"{target_metric}_std")

    out = tqa[tqa_cols].merge(
        tgt[tgt_cols],
        on=group_cols,
        how="inner",
    )

    out = out.rename(
        columns={
            f"{tqa_metric}_mean": "TQA_mean",
            f"{tqa_metric}_std": "TQA_std",
            f"{target_metric}_mean": f"{task_name}_mean",
            f"{target_metric}_std": f"{task_name}_std",
        }
    )

    out["Task"] = task_name
    return out


def add_curve_delta_columns(
    df: pd.DataFrame,
    task_name: str,
) -> pd.DataFrame:
    """
    Add deltas relative to Original within each model.

    Original is duplicated across T, so we first collapse it to one baseline
    per model.
    """
    out = df.copy()

    baseline = (
        out[out["Method"] == "Original"]
        .groupby("ModelDisplay", as_index=False)
        .agg(
            TQA_baseline=("TQA_mean", "mean"),
            OOD_baseline=(f"{task_name}_mean", "mean"),
        )
    )

    out = out.merge(baseline, on="ModelDisplay", how="left")
    out["TQA_delta"] = out["TQA_mean"] - out["TQA_baseline"]
    out[f"{task_name}_delta"] = out[f"{task_name}_mean"] - out["OOD_baseline"]

    return out


def pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> pd.DataFrame:
    """
    Return non-dominated points.

    For our case, larger TQA and larger OOD are better.
    """
    points = df.dropna(subset=[x_col, y_col]).copy()

    if points.empty:
        return points

    is_pareto = []

    xs = points[x_col].to_numpy()
    ys = points[y_col].to_numpy()

    for i, (x, y) in enumerate(zip(xs, ys)):
        if maximize_x:
            better_or_equal_x = xs >= x
            strictly_better_x = xs > x
        else:
            better_or_equal_x = xs <= x
            strictly_better_x = xs < x

        if maximize_y:
            better_or_equal_y = ys >= y
            strictly_better_y = ys > y
        else:
            better_or_equal_y = ys <= y
            strictly_better_y = ys < y

        dominated = (
            better_or_equal_x
            & better_or_equal_y
            & (strictly_better_x | strictly_better_y)
        ).any()

        is_pareto.append(not dominated)

    front = points.loc[is_pareto].copy()
    front = front.sort_values(x_col)
    return front


def plot_tqa_vs_ood_curves_grid(
    mmlu_curve_df: pd.DataFrame,
    gsm8k_curve_df: pd.DataFrame,
    model_order: list[str] | None = None,
    method_order: list[str] | None = None,
    delta: bool = False,
    save_path: str | Path | None = None,
    annotate: bool = False,
    annotate_t: bool = False,
    show_pareto: bool = True,
    show_original: bool = True,
) -> None:
    """
    Create a 2 x 4 curve grid.

    Columns: models
    Row 1: TQA vs MMLU
    Row 2: TQA vs GSM8K

    Each method is a curve across T.

    If delta=True:
      x = Δ TQA over Original
      y = Δ OOD over Original

    If delta=False:
      x = absolute TQA score
      y = absolute OOD score
    """
    if model_order is None:
        model_order = MODEL_ORDER
    if method_order is None:
        method_order = METHOD_ORDER

    mmlu_df = add_curve_delta_columns(mmlu_curve_df, "MMLU")
    gsm8k_df = add_curve_delta_columns(gsm8k_curve_df, "GSM8K")

    task_dfs = [
        ("MMLU", mmlu_df),
        ("GSM8K", gsm8k_df),
    ]

    fig, axes = plt.subplots(
        2,
        len(model_order),
        figsize=(5.0 * len(model_order), 8.5),
        sharex=False,
        sharey=False,
    )

    title_suffix = "Delta over Original" if delta else "Absolute Scores"
    fig.suptitle(f"TQA–OOD Steering Tradeoff Curves: {title_suffix}", fontsize=18)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_to_color = {
        method: color_cycle[i % len(color_cycle)]
        for i, method in enumerate(method_order)
    }

    for row_idx, (task_name, task_df) in enumerate(task_dfs):
        if delta:
            x_col = "TQA_delta"
            y_col = f"{task_name}_delta"
            x_label = "Δ TruthfulQA True * Info"
            y_label = f"Δ {task_name} Accuracy"
        else:
            x_col = "TQA_mean"
            y_col = f"{task_name}_mean"
            x_label = "TruthfulQA True * Info"
            y_label = f"{task_name} Accuracy"

        for col_idx, model in enumerate(model_order):
            ax = axes[row_idx, col_idx]
            sub = task_df[task_df["ModelDisplay"] == model].copy()

            if sub.empty:
                ax.set_title(model)
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Optional Pareto frontier across all methods/T values.
            if show_pareto:
                front = pareto_frontier(sub, x_col=x_col, y_col=y_col)
                if not front.empty:
                    ax.plot(
                        front[x_col],
                        front[y_col],
                        linestyle="--",
                        marker="",
                        linewidth=2.0,
                        color="black",
                        alpha=0.5,
                        label="Pareto frontier" if row_idx == 0 and col_idx == 0 else None,
                        zorder=1,
                    )

            # Plot each method as a T curve.
            for method in method_order:
                msub = sub[sub["Method"] == method].sort_values("T_num").copy()
                if msub.empty:
                    continue

                if method == "Original":
                    if not show_original:
                        continue

                    # Original is duplicated across T; plot only one star.
                    row = msub.iloc[0]
                    ax.scatter(
                        row[x_col],
                        row[y_col],
                        s=180,
                        marker="*",
                        color=method_to_color.get(method, None),
                        edgecolors="black",
                        linewidth=1.2,
                        label=method if row_idx == 0 and col_idx == 0 else None,
                        zorder=5,
                    )

                    if annotate:
                        ax.annotate(
                            method,
                            (row[x_col], row[y_col]),
                            textcoords="offset points",
                            xytext=(5, 5),
                            fontsize=8,
                        )
                    continue

                linewidth = 2.6 if method == "COBRAS (Ours)" else 1.5
                markersize = 7 if method == "COBRAS (Ours)" else 5
                alpha = 1.0 if method == "COBRAS (Ours)" else 0.75
                zorder = 4 if method == "COBRAS (Ours)" else 3

                ax.plot(
                    msub[x_col],
                    msub[y_col],
                    marker="o",
                    markersize=markersize,
                    linewidth=linewidth,
                    alpha=alpha,
                    color=method_to_color.get(method, None),
                    label=method if row_idx == 0 and col_idx == 0 else None,
                    zorder=zorder,
                )

                # Mark start/end of the T sweep.
                first = msub.iloc[0]
                last = msub.iloc[-1]

                ax.scatter(
                    first[x_col],
                    first[y_col],
                    marker=".",
                    s=80,
                    color=method_to_color.get(method, None),
                    alpha=alpha,
                    zorder=zorder,
                )
                ax.scatter(
                    last[x_col],
                    last[y_col],
                    marker=">",
                    s=80,
                    color=method_to_color.get(method, None),
                    alpha=alpha,
                    zorder=zorder,
                )

                if annotate:
                    ax.annotate(
                        method,
                        (last[x_col], last[y_col]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                    )

                if annotate_t:
                    for _, r in msub.iterrows():
                        ax.annotate(
                            f"T={r['T_num']:.0f}" if float(r["T_num"]).is_integer() else f"T={r['T_num']:.1f}",
                            (r[x_col], r[y_col]),
                            textcoords="offset points",
                            xytext=(4, 4),
                            fontsize=7,
                            alpha=0.8,
                        )

            if delta:
                ax.axhline(0.0, linewidth=1.0, alpha=0.5)
                ax.axvline(0.0, linewidth=1.0, alpha=0.5)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            if row_idx == 0:
                ax.set_title(model)

            ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.995, 0.5),
            frameon=False,
        )

    plt.tight_layout(rect=[0, 0, 0.94, 0.94])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Plot deltas over Original instead of absolute scores.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate each curve endpoint with the method name.",
    )
    parser.add_argument(
        "--annotate-t",
        action="store_true",
        help="Annotate each point with its T value.",
    )
    parser.add_argument(
        "--no-pareto",
        action="store_true",
        help="Do not draw the Pareto frontier.",
    )
    parser.add_argument(
        "--hide-original",
        action="store_true",
        help="Do not show the Original baseline star.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Where to save the figure, e.g. results/plots/tqa_vs_ood_curves.pdf",
    )
    args = parser.parse_args()

    tqa_df = load_results("truthfulqa")
    mmlu_df = load_results("mmlu")
    gsm8k_df = load_results("gsm8k")

    tqa_agg = aggregate_truthfulqa(tqa_df)
    mmlu_agg = aggregate_mmlu(mmlu_df)
    gsm8k_agg = aggregate_mmlu(gsm8k_df)

    tqa_agg_shortlisted = filter_to_shortlisted_t(tqa_agg, SHORTLISTED_T_MAP)
    mmlu_agg_shortlisted = filter_to_shortlisted_t(mmlu_agg, SHORTLISTED_T_MAP)
    gsm8k_agg_shortlisted = filter_to_shortlisted_t(gsm8k_agg, SHORTLISTED_T_MAP)

    mmlu_curve_df = build_ood_curve_df(
        tqa_agg=tqa_agg_shortlisted,
        target_agg=mmlu_agg_shortlisted,
        task_name="MMLU",
        tqa_metric="True * Info",
        target_metric="Accuracy",
    )

    gsm8k_curve_df = build_ood_curve_df(
        tqa_agg=tqa_agg_shortlisted,
        target_agg=gsm8k_agg_shortlisted,
        task_name="GSM8K",
        tqa_metric="True * Info",
        target_metric="Accuracy",
    )

    print("\nMMLU curve rows:")
    print(
        mmlu_curve_df[
            ["ModelDisplay", "Method", "T_num", "TQA_mean", "MMLU_mean"]
        ]
        .sort_values(["ModelDisplay", "Method", "T_num"])
        .to_string(index=False)
    )

    print("\nGSM8K curve rows:")
    print(
        gsm8k_curve_df[
            ["ModelDisplay", "Method", "T_num", "TQA_mean", "GSM8K_mean"]
        ]
        .sort_values(["ModelDisplay", "Method", "T_num"])
        .to_string(index=False)
    )

    plot_tqa_vs_ood_curves_grid(
        mmlu_curve_df=mmlu_curve_df,
        gsm8k_curve_df=gsm8k_curve_df,
        model_order=MODEL_ORDER,
        method_order=METHOD_ORDER,
        delta=args.delta,
        save_path=args.save_path,
        annotate=args.annotate,
        annotate_t=args.annotate_t,
        show_pareto=not args.no_pareto,
        show_original=not args.hide_original,
    )
    

if __name__ == "__main__":
    main()