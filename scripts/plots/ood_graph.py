#!/usr/bin/env python3
"""
    Run it using 
    uv run python -u scripts/plots/ood_graph.py \
        --save-path results/plots/tqa_vs_ood_absolute.png \
        --delta \
        --annotate 
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from odesteer.utils import get_project_dir


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
def build_ood_scatter_df(
    best_t_from_tqa: pd.DataFrame,
    target_at_best_tqa_t: pd.DataFrame,
    task_name: str,
    tqa_metric: str = "True * Info",
    target_metric: str = "Accuracy",
) -> pd.DataFrame:
    """
    Build one row per (model, method) for plotting:

      x = TruthfulQA metric at T selected by TruthfulQA
      y = target task metric at that same selected T

    task_name should be "MMLU" or "GSM8K".
    """
    group_cols = ["ModelDisplay", "Method"]

    tqa_cols = group_cols + [
        "BestT",
        "BestT_num",
        f"Best_{tqa_metric}_mean",
    ]

    tqa_std_col = f"Best_{tqa_metric}_std"
    if tqa_std_col in best_t_from_tqa.columns:
        tqa_cols.append(tqa_std_col)

    target_cols = group_cols + [
        "T",
        "T_num",
        "BestT",
        "BestT_num",
        f"{target_metric}_mean",
    ]

    target_std_col = f"{target_metric}_std"
    if target_std_col in target_at_best_tqa_t.columns:
        target_cols.append(target_std_col)

    out = target_at_best_tqa_t[target_cols].merge(
        best_t_from_tqa[tqa_cols],
        on=group_cols + ["BestT", "BestT_num"],
        how="inner",
    )

    out = out.rename(
        columns={
            f"Best_{tqa_metric}_mean": "TQA_mean",
            f"Best_{tqa_metric}_std": "TQA_std",
            f"{target_metric}_mean": f"{task_name}_mean",
            f"{target_metric}_std": f"{task_name}_std",
        }
    )

    out["Task"] = task_name
    return out


def add_delta_columns(
    df: pd.DataFrame,
    task_name: str,
) -> pd.DataFrame:
    """
    Add delta columns relative to Original within each model.

    x_delta = TQA(method) - TQA(Original)
    y_delta = OOD(method) - OOD(Original)
    """
    out = df.copy()

    baseline = (
        out[out["Method"] == "Original"]
        [["ModelDisplay", "TQA_mean", f"{task_name}_mean"]]
        .rename(
            columns={
                "TQA_mean": "TQA_baseline",
                f"{task_name}_mean": f"{task_name}_baseline",
            }
        )
    )

    out = out.merge(baseline, on="ModelDisplay", how="left")

    out["TQA_delta"] = out["TQA_mean"] - out["TQA_baseline"]
    out[f"{task_name}_delta"] = out[f"{task_name}_mean"] - out[f"{task_name}_baseline"]

    return out


def plot_tqa_vs_ood_grid(
    mmlu_plot_df: pd.DataFrame,
    gsm8k_plot_df: pd.DataFrame,
    model_order: list[str] | None = None,
    method_order: list[str] | None = None,
    delta: bool = False,
    save_path: str | Path | None = None,
    annotate: bool = False,
    show_std: bool = False,
) -> None:
    """
    Create a 2 x 4 scatter grid.

    Columns: models
    Row 1: TQA vs MMLU
    Row 2: TQA vs GSM8K

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

    mmlu_df = add_delta_columns(mmlu_plot_df, "MMLU")
    gsm8k_df = add_delta_columns(gsm8k_plot_df, "GSM8K")

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
    fig.suptitle(f"TQA-selected Steering Transfer: {title_suffix}", fontsize=18)

    # Stable colors by method.
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_to_color = {
        method: color_cycle[i % len(color_cycle)]
        for i, method in enumerate(method_order)
    }

    for row_idx, (task_name, task_df) in enumerate(task_dfs):
        y_abs_col = f"{task_name}_mean"
        y_delta_col = f"{task_name}_delta"

        for col_idx, model in enumerate(model_order):
            ax = axes[row_idx, col_idx]
            sub = task_df[task_df["ModelDisplay"] == model].copy()

            if sub.empty:
                ax.set_title(f"{model}")
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            for method in method_order:
                msub = sub[sub["Method"] == method]
                if msub.empty:
                    continue

                row = msub.iloc[0]

                if delta:
                    x = row["TQA_delta"]
                    y = row[y_delta_col]
                    xerr = None
                    yerr = None
                else:
                    x = row["TQA_mean"]
                    y = row[y_abs_col]

                    if show_std:
                        xerr = row.get("TQA_std", None)
                        yerr = row.get(f"{task_name}_std", None)
                    else:
                        xerr = None
                        yerr = None

                marker = "*" if method == "Original" else "o"
                size = 170 if method == "Original" else 80
                size = 130 if method == "COBRAS (Ours)" else size
                linewidth = 1.8 if method in {"Original", "COBRAS (Ours)"} else 1.0

                if show_std and not delta:
                    ax.errorbar(
                        x,
                        y,
                        xerr=xerr,
                        yerr=yerr,
                        fmt=marker,
                        markersize=9 if method == "Original" else 7,
                        color=method_to_color.get(method, None),
                        label=method if row_idx == 0 and col_idx == 0 else None,
                        capsize=2,
                        linewidth=linewidth,
                    )
                else:
                    ax.scatter(
                        x,
                        y,
                        s=size,
                        marker=marker,
                        color=method_to_color.get(method, None),
                        label=method if row_idx == 0 and col_idx == 0 else None,
                        linewidth=linewidth,
                        edgecolors="black" if method in {"Original", "COBRAS (Ours)"} else None,
                    )

                if annotate:
                    ax.annotate(
                        method,
                        (x, y),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=8,
                    )

            if delta:
                ax.axhline(0.0, linewidth=1.0, alpha=0.5)
                ax.axvline(0.0, linewidth=1.0, alpha=0.5)
                ax.set_xlabel("Δ TruthfulQA True * Info")
                ax.set_ylabel(f"Δ {task_name} Accuracy")
            else:
                ax.set_xlabel("TruthfulQA True * Info")
                ax.set_ylabel(f"{task_name} Accuracy")

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
        help="Annotate each scatter point with the method name.",
    )
    parser.add_argument(
        "--show-std",
        action="store_true",
        help="Show std error bars for absolute plots. Ignored for delta plots.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Where to save the figure, e.g. results/figures/tqa_vs_ood.pdf",
    )
    args = parser.parse_args()

    uf_df = load_results("ultrafeedback")
    tqa_df = load_results("truthfulqa")
    mmlu_df = load_results("mmlu")
    gsm8k_df = load_results("gsm8k")

    uf_agg = aggregate_ultrafeedback(uf_df)
    tqa_agg = aggregate_truthfulqa(tqa_df)
    mmlu_agg = aggregate_mmlu(mmlu_df)
    gsm8k_agg = aggregate_mmlu(gsm8k_df)

    tqa_agg_shortlisted = filter_to_shortlisted_t(tqa_agg, SHORTLISTED_T_MAP)
    mmlu_agg_shortlisted = filter_to_shortlisted_t(mmlu_agg, SHORTLISTED_T_MAP)
    gsm8k_agg_shortlisted = filter_to_shortlisted_t(gsm8k_agg, SHORTLISTED_T_MAP)

    # Select best T using TruthfulQA, then evaluate MMLU at that same T.
    best_t_from_tqa_mmlu, mmlu_at_best_tqa_t = select_best_t_and_filter_target(
        reference_df=tqa_agg_shortlisted,
        target_df=mmlu_agg_shortlisted,
        metric="True * Info",
        higher_is_better=True,
    )

    # Select best T using TruthfulQA, then evaluate GSM8K at that same T.
    best_t_from_tqa_gsm8k, gsm8k_at_best_tqa_t = select_best_t_and_filter_target(
        reference_df=tqa_agg_shortlisted,
        target_df=gsm8k_agg_shortlisted,
        metric="True * Info",
        higher_is_better=True,
    )

    mmlu_plot_df = build_ood_scatter_df(
        best_t_from_tqa=best_t_from_tqa_mmlu,
        target_at_best_tqa_t=mmlu_at_best_tqa_t,
        task_name="MMLU",
        tqa_metric="True * Info",
        target_metric="Accuracy",
    )

    gsm8k_plot_df = build_ood_scatter_df(
        best_t_from_tqa=best_t_from_tqa_gsm8k,
        target_at_best_tqa_t=gsm8k_at_best_tqa_t,
        task_name="GSM8K",
        tqa_metric="True * Info",
        target_metric="Accuracy",
    )

    print("\nMMLU plotting rows:")
    print(
        mmlu_plot_df[
            ["ModelDisplay", "Method", "BestT", "TQA_mean", "MMLU_mean"]
        ]
        .sort_values(["ModelDisplay", "Method"])
        .to_string(index=False)
    )

    print("\nGSM8K plotting rows:")
    print(
        gsm8k_plot_df[
            ["ModelDisplay", "Method", "BestT", "TQA_mean", "GSM8K_mean"]
        ]
        .sort_values(["ModelDisplay", "Method"])
        .to_string(index=False)
    )

    plot_tqa_vs_ood_grid(
        mmlu_plot_df=mmlu_plot_df,
        gsm8k_plot_df=gsm8k_plot_df,
        model_order=MODEL_ORDER,
        method_order=METHOD_ORDER,
        delta=args.delta,
        save_path=args.save_path,
        annotate=args.annotate,
        show_std=args.show_std,
    )
    

if __name__ == "__main__":
    main()