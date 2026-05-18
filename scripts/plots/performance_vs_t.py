#!/usr/bin/env python3
from __future__ import annotations

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


def _prepare_metric_plot_df(
    agg_df: pd.DataFrame,
    metric: str,
    shortlisted_t_map: dict[str, list[int | float]],
) -> pd.DataFrame:
    """
    Return a filtered dataframe with only the columns needed for plotting:
    ModelDisplay, Method, T_num, <metric>_mean, <metric>_std

    Keeps only T values listed in shortlisted_t_map for each model.
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in agg_df.columns:
        raise KeyError(f"Missing column: {mean_col}")
    if std_col not in agg_df.columns:
        raise KeyError(f"Missing column: {std_col}")

    plot_df = agg_df.copy()

    # Convert T to numeric for sorting/plotting
    plot_df["T_num"] = pd.to_numeric(plot_df["T"], errors="coerce")
    plot_df = plot_df.dropna(subset=["T_num"])

    # Keep only shortlisted T values for each model
    keep_mask = pd.Series(False, index=plot_df.index)
    for model, allowed_ts in shortlisted_t_map.items():
        model_mask = plot_df["ModelDisplay"] == model
        t_mask = plot_df["T_num"].isin([float(x) for x in allowed_ts])
        keep_mask |= (model_mask & t_mask)

    plot_df = plot_df.loc[keep_mask].copy()
    plot_df = plot_df.sort_values(["ModelDisplay", "Method", "T_num"])

    return plot_df[["ModelDisplay", "Method", "T_num", mean_col, std_col]].copy()


def plot_shortlisted_t_sweeps(
    tqa_agg: pd.DataFrame,
    uf_agg: pd.DataFrame,
    shortlisted_t_map: dict[str, list[int | float]],
    model_order: list[str] | None = None,
    method_order: list[str] | None = None,
    save_path: str | Path | None = None,
    show_std: bool = True,
) -> None:
    """
    Create an 8-subplot figure:
      - top row: TruthfulQA True * Info vs T for 4 models
      - bottom row: UltraFeedback RM Mean vs T for 4 models

    One line per method.
    """
    if model_order is None:
        model_order = MODEL_ORDER
    if method_order is None:
        method_order = METHOD_ORDER

    tqa_metric = "True * Info"
    uf_metric = "RM Mean"

    tqa_plot_df = _prepare_metric_plot_df(tqa_agg, tqa_metric, shortlisted_t_map)
    uf_plot_df = _prepare_metric_plot_df(uf_agg, uf_metric, shortlisted_t_map)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=False)
    fig.suptitle("Performance vs Steering Strength T", fontsize=20)

    # Top row: TruthfulQA
    for j, model in enumerate(model_order):
        ax = axes[0, j]
        sub = tqa_plot_df[tqa_plot_df["ModelDisplay"] == model].copy()

        if sub.empty:
            ax.set_title(f"{model} (TruthfulQA)")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("T")
            ax.set_ylabel("True * Info")
            continue

        for method in method_order:
            msub = sub[sub["Method"] == method].sort_values("T_num")
            if msub.empty:
                continue

            ax.plot(
                msub["T_num"],
                msub[f"{tqa_metric}_mean"],
                marker="o",
                label=method,
            )

            if show_std:
                ax.fill_between(
                    msub["T_num"],
                    msub[f"{tqa_metric}_mean"] - msub[f"{tqa_metric}_std"],
                    msub[f"{tqa_metric}_mean"] + msub[f"{tqa_metric}_std"],
                    alpha=0.15,
                )

        ax.set_title(f"{model} — TruthfulQA")
        ax.set_xlabel("T")
        ax.set_ylabel("True * Info")
        ax.grid(True, alpha=0.3)

    # Bottom row: UltraFeedback
    for j, model in enumerate(model_order):
        ax = axes[1, j]
        sub = uf_plot_df[uf_plot_df["ModelDisplay"] == model].copy()

        if sub.empty:
            ax.set_title(f"{model} (UltraFeedback)")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("T")
            ax.set_ylabel("RM Mean")
            continue

        for method in method_order:
            msub = sub[sub["Method"] == method].sort_values("T_num")
            if msub.empty:
                continue

            ax.plot(
                msub["T_num"],
                msub[f"{uf_metric}_mean"],
                marker="o",
                label=method,
            )

            if show_std:
                ax.fill_between(
                    msub["T_num"],
                    msub[f"{uf_metric}_mean"] - msub[f"{uf_metric}_std"],
                    msub[f"{uf_metric}_mean"] + msub[f"{uf_metric}_std"],
                    alpha=0.15,
                )

        ax.set_title(f"{model} — UltraFeedback")
        ax.set_xlabel("T")
        ax.set_ylabel("RM Mean")
        ax.grid(True, alpha=0.3)

    # # Put one shared legend outside the grid
    # handles, labels = axes[0, 0].get_legend_handles_labels()
    # if handles:
    #     fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    # plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    # Put one shared legend outside the grid
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.97, 0.5),
            frameon=False,
        )

    plt.tight_layout(rect=[0, 0, 0.96, 0.95])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def main() -> None:
    uf_df = load_results("ultrafeedback")
    tqa_df = load_results("truthfulqa")
    mmlu_df = load_results("mmlu")
    gsm8k_df = load_results("gsm8k")

    uf_agg = aggregate_ultrafeedback(uf_df)
    tqa_agg = aggregate_truthfulqa(tqa_df)

    plot_shortlisted_t_sweeps(
        tqa_agg=tqa_agg,
        uf_agg=uf_agg,
        shortlisted_t_map=SHORTLISTED_T_MAP,
        model_order=MODEL_ORDER,
        method_order=[x for x in METHOD_ORDER if x != "SphericalSteer"],
        save_path=get_project_dir() / "results" / "plots" / "performance_vs_t.png",
        show_std=True,
    )
    

if __name__ == "__main__":
    main()