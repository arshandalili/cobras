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


def t_sort_key(t: str) -> float:
    return float(t)


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


def merge_aggregates(uf_df: pd.DataFrame, tqa_df: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(
        uf_df,
        tqa_df,
        on=["T", "ModelDisplay", "Method"],
        how="outer",
    )


def build_row(
    method: str,
    model: str,
    n_rows: int,
    win: str,
    rm_mean: str,
    rm_p90: str,
    ti: str,
    true: str,
    info: str,
) -> str:
    method_tex = "\\textbf{COBRAS (Ours)}" if method == "COBRAS (Ours)" else method

    if method == "Original":
        return (
            rf"\midrule" "\n"
            rf"{method_tex} & \multirow{{{n_rows}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{model}}}}}" "\n"
            rf"& {win} & {rm_mean} & {rm_p90} " "\n"
            rf"& {ti} & {true} & {info} " "\n"
            rf"&  &  &  \\"
            "\n"
            rf"\midrule"
        )

    return (
        rf"{method_tex} &" "\n"
        rf"& {win} & {rm_mean} & {rm_p90} " "\n"
        rf"& {ti} & {true} & {info} " "\n"
        rf"&  &  &  \\"
    )


def generate_rows_for_model(sub: pd.DataFrame, model: str) -> list[str]:
    lines: list[str] = []
    n_rows = len(METHOD_ORDER)

    for method in METHOD_ORDER:
        row = sub[sub["Method"] == method]

        if row.empty:
            win = rm_mean = rm_p90 = ""
            ti = true = info = ""
        else:
            row = row.iloc[0]

            win = format_mean_std(
                row.get("RM Win-Rate vs NoSteer_mean", float("nan")),
                row.get("RM Win-Rate vs NoSteer_std", float("nan")),
                mean_precision=1,
                std_precision=3,
            )
            rm_mean = format_mean_std(
                row.get("RM Mean_mean", float("nan")),
                row.get("RM Mean_std", float("nan")),
                mean_precision=3,
                std_precision=3,
            )
            rm_p90 = format_mean_std(
                row.get("RM P90_mean", float("nan")),
                row.get("RM P90_std", float("nan")),
                mean_precision=3,
                std_precision=3,
            )

            ti = format_mean_std(
                row.get("True * Info_mean", float("nan")),
                row.get("True * Info_std", float("nan")),
                mean_precision=1,
                std_precision=3,
            )
            true = format_mean_std(
                row.get("Truthfulness_mean", float("nan")),
                row.get("Truthfulness_std", float("nan")),
                mean_precision=1,
                std_precision=3,
            )
            info = format_mean_std(
                row.get("Informativeness_mean", float("nan")),
                row.get("Informativeness_std", float("nan")),
                mean_precision=1,
                std_precision=3,
            )

        lines.append(
            build_row(
                method=method,
                model=model,
                n_rows=n_rows,
                win=win,
                rm_mean=rm_mean,
                rm_p90=rm_p90,
                ti=ti,
                true=true,
                info=info,
            )
        )
        lines.append("")

    return lines


def generate_rows_text(agg_df: pd.DataFrame) -> str:
    blocks: list[str] = []

    t_values = sorted(
        [t for t in agg_df["T"].dropna().unique()],
        key=t_sort_key,
    )

    for t in t_values:
        blocks.append(f"% ==================== T = {t} ====================")
        blocks.append("")

        t_df = agg_df[agg_df["T"] == t].copy()

        for model in MODEL_ORDER:
            sub = t_df[t_df["ModelDisplay"] == model].copy()

            # Still emit blank model block only if this model exists somewhere at this T.
            # If you want every model always, remove this if.
            if sub.empty:
                continue

            blocks.append(f"% {model}")
            blocks.extend(generate_rows_for_model(sub, model))
            blocks.append("")

    return "\n".join(blocks).rstrip() + "\n"


def mean_std_col(df: pd.DataFrame, metric: str, decimals: int = 2) -> pd.Series:
    return (
        df[f'{metric}_mean'].map(lambda x: f"{x:.{decimals}f}") +
        " ± " +
        df[f'{metric}_std'].map(lambda x: f"{x:.{decimals}f}")
    )



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


def main() -> None:
    uf_df = load_results("ultrafeedback")
    tqa_df = load_results("truthfulqa")
    mmlu_df = load_results("mmlu")
    gsm8k_df = load_results("gsm8k")

    uf_agg = aggregate_ultrafeedback(uf_df)
    tqa_agg = aggregate_truthfulqa(tqa_df)
    mmlu_agg = aggregate_mmlu(mmlu_df) 
    # can reuse mmlu fn because gsm8k and mmlu share the same metrics
    gsm8k_agg = aggregate_mmlu(gsm8k_df) 

    agg_df = merge_aggregates(uf_agg, tqa_agg)

    gagg = (
        agg_df.groupby(['T', 'Method', 'ModelDisplay'], as_index=False)
        .mean()
        .sort_values(['ModelDisplay', 'Method', 'T'])
    )

    selected_models = [
        "Falcon-7B",
    ]

    gagg = gagg[gagg['ModelDisplay'].isin(selected_models)].copy()

    display_df = gagg[['ModelDisplay', 'Method', 'T']].copy()

    for col in ULTRAFEEDBACK_METRICS + TRUTHFULQA_METRICS:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        if mean_col in gagg.columns and std_col in gagg.columns:
            display_df[col] = mean_std_col(gagg, col)

    print(display_df.sort_values(['ModelDisplay', 'Method', 'T']).to_string(index=False))

    tqa_agg_shortlisted = filter_to_shortlisted_t(tqa_agg, SHORTLISTED_T_MAP)
    mmlu_agg_shortlisted = filter_to_shortlisted_t(mmlu_agg, SHORTLISTED_T_MAP)

    best_t_from_tqa, mmlu_at_best_tqa_t = select_best_t_and_filter_target(
        reference_df=tqa_agg_shortlisted,
        target_df=mmlu_agg_shortlisted,
        metric="True * Info",
        higher_is_better=True,
    )

    best_t_from_tqa["TQA_score_at_BestT"] = format_best_metric(best_t_from_tqa, "True * Info")

    print("Best T chosen from TruthfulQA:")
    print(
        best_t_from_tqa[
            ["ModelDisplay", "Method", "BestT", "TQA_score_at_BestT"]
        ].sort_values(["ModelDisplay", "Method"]).to_string(index=False)
    )

    print("\nMMLU rows at T selected from TruthfulQA:")
    print(
        mmlu_at_best_tqa_t.sort_values(["ModelDisplay", "Method", "T"]).to_string(index=False)
    )



    tqa_agg_shortlisted = filter_to_shortlisted_t(tqa_agg, SHORTLISTED_T_MAP)
    gsm8k_agg_shortlisted = filter_to_shortlisted_t(gsm8k_agg, SHORTLISTED_T_MAP)

    best_t_from_tqa, gsm8k_at_best_tqa_t = select_best_t_and_filter_target(
        reference_df=tqa_agg_shortlisted,
        target_df=gsm8k_agg_shortlisted,
        metric="True * Info",
        higher_is_better=True,
    )

    best_t_from_tqa["TQA_score_at_BestT"] = format_best_metric(best_t_from_tqa, "True * Info")

    print("Best T chosen from TruthfulQA:")
    print(
        best_t_from_tqa[
            ["ModelDisplay", "Method", "BestT", "TQA_score_at_BestT"]
        ].sort_values(["ModelDisplay", "Method"]).to_string(index=False)
    )

    print("\nGSM8k rows at T selected from TruthfulQA:")
    print(
        gsm8k_at_best_tqa_t.sort_values(["ModelDisplay", "Method", "T"]).to_string(index=False)
    )

    plot_shortlisted_t_sweeps(
        tqa_agg=tqa_agg,
        uf_agg=uf_agg,
        shortlisted_t_map=SHORTLISTED_T_MAP,
        model_order=MODEL_ORDER,
        method_order=[x for x in METHOD_ORDER if x != "SphericalSteer"],
        save_path=get_project_dir() / "results" / "shortlisted_t_sweeps.png",
        show_std=True,
    )
    

if __name__ == "__main__":
    main()