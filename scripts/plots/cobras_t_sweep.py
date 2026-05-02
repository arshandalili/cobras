#!/usr/bin/env python3
"""]
    Run it with 
    uv run python scripts/plots/cobras_t_sweep.py 
"""
from __future__ import annotations

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

T_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EPSILON_VALUES = [0, 0.001, 0.01, 0.05, 0.1]

T_REGEX = re.compile(r"-T(?P<T>\d+(?:\.\d+)?)$")
EPS_REGEX = re.compile(r"-eps(?P<eps>\d+(?:\.\d+)?)(?:-|$)")


def normalize_model(raw: str) -> str:
    return MODEL_DISPLAY_MAP.get(raw, raw)


def parse_cobras_t(raw: str) -> float | None:
    match = T_REGEX.search(raw)
    if match is None:
        return None
    return float(match.group("T"))


def parse_cobras_eps(raw: str) -> float | None:
    match = EPS_REGEX.search(raw)
    if match is None:
        return None
    return float(match.group("eps"))


def is_cobras(raw: str) -> bool:
    return raw.startswith("COBRAS")


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

    raise FileNotFoundError(f"No CSV files found in any of: {candidate_dirs}")


def prepare_cobras_df(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    df = df[df["Steering Method"].apply(is_cobras)].copy()

    df["ModelDisplay"] = df["Model"].apply(normalize_model)
    df["T"] = df["Steering Method"].apply(parse_cobras_t)
    df["epsilon"] = df["Steering Method"].apply(parse_cobras_eps)

    df = df.dropna(subset=["T", "epsilon"])

    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def aggregate_cobras(
    df: pd.DataFrame,
    metric_cols: list[str],
    task_name: str,
) -> pd.DataFrame:
    df = prepare_cobras_df(df, metric_cols)

    if task_name == "truthfulqa":
        df[metric_cols] = df[metric_cols] * 100.0

    elif task_name == "ultrafeedback":
        # Win-rate is a fraction, but RM Mean / RM P90 are already raw reward scores.
        if "RM Win-Rate vs NoSteer" in metric_cols:
            df["RM Win-Rate vs NoSteer"] = df["RM Win-Rate vs NoSteer"] * 100.0

    grouped = (
        df.groupby(["ModelDisplay", "T", "epsilon"], as_index=False)[metric_cols]
        .agg(["mean", "std", "count"])
    )

    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    std_cols = [c for c in grouped.columns if c.endswith("_std")]
    grouped[std_cols] = grouped[std_cols].fillna(0.0)

    return grouped


def _filter_values(
    df: pd.DataFrame,
    x_col: str,
    allowed_values: list[float],
) -> pd.DataFrame:
    df = df.copy()
    allowed = [float(x) for x in allowed_values]
    return df[df[x_col].round(10).isin([round(x, 10) for x in allowed])].copy()


def plot_cobras_vs_x(
    agg_df: pd.DataFrame,
    metric: str,
    x_col: str,
    fixed_col: str,
    fixed_value: float,
    allowed_x_values: list[float],
    title: str,
    ylabel: str,
    xlabel: str,
    save_path: str | Path,
    model_order: list[str] = MODEL_ORDER,
    show_std: bool = True,
) -> None:
    """
    Plots COBRAS metric vs one hyperparameter while holding the other fixed.

    Example:
      x_col="T", fixed_col="epsilon", fixed_value=0.01
      => performance vs T at epsilon=0.01

      x_col="epsilon", fixed_col="T", fixed_value=0.5
      => performance vs epsilon at T=0.5
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    count_col = f"{metric}_count"

    for col in [mean_col, std_col, count_col]:
        if col not in agg_df.columns:
            raise KeyError(f"Missing column: {col}")

    plot_df = agg_df.copy()

    plot_df = plot_df[plot_df[fixed_col].round(10) == round(float(fixed_value), 10)].copy()
    plot_df = _filter_values(plot_df, x_col=x_col, allowed_values=allowed_x_values)
    plot_df = plot_df.sort_values(["ModelDisplay", x_col])

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharex=False)
    fig.suptitle(title, fontsize=18)

    for j, model in enumerate(model_order):
        ax = axes[j]
        sub = plot_df[plot_df["ModelDisplay"] == model].sort_values(x_col)

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
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            continue

        ax.plot(
            sub[x_col],
            sub[mean_col],
            marker="o",
            label="COBRAS",
        )

        if show_std:
            ax.fill_between(
                sub[x_col],
                sub[mean_col] - sub[std_col],
                sub[mean_col] + sub[std_col],
                alpha=0.15,
            )

        # Show how many seeds/runs contributed to each point.
        # for _, row in sub.iterrows():
        #     ax.annotate(
        #         f"n={int(row[count_col])}",
        #         xy=(row[x_col], row[mean_col]),
        #         xytext=(0, 6),
        #         textcoords="offset points",
        #         ha="center",
        #         fontsize=8,
        #     )

        ax.set_title(model)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if x_col == "epsilon":
            ax.set_xscale("symlog", linthresh=0.001)

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_dir = get_project_dir() / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    uf_df = load_results("ultrafeedback")
    tqa_df = load_results("truthfulqa")

    tqa_agg = aggregate_cobras(
        tqa_df,
        metric_cols=TRUTHFULQA_METRICS,
        task_name="truthfulqa",
    )

    uf_agg = aggregate_cobras(
        uf_df,
        metric_cols=ULTRAFEEDBACK_METRICS,
        task_name="ultrafeedback",
    )

    # ------------------------------------------------------------------
    # Choose defaults for the held-fixed hyperparameter.
    #
    # For T sweep:
    #   plot performance vs T at epsilon = 0.01
    #
    # For epsilon sweep:
    #   plot performance vs epsilon at T = 0.5
    #
    # Change these if your canonical setting differs.
    # ------------------------------------------------------------------
    fixed_epsilon_for_t_sweep = 0.01
    fixed_t_for_epsilon_sweep = 0.6

    # 1. TruthfulQA performance vs T
    plot_cobras_vs_x(
        agg_df=tqa_agg,
        metric="True * Info",
        x_col="T",
        fixed_col="epsilon",
        fixed_value=fixed_epsilon_for_t_sweep,
        allowed_x_values=T_VALUES,
        title=f"COBRAS TruthfulQA vs T, epsilon={fixed_epsilon_for_t_sweep}",
        ylabel="True * Info",
        xlabel="T",
        save_path=plot_dir / "cobras_tqa_vs_t.png",
        show_std=True,
    )

    # 2. UltraFeedback performance vs T
    plot_cobras_vs_x(
        agg_df=uf_agg,
        metric="RM Mean",
        x_col="T",
        fixed_col="epsilon",
        fixed_value=fixed_epsilon_for_t_sweep,
        allowed_x_values=T_VALUES,
        title=f"COBRAS UltraFeedback vs T, epsilon={fixed_epsilon_for_t_sweep}",
        ylabel="RM Mean",
        xlabel="T",
        save_path=plot_dir / "cobras_uf_vs_t.png",
        show_std=True,
    )

    # 3. TruthfulQA performance vs epsilon
    plot_cobras_vs_x(
        agg_df=tqa_agg,
        metric="True * Info",
        x_col="epsilon",
        fixed_col="T",
        fixed_value=fixed_t_for_epsilon_sweep,
        allowed_x_values=EPSILON_VALUES,
        title=f"COBRAS TruthfulQA vs epsilon, T={fixed_t_for_epsilon_sweep}",
        ylabel="True * Info",
        xlabel="epsilon",
        save_path=plot_dir / "cobras_tqa_vs_epsilon.png",
        show_std=True,
    )

    # 4. UltraFeedback performance vs epsilon
    plot_cobras_vs_x(
        agg_df=uf_agg,
        metric="RM Mean",
        x_col="epsilon",
        fixed_col="T",
        fixed_value=fixed_t_for_epsilon_sweep,
        allowed_x_values=EPSILON_VALUES,
        title=f"COBRAS UltraFeedback vs epsilon, T={fixed_t_for_epsilon_sweep}",
        ylabel="RM Mean",
        xlabel="epsilon",
        save_path=plot_dir / "cobras_uf_vs_epsilon.png",
        show_std=True,
    )

    print("Saved plots:")
    print(plot_dir / "cobras_tqa_vs_t.png")
    print(plot_dir / "cobras_uf_vs_t.png")
    print(plot_dir / "cobras_tqa_vs_epsilon.png")
    print(plot_dir / "cobras_uf_vs_epsilon.png")


if __name__ == "__main__":
    main()