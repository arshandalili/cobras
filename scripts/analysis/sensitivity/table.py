"""Q5 (reviewer 4oei): summarize the never-tuned-parameter sensitivity sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cobras.utils import get_project_dir


GROUPS = {
    "k_bw (KDE bandwidth neighbours)": [
        ("3", "q5-kbw3"),
        ("5 (shipped)", "q5-ref"),
        ("10", "q5-kbw10"),
        ("20", "q5-kbw20"),
    ],
    "n_sinkhorn (Sinkhorn iterations)": [
        ("1", "q5-sink1"),
        ("5 (shipped)", "q5-ref"),
        ("20", "q5-sink20"),
    ],
    "vmf_kappa (strength gate)": [
        ("10", "q5-kappa10"),
        ("20 (shipped)", "q5-ref"),
        ("40", "q5-kappa40"),
    ],
    "alpha_sigma (tangent damping)": [
        ("1e-4", "q5-alpha1em4"),
        ("1e-3 (shipped)", "q5-ref"),
        ("1e-2", "q5-alpha1em2"),
    ],
}


def main() -> None:
    csv = get_project_dir() / "results" / "analysis" / "q5x" / "q5x_sensitivity.csv"
    df = pd.read_csv(csv)
    for c in ("True * Info", "Truthfulness", "Informativeness"):
        df[c] = df[c] * 100.0

    g = df.groupby("Variant").agg(
        txi_mean=("True * Info", "mean"),
        txi_std=("True * Info", "std"),
        true_mean=("Truthfulness", "mean"),
        info_mean=("Informativeness", "mean"),
        seeds=("Seed", "count"),
        N=("N", "max"),
    )

    ref = g.loc["q5-ref"]
    print(f"reference (shipped) True x Info = {ref.txi_mean:.2f} +/- {ref.txi_std:.2f} "
          f"over {int(ref.seeds)} seeds, N = {int(ref.N)}\n")

    spread = []
    for title, rows in GROUPS.items():
        print(title)
        print(f"  {'value':<16}{'T x I':>16}{'True':>9}{'Info':>9}{'d(TxI)':>9}{'seeds':>7}")
        vals = []
        for label, var in rows:
            if var not in g.index:
                print(f"  {label:<16}{'(pending)':>16}")
                continue
            r = g.loc[var]
            d = r.txi_mean - ref.txi_mean
            vals.append(r.txi_mean)
            print(
                f"  {label:<16}{r.txi_mean:>10.2f} +/-{r.txi_std:>4.2f}"
                f"{r.true_mean:>9.2f}{r.info_mean:>9.2f}{d:>+9.2f}{int(r.seeds):>7}"
            )
        if len(vals) == len(rows):
            spread.append((title, max(vals) - min(vals)))
        print()

    print("range of True x Info across each parameter's grid:")
    for title, s in spread:
        print(f"  {title:<36}{s:>6.2f} points")
    if spread:
        print(f"\n  worst-case range over all four parameters: {max(s for _, s in spread):.2f} points")
    print(f"  seed-to-seed std at the shipped setting:   {ref.txi_std:.2f} points")

    # per-seed spread at the shipped setting, for the noise floor
    r = df[df.Variant == "q5-ref"].sort_values("Seed")
    print("\n  shipped setting per seed: "
          + ", ".join(f"{int(s)}={v:.2f}" for s, v in zip(r.Seed, r["True * Info"])))

    # spread of every non-reference variant against the reference, pooled
    others = g.drop(index="q5-ref")
    print(f"\n  all {len(others)} off-shipped settings: True x Info in "
          f"[{others.txi_mean.min():.2f}, {others.txi_mean.max():.2f}], "
          f"max |delta| from shipped = {np.abs(others.txi_mean - ref.txi_mean).max():.2f}")


if __name__ == "__main__":
    main()
