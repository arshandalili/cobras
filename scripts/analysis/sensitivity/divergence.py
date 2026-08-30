"""Q5 (reviewer 4oei): how much each never-tuned parameter changes the actual generations.

The point of the sensitivity sweep is that True x Info does not move. That is only
interesting if the parameter is doing something, so this counts how many of the 817
TruthfulQA responses differ from the shipped configuration at the same seed.
"""

from __future__ import annotations

import json
from pathlib import Path

from cobras.utils import get_project_dir


MODEL, LAYER = "Llama3.1-8B-Base", 13
VARIANTS = [
    "q5-kbw3", "q5-kbw10", "q5-kbw20",
    "q5-sink1", "q5-sink20",
    "q5-kappa10", "q5-kappa40",
    "q5-alpha1em4", "q5-alpha1em2",
]


def outputs(path: Path) -> list[str]:
    return [json.loads(line)["output"] for line in open(path)]


def main() -> None:
    gen = get_project_dir() / "results" / "analysis" / "q5x" / "gen"

    def p(variant: str, seed: int) -> Path:
        return gen / f"{MODEL}-l{LAYER}-{variant}-TruthfulQA-seed{seed}.jsonl"

    print(f"{'variant':<14}{'seed':>6}{'responses changed vs shipped':>32}")
    for variant in VARIANTS:
        for seed in (42, 43, 44):
            a, b = p("q5-ref", seed), p(variant, seed)
            if not (a.exists() and b.exists()):
                continue
            oa, ob = outputs(a), outputs(b)
            n = sum(x != y for x, y in zip(oa, ob))
            print(f"{variant:<14}{seed:>6}{n:>22} / {len(oa)}  ({100 * n / len(oa):.1f}%)")


if __name__ == "__main__":
    main()
