"""Robustness check: keep only the first line of each free-form answer, then re-judge.

The official HalluQA free-form prompt is a six-shot "Q: ... \\nA: ..." block and expects a
one-line answer. HuggingFaceLM.generate cuts a response at the next "\\nQ:", which leaves
the answer intact whenever the model starts a new question, but a base model sometimes
continues with "Human:" or a fresh unrelated paragraph instead, and that trailing text is
then shown to the judge. This script writes a copy of a generation file with each response
cut at its first newline, so the same judge can be run again on answers only. It changes
roughly one response in eight; the point is to check that the ranking of the methods does
not depend on how the trailing text is handled.
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_file', required=True)
    args = ap.parse_args()

    df = pd.read_json(args.gen_file, lines=True, orient='records')
    for col in ('non_hallucinated', 'margin'):
        if col in df.columns:
            df = df.drop(columns=col)
    before = df.response.copy()
    df['response'] = df.response.str.split('\n').str[0].str.strip()
    changed = float((before.str.strip() != df.response).mean())

    out = Path(str(args.gen_file).replace('.jsonl', '_firstline.jsonl'))
    df.to_json(out, orient='records', lines=True, force_ascii=False)
    print(f'wrote {out} ({len(df)} rows, {changed * 100:.1f}% of responses truncated)')


if __name__ == '__main__':
    main()
