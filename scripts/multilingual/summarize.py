"""Aggregate judged HalluQA generations into the reported table.

Reports, per method, the non-hallucination rate over the pooled two-fold test set (every
one of the 450 questions is answered exactly once, by a steer model fitted on the other
fold), the per-fold rates, and a paired bootstrap over questions for the difference
against NoSteer. Pairing is by question id, which is the right unit because the same
question is answered by every method.
"""

import argparse

import numpy as np
import pandas as pd


def bootstrap_diff(df, method, base='NoSteer', n_boot=10000, seed=0):
    """Paired bootstrap over question ids of rate(method) - rate(base)."""
    piv = df.pivot_table(index='question_id', columns='method',
                         values='non_hallucinated', aggfunc='mean')
    if method not in piv.columns or base not in piv.columns:
        return np.nan, np.nan, np.nan
    d = (piv[method] - piv[base]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    return float(d.mean() * 100), float(np.percentile(boots, 2.5) * 100), \
        float(np.percentile(boots, 97.5) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judged', nargs='+', required=True,
                    help='one judged jsonl per generation seed')
    ap.add_argument('--order', nargs='+',
                    default=['NoSteer', 'CAA', 'ITI', 'COBRAS'])
    args = ap.parse_args()

    frames = []
    for f in args.judged:
        d = pd.read_json(f, lines=True, orient='records')
        d['file'] = f
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    print(f'{len(df)} judged generations, seeds {sorted(df.seed.unique())}')

    print('\n=== per method, pooled over both test folds ===')
    print(f"{'method':10s} {'T fold0':>9s} {'T fold1':>9s} {'delta':>6s} "
          f"{'n':>5s} {'NonHallu%':>10s} {'fold0':>7s} {'fold1':>7s} "
          f"{'seed sd':>8s} {'vs NoSteer (95% CI)':>26s}")
    for m in args.order:
        g = df[df.method == m]
        if not len(g):
            continue
        rate = g.non_hallucinated.mean() * 100
        f0 = g[g.fold == 0].non_hallucinated.mean() * 100
        f1 = g[g.fold == 1].non_hallucinated.mean() * 100
        per_seed = g.groupby('seed').non_hallucinated.mean() * 100
        sd = per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan
        T0 = g[g.fold == 0]['T'].iloc[0]
        T1 = g[g.fold == 1]['T'].iloc[0]
        delta = g['delta'].iloc[0]
        diff, lo, hi = bootstrap_diff(df, m)
        ci = '-' if m == 'NoSteer' else f'{diff:+.2f} [{lo:+.2f}, {hi:+.2f}]'
        print(f'{m:10s} {T0:9.4f} {T1:9.4f} {delta:6.2f} {len(g):5d} '
              f'{rate:10.2f} {f0:7.2f} {f1:7.2f} '
              f'{"" if np.isnan(sd) else f"{sd:8.2f}":>8s} {ci:>26s}')

    print('\n=== per method x category ===')
    cat = (df.pivot_table(index='method', columns='category',
                          values='non_hallucinated', aggfunc='mean') * 100)
    print(cat.reindex(args.order).round(2).to_string())

    print('\n=== per method x seed ===')
    sd = (df.pivot_table(index='method', columns='seed',
                         values='non_hallucinated', aggfunc='mean') * 100)
    print(sd.reindex(args.order).round(2).to_string())


if __name__ == '__main__':
    main()
