"""Pick the steering strength for the free-form task from the judged validation sweep.

Same rule as select_delta_mc.py: the target displacement with the highest mean validation
non-hallucination rate over both folds, ties broken toward the smaller displacement.
"""

import argparse
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judged', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df = pd.read_json(args.judged, lines=True, orient='records')
    grouped = (df.groupby(['method', 'delta'])['non_hallucinated']
                 .agg(['mean', 'count']).reset_index())
    grouped['mean'] = (grouped['mean'] * 100).round(2)
    print(grouped.to_string(index=False))

    selected = {}
    for method, g in grouped.groupby('method'):
        if method == 'NoSteer':
            continue
        g = g.sort_values(['mean', 'delta'], ascending=[False, True])
        selected[method] = float(g.iloc[0].delta)
    print('\nselected target displacements:', selected)
    json.dump(selected, open(args.out, 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
