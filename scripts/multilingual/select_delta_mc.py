"""Pick the steering strength for each method from the validation sweep.

Selection rule, fixed before the sweep was run: the target displacement whose mean
validation accuracy over both folds and all choice-order seeds is highest, ties broken
toward the smaller displacement (the least intervention). Validation questions are the
10% held out inside each training fold; the test fold is never touched.
"""

import argparse
import json

import pandas as pd

from cobras.utils import get_project_dir

OUT_DIR = get_project_dir() / 'results' / 'q5_halluqa'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.val_csv)
    grouped = df.groupby(['method', 'delta'])['acc'].mean().reset_index()
    print(grouped.assign(acc=lambda d: (d.acc * 100).round(2)).to_string(index=False))

    selected = {}
    for method, g in grouped.groupby('method'):
        if method == 'NoSteer':
            continue
        g = g.sort_values(['acc', 'delta'], ascending=[False, True])
        selected[method] = float(g.iloc[0].delta)
    print('\nselected target displacements:', selected)
    json.dump(selected, open(args.out, 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
