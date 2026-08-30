"""How much displacement does each method actually apply while decoding?

The tuning grid in halluqa_mc.py matches methods on the median relative displacement
||steer(h)-h||/||h|| measured at the *final prompt token*, the activation the steer model
sees when the first answer token is produced. Steering is then applied at every decoded
token, whose activations are not the ones the budget was calibrated on. For a fixed vector
(CAA, ITI) the displacement is T||v||/||h|| and can only drift as far as the norm drifts.
COBRAS is query-adaptive, so its step depends on where the current activation sits relative
to the contrastive data, and it can realise a different budget once decoding leaves the
prompt.

This script wraps steer() to log the realised displacement of every call, separating the
first call (prompt token, the calibrated one) from the later calls (decoded tokens).
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from transformers import GenerationConfig

from cobras.lm import HuggingFaceLM, batch_generate
from cobras.steer import get_steer_model
from cobras.utils import get_project_dir

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from halluqa_mc import (  # noqa: E402
    STEER_KWARGS, fit_val_question_split, load_fold, query_activations, solve_T_grid,
)
from halluqa_generate import QA_FEWSHOT  # noqa: E402

OUT_DIR = get_project_dir() / 'results' / 'q5_halluqa'


class Probe:
    """Records the relative displacement of every steer() call during generation.

    HuggingFaceLM.generate calls register_steer_hook exactly once per batch, so wrapping
    it gives a reliable marker for "this is the first steered position of a new batch",
    which is the final prompt token. Every later call in that batch is a decoded token.
    Counting on max_new_tokens instead would misalign whenever a batch stops early.
    """

    def __init__(self, lm):
        self.lm = lm
        self.inner_steer = lm.steer_model.steer
        self.inner_register = lm.register_steer_hook
        self.records = []      # (call index within batch, relative displacement)
        self.call_idx = 0

    def __enter__(self):
        def wrapped_steer(X, **kw):
            out = self.inner_steer(X, **kw)
            rel = ((out - X).norm(dim=-1) / X.norm(dim=-1).clamp(min=1e-8))
            for v in rel.detach().float().cpu().tolist():
                self.records.append((self.call_idx, v))
            self.call_idx += 1
            return out

        def wrapped_register(*a, **kw):
            self.call_idx = 0
            return self.inner_register(*a, **kw)

        self.lm.steer_model.steer = wrapped_steer
        self.lm.register_steer_hook = wrapped_register
        return self

    def __exit__(self, *a):
        self.lm.steer_model.steer = self.inner_steer
        self.lm.register_steer_hook = self.inner_register


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', default='Qwen2.5-7B-Base')
    ap.add_argument('-l', '--layer_idx', type=int, default=13)
    ap.add_argument('--methods', nargs='+', default=['CAA', 'ITI', 'COBRAS'])
    ap.add_argument('--selected_delta', required=True)
    ap.add_argument('--n_questions', type=int, default=60)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--split_seed', type=int, default=42)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=15)
    ap.add_argument('--max_new_tokens', type=int, default=50)
    ap.add_argument('--extra_deltas', nargs='+', type=float, default=[],
                    help='additional target displacements to probe for every method')
    args = ap.parse_args()

    selected = json.load(open(args.selected_delta))
    gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=True,
                               temperature=0.7, top_p=0.9, repetition_penalty=1.1,
                               seed=args.seed)
    lm = HuggingFaceLM(args.model, steer_name=None, default_generation_config=gen_cfg,
                       steer_layer_idx=args.layer_idx, device='auto', dtype=torch.float32)

    rows = []
    for test_fold in [0, 1]:
        train_fold = 1 - test_fold
        qa_tr, pos_X, neg_X, pos_q, neg_q = load_fold(args.model, args.layer_idx, train_fold)
        fit_q, val_q = fit_val_question_split(len(qa_tr), args.val_ratio, args.split_seed)
        fit_pos = pos_X[np.isin(pos_q, fit_q)]
        fit_neg = neg_X[np.isin(neg_q, fit_q)]
        cal_prompts = [f"{QA_FEWSHOT}\n\nQ: {qa_tr.question.iloc[i]}\nA:" for i in val_q]
        cal_act = query_activations(lm, cal_prompts, args.layer_idx, 4)

        qa_eval = load_fold(args.model, args.layer_idx, test_fold)[0].head(args.n_questions)
        prompts = [f"{QA_FEWSHOT}\n\nQ: {q}\nA:" for q in qa_eval.question]

        for method in args.methods:
            lm.steer_model = get_steer_model(method, **STEER_KWARGS[method])
            lm.fit_steer_model(fit_pos, fit_neg)
            targets = [float(selected[method])] + list(args.extra_deltas)
            Ts = solve_T_grid(lm.steer_model, cal_act, method, targets=targets)
            for delta, T in zip(targets, Ts):
                torch.manual_seed(args.seed)
                with Probe(lm) as p:
                    batch_generate(lm, prompts, T=T, batch_size=args.batch_size)
                    df = pd.DataFrame(p.records, columns=['call', 'rel'])
                # call 0 of each batch is the prompt's final token, the position the
                # budget was calibrated on; every later call is a decoded token
                first = df[df.call == 0]
                later = df[df.call > 0]
                rows.append(dict(
                    fold=test_fold, method=method, target_delta=delta, T=T,
                    prompt_token_median=float(first.rel.median()),
                    decoded_token_median=float(later.rel.median()),
                    decoded_token_p90=float(later.rel.quantile(0.9)),
                    decoded_over_target=float(later.rel.median()) / delta,
                    n_calls=len(df)))
                print(f'fold {test_fold} {method:7s} target={delta:<5} T={T:<8} '
                      f'prompt={rows[-1]["prompt_token_median"]:.4f} '
                      f'decoded={rows[-1]["decoded_token_median"]:.4f} '
                      f'(x{rows[-1]["decoded_over_target"]:.2f})', flush=True)
            lm.steer_model = None

    out = OUT_DIR / f'q5_displacement_{args.model}_l{args.layer_idx}.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print('\nwrote', out)
    print(pd.DataFrame(rows).groupby(['method', 'target_delta'])[
        ['prompt_token_median', 'decoded_token_median', 'decoded_over_target']
    ].mean().round(4).to_string())


if __name__ == '__main__':
    main()
