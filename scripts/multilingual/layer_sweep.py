"""Pick the steering layer on the validation split.

Layer 13 was inherited from the paper's English setup, which follows CAA's choice of layer
13 out of 32 on LLaMA-2. Qwen2.5-7B has 28 layers, and the Chinese steering literature
picks much later layers for this architecture, so the layer has to be re-selected rather
than assumed.

To keep the sweep affordable this stage uses each method's *raw* default strength from
`confs/steer/` (COBRAS 0.5, CAA 1.0, ITI 1.0) instead of solving the matched-displacement
grid at every layer, which would need one 60-point ODE probe per layer and fold. The layer
is chosen here, and the matched-displacement tuning is then run in full at the chosen
layer by halluqa_generate.py. Only validation questions are used, 45 per fold.
"""

import argparse
import json

import numpy as np
import torch
from transformers import GenerationConfig

from cobras.lm import HuggingFaceLM, batch_generate
from cobras.steer import get_steer_model
from cobras.utils import get_project_dir

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from halluqa_mc import STEER_KWARGS, fit_val_question_split, load_fold  # noqa: E402
from halluqa_generate import QA_FEWSHOT  # noqa: E402

OUT_DIR = get_project_dir() / 'results' / 'q5_halluqa'

# raw defaults from confs/steer/*.yaml
DEFAULT_T = {'CAA': 1.0, 'ITI': 1.0, 'COBRAS': 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('-l', '--layers', nargs='+', type=int,
                    default=[10, 13, 16, 19, 21, 23, 24, 25, 26])
    ap.add_argument('--methods', nargs='+', default=['CAA', 'ITI', 'COBRAS'])
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--split_seed', type=int, default=42)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=15)
    ap.add_argument('--max_new_tokens', type=int, default=50)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=True,
                               temperature=0.7, top_p=0.9, repetition_penalty=1.1,
                               seed=args.seed)
    lm = HuggingFaceLM(args.model, steer_name=None, default_generation_config=gen_cfg,
                       steer_layer_idx=args.layers[0], device='auto', dtype=torch.float32)

    records = []
    for test_fold in (0, 1):
        train_fold = 1 - test_fold
        qa_tr = load_fold(args.model, args.layers[0], train_fold)[0]
        _, val_q = fit_val_question_split(len(qa_tr), args.val_ratio, args.split_seed)
        prompts = [f"{QA_FEWSHOT}\n\nQ: {qa_tr.question.iloc[i]}\nA:" for i in val_q]
        rows = qa_tr.iloc[val_q].reset_index(drop=True)
        print(f'\n=== fold {test_fold}: {len(prompts)} validation questions ===', flush=True)

        # the unsteered run does not depend on the layer, so it is done once per fold
        lm.steer_model = None
        torch.manual_seed(args.seed)
        outs = batch_generate(lm, prompts, T=0.0, batch_size=args.batch_size)
        for (_, r), o in zip(rows.iterrows(), outs):
            records.append(dict(fold=test_fold, layer=-1, method='NoSteer', delta=0.0,
                                T=0.0, seed=args.seed, question_id=int(r.question_id),
                                question=r.question, category=r.category,
                                best_answer=r.best_answer,
                                correct_answers=list(r.correct_answers),
                                response=o.strip()))

        for layer in args.layers:
            _, pos_X, neg_X, pos_q, neg_q = load_fold(args.model, layer, train_fold)
            fit_q, _ = fit_val_question_split(len(qa_tr), args.val_ratio, args.split_seed)
            fit_pos = pos_X[np.isin(pos_q, fit_q)]
            fit_neg = neg_X[np.isin(neg_q, fit_q)]
            lm.steer_layer_idx = layer
            for method in args.methods:
                T = DEFAULT_T[method]
                lm.steer_model = get_steer_model(method, **STEER_KWARGS[method])
                lm.fit_steer_model(fit_pos, fit_neg)
                torch.manual_seed(args.seed)
                print(f'  layer {layer} {method} T={T}', flush=True)
                outs = batch_generate(lm, prompts, T=T, batch_size=args.batch_size)
                for (_, r), o in zip(rows.iterrows(), outs):
                    records.append(dict(
                        fold=test_fold, layer=layer, method=method, delta=0.0, T=T,
                        seed=args.seed, question_id=int(r.question_id),
                        question=r.question, category=r.category,
                        best_answer=r.best_answer,
                        correct_answers=list(r.correct_answers), response=o.strip()))
                lm.steer_model = None

    name = args.model.replace('/', '-')
    out = OUT_DIR / f'q5_layersweep_{name}{args.tag}_seed{args.seed}.jsonl'
    with open(out, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\nwrote {out} ({len(records)} rows)')


if __name__ == '__main__':
    main()
