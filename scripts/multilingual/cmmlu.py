"""Chinese analogue of the paper's MMLU out-of-distribution capability check.

Steer models are fitted on the HalluQA contrastive pairs (the whole fit set, both folds,
mirroring scripts/mmlu/mmlu_generate.py which fits on all of TruthfulQA) and then applied
unchanged to CMMLU (https://huggingface.co/datasets/haonan-li/cmmlu), a 67-subject Chinese
multiple-choice benchmark. Everything else follows scripts/mmlu/mmlu_generate.py: five-shot
prompt from the dev split, answer read from the letter logits, steering at the final prompt
token.
"""

import argparse
import io
import json
import zipfile

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from cobras.lm import HuggingFaceLM
from cobras.steer import get_steer_model
from cobras.utils import get_project_dir

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from halluqa_mc import (  # noqa: E402
    STEER_KWARGS, fit_val_question_split, load_fold,
    query_activations, solve_T_grid,
)
from halluqa_generate import QA_FEWSHOT  # noqa: E402

OUT_DIR = get_project_dir() / 'results' / 'q5_halluqa'
LETTERS = ['A', 'B', 'C', 'D']
HEADER = "以下是关于各个学科的单项选择题，请选出正确答案。\n\n"


def fmt(question, choices):
    s = f"题目：{question}\n"
    for l, c in zip(LETTERS, choices):
        s += f"{l}) {c}\n"
    return s


def load_cmmlu(n_subjects=None):
    """Read the official CMMLU v1.0.1 archive from the Hub.

    datasets.load_dataset('haonan-li/cmmlu', subject) needs trust_remote_code, so the
    release zip is downloaded and its per-subject CSVs are read directly. Columns are
    Question, A, B, C, D, Answer, exactly what the loading script exposes. Subjects are
    visited in sorted order so the five-shot prompt is deterministic."""
    path = hf_hub_download('haonan-li/cmmlu', 'cmmlu_v1_0_1.zip', repo_type='dataset')
    z = zipfile.ZipFile(path)
    subjects = sorted(
        n.split('/')[1][:-4] for n in z.namelist()
        if n.startswith('test/') and n.endswith('.csv')
    )
    if n_subjects:
        subjects = subjects[:n_subjects]
    dev, test = [], []
    for s in subjects:
        for split, bucket in (('dev', dev), ('test', test)):
            df = pd.read_csv(io.BytesIO(z.read(f'{split}/{s}.csv')))
            for r in df.to_dict('records'):
                r['subject'] = s
                bucket.append(r)
    return dev, test


@torch.no_grad()
def score(lm, prompts, letter_ids, T, bs):
    steer = lm.steer_model is not None
    preds = []
    for i in range(0, len(prompts), bs):
        inputs = lm.tokenizer(prompts[i:i + bs], return_tensors='pt',
                              padding=True).to(lm.model.device)
        if steer:
            lm.register_steer_hook(-1, dict(T=T))
            logits = lm.model(**inputs, logits_to_keep=1).logits[:, -1]
            lm.remove_steer_hook()
        else:
            logits = lm.model(**inputs, logits_to_keep=1).logits[:, -1]
        preds.extend([LETTERS[k] for k in logits[:, letter_ids].argmax(-1).tolist()])
        if (i // bs) % 25 == 0:
            print(f'  [{i + len(prompts[i:i + bs])}/{len(prompts)}]', flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', default='Qwen2.5-7B-Base')
    ap.add_argument('-l', '--layer_idx', type=int, default=13)
    ap.add_argument('--methods', nargs='+', default=['NoSteer', 'CAA', 'ITI', 'COBRAS'])
    ap.add_argument('--selected_delta', required=True)
    ap.add_argument('--num_examples', type=int, default=3000)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--split_seed', type=int, default=42)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = json.load(open(args.selected_delta))

    dev, test = load_cmmlu()
    few = HEADER + "".join(
        fmt(r['Question'], [r[l] for l in LETTERS]) + f"答案：{r['Answer']}\n\n"
        for r in dev[:5]
    )
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(test), size=min(args.num_examples, len(test)), replace=False)
    test = [test[i] for i in idx]
    prompts = [few + fmt(r['Question'], [r[l] for l in LETTERS]) + "答案：" for r in test]
    gold = [r['Answer'] for r in test]
    print(f'CMMLU: {len(prompts)} test questions')

    lm = HuggingFaceLM(args.model, steer_name=None, steer_layer_idx=args.layer_idx,
                       device='auto', dtype=torch.float32)
    letter_ids = torch.tensor(
        [lm.tokenizer.encode(l, add_special_tokens=False)[-1] for l in LETTERS]
    ).to(lm.model.device)

    # fit on the union of both folds' fit questions, no test-fold leakage from HalluQA
    pos_all, neg_all, cal_prompts = [], [], []
    for fold in [0, 1]:
        qa, pos_X, neg_X, pos_q, neg_q = load_fold(args.model, args.layer_idx, fold)
        fit_q, val_q = fit_val_question_split(len(qa), args.val_ratio, args.split_seed)
        pos_all.append(pos_X[np.isin(pos_q, fit_q)])
        neg_all.append(neg_X[np.isin(neg_q, fit_q)])
        # the strength was selected on the free-form HalluQA task, so T is solved on the
        # same free-form validation prompts and then applied to CMMLU unchanged
        cal_prompts += [f"{QA_FEWSHOT}\n\nQ: {qa.question.iloc[i]}\nA:" for i in val_q]
    pos_all, neg_all = torch.cat(pos_all), torch.cat(neg_all)
    print(f'steer fit on {len(pos_all)} D+ / {len(neg_all)} D- HalluQA activations')
    # T is the strength selected on the in-distribution HalluQA task, applied unchanged
    cal_act = query_activations(lm, cal_prompts, args.layer_idx, 4)

    rows = []
    for method in args.methods:
        if method == 'NoSteer':
            lm.steer_model, T = None, 0.0
        else:
            lm.steer_model = get_steer_model(method, **STEER_KWARGS[method])
            lm.fit_steer_model(pos_all, neg_all)
            T = solve_T_grid(lm.steer_model, cal_act, method,
                             targets=[float(selected[method])])[0]
        print(f'scoring {method} T={T}', flush=True)
        preds = score(lm, prompts, letter_ids, T, args.batch_size)
        acc = float(np.mean([p == g for p, g in zip(preds, gold)]))
        print(f'  {method}: CMMLU acc = {acc * 100:.2f}')
        rows.append(dict(model=args.model, method=method, T=T, n=len(preds), acc=acc))
        lm.steer_model = None

    out = OUT_DIR / f'q5_cmmlu_{args.model}_l{args.layer_idx}.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)


if __name__ == '__main__':
    main()
