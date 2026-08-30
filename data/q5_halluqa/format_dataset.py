"""Build the Chinese (HalluQA) analogue of data/truthfulqa/format_dataset.py.

HalluQA (Cheng et al., 2023, arXiv:2310.03368; https://github.com/OpenMOSS/HalluQA) is a
Chinese hallucination benchmark modelled on TruthfulQA: 450 adversarial questions, each with
1-4 reference correct answers ("Best AnswerN") and 1-4 hallucinated answers ("Wrong_AnswerN").
That is exactly the structure data/truthfulqa/format_dataset.py consumes, so D+/D- are built
the same way and the two-fold 50/50 question split uses the same seed.

Outputs (all under data/q5_halluqa/q5_texts/):
  q5_pos_{0,1}.jsonl   idx, question, answer   -- D+ contrastive pairs
  q5_neg_{0,1}.jsonl   idx, question, answer   -- D- contrastive pairs
  q5_qa_{0,1}.jsonl    idx, question_id, question, best_answer, correct_answers,
                       wrong_answers, mc_choices, mc_answer, category
"""

import json
import re

import pandas as pd
from datasets import Dataset

from cobras.utils import get_project_dir

DATA_DIR = get_project_dir() / 'data' / 'q5_halluqa'
RAW_DIR = DATA_DIR / 'q5_raw'
TEXT_DIR = DATA_DIR / 'q5_texts'


def _parse_mc(mc_question: str) -> dict[str, str]:
    """Split the official HalluQA_mc.json question string into its five lettered choices."""
    spans = [(m.start(), m.group(1)) for m in re.finditer(r'([A-E]):', mc_question)]
    choices = {}
    for i, (start, letter) in enumerate(spans):
        end = spans[i + 1][0] if i + 1 < len(spans) else len(mc_question)
        choices[letter] = mc_question[start + 2:end].strip()
    return choices


def format_halluqa():
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_DIR / 'q5_HalluQA.json') as f:
        raw = json.load(f)
    with open(RAW_DIR / 'q5_HalluQA_mc.json') as f:
        mc_raw = {x['question_id']: x for x in json.load(f)}

    records = []
    for item in raw:
        correct = [item[k].strip() for k in
                   ('Best Answer1', 'Best Answer2', 'Best Answer3', 'Best Answer4')
                   if item.get(k, '').strip()]
        wrong = [item[k].strip() for k in
                 ('Wrong_Answer1', 'Wrong_Answer2', 'Wrong_Answer3', 'Wrong_Answer4')
                 if item.get(k, '').strip()]
        mc = mc_raw[item['question_id']]
        choices = _parse_mc(mc['question'])
        records.append({
            'question_id': item['question_id'],
            'question': item['Question'].strip(),
            'best_answer': item['Best Answer1'].strip(),
            'correct_answers': correct,
            'wrong_answers': wrong,
            # 421 questions have 5 choices, the 29 with a single wrong answer have 2
            'mc_choices': [choices[l] for l in 'ABCDE' if l in choices],
            'mc_answer': mc['answer'].replace('Answer:', '').strip(),
            'category': item['Category'],
        })

    # same two-fold 50/50 question split protocol as TruthfulQA (seed 42)
    ds = Dataset.from_list(records).train_test_split(test_size=0.5, seed=42)
    df0, df1 = ds['train'].to_pandas(), ds['test'].to_pandas()

    for split_idx, df in enumerate([df0, df1]):
        df = df.reset_index(drop=True)
        pos_pairs, neg_pairs = [], []
        for idx, row in df.iterrows():
            for a in row.correct_answers:
                pos_pairs.append({'idx': idx, 'question': row.question, 'answer': a})
            for a in row.wrong_answers:
                neg_pairs.append({'idx': idx, 'question': row.question, 'answer': a})
        pos_df, neg_df = pd.DataFrame(pos_pairs), pd.DataFrame(neg_pairs)
        pos_df.to_json(TEXT_DIR / f'q5_pos_{split_idx}.jsonl', orient='records', lines=True,
                       force_ascii=False)
        neg_df.to_json(TEXT_DIR / f'q5_neg_{split_idx}.jsonl', orient='records', lines=True,
                       force_ascii=False)
        df['idx'] = df.index
        df.to_json(TEXT_DIR / f'q5_qa_{split_idx}.jsonl', orient='records', lines=True,
                   force_ascii=False)
        print(f'split {split_idx}: {len(df)} questions, '
              f'{len(pos_df)} positive pairs, {len(neg_df)} negative pairs')


if __name__ == '__main__':
    format_halluqa()
