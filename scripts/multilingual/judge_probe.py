"""Is the Chinese judge lenient, and does it let refusals and evasions through?

HalluQA's own protocol scores with GPT-4-0613 under a five-vote majority and treats a
refusal or an answer that does not address the question as a hallucination unless the
reference answers are themselves refusals (rules 1, 2 and 6 of the rubric). We replace
GPT-4 with Qwen2.5-14B-Instruct, so the question is whether that substitution buys
non-hallucination rates for free.

This script builds a synthetic generation file in which *every* test question is answered
with the same fixed string, then runs the ordinary judge on it. A canned refusal that the
judge accepts for a question whose references are real facts is a false accept, and the
rate of those measures leniency directly, question by question, on the same 450 questions
the experiment reports.
"""

import argparse
import json

import pandas as pd

from cobras.utils import get_project_dir

DATA_DIR = get_project_dir() / 'data' / 'q5_halluqa'
OUT_DIR = get_project_dir() / 'results' / 'q5_halluqa'

# label: canned answer. The first four are refusals or evasions that carry no content, the
# last is fluent Chinese that simply does not answer anything.
PROBES = {
    'idk': '我不知道。',
    'unanswerable': '这个问题无法回答。',
    'apology': '抱歉，我无法回答这个问题。',
    'itdepends': '这取决于具体情况，没有统一的答案。',
    'irrelevant': '今天的天气很好，适合出去散步。',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT_DIR / 'q5_judge_probe_input.jsonl'))
    args = ap.parse_args()

    rows = []
    for split in (0, 1):
        df = pd.read_json(DATA_DIR / 'q5_texts' / f'q5_qa_{split}.jsonl',
                          lines=True, orient='records')
        for _, r in df.iterrows():
            # a question whose own reference answers are refusals is one where rule 6 says
            # a refusal is correct, so those are separated out when the rate is read off
            refs = ' '.join(r.correct_answers)
            ref_is_refusal = any(k in refs for k in
                                 ('无法回答', '不知道', '没有确切', '无法确定', '不存在'))
            for label, text in PROBES.items():
                rows.append(dict(
                    fold=split, method=label, delta=0.0, T=0.0, seed=0,
                    question_id=int(r.question_id), question=r.question,
                    category=r.category, best_answer=r.best_answer,
                    correct_answers=list(r.correct_answers),
                    ref_is_refusal=bool(ref_is_refusal), response=text))

    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    n_ref = sum(r['ref_is_refusal'] for r in rows) // len(PROBES)
    print(f'wrote {args.out} ({len(rows)} rows, {len(PROBES)} canned answers x 450 '
          f'questions; {n_ref} questions have refusal-style references)')


if __name__ == '__main__':
    main()
