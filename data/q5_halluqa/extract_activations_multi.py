"""Extract HalluQA D+/D- activations at many layers in a single forward pass.

Same procedure as extract_activations.py, but HuggingFaceLM.extract_prompt_eos_activations
already accepts a list of layer indices and returns a dict, so a layer sweep does not need
one full pass over the data per layer. Files are written in exactly the layout
q5_halluqa_mc.load_fold expects, one file per layer.
"""

import argparse

import pandas as pd
import torch
from tqdm import trange

from cobras.lm import HuggingFaceLM
from cobras.utils import get_project_dir


@torch.no_grad()
def extract_base(model, questions, answers, layers):
    prompts = [f'Q: {q}\nA: {a}' for q, a in zip(questions, answers)]
    return model.extract_prompt_eos_activations(prompts, layers)


@torch.no_grad()
def extract_chat(model, questions, answers, layers):
    messages = [[{"role": "user", "content": q}, {"role": "assistant", "content": a}]
                for q, a in zip(questions, answers)]
    return model.extract_message_eos_activations(messages, layer_idx=layers)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('-l', '--layers', nargs='+', type=int,
                    default=list(range(10, 27)))
    ap.add_argument('-b', '--batch_size', type=int, default=8)
    args = ap.parse_args()

    data_dir = get_project_dir() / 'data' / 'q5_halluqa'
    texts_dir = data_dir / 'q5_texts'
    act_dir = data_dir / 'q5_activations' / args.model
    act_dir.mkdir(parents=True, exist_ok=True)

    model = HuggingFaceLM(args.model, device='auto', dtype=torch.float32)
    # same branch as data/truthfulqa/extract_activations.py: everything except a
    # non-Qwen base model goes through the chat-template path
    extract = (extract_base if ('Base' in args.model and 'Qwen' not in args.model)
               else extract_chat)

    for file in sorted(texts_dir.glob('q5_[pn]*_[01].jsonl')):
        todo = [l for l in args.layers
                if not (act_dir / f'{file.stem}_activations_layer{l}.pt').exists()]
        if not todo:
            print(f'exists: {file.stem}')
            continue
        print(f'processing {file.stem} on layers {todo}', flush=True)
        df = pd.read_json(file, lines=True, orient='records')
        questions, answers = df['question'].tolist(), df['answer'].tolist()
        acc = {l: [] for l in todo}
        n_batches = (len(df) + args.batch_size - 1) // args.batch_size
        for i in trange(n_batches):
            s, e = i * args.batch_size, min((i + 1) * args.batch_size, len(df))
            out = extract(model, questions[s:e], answers[s:e], todo)
            for l in todo:
                acc[l].append(out[l].cpu())
        torch.save(df.idx.values, act_dir / f'{file.stem}_question_idx.pt')
        for l in todo:
            X = torch.cat(acc[l], dim=0)
            torch.save(X, act_dir / f'{file.stem}_activations_layer{l}.pt')
        print(f'saved {file.stem} {tuple(X.shape)} for {len(todo)} layers', flush=True)
