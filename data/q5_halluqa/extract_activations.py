"""Chinese (HalluQA) analogue of data/truthfulqa/extract_activations.py.

Identical procedure: last-token residual-stream activation of the formatted
question+answer pair, at a fixed layer, for D+ and D-. Base models use the
"Q: {q}\\nA: {a}" template (which is also the template of the official HalluQA
few-shot prompt, prompts/Chinese_QA_prompt.txt), Qwen uses its chat template,
matching the branch in data/truthfulqa/extract_activations.py.
"""

import argparse

import pandas as pd
import torch
from tqdm import trange

from cobras.lm import HuggingFaceLM
from cobras.utils import get_project_dir


@torch.no_grad()
def extract_base_activations(model, questions, answers, layer_idx):
    full_prompts = [f'Q: {q}\nA: {a}' for q, a in zip(questions, answers)]
    return model.extract_prompt_eos_activations(full_prompts, layer_idx)


@torch.no_grad()
def extract_chat_activations(model, questions, answers, layer_idx):
    messages = [[{"role": "user", "content": q}, {"role": "assistant", "content": a}]
                for q, a in zip(questions, answers)]
    return model.extract_message_eos_activations(messages, layer_idx=layer_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='Qwen2.5-7B-Base')
    parser.add_argument('-l', '--layer_idx', type=int, default=13)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    args = parser.parse_args()

    data_dir = get_project_dir() / 'data' / 'q5_halluqa'
    texts_dir = data_dir / 'q5_texts'
    act_dir = data_dir / 'q5_activations' / args.model
    act_dir.mkdir(parents=True, exist_ok=True)

    model = HuggingFaceLM(args.model, device="auto", dtype=torch.float32)
    layer_idx = args.layer_idx

    if 'Base' in args.model and 'Qwen' not in args.model:
        extract_func = extract_base_activations
    else:
        extract_func = extract_chat_activations

    for file in sorted(texts_dir.glob('q5_[pn]*_[01].jsonl')):
        save_path = act_dir / f'{file.stem}_activations_layer{layer_idx}.pt'
        if save_path.exists():
            print(f'exists: {save_path.name}')
            continue
        print(f'processing {file.stem} on layer {layer_idx}')
        df = pd.read_json(file, lines=True, orient='records')
        questions, answers = df['question'].tolist(), df['answer'].tolist()
        n_batches = (len(df) + args.batch_size - 1) // args.batch_size
        acts = []
        for i in trange(n_batches):
            s, e = i * args.batch_size, min((i + 1) * args.batch_size, len(df))
            acts.append(extract_func(model, questions[s:e], answers[s:e], layer_idx).cpu())
        torch.save(df.idx.values, act_dir / f'{file.stem}_question_idx.pt')
        torch.save(torch.cat(acts, dim=0), save_path)
        print(f'saved {save_path.name} {torch.cat(acts, dim=0).shape}')
