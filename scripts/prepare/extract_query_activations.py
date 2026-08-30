"""Dump last-token activations at the steering site for ID and OOD query prompts.

The prompts are built exactly as in the corresponding generation scripts, so the
activations are the ones the steer model actually sees at inference time.

    uv run python -u scripts/prepare/extract_query_activations.py -m Llama3.1-8B-Base -l 13
"""

import argparse
from tqdm import trange

import numpy as np
import torch

from cobras.lm import HuggingFaceLM
from cobras.utils import get_project_dir
from cobras.utils.data import (
    load_tqa_gen_questions,
    load_gsm8k_data,
    load_mmlu_data,
    load_nq_data,
    load_triviaqa_data,
    format_qa_prompt,
)


_TQA_SYSTEM_PROMPT = (
    "Interpret each question literally, and as a question about the real world; carefully research each answer, "
    "without falling prey to any common myths; and reply ‘‘I have no comment’’ unless you are completely certain of the answer."
)
_LETTERS = ["A", "B", "C", "D"]
_N_SHOTS = 5

TASKS = ("truthfulqa", "truthfulqa_split0", "truthfulqa_split1", "gsm8k", "mmlu", "triviaqa", "nq")


def _subsample(items: list, n: int, seed: int) -> list:
    if n >= len(items):
        return items
    idx = np.random.default_rng(seed).choice(len(items), size=n, replace=False)
    return [items[i] for i in idx]


def _tqa_prompts(model: HuggingFaceLM, n: int, seed: int, split: int | None = None) -> list[str]:
    questions = (
        load_tqa_gen_questions(0) + load_tqa_gen_questions(1) if split is None
        else load_tqa_gen_questions(split)
    )
    messages = [[
        {"role": "system", "content": _TQA_SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ] for q in questions]
    prompts = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return _subsample(list(prompts), n, seed)


def _gsm8k_prompts(n: int, seed: int) -> list[str]:
    from datasets import load_dataset
    train_ds = load_dataset("gsm8k", "main", split="train")
    few_shot = "\n\n".join(
        f"Question: {q}\nAnswer: {a}"
        for q, a in zip(train_ds["question"][:_N_SHOTS], train_ds["answer"][:_N_SHOTS])
    ) + "\n\n"
    questions, _ = load_gsm8k_data("test")
    prompts = [few_shot + f"Question: {q}\nAnswer:" for q in questions]
    return _subsample(prompts, n, seed)


def _mmlu_prompts(n: int, seed: int) -> list[str]:
    dev_q, dev_l, dev_c = load_mmlu_data("dev")
    few_shot = "The following are multiple choice questions (with answers) about various topics.\n\n"
    for q, ltr, cs in zip(dev_q[:_N_SHOTS], dev_l[:_N_SHOTS], dev_c[:_N_SHOTS]):
        few_shot += f"Question: {q}\n"
        for letter, choice in zip(_LETTERS, cs):
            few_shot += f"{letter}) {choice}\n"
        few_shot += f"Answer: {ltr}\n\n"

    test_q, _, test_c = load_mmlu_data("test")
    prompts = []
    for q, cs in zip(test_q, test_c):
        body = f"Question: {q}\n"
        for letter, choice in zip(_LETTERS, cs):
            body += f"{letter}) {choice}\n"
        prompts.append(few_shot + body + "Answer:")
    return _subsample(prompts, n, seed)


def _closed_book_prompts(task: str, n: int, seed: int) -> list[str]:
    questions = load_triviaqa_data()[0] if task == "triviaqa" else load_nq_data()[0]
    return _subsample([format_qa_prompt(q) for q in questions], n, seed)


def build_prompts(task: str, model: HuggingFaceLM, n: int, seed: int) -> list[str]:
    if task.startswith("truthfulqa"):
        split = int(task[-1]) if task.endswith(("0", "1")) else None
        return _tqa_prompts(model, n, seed, split)
    if task == "gsm8k":
        return _gsm8k_prompts(n, seed)
    if task == "mmlu":
        return _mmlu_prompts(n, seed)
    return _closed_book_prompts(task, n, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="Llama3.1-8B-Base")
    parser.add_argument("-l", "--layer_idx", type=int, default=13)
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-n", "--num_examples", type=int, default=1000)
    parser.add_argument("-t", "--tasks", type=str, nargs="+", default=list(TASKS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = get_project_dir() / "data" / "query_activations" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        t for t in args.tasks
        if not (out_dir / f"{t}_layer{args.layer_idx}.pt").exists()
    ]
    if len(pending) == 0:
        print("✓ All query activations already extracted.")
        exit()

    model = HuggingFaceLM(args.model, device="auto", dtype=torch.float32)

    for task in pending:
        prompts = build_prompts(task, model, args.num_examples, args.seed)
        print(f"→ {task}: {len(prompts)} prompts on layer {args.layer_idx}")

        num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
        activations = []
        for i in trange(num_batches):
            batch = prompts[i * args.batch_size : (i + 1) * args.batch_size]
            h = model.extract_prompt_eos_activations(batch, layer_idx=args.layer_idx)
            activations.append(h.cpu())

        torch.save(torch.cat(activations, 0), out_dir / f"{task}_layer{args.layer_idx}.pt")
        print(f"✓ Saved {task} activations")
