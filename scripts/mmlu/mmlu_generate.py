import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json
import gc

import numpy as np
import torch
from lightning import seed_everything

from cobras.lm import HuggingFaceLM
from cobras.utils import get_project_dir
from cobras.utils.data import (
    load_tqa_gen_data_all_splits,
    load_mmlu_data,
)

_LETTERS = ["A", "B", "C", "D"]


def build_few_shot_str(
    dev_questions: list[str],
    dev_letters: list[str],
    dev_choices: list[list[str]],
    n_shots: int = 5,
) -> str:
    header = "The following are multiple choice questions (with answers) about various topics.\n\n"
    shots = ""
    for q, ltr, cs in zip(dev_questions[:n_shots], dev_letters[:n_shots], dev_choices[:n_shots]):
        shots += f"Question: {q}\n"
        for letter, choice in zip(_LETTERS, cs):
            shots += f"{letter}) {choice}\n"
        shots += f"Answer: {ltr}\n\n"
    return header + shots


def format_mmlu_prompt(question: str, choices: list[str], few_shot_str: str) -> str:
    body = f"Question: {question}\n"
    for letter, choice in zip(_LETTERS, choices):
        body += f"{letter}) {choice}\n"
    body += "Answer:"
    return few_shot_str + body


@torch.no_grad()
def batch_score_mcq(
    model: HuggingFaceLM,
    prompts: list[str],
    batch_size: int,
    steer_kwargs: dict,
) -> list[str]:
    choice_toks = torch.tensor([
        model.tokenizer.encode(f" {l}", add_special_tokens=False)[-1]
        for l in _LETTERS
    ]).to(model.model.device)
    use_steer = model.steer_model is not None
    preds = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = model.tokenizer(batch, return_tensors="pt", padding=True).to(model.model.device)
        if use_steer:
            model.register_steer_hook(-1, steer_kwargs)
        logits = model.model(**inputs).logits
        if use_steer:
            model.remove_steer_hook()
        last_logits = logits[:, -1, :]
        scores = last_logits[:, choice_toks]
        preds.extend([_LETTERS[idx] for idx in scores.argmax(dim=-1).tolist()])
        if (i // batch_size) % 10 == 0:
            print(f"  [{i + len(batch)}/{len(prompts)}]")
    return preds


@hydra.main(
    config_path=str(get_project_dir() / "confs"),
    config_name="mmlu",
    version_base="1.3",
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    output_dir: Path = get_project_dir() / "results" / "mmlu" / "raw_outputs" / cfg.model
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{cfg.model}-l{cfg.layer_idx}-{cfg.steer.name}-MMLU-seed{cfg.seed}.jsonl"
    if (output_dir / filename).exists():
        print(f"✓ Output file {filename} already exists. Skipping.")
        exit()

    try:
        print(f"→ Running MMLU OOD eval for {cfg.model}-{cfg.steer.name} on layer {cfg.layer_idx}")
        steer_model_kwargs = OmegaConf.to_container(cfg.steer.kwargs, resolve=True)

        model = HuggingFaceLM(
            cfg.model, cfg.steer.type,
            steer_model_kwargs=steer_model_kwargs,
            steer_layer_idx=cfg.layer_idx,
            device="auto", dtype=torch.float32,
        )

        print("→ Fitting steer on all TruthfulQA data ...")
        pos_X, neg_X = load_tqa_gen_data_all_splits(cfg.model, cfg.layer_idx)
        model.fit_steer_model(pos_X, neg_X)

        print("→ Loading MMLU dev (5-shot) and test data ...")
        dev_questions, dev_letters, dev_choices = load_mmlu_data("dev")
        test_questions, test_letters, test_choices = load_mmlu_data("test")
        few_shot_str = build_few_shot_str(dev_questions, dev_letters, dev_choices, n_shots=5)

        rng = np.random.default_rng(cfg.seed)
        indices = rng.choice(len(test_questions), size=min(cfg.num_examples, len(test_questions)), replace=False)
        test_questions = [test_questions[i] for i in indices]
        test_letters = [test_letters[i] for i in indices]
        test_choices = [test_choices[i] for i in indices]

        prompts = [format_mmlu_prompt(q, cs, few_shot_str) for q, cs in zip(test_questions, test_choices)]

        print(f"→ Scoring {len(prompts)} MMLU questions with T={cfg.steer.T} ...")
        preds = batch_score_mcq(model, prompts, cfg.batch_size, dict(T=cfg.steer.T))

        del model
        gc.collect()
        torch.cuda.empty_cache()

        accuracy = sum(p == c for p, c in zip(preds, test_letters)) / len(preds)
        print(f"→ MMLU Accuracy: {accuracy:.4f}")

        print(f"→ Saving {len(preds)} records to {filename} ...")
        with open(output_dir / filename, "w") as f:
            for prompt, pred, correct in zip(prompts, preds, test_letters):
                f.write(json.dumps({
                    "prompt": prompt,
                    "output": pred,
                    "correct": correct,
                    "generator": f"{cfg.model}-{cfg.steer.name}",
                    "dataset": "MMLU",
                    "T": cfg.steer.T,
                }) + "\n")

        print(f"✓ Completed {cfg.model}-{cfg.steer.name} on MMLU")
        print("-" * 120)

    except Exception as e:
        print(f"→ Error: {e}")
        import traceback
        traceback.print_exc()
        exit()


if __name__ == "__main__":
    main()
