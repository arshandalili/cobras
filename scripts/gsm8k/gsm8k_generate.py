import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json
import gc

import numpy as np
import torch
from transformers import GenerationConfig
from lightning import seed_everything

from odesteer.lm import HuggingFaceLM, batch_generate
from odesteer.utils import get_project_dir
from odesteer.utils.data import (
    load_tqa_gen_data_all_splits,
    load_tqa_gen_data_all_splits_with_idx,
    load_gsm8k_data,
)

_N_SHOTS = 8


def build_few_shot_str(train_questions: list[str], train_answers_raw: list[str]) -> str:
    parts = []
    for q, a in zip(train_questions[:_N_SHOTS], train_answers_raw[:_N_SHOTS]):
        parts.append(f"Question: {q}\nAnswer: {a}")
    return "\n\n".join(parts) + "\n\n"


def format_gsm8k_prompt(question: str, few_shot_str: str) -> str:
    return few_shot_str + f"Question: {question}\nAnswer:"


@hydra.main(
    config_path=str(get_project_dir() / "confs"),
    config_name="gsm8k",
    version_base="1.3",
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    output_dir: Path = get_project_dir() / "results" / "gsm8k" / "raw_outputs" / cfg.model
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{cfg.model}-l{cfg.layer_idx}-{cfg.steer.name}-GSM8K-seed{cfg.seed}.jsonl"
    if (output_dir / filename).exists():
        print(f"✓ Output file {filename} already exists. Skipping.")
        exit()

    try:
        print(f"→ Running GSM8K OOD eval for {cfg.model}-{cfg.steer.name} on layer {cfg.layer_idx}")
        steer_model_kwargs = OmegaConf.to_container(cfg.steer.kwargs, resolve=True)

        default_generation_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.1,
        )

        model = HuggingFaceLM(
            cfg.model, cfg.steer.type,
            default_generation_config=default_generation_config,
            steer_model_kwargs=steer_model_kwargs,
            steer_layer_idx=cfg.layer_idx,
            device="auto", dtype=torch.float32,
        )

        print("→ Fitting steer on all TruthfulQA data ...")
        uses_paired = getattr(model.steer_model, "needs_paired_idx", False)
        if uses_paired:
            pos_X, neg_X, pos_q_idx, neg_q_idx = load_tqa_gen_data_all_splits_with_idx(
                cfg.model, cfg.layer_idx
            )
            model.fit_steer_model(pos_X, neg_X, pos_q_idx=pos_q_idx, neg_q_idx=neg_q_idx)
        else:
            pos_X, neg_X = load_tqa_gen_data_all_splits(cfg.model, cfg.layer_idx)
            model.fit_steer_model(pos_X, neg_X)

        print("→ Loading GSM8K data ...")
        from datasets import load_dataset
        train_ds = load_dataset("gsm8k", "main", split="train")
        train_questions_raw = [r["question"] for r in train_ds]
        train_answers_raw = [r["answer"] for r in train_ds]
        few_shot_str = build_few_shot_str(train_questions_raw, train_answers_raw)

        test_questions, test_answers = load_gsm8k_data("test")

        rng = np.random.default_rng(cfg.seed)
        indices = rng.choice(len(test_questions), size=min(cfg.num_examples, len(test_questions)), replace=False)
        test_questions = [test_questions[i] for i in indices]
        test_answers = [test_answers[i] for i in indices]

        prompts = [format_gsm8k_prompt(q, few_shot_str) for q in test_questions]

        print(f"→ Generating {len(prompts)} GSM8K responses with T={cfg.steer.T} ...")
        outputs = batch_generate(model, prompts, T=cfg.steer.T, batch_size=cfg.batch_size)
        outputs = [o.split("\nQuestion:")[0] for o in outputs]

        del model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"→ Saving {len(outputs)} records to {filename} ...")
        with open(output_dir / filename, "w") as f:
            for prompt, output, correct in zip(prompts, outputs, test_answers):
                f.write(json.dumps({
                    "prompt": prompt,
                    "output": output,
                    "correct_answer": correct,
                    "generator": f"{cfg.model}-{cfg.steer.name}",
                    "dataset": "GSM8K",
                    "T": cfg.steer.T,
                }) + "\n")

        print(f"✓ Completed {cfg.model}-{cfg.steer.name} on GSM8K")
        print("-" * 120)

    except Exception as e:
        print(f"→ Error: {e}")
        import traceback
        traceback.print_exc()
        exit()


if __name__ == "__main__":
    main()
