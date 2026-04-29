import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json
import gc

import numpy as np
import torch
import torch.distributed as dist
from transformers import GenerationConfig
from lightning import seed_everything
from accelerate import Accelerator

from odesteer.lm import HuggingFaceLM, batch_generate
from odesteer.utils import get_project_dir
from odesteer.utils.data import (
    load_tqa_gen_data_all_splits,
    load_gsm8k_data,
)

_N_SHOTS = 5


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
    accelerator = Accelerator()

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device

    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    is_main_process = accelerator.is_main_process

    # Use same seed for data subset selection across all ranks.
    # Add rank for any per-process randomness.
    seed_everything(cfg.seed + rank)

    output_dir: Path = get_project_dir() / "results" / "gsm8k" / "raw_rank_outputs" / cfg.model
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{cfg.model}-l{cfg.layer_idx}-{cfg.steer.name}"
        f"-GSM8K-seed{cfg.seed}-rank{rank}-of-{world_size}.jsonl"
    )

    output_path = output_dir / filename

    if output_path.exists():
        print(f"✓ Rank {rank}: Output file {filename} already exists. Skipping.")
        return

    try:
        print(
            f"→ Rank {rank}/{world_size}: Running GSM8K OOD eval for "
            f"{cfg.model}-{cfg.steer.name} on layer {cfg.layer_idx} using {device}"
        )

        steer_model_kwargs = OmegaConf.to_container(cfg.steer.kwargs, resolve=True)

        default_generation_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.1,
            use_cache=True,
        )

        model = HuggingFaceLM(
            cfg.model,
            cfg.steer.type,
            default_generation_config=default_generation_config,
            steer_model_kwargs=steer_model_kwargs,
            steer_layer_idx=cfg.layer_idx,
            device=str(device),
            dtype=torch.float32,
        )

        print(f"→ Rank {rank}: Fitting steer on all TruthfulQA data ...")

        pos_X, neg_X = load_tqa_gen_data_all_splits(
            cfg.model,
            cfg.layer_idx,
        )
        model.fit_steer_model(pos_X, neg_X)

        # Free steering training activations if no longer needed.
        del pos_X, neg_X
        gc.collect()
        torch.cuda.empty_cache()

        print(f"→ Rank {rank}: Loading GSM8K data ...")

        from datasets import load_dataset

        train_ds = load_dataset("gsm8k", "main", split="train")
        train_questions_raw = [r["question"] for r in train_ds]
        train_answers_raw = [r["answer"] for r in train_ds]
        few_shot_str = build_few_shot_str(train_questions_raw, train_answers_raw)

        test_questions, test_answers = load_gsm8k_data("test")

        # Important: use cfg.seed, not cfg.seed + rank, so all ranks produce
        # the same global subset before sharding.
        rng = np.random.default_rng(cfg.seed)
        indices = rng.choice(
            len(test_questions),
            size=min(cfg.num_examples, len(test_questions)),
            replace=False,
        )

        # Shard across accelerate processes.
        rank_indices = indices[rank::world_size]

        test_questions = [test_questions[i] for i in rank_indices]
        test_answers = [test_answers[i] for i in rank_indices]

        prompts = [format_gsm8k_prompt(q, few_shot_str) for q in test_questions]

        print(
            f"→ Rank {rank}/{world_size}: Generating {len(prompts)} "
            f"GSM8K responses with T={cfg.steer.T} ..."
        )

        outputs = batch_generate(
            model,
            prompts,
            T=cfg.steer.T,
            batch_size=cfg.batch_size,
        )

        outputs = [o.split("\nQuestion:")[0] for o in outputs]

        del model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"→ Rank {rank}: Saving {len(outputs)} records to {filename} ...")

        with open(output_path, "w") as f:
            for idx, prompt, output, correct in zip(
                rank_indices,
                prompts,
                outputs,
                test_answers,
            ):
                f.write(
                    json.dumps(
                        {
                            "idx": int(idx),
                            "prompt": prompt,
                            "output": output,
                            "correct_answer": correct,
                            "generator": f"{cfg.model}-{cfg.steer.name}",
                            "dataset": "GSM8K",
                            "T": cfg.steer.T,
                            "rank": rank,
                            "world_size": world_size,
                        }
                    )
                    + "\n"
                )

        print(
            f"✓ Rank {rank}/{world_size}: Completed "
            f"{cfg.model}-{cfg.steer.name} on GSM8K"
        )
        print("-" * 120)

        if dist.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])

        if is_main_process:
            print("✓ All ranks completed.")

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    except Exception as e:
        print(f"→ Rank {rank}: Error: {e}")
        import traceback

        traceback.print_exc()
        raise e


if __name__ == "__main__":
    main()