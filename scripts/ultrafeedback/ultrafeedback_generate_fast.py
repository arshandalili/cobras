import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json
import gc

import torch
import torch.distributed as dist
from transformers import GenerationConfig
from lightning import seed_everything
from accelerate import Accelerator

from odesteer.lm import HuggingFaceLM, batch_generate
from odesteer.utils import get_project_dir
from odesteer.utils.data import load_ultrafeedback_data, load_ultrafeedback_prompts


@hydra.main(
    config_path=str(get_project_dir() / "confs"),
    config_name="ultrafeedback",
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

    # Same global seed, with rank offset for per-process randomness.
    seed_everything(cfg.seed + rank)

    output_dir: Path = (
        get_project_dir()
        / "results"
        / "ultrafeedback"
        / "raw_rank_outputs"
        / cfg.model
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{cfg.model}-l{cfg.layer_idx}-{cfg.steer.name}"
        f"-UltrafeedbackBinarized-seed{cfg.seed}"
        f"-rank{rank}-of-{world_size}.jsonl"
    )

    output_path = output_dir / filename

    if output_path.exists():
        print(f"✓ Rank {rank}: Output file {filename} already exists. Skipping.")
        return

    try:
        print(
            f"→ Rank {rank}/{world_size}: Running {cfg.model}-{cfg.steer.name} "
            f"on UltrafeedbackBinarized layer {cfg.layer_idx} using {device}"
        )

        print(f"→ Rank {rank}: Loading LLM & fitting steer model ...")

        default_generation_config = GenerationConfig(
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            seed=cfg.seed,
            use_cache=True,
        )

        steer_model_kwargs = OmegaConf.to_container(
            cfg.steer.kwargs,
            resolve=True,
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

        pos_train, neg_train = load_ultrafeedback_data(
            cfg.model,
            cfg.layer_idx,
            "train",
        )
        model.fit_steer_model(pos_train, neg_train)

        # Free steering activations after fitting.
        del pos_train, neg_train
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"→ Rank {rank}: Loading test prompts ...")

        prompts = load_ultrafeedback_prompts("test")

        # Shard prompts across accelerate processes.
        all_indices = list(range(len(prompts)))
        rank_indices = all_indices[rank::world_size]
        rank_prompts = [prompts[i] for i in rank_indices]

        print(
            f"→ Rank {rank}/{world_size}: Generating {len(rank_prompts)} "
            f"responses with T={cfg.steer.T} ..."
        )

        outputs = batch_generate(
            model,
            rank_prompts,
            T=cfg.steer.T,
            batch_size=cfg.batch_size,
        )

        print(f"→ Rank {rank}: Generated {len(outputs)} outputs")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"→ Rank {rank}: Saving outputs to {filename} ...")

        with open(output_path, "w") as f:
            for idx, prompt, output in zip(rank_indices, rank_prompts, outputs):
                f.write(
                    json.dumps(
                        {
                            "idx": int(idx),
                            "prompt": prompt,
                            "output": output,
                            "generator": f"{cfg.model}-{cfg.steer.name}",
                            "dataset": "UltrafeedbackBinarized",
                            "T": cfg.steer.T,
                            "rank": rank,
                            "world_size": world_size,
                        }
                    )
                    + "\n"
                )

        print(
            f"✓ Rank {rank}/{world_size}: Completed "
            f"{cfg.model}-{cfg.steer.name} on UltrafeedbackBinarized"
        )
        print(f"  Rank responses generated: {len(outputs)}")
        print(f"  Configuration: T={cfg.steer.T}")
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