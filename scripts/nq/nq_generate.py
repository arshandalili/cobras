import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json
import gc

import numpy as np
import torch
from lightning import seed_everything

from cobras.lm import HuggingFaceLM, batch_score_answers
from cobras.utils import get_project_dir
from cobras.utils.data import (
    load_tqa_gen_data_all_splits,
    load_query_activations,
    load_nq_data,
    format_qa_prompt,
)


@hydra.main(
    config_path=str(get_project_dir() / "confs"),
    config_name="nq",
    version_base="1.3",
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    output_dir: Path = get_project_dir() / "results" / "nq" / "raw_outputs" / cfg.model
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{cfg.model}-l{cfg.layer_idx}-{cfg.steer.name}-NQ-seed{cfg.seed}.jsonl"
    if (output_dir / filename).exists():
        print(f"✓ Output file {filename} already exists. Skipping.")
        exit()

    try:
        print(f"→ Running NQ OOD eval for {cfg.model}-{cfg.steer.name} on layer {cfg.layer_idx}")
        steer_model_kwargs = OmegaConf.to_container(cfg.steer.kwargs, resolve=True)

        model = HuggingFaceLM(
            cfg.model, cfg.steer.type,
            steer_model_kwargs=steer_model_kwargs,
            steer_layer_idx=cfg.layer_idx,
            device="auto", dtype=torch.float32,
        )

        print("→ Fitting steer on all TruthfulQA data ...")
        pos_X, neg_X = load_tqa_gen_data_all_splits(cfg.model, cfg.layer_idx)
        ref_X = load_query_activations(cfg.model, cfg.layer_idx, "truthfulqa")
        model.fit_steer_model(pos_X, neg_X, ref_X=ref_X)

        print("→ Loading Natural Questions data ...")
        questions, correct_answers, false_answers = load_nq_data()

        rng = np.random.default_rng(cfg.seed)
        indices = rng.choice(len(questions), size=min(cfg.num_examples, len(questions)), replace=False)
        questions = [questions[i] for i in indices]
        correct_answers = [correct_answers[i] for i in indices]
        false_answers = [false_answers[i] for i in indices]

        prompts = [format_qa_prompt(q) for q in questions]
        candidates = [correct + [false] for correct, false in zip(correct_answers, false_answers)]

        print(f"→ Scoring {len(prompts)} NQ questions with T={cfg.steer.T} ...")
        scores = batch_score_answers(model, prompts, candidates, T=cfg.steer.T, batch_size=cfg.batch_size)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        # a question counts as correct when a gold answer ranks first among the candidates
        preds = [cs[int(np.argmax(s))] for cs, s in zip(candidates, scores)]
        corrects = [pred in golds for pred, golds in zip(preds, correct_answers)]
        accuracy = sum(corrects) / len(corrects)
        print(f"→ NQ Accuracy: {accuracy:.4f}")

        print(f"→ Saving {len(preds)} records to {filename} ...")
        with open(output_dir / filename, "w") as f:
            for prompt, pred, correct, false, is_correct in zip(
                prompts, preds, correct_answers, false_answers, corrects
            ):
                f.write(json.dumps({
                    "prompt": prompt,
                    "output": pred,
                    "answers": correct,
                    "false_answer": false,
                    "correct": bool(is_correct),
                    "generator": f"{cfg.model}-{cfg.steer.name}",
                    "dataset": "NQ",
                    "T": cfg.steer.T,
                }) + "\n")

        print(f"✓ Completed {cfg.model}-{cfg.steer.name} on NQ")
        print("-" * 120)

    except Exception as e:
        print(f"→ Error: {e}")
        import traceback
        traceback.print_exc()
        exit()


if __name__ == "__main__":
    main()
