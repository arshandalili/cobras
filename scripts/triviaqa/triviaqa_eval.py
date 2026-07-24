import argparse
from pathlib import Path

import pandas as pd

from cobras.utils import get_project_dir

triviaqa_df_cols = ["Model", "Steering Method", "Accuracy", "N"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="Llama3.1-8B-Base")
    parser.add_argument("-l", "--layer_idx", type=int, default=13)
    parser.add_argument("-s", "--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = get_project_dir() / "results" / "triviaqa"
    raw_dir = output_dir / "raw_outputs" / args.model
    eval_path = (
        output_dir / "eval_results"
        / f"{args.model}-l{args.layer_idx}-TriviaQA-seed{args.seed}.csv"
    )
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame(columns=triviaqa_df_cols)

    prefix = f"{args.model}-l{args.layer_idx}-"
    suffix = f"-TriviaQA-seed{args.seed}"
    pattern = f"{prefix}*{suffix}.jsonl"

    for file_path in raw_dir.glob(pattern):
        steer_method = file_path.stem[len(prefix) : -len(suffix)]
        if steer_method in eval_df["Steering Method"].values:
            continue

        df = pd.read_json(file_path, orient="records", lines=True)
        correct = int(df["correct"].sum())
        total = len(df)
        accuracy = correct / total

        print(f"{args.model} | {steer_method}: Accuracy = {accuracy:.4f} ({correct}/{total})")
        eval_df.loc[len(eval_df)] = [args.model, steer_method, accuracy, total]
        eval_df = eval_df.sort_values(by="Steering Method")
        eval_df.to_csv(eval_path, index=False)

    eval_df.to_csv(eval_path, index=False)
    print(f"\nResults saved to {eval_path}")

if __name__ == "__main__":
    main()
