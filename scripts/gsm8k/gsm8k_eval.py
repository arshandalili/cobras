import argparse
import re
from pathlib import Path

import pandas as pd

from odesteer.utils import get_project_dir

gsm8k_df_cols = ["Model", "Steering Method", "Accuracy", "N", "N_parsed"]

def extract_answer(output: str) -> str | None:
    output = output.split("\nQuestion:")[0]
    if "####" in output:
        after = output.split("####")[1].strip()
        m = re.search(r"-?[\d,]+(?:\.\d+)?", after)
        if m:
            return m.group(0).replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", output)
    return nums[-1].replace(",", "") if nums else None

def normalize(ans: str) -> str:
    ans = ans.replace(",", "").strip()
    try:
        v = float(ans)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return ans

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="Llama3.1-8B-Base")
    parser.add_argument("-l", "--layer_idx", type=int, default=13)
    parser.add_argument("-s", "--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = get_project_dir() / "results" / "gsm8k"
    raw_dir = output_dir / "raw_outputs" / args.model
    eval_path = (
        output_dir / "eval_results"
        / f"{args.model}-l{args.layer_idx}-GSM8K-seed{args.seed}.csv"
    )
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame(columns=gsm8k_df_cols)

    prefix = f"{args.model}-l{args.layer_idx}-"
    suffix = f"-GSM8K-seed{args.seed}"
    pattern = f"{prefix}*{suffix}.jsonl"

    for file_path in raw_dir.glob(pattern):
        steer_method = file_path.stem[len(prefix) : -len(suffix)]
        if steer_method in eval_df["Steering Method"].values:
            continue

        df = pd.read_json(file_path, orient="records", lines=True)

        correct_count = 0
        parsed_count = 0
        for _, row in df.iterrows():
            pred = extract_answer(str(row["output"]))
            if pred is not None:
                parsed_count += 1
                if normalize(pred) == normalize(str(row["correct_answer"])):
                    correct_count += 1

        total = len(df)
        accuracy = correct_count / total

        print(
            f"{args.model} | {steer_method}: Accuracy = {accuracy:.4f} "
            f"({correct_count}/{total}), parsed {parsed_count}/{total}"
        )
        eval_df.loc[len(eval_df)] = [args.model, steer_method, accuracy, total, parsed_count]
        eval_df = eval_df.sort_values(
            by="Steering Method"
        )
        eval_df.to_csv(eval_path, index=False)

    eval_df.to_csv(eval_path, index=False)
    print(f"\nResults saved to {eval_path}")

if __name__ == "__main__":
    main()
