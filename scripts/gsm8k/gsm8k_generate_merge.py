import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
from lightning import seed_everything

from odesteer.utils import get_project_dir


@hydra.main(
    config_path=str(get_project_dir() / "confs"),
    config_name="gsm8k",
    version_base="1.3",
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    input_dir: Path = get_project_dir() / "results" / "gsm8k" / "raw_rank_outputs" / cfg.model
    output_dir: Path = get_project_dir() / "results" / "gsm8k" / "raw_outputs" / cfg.model
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = (
        f"{cfg.model}-l{cfg.layer_idx}-{cfg.steer.name}"
        f"-GSM8K-seed{cfg.seed}-rank*-of-*.jsonl"
    )

    paths = sorted(input_dir.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No rank output files found matching: {input_dir / pattern}")

    records = []
    seen_world_sizes = set()

    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                records.append(r)
                seen_world_sizes.add(r.get("world_size"))

    if len(seen_world_sizes) > 1:
        raise ValueError(f"Found mixed world sizes in rank files: {seen_world_sizes}")

    world_size = seen_world_sizes.pop()
    print(f"Found {len(paths)} rank files from world_size={world_size}")

    if len(paths) != world_size:
        raise ValueError(f"Expected {world_size} rank files, found {len(paths)}")

    records = sorted(records, key=lambda r: r["idx"])

    merged_path = output_dir / (
        f"{cfg.model}-l{cfg.layer_idx}-{cfg.steer.name}-GSM8K-seed{cfg.seed}.jsonl"
    )

    with open(merged_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(records)} records to {merged_path}")


if __name__ == "__main__":
    main()