"""Q4 (Reviewer 4oei): run one of the read-only generation scripts with TF32 matmuls on.

The generation scripts under scripts/{truthfulqa,gsm8k,mmlu}/ are not to be edited, so this
wrapper flips the two TF32 switches and then executes the requested script as __main__, with
the remaining arguments forwarded to hydra untouched.

Usage:
  CUDA_VISIBLE_DEVICES=5 uv run python -u scripts/analysis/ablation/run_tf32.py \
      scripts/mmlu/mmlu_generate.py model=Llama3.1-8B-Base steer=Ablation-Full ...
"""

from __future__ import annotations

import runpy
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]
    print(f"[q4x] TF32 matmul={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn={torch.backends.cudnn.allow_tf32} -> {script}")
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
