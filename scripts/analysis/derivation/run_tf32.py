"""Run one of the read-only generation scripts with TF32 matmuls enabled.

The generation scripts under scripts/{mmlu,gsm8k,...} are not to be edited, and TF32 has to be
switched on before torch is used. This sets the two flags and then executes the target script as
__main__ with the remaining arguments, so hydra sees exactly the command line it would have.

    uv run python -u scripts/analysis/derivation/run_tf32.py scripts/mmlu/mmlu_generate.py model=... steer=...
"""

import runpy
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:  # newer torch spells it this way as well
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

target = sys.argv[1]
sys.argv = [target] + sys.argv[2:]
print(f"[q6_run_tf32] TF32 on (matmul={torch.backends.cuda.matmul.allow_tf32}, "
      f"cudnn={torch.backends.cudnn.allow_tf32}) -> {target}", flush=True)
runpy.run_path(target, run_name="__main__")
