"""Run scripts/mmlu/mmlu_generate.py with TF32 matmuls enabled.

Full-precision MMLU scoring costs ~92 min per row; with TF32 it costs ~18 min at 99.8%
prediction agreement.  scripts/mmlu is read-only for this workstream, so TF32 is turned on
here and the generation script is executed unmodified.  Every row of the Q1/Q2 table is run
through this wrapper, including the unsteered reference.
"""

import runpy
import torch

from cobras.utils import get_project_dir

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

runpy.run_path(str(get_project_dir() / "scripts" / "mmlu" / "mmlu_generate.py"), run_name="__main__")
