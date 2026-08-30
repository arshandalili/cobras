"""Does every N=100 GSM8K/MMLU result have a full-size re-run queued for it?

The interactive rebuttal pass ran GSM8K and MMLU at num_examples=100. Each question section
ships an experiments/q<N>_fullsize.sh that is supposed to redo those rows at the full 1319 and
14042 test splits on a faster machine. This checks that claim instead of trusting it: it takes
each section's DRY_RUN log, resolves the steer name every planned command would produce, and
diffs that against the N=100 outputs actually on disk.

    for q in q1 q4 q6; do
      DRY_RUN=1 ARCHIVE_N100=0 bash experiments/${q}_fullsize.sh > /tmp/dry_$q.log 2>&1
    done
    uv run python scripts/analysis/coverage_audit.py /tmp/dry_q1.log /tmp/dry_q4.log /tmp/dry_q6.log

Exit status is 1 if any N=100 output is uncovered, or if any planned run is not at full size.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from cobras.utils import get_project_dir

ROOT = get_project_dir()
FULL = {"gsm8k": 1319, "mmlu": 14042}
TAG = {"gsm8k": "GSM8K", "mmlu": "MMLU"}


def steer_name(cfg: str, T: str | None, overrides: dict[str, str]) -> str | None:
    path = ROOT / "confs" / "steer" / f"{cfg}.yaml"
    if not path.exists():
        return None
    root = OmegaConf.create({"steer": OmegaConf.load(path)})
    if T is not None:
        root.steer.T = T
    for k, v in overrides.items():
        try:
            v = float(v) if ("." in v or "e-" in v) else int(v)
        except ValueError:
            if v in ("null", "None"):
                v = None
            elif v in ("true", "false"):
                v = v == "true"
        OmegaConf.update(root, k, v)
    return OmegaConf.select(root, "steer.name")


def parse_log(path: Path):
    """(task, resolved steer name, num_examples) for every generation the log plans."""
    planned = []
    for line in path.read_text(errors="replace").splitlines():
        if "_generate.py" not in line and "mmlu" not in line.lower():
            continue
        if "steer=" not in line:
            continue
        if "mmlu" in line.lower():
            task = "mmlu"
        elif "gsm8k" in line.lower():
            task = "gsm8k"
        else:
            continue  # truthfulqa, not part of this audit
        cfg = re.search(r"steer=(\S+)", line)
        T = re.search(r"steer\.T=(\S+)", line)
        n = re.search(r"num_examples=(\d+)", line)
        ov = dict(re.findall(r"(steer\.kwargs\.[\w.]+)=(\S+)", line))
        if not cfg:
            continue
        name = steer_name(cfg.group(1), T.group(1) if T else None, ov)
        if name:
            planned.append((task, name, int(n.group(1)) if n else None))
    return planned


def on_disk():
    """(task, steer name) for every GSM8K/MMLU output with at most 200 records."""
    small = defaultdict(set)
    for task, tag in TAG.items():
        for f in (ROOT / "results" / task / "raw_outputs").rglob("*.jsonl"):
            if sum(1 for _ in f.open()) > 200:
                continue
            m = re.match(rf".*-l\d+-(.+)-{tag}-seed\d+\.jsonl$", f.name)
            if m:
                small[m.group(1).split("-")[0]].add((task, m.group(1)))
    return small


def main() -> int:
    logs = [Path(p) for p in sys.argv[1:]]
    if not logs:
        print(__doc__)
        return 2

    small = on_disk()
    bad = 0
    for log in logs:
        section = re.search(r"(q\d+)", log.stem)
        section = section.group(1) if section else log.stem
        planned = parse_log(log)
        # the full-size runs deliberately drop the "-n100" marker some sections put in the name
        plan_set = {(t, n.replace("-n100", "")) for t, n, _ in planned}
        have_set = {(t, n.replace("-n100", "")) for t, n in small.get(section, set())}

        undersized = [(t, n, k) for t, n, k in planned if k != FULL[t]]
        missing = sorted(have_set - plan_set)

        print(f"=== {section} ===")
        print(f"  N<=200 outputs on disk : {len(have_set)}")
        print(f"  full-size runs planned : {len(plan_set)}")
        print(f"  extra rows planned     : {len(plan_set - have_set)}")
        if missing:
            bad += 1
            print(f"  NOT COVERED ({len(missing)}):")
            for t, n in missing:
                print(f"      {t:6s} {n}")
        else:
            print("  every N=100 output is covered")
        if undersized:
            bad += 1
            print(f"  NOT AT FULL SIZE ({len(undersized)}):")
            for t, n, k in undersized[:10]:
                print(f"      {t:6s} num_examples={k} (want {FULL[t]})  {n}")
        print()

    print("AUDIT PASSED" if not bad else "AUDIT FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
