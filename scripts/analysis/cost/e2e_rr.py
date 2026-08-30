"""Q4 (Reviewer EJdD): end-to-end decoding cost, round-robin repeats.

Same protocol as scripts/analysis/cost/e2e_latency.py (same model, precision, batch
size, prompt set, greedy fixed-length decoding) with two changes that make it
robust on a shared node:

  * methods are timed in round-robin (repeat 1: all methods, repeat 2: all
    methods, ...) instead of one method to completion, so a burst of CPU load
    part-way through the job perturbs every method rather than one of them;
  * the reported number per method is the MEDIAN over repeats of the mean
    seconds per batch, and the first repeat is discarded as warm-up.

This matters because the steering hook is kernel-launch bound (K sequential small
kernels per token), so it is sensitive to CPU contention in a way the unsteered
forward pass is not.

build() also accepts several tags joined by '+' (e.g. COBRAS-sub25+K2), matching
the variant names used by scripts/analysis/cost/quality_run.py.

Writes only results/analysis/q4/q4_e2e_rr.json.

CUDA_VISIBLE_DEVICES=6 ./.venv/bin/python scripts/analysis/cost/e2e_rr.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
from transformers import GenerationConfig

from cobras.lm import HuggingFaceLM
from cobras.steer import COBRAS
from cobras.utils import get_project_dir
from cobras.utils.data import load_tqa_gen_data, load_tqa_gen_questions
from variants import TopKCOBRAS, subsample

KW = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0,
          vmf_kappa=20, vmf_beta=0.0, abstain_percentile=1.0, abstain_k=32,
          abstain_sharpness=50.0)

SYS = ("Interpret each question literally, and as a question about the real world; "
       "carefully research each answer, without falling prey to any common myths; and "
       "reply ‘‘I have no comment’’ unless you are completely certain of the answer.")


def build(name, pos, neg):
    """(steer object, T). `name` is NoSteer or COBRAS[-tag[+tag...]]."""
    if name == "NoSteer":
        return None, 0.0
    if not name.startswith("COBRAS"):
        raise ValueError(name)
    tag = name[len("COBRAS"):].lstrip("-")
    cls, ckw, kw, p, n = COBRAS, {}, dict(max_iters=10), pos, neg
    for t in [x for x in tag.split("+") if x]:
        if t == "full":
            pass
        elif t.startswith("sub"):
            f = int(t[3:]) / 100.0
            p, n = subsample(pos, f, seed=0), subsample(neg, f, seed=1)
        elif t.startswith("topk"):
            cls, ckw = TopKCOBRAS, dict(topk=int(t[4:]))
        elif t.startswith("K"):
            kw["max_iters"] = int(t[1:])
        elif t == "plain":
            kw["potentials"] = "plain"
        else:
            raise ValueError(name)
    return cls(**ckw, **kw, **KW).fit(p, n), 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama3.1-8B-Base")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5, help="incl. 1 discarded warm-up")
    ap.add_argument("--new_tokens", type=int, default=50)
    ap.add_argument("--out", default="results/analysis/q4/q4_e2e_rr.json")
    ap.add_argument("--methods", default=(
        "NoSteer,COBRAS,COBRAS-K5,COBRAS-K2,COBRAS-K1,COBRAS-sub25,"
        "COBRAS-sub25+K2,COBRAS-plain,COBRAS-topk64"))
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    gen_cfg = GenerationConfig(
        max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
        do_sample=False, repetition_penalty=1.1,
    )
    model = HuggingFaceLM(args.model, None, default_generation_config=gen_cfg,
                          steer_layer_idx=args.layer, device="cuda", dtype=torch.float32)
    dev = model.model.device
    pos, neg = load_tqa_gen_data(args.model, args.layer, 0)
    pos, neg = pos.to(dev), neg.to(dev)
    qs = load_tqa_gen_questions(1)[: args.n_prompts]
    msgs = [[{"role": "system", "content": SYS}, {"role": "user", "content": q}] for q in qs]
    names = args.methods.split(",")
    nb = (len(msgs) + args.batch - 1) // args.batch
    print(f"{len(msgs)} prompts, batch {args.batch} ({nb} batches), "
          f"{args.new_tokens} new tokens, N+={pos.size(0)} N-={neg.size(0)}, "
          f"{args.repeats} repeats", flush=True)

    built = {}
    for name in names:
        steer, T = build(name, pos, neg)
        built[name] = (steer, T)
    sizes = {n: (0, 0) if s is None else (int(s.h_pos.size(0)), int(s.h_neg.size(0)))
             for n, (s, _) in built.items()}

    per_repeat = {n: [] for n in names}
    load_log = []
    for r in range(args.repeats):
        load_log.append(os.getloadavg()[0])
        for name in names:
            steer, T = built[name]
            model.steer_model = steer
            times = []
            for i in range(nb):
                batch = msgs[i * args.batch : (i + 1) * args.batch]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                model.chat(batch, steer=steer is not None, steer_kwargs=dict(T=T))
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            per_repeat[name].append(sum(times) / len(times))
            model.steer_model = None
        print(f"repeat {r}: load1={load_log[-1]:.0f}  " +
              "  ".join(f"{n}={per_repeat[n][-1]:.3f}" for n in names), flush=True)

    rows = []
    for name in names:
        vals = per_repeat[name][1:]          # drop repeat 0 as warm-up
        med = statistics.median(vals)
        rows.append(dict(
            method=name, T=built[name][1], B=args.batch, new_tokens=args.new_tokens,
            batches_per_repeat=nb, repeats_used=len(vals),
            N_pos_used=sizes[name][0], N_neg_used=sizes[name][1],
            s_per_batch_median=round(med, 4),
            s_per_batch_min=round(min(vals), 4),
            s_per_batch_max=round(max(vals), 4),
            ms_per_decode_step=round(med / args.new_tokens * 1e3, 4),
            ms_per_decode_step_min=round(min(vals) / args.new_tokens * 1e3, 4),
        ))

    base = next(r["ms_per_decode_step"] for r in rows if r["method"] == "NoSteer")
    base_min = next(r["ms_per_decode_step_min"] for r in rows if r["method"] == "NoSteer")
    for r in rows:
        r["added_ms_per_step"] = round(r["ms_per_decode_step"] - base, 4)
        r["overhead_pct"] = round(100.0 * (r["ms_per_decode_step"] - base) / base, 2)
        r["overhead_pct_min"] = round(
            100.0 * (r["ms_per_decode_step_min"] - base_min) / base_min, 2)

    out = get_project_dir() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=dict(
        gpu=torch.cuda.get_device_name(0), dtype="float32", device="cuda",
        torch=torch.__version__, model=args.model, layer=args.layer,
        decoding="greedy, fixed length", n_prompts=len(msgs),
        schedule="round-robin over methods, median over repeats, repeat 0 discarded",
        repeats=args.repeats, loadavg_at_repeat_start=load_log,
        N_pos=int(pos.size(0)), N_neg=int(neg.size(0)),
    ), rows=rows), indent=2))
    print(f"wrote {out}")
    for r in rows:
        print(f"{r['method']:>20}  {r['ms_per_decode_step']:8.2f} ms/step  "
              f"+{r['added_ms_per_step']:7.2f} ms  ({r['overhead_pct']:+.1f}%)  "
              f"[min-load {r['overhead_pct_min']:+.1f}%]")


if __name__ == "__main__":
    main()
