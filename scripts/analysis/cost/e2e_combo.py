"""Q4 (Reviewer EJdD): end-to-end decoding cost for '+'-combined COBRAS variants.

Identical protocol to scripts/analysis/cost/e2e_latency.py (same model, precision,
batch size, prompt set, greedy fixed-length decoding, same warm-up handling), but
build() accepts several tags joined by '+' (e.g. COBRAS-sub25+K2), which the
original single-tag parser cannot express. NoSteer / COBRAS / COBRAS-K2 /
COBRAS-sub25 are re-measured here as a consistency check against the rows already
in q4_e2e_latency.json.

Writes only results/analysis/q4/q4_e2e_combo.json.

CUDA_VISIBLE_DEVICES=6 ./.venv/bin/python scripts/analysis/cost/e2e_combo.py
"""
from __future__ import annotations

import argparse
import gc
import json
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
    """(steer object, T). `name` is NoSteer or COBRAS[-tag[+tag...]].

    Tag semantics are exactly those of q4_quality_run.build, so a latency row here
    corresponds to the generation run of the same variant name."""
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
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--warmup_batches", type=int, default=2)
    ap.add_argument("--new_tokens", type=int, default=50)
    ap.add_argument("--out", default="results/analysis/q4/q4_e2e_combo.json")
    ap.add_argument("--methods",
                    default="NoSteer,COBRAS,COBRAS-K2,COBRAS-sub25,COBRAS-sub25+K2")
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
    print(f"{len(msgs)} prompts, batch {args.batch}, {args.new_tokens} new tokens, "
          f"N+={pos.size(0)} N-={neg.size(0)}", flush=True)

    rows = []
    for name in args.methods.split(","):
        steer, T = build(name, pos, neg)
        model.steer_model = steer
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        per_batch = []
        nb = (len(msgs) + args.batch - 1) // args.batch
        for i in range(nb):
            batch = msgs[i * args.batch : (i + 1) * args.batch]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            model.chat(batch, steer=steer is not None, steer_kwargs=dict(T=T))
            torch.cuda.synchronize()
            per_batch.append(time.perf_counter() - t0)
        t = torch.tensor(per_batch[args.warmup_batches :])
        row = dict(
            method=name, T=T, batches=int(t.numel()), B=args.batch,
            new_tokens=args.new_tokens,
            N_pos_used=int(steer.h_pos.size(0)) if steer is not None else 0,
            N_neg_used=int(steer.h_neg.size(0)) if steer is not None else 0,
            s_per_batch=round(t.mean().item(), 4),
            ms_per_decode_step=round(t.mean().item() / args.new_tokens * 1e3, 4),
            peak_mem_GB=round(torch.cuda.max_memory_allocated() / 2**30, 3),
        )
        print(json.dumps(row), flush=True); rows.append(row)
        del steer; model.steer_model = None
        gc.collect(); torch.cuda.empty_cache()

    base = next(r["ms_per_decode_step"] for r in rows if r["method"] == "NoSteer")
    for r in rows:
        r["added_ms_per_step"] = round(r["ms_per_decode_step"] - base, 4)
        r["overhead_pct"] = round(100.0 * (r["ms_per_decode_step"] - base) / base, 2)

    out = get_project_dir() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=dict(
        gpu=torch.cuda.get_device_name(0), dtype="float32", device="cuda",
        torch=torch.__version__, model=args.model, layer=args.layer,
        decoding="greedy, fixed length", n_prompts=len(msgs),
        warmup_batches=args.warmup_batches,
        N_pos=int(pos.size(0)), N_neg=int(neg.size(0)),
    ), rows=rows), indent=2))
    print(f"wrote {out}")
    for r in rows:
        print(f"{r['method']:>18}  {r['ms_per_decode_step']:8.2f} ms/step  "
              f"+{r['added_ms_per_step']:7.2f} ms  ({r['overhead_pct']:+.1f}%)")


if __name__ == "__main__":
    main()
