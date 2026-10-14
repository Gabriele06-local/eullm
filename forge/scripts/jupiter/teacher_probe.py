#!/usr/bin/env python3
"""The distillation step and the teacher's generation on GH200, measured
before any corpus is on JUPITER (sbatch_teacher_probe.slurm).

Two modes:

``step``: one micro-batch of the Leonardo recipe
(``forge/training/configs/leonardo/distill_qwen3_30b_a3b_to_4b_split.yaml``),
the way ``distill.py`` runs it, on two GPUs: the teacher (Qwen3-30B-A3B-Base)
in BF16 alone on the first, with no gradient; the student (Qwen3-4B-Base)
with LoRA r128/alpha256 on all seven projections and gradient checkpointing
on the second; ``cutoff_len`` 2048, batch 1, the KL at temperature 2 mixed
with the cross-entropy at alpha 0.7, AdamW. Leonardo's optimiser step is 16
of these (``gradient_accumulation_steps``), at ~380 steps an hour with the
teacher split over two A100s; ``steps/h`` here is 3600 / (16 x the median
micro-batch). The tokens are random: the time of a step does not depend on
the text, and no corpus is on JUPITER yet. Without the Leonardo adapter of
the teacher (continued pre-training), which adds a LoRA's worth of compute.

``generate``: the teacher writing 256 tokens per sequence with plain
Transformers, greedy, batches of 1, 8 and 32: the backend Leonardo measured
at 58-64 tok/s (deliverable 5's baseline), here with the base model.

One JSON line per result on stdout, prefixed ``RESULT``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

TEACHER = "Qwen/Qwen3-30B-A3B-Base"
STUDENT = "Qwen/Qwen3-4B-Base"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PROMPT = ("Ai sensi dell'articolo 1 della Costituzione della Repubblica italiana, "
          "la sovranità appartiene al popolo, che la esercita nelle forme e nei limiti")


def result(**kv) -> None:
    kv.setdefault("host", os.uname().nodename)
    kv.setdefault("job", os.environ.get("SLURM_JOB_ID"))
    kv.setdefault("cuda_visible", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("RESULT " + json.dumps(kv), flush=True)


def sync(*devices) -> None:
    for d in devices:
        if str(d).startswith("cuda"):
            torch.cuda.synchronize(d)


def mem_gib(device, peak=False) -> float:
    if not str(device).startswith("cuda"):
        return 0.0
    f = torch.cuda.max_memory_allocated if peak else torch.cuda.memory_allocated
    return round(f(device) / 2**30, 1)


def load(model_id: str, device: str):
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map={"": device}, local_files_only=True)
    sync(device)
    return model, time.time() - t0


def step_mode(args) -> None:
    from peft import LoraConfig, get_peft_model

    t_dev, s_dev = args.devices
    teacher, t_load = load(args.teacher, t_dev)
    teacher.eval()
    student, s_load = load(args.student, s_dev)
    student.gradient_checkpointing_enable()
    student.enable_input_require_grads()
    student = get_peft_model(student, LoraConfig(
        r=128, lora_alpha=256, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=5e-5)
    result(mode="step", phase="load", teacher_s=round(t_load, 1), student_s=round(s_load, 1),
           teacher_gib=mem_gib(t_dev), student_gib=mem_gib(s_dev))

    vocab = min(teacher.config.vocab_size, student.config.vocab_size)
    temp, alpha = 2.0, 0.7
    times = []
    for i in range(args.warmup + args.iters):
        ids = torch.randint(0, vocab, (1, args.seq), device=t_dev)
        sync(t_dev, s_dev)
        t0 = time.time()
        with torch.no_grad():
            t_logits = teacher(input_ids=ids).logits[..., :vocab].to(s_dev)
        s_ids = ids.to(s_dev)
        s_logits = student(input_ids=s_ids).logits[..., :vocab]
        kl = F.kl_div(F.log_softmax(s_logits.float() / temp, -1),
                      F.log_softmax(t_logits.float() / temp, -1),
                      log_target=True, reduction="batchmean") * temp**2 / args.seq
        ce = F.cross_entropy(s_logits[:, :-1].reshape(-1, vocab).float(), s_ids[:, 1:].reshape(-1))
        loss = alpha * kl + (1 - alpha) * ce
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync(t_dev, s_dev)
        if i >= args.warmup:
            times.append(time.time() - t0)
    med = statistics.median(times)
    result(mode="step", phase="micro-batch", seq=args.seq, iters=len(times),
           median_s=round(med, 3), min_s=round(min(times), 3), max_s=round(max(times), 3),
           steps_per_h_at_16=round(3600 / (16 * med), 1),
           teacher_peak_gib=mem_gib(t_dev, peak=True), student_peak_gib=mem_gib(s_dev, peak=True))


def generate_mode(args) -> None:
    from transformers import AutoTokenizer

    dev = args.devices[0]
    tok = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher, t_load = load(args.teacher, dev)
    teacher.eval()
    result(mode="generate", phase="load", teacher_s=round(t_load, 1))
    for batch in args.batches:
        enc = tok([PROMPT] * batch, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            teacher.generate(**enc, max_new_tokens=8, do_sample=False)  # warm-up
            sync(dev)
            t0 = time.time()
            out = teacher.generate(**enc, max_new_tokens=args.new_tokens,
                                   min_new_tokens=args.new_tokens, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            sync(dev)
        dt = time.time() - t0
        new = (out.shape[1] - enc["input_ids"].shape[1]) * batch
        result(mode="generate", batch=batch, new_tokens=new, seconds=round(dt, 1),
               tok_s=round(new / dt, 1), tok_s_per_sequence=round(new / dt / batch, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("step", "generate"))
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--student", default=STUDENT)
    ap.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"],
                    help="teacher's and student's device (generate uses the first)")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--new-tokens", type=int, default=256)
    args = ap.parse_args()
    print(f"torch {torch.__version__}, {torch.cuda.device_count()} GPUs visible "
          f"({os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)
    if args.mode == "step":
        step_mode(args)
    else:
        generate_mode(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
