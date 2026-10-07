#!/usr/bin/env python3
"""On-policy distillation with a privileged teacher, as a LoRA adapter.

    python forge/scripts/opd_train.py \\
        --student $WORK/eullm_runs/grpo/v04/merged \\
        --teacher $HF_HOME/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/<rev> \\
        --prompts $WORK/eullm_runs/opd/prompts.jsonl --out $WORK/eullm_runs/opd/v04 \\
        --student-device cuda:0 --teacher-devices 1,2

Step 4 of the case-law plan (research report of 2026-10-05). Each prompts
row holds the student's messages and the teacher's (make_opd_prompts.py):
the same question and retrieved passages, the teacher's with the source
ruling in front. Every step the student samples an answer to a batch of its
prompts; the teacher reads each answer behind its own prompt; the LoRA
moves the student's next-token distributions towards the teacher's at the
answer positions (`eullm_forge.opd.reverse_kl`).

For Ministral 8B the teacher is Ministral-3-14B, which has the same
vocabulary (checked on 2026-10-06). The student's own weights can teach it
too (self-distillation), but the pilot of 2026-10-06 showed why not to: its
KL stayed at 0.04 for 60 steps, with nothing to learn.

``--teacher-note`` goes into the teacher's prompt only. Without one, the
4B pilot of 2026-10-06 learnt to write 2.3 times longer: the teacher, with
the whole ruling in front of it, always has more to say, and the student's
answers grew from 174 tokens to the 384-token limit. A note asking the
teacher to answer briefly keeps what it knows and drops the length.

One process: the student on ``--student-device``, the teacher spread over
``--teacher-devices`` (bf16 Qwen3-30B-A3B is 61 GB: two 64 GB A100s). The
adapter, the optimizer and the step are saved every ``--save-every`` steps
and on ``--stop-after``, and a run with the same --out carries on, so the
work fits two-hour links. The final adapter lands in ``--out/adapter``,
merged like any other (merge_identity_adapter).

It prints step, loss, answer length and seconds per step; never a prompt
or an answer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.identity import load_text_model, lora_target_modules  # noqa: E402
from eullm_forge.opd import reverse_kl, same_vocabulary  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise SystemExit(f"[opd] no prompts in {path}")
    for r in rows[:1]:
        if not (isinstance(r.get("student"), list) and isinstance(r.get("teacher"), list)):
            raise SystemExit("[opd] rows need 'student' and 'teacher' message lists "
                             "(make_opd_prompts.py)")
    return rows


def template(tok, messages: list[dict]) -> list[int]:
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                  enable_thinking=False)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)


def with_note(messages: list[dict], note: str) -> list[dict]:
    """The messages with ``note`` after the last user turn's text (the teacher's only)."""
    if not note:
        return messages
    out = [dict(m) for m in messages]
    last = max(i for i, m in enumerate(out) if m["role"] == "user")
    out[last]["content"] = f"{out[last]['content']}\n\n{note}"
    return out


def answer_logits(model, prompt_ids: list[int], answer: list[int], device):
    """Logits that predict each answer token, [len(answer), V]."""
    import torch

    ids = torch.tensor([prompt_ids + answer], device=device)
    n = len(answer)
    try:
        out = model(input_ids=ids, logits_to_keep=n + 1)
        logits = out.logits[0, :n]
    except TypeError:                       # a model without logits_to_keep
        out = model(input_ids=ids)
        logits = out.logits[0, len(prompt_ids) - 1:len(prompt_ids) - 1 + n]
    return logits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--student-device", default="cuda:0")
    ap.add_argument("--teacher-devices", default="1,2",
                    help="GPU indices for the teacher, or 'cpu'")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8, help="prompts per step")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--stop-after", type=float, default=0, help="seconds, then save and exit")
    ap.add_argument("--teacher-note", default="",
                    help="text added to the teacher's prompt only, e.g. how long to answer: "
                         "the teacher holds the source and, unprompted, never stops writing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    done = args.out / "adapter" / "adapter_config.json"
    # Size, not existence, as in grpo_train.py and stage3_sft.py: a 0-byte
    # adapter_config.json -- an interrupted save, not a finished adapter --
    # used to count as done and exit 0, skipping the whole run. The run's own
    # log then said "nothing left to do", which is what a finished adapter
    # says too, so a killed link looked exactly like a completed one.
    if done.is_file() and done.stat().st_size > 0:
        print(f"[opd] adapter already at {done.parent}: nothing left to do", flush=True)
        return 0
    rows = load_rows(args.prompts)
    t0 = time.monotonic()

    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.student)
    ttok = AutoTokenizer.from_pretrained(args.teacher)
    if not same_vocabulary(tok, ttok):
        raise SystemExit("[opd] student and teacher tokenizers differ: the loss compares their "
                         "distributions token by token. Use the student itself as teacher.")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    end_ids = {i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
               if isinstance(i, int) and i >= 0}

    cuda = torch.cuda.is_available() and args.student_device != "cpu"
    dtype = torch.bfloat16 if cuda else torch.float32
    sdev = torch.device(args.student_device if cuda else "cpu")
    student = load_text_model(args.student, dtype=dtype).to(sdev)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()
    student = get_peft_model(student, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
        target_modules=lora_target_modules(student), task_type="CAUSAL_LM"))
    if args.teacher_devices == "cpu" or not cuda:
        teacher = load_text_model(args.teacher, dtype=dtype)
        tdev = torch.device("cpu")
    else:
        gpus = [int(g) for g in args.teacher_devices.split(",")]
        mem = {g: "60GiB" for g in gpus}
        teacher = load_text_model(args.teacher, dtype=dtype, device_map="auto", max_memory=mem)
        tdev = torch.device(f"cuda:{gpus[0]}")
    teacher.eval().requires_grad_(False)

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    ckpt = args.out / "ckpt"
    step = 0
    if (ckpt / "state.pt").is_file():
        state = torch.load(ckpt / "state.pt", map_location="cpu", weights_only=False)
        set_peft_model_state_dict(student, state["adapter"])
        opt.load_state_dict(state["optimizer"])
        step = state["step"]
        print(f"[opd] resuming at step {step}", flush=True)
    print(f"[opd] {len(rows):,} prompts, {args.batch} a step, steps {step}->{args.steps}; "
          f"loaded in {time.monotonic() - t0:.0f} s", flush=True)

    def save(final: bool = False) -> None:
        from peft import get_peft_model_state_dict
        args.out.mkdir(parents=True, exist_ok=True)
        if final:
            student.save_pretrained(str(args.out / "adapter"))
            tok.save_pretrained(str(args.out / "adapter"))
            return
        ckpt.mkdir(parents=True, exist_ok=True)
        tmp = ckpt / "state.pt.partial"
        torch.save({"adapter": get_peft_model_state_dict(student),
                    "optimizer": opt.state_dict(), "step": step}, tmp)
        tmp.replace(ckpt / "state.pt")

    tok.padding_side = "left"
    while step < args.steps:
        if args.stop_after and time.monotonic() - t0 > args.stop_after:
            save()
            print(f"[opd] time is up at step {step}: saved, the next link carries on", flush=True)
            return 0
        ts = time.monotonic()
        rng = random.Random(args.seed * 1_000_003 + step)    # the same batch on a resumed step
        batch = rng.sample(rows, min(args.batch, len(rows)))
        s_prompts = [template(tok, r["student"]) for r in batch]
        t_prompts = [template(ttok, with_note(r["teacher"], args.teacher_note)) for r in batch]

        student.eval()
        width = max(len(p) for p in s_prompts)
        ids = torch.tensor([[tok.pad_token_id] * (width - len(p)) + p for p in s_prompts],
                           device=sdev)
        attn = (torch.arange(width, device=sdev)[None, :]
                >= torch.tensor([width - len(p) for p in s_prompts], device=sdev)[:, None]).long()
        with torch.no_grad():
            gen = student.generate(input_ids=ids, attention_mask=attn, do_sample=True,
                                   temperature=args.temperature, top_p=1.0,
                                   max_new_tokens=args.max_new_tokens,
                                   pad_token_id=tok.pad_token_id, eos_token_id=list(end_ids))
        answers = []
        for row in gen[:, width:].tolist():
            cut = next((j for j, t in enumerate(row) if t in end_ids), None)
            answers.append(row[:cut + 1] if cut is not None else row)

        student.train()
        total, n = 0.0, 0
        for sp, tp, ans in zip(s_prompts, t_prompts, answers):
            if not ans:
                continue
            with torch.no_grad():
                t_logits = answer_logits(teacher, tp, ans, tdev).to(sdev)
            s_logits = answer_logits(student, sp, ans, sdev)
            loss = reverse_kl(s_logits, t_logits)
            if not math.isfinite(loss.item()):
                raise SystemExit(f"[opd] non-finite loss at step {step}: stopping, nothing saved")
            (loss / len(batch)).backward()
            total += loss.item()
            n += 1
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        mean_len = sum(len(a) for a in answers) / max(1, len(answers))
        print(f"[opd] step {step}/{args.steps} kl {total / max(1, n):.4f} len {mean_len:.0f} "
              f"{time.monotonic() - ts:.0f}s/step", flush=True)
        if step % args.save_every == 0:
            save()
    save()
    save(final=True)
    print(f"[opd] adapter {args.out / 'adapter'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
