#!/usr/bin/env python3
"""GRPO with verifiable rewards on an open-book model, as a LoRA adapter.

    accelerate launch --num_processes 4 forge/scripts/grpo_train.py \\
        --model  $WORK/eullm_runs/stage3/sft-v03-instruct-ob/merged \\
        --prompts $WORK/eullm_runs/grpo/prompts.jsonl \\
        --out    $WORK/eullm_runs/grpo/v03

For each prompt the model writes ``--num-generations`` answers; each is
scored by `eullm_forge.rl.answer_reward` (1 right, 0 wrong), and the update
favours the answers that beat their group's mean. Starting from a model that
already answers in the open-book format (a merged stage-3 model), not from a
base model: RL sharpens what the model can already do sometimes, it does not
teach a format from nothing.

The adapter lands in ``--out/adapter`` like stage 3's, so
leonardo/sbatch_stage3_package.slurm merges and packages it unchanged, with
``S3_BASE`` the model given here as ``--model``.

Resumable: a rerun with the same --out continues from its last checkpoint,
and one whose adapter is already there stops at once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.identity import load_text_model, lora_target_modules  # noqa: E402
from eullm_forge.rl import answer_reward  # noqa: E402


def load_prompts(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise SystemExit(f"[grpo] no prompts in {path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="merged open-book model to start from")
    ap.add_argument("--prompts", required=True, type=Path, help="make_grpo_prompts.py output")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rank", type=int, default=32)
    # RL moves a model that already works; a stage-3 learning rate would
    # undo the format it learnt there in a few dozen steps.
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=8,
                    help="completions per GPU per step; "
                         "prompts per step = this x GPUs / generations")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-completion", type=int, default=384,
                    help="the exam allows 300 new tokens; a little room, not a license to ramble")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.02,
                    help="KL to the starting model: keeps it from drifting off what it could do")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N prompts (smoke tests)")
    args = ap.parse_args(argv)

    done = args.out / "adapter" / "adapter_config.json"
    if done.is_file() and done.stat().st_size > 0:
        print(f"[grpo] adapter already at {done.parent}: nothing left to do "
              "(move it away to train again)", flush=True)
        return 0

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    rows = load_prompts(args.prompts)
    if args.limit:
        rows = rows[:args.limit]
    dataset = Dataset.from_list(
        [{"prompt": r["prompt"], "tipo": r["tipo"], "keywords": r["keywords"]} for r in rows])

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_text_model(args.model, dtype=torch.bfloat16 if cuda else torch.float32)

    config = GRPOConfig(
        output_dir=str(args.out),
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        max_steps=args.max_steps,
        beta=args.beta,
        temperature=args.temperature,
        # The exam and the engine ask with thinking off (legal_eval.chat_prompt).
        chat_template_kwargs={"enable_thinking": False},
        bf16=cuda,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        use_cpu=not cuda,
    )
    peft_config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                             target_modules=lora_target_modules(model), task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=model, reward_funcs=[answer_reward], args=config,
                          train_dataset=dataset, processing_class=tok, peft_config=peft_config)

    last = get_last_checkpoint(str(args.out)) if args.out.is_dir() else None
    if last:
        print(f"[grpo] resuming from {last}", flush=True)
    print(f"[grpo] {len(dataset)} prompts, {args.num_generations} answers each, "
          f"up to {args.max_steps} steps", flush=True)
    trainer.train(resume_from_checkpoint=last)

    if trainer.accelerator.is_main_process:
        trainer.model.save_pretrained(str(args.out / "adapter"))
        tok.save_pretrained(str(args.out / "adapter"))
        print(f"[grpo] adapter {args.out / 'adapter'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
