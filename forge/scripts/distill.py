#!/usr/bin/env python3
"""Phase 2 — Knowledge distillation Qwen3-32B-legal-it → Qwen3-7B-Base.

A frozen teacher (the Phase-1 LoRA-tuned 32B) generates a probability
distribution over each next token; the student (7B) is trained to
imitate that distribution via KL divergence (soft labels), plus a
fraction of the standard cross-entropy on the ground-truth tokens
(hard labels).

Loss = α · KL(student || teacher) · T² + (1-α) · CE(student, y)

Training is single-process. Two hardware layouts are supported:

* ``teacher_device_map: single`` (default) — teacher and student share
  one ~96 GB device (H100 NVL, RTX PRO 6000 Blackwell):

      Teacher 32B (frozen, no grad):     ~64 GB
      Student 7B + grad + 8-bit Adam:    ~28 GB
      Activations (seq 2048, both nets): ~6 GB
      Headroom:                          ~variable

* ``teacher_device_map: auto`` — multi-GPU nodes where no single GPU
  fits the BF16 teacher (Leonardo Booster: 4x A100 64 GB). The frozen
  teacher is sharded across every visible GPU via accelerate, capped by
  a max_memory map that reserves the student's GPU (default cuda:0)
  for the student + optimizer + activations. The teacher forward is
  pipelined across GPUs; the student trains entirely on its own GPU.

If the 32B teacher does not fit, pass --teacher-load-in-8bit (bitsandbytes
NF4/INT8) to drop the teacher to ~16 GB at the cost of slightly noisier
logits.

The script is checkpoint-resumable: pass --resume-from <dir> or just
re-run with the same --output-dir and the latest checkpoint inside it
will be picked up automatically.

Usage:
    python forge/scripts/distill.py \\
        --config forge/training/configs/distill_qwen3_32b_to_7b.yaml

Or with explicit args (overrides YAML if both are present):
    python forge/scripts/distill.py \\
        --teacher-model Qwen/Qwen3-32B-Base \\
        --teacher-adapter ~/checkpoints/qwen3_32b_legal_it_continued_pt \\
        --student-model Qwen/Qwen3-7B-Base \\
        --dataset-dir ~/datasets/legal_it \\
        --output-dir ~/checkpoints/qwen3_7b_legal_it_distilled \\
        ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)

# The script runs from a repo checkout, not necessarily with eullm_forge
# pip-installed — make the package importable from its source tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eullm_forge.distill import (  # noqa: E402
    build_teacher_max_memory,
    build_teacher_split_memory,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DistillConfig:
    # Models
    teacher_model: str = "Qwen/Qwen3-32B-Base"
    teacher_adapter: Optional[str] = None  # Phase-1 LoRA adapter dir
    student_model: str = "Qwen/Qwen3-7B-Base"
    teacher_load_in_8bit: bool = False     # bitsandbytes 8-bit teacher
    teacher_load_in_4bit: bool = False     # bitsandbytes 4-bit teacher (NF4)

    # Teacher placement:
    #   "single" — teacher lives on student_device (one 96 GB-class GPU).
    #   "auto"   — teacher sharded across all visible GPUs (accelerate
    #              device_map) with a max_memory cap that keeps the
    #              student's GPU free. Required on nodes where no single
    #              GPU fits the BF16 teacher (Leonardo: 4x A100 64 GB).
    teacher_device_map: str = "single"
    # design B only: the GPUs the teacher owns outright. The student takes
    # whatever is left, and `student_device` must not be one of these.
    teacher_gpus: tuple = (0, 1)
    teacher_gib_per_gpu: int = 58          # teacher budget on non-student GPUs
    teacher_gib_on_student_gpu: int = 8    # teacher budget on the student GPU
    student_device: str = "cuda:0"

    # Student fine-tuning method:
    #   "lora" (default) — wrap student in PEFT LoRA, only LoRA params
    #                     trainable. Keeps total VRAM ≤ 95 GB on a 94-96 GB
    #                     GPU even with a 32B BF16 teacher in the same
    #                     process. Recommended for v0.1 PoC.
    #   "full" — train all 7B params. Needs ≥ 130 GB total VRAM (so
    #            multi-GPU FSDP, or H200 141 GB + teacher quantised).
    student_finetune: str = "lora"
    student_lora_rank: int = 128
    student_lora_alpha: int = 256
    student_lora_dropout: float = 0.0
    # Default LoRA targets cover attention + MLP — empirical sweet spot
    # for Qwen-family domain adaptation.
    student_lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Data
    dataset_dir: str = "~/datasets/legal_it"
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"
    cutoff_len: int = 2048
    max_train_samples: Optional[int] = None  # None = all

    # Training
    output_dir: str = "~/checkpoints/qwen3_7b_legal_it_distilled"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: int = 1
    max_steps: int = -1                    # -1 = unlimited (use epochs)
    warmup_steps: int = 1000
    save_steps: int = 1000
    # Checkpoints kept on disk, newest first; <= 0 keeps every one. Needed
    # because `save_steps` is the knob that decides how much work a walltime
    # kill throws away, and lowering it without bounding the directory turns
    # a 24 h chain into tens of gigabytes of optimizer state on $WORK.
    save_total_limit: int = 3
    # Consecutive optimizer updates that may be skipped for non-finite loss
    # or gradients before the run is declared diverged. See `train`.
    max_skipped_updates: int = 25
    eval_steps: int = 1000
    # Validation batches scored per eval. Capped on purpose: see `evaluate`.
    eval_max_batches: int = 200
    logging_steps: int = 20

    # Distillation
    kl_temperature: float = 2.0
    kl_alpha: float = 0.7                  # weight of soft (KL) loss
    # ce_alpha is implicitly (1 - kl_alpha)

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42

    # Resume
    resume_from: Optional[str] = None  # path to a checkpoint dir; auto if None


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _parse_args() -> DistillConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config path (CLI args override YAML).")
    # Allow every DistillConfig field as a CLI arg.
    for f in DistillConfig.__dataclass_fields__.values():
        flag = "--" + f.name.replace("_", "-")
        kw = {"default": None}
        if f.type is bool:
            kw["action"] = argparse.BooleanOptionalAction
        elif f.type is int:
            kw["type"] = int
        elif f.type is float:
            kw["type"] = float
        else:
            kw["type"] = str
        parser.add_argument(flag, **kw)
    raw = parser.parse_args()

    cfg_dict: dict = {}
    if raw.config:
        with open(raw.config) as f:
            cfg_dict = yaml.safe_load(f) or {}
    for f in DistillConfig.__dataclass_fields__.values():
        v = getattr(raw, f.name)
        if v is not None:
            cfg_dict[f.name] = v

    cfg = DistillConfig(**cfg_dict)
    cfg.dataset_dir = os.path.expanduser(cfg.dataset_dir)
    cfg.output_dir = os.path.expanduser(cfg.output_dir)
    if cfg.teacher_adapter:
        cfg.teacher_adapter = os.path.expanduser(cfg.teacher_adapter)
    if cfg.resume_from:
        cfg.resume_from = os.path.expanduser(cfg.resume_from)
    return cfg


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _quantization_config(load_in_8bit: bool, load_in_4bit: bool):
    if not (load_in_8bit or load_in_4bit):
        return None
    from transformers import BitsAndBytesConfig
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_teacher(cfg: DistillConfig, dtype: torch.dtype, device: str):
    print(f"[teacher] loading {cfg.teacher_model} "
          f"(8bit={cfg.teacher_load_in_8bit}, 4bit={cfg.teacher_load_in_4bit}, "
          f"device_map={cfg.teacher_device_map})",
          file=sys.stderr)
    quant = _quantization_config(cfg.teacher_load_in_8bit,
                                 cfg.teacher_load_in_4bit)
    device_map = {"": device} if quant is None else "auto"
    max_memory = None
    if cfg.teacher_device_map == "split":
        # ADR-001 design B. The teacher owns `teacher_gpus` outright and the
        # student owns the rest; they never share a card, which is the whole
        # reason this mode exists.
        n_gpus = torch.cuda.device_count()
        student_idx = torch.device(cfg.student_device).index or 0
        # YAML gives a list, the generated CLI flag gives a string like
        # "0,1". Normalise before anything compares against it, or
        # `student_idx in "0,1"` raises a TypeError after the teacher has
        # already been named in the log and the operator thinks it loaded.
        teacher_gpus = cfg.teacher_gpus
        if isinstance(teacher_gpus, str):
            teacher_gpus = [int(x) for x in teacher_gpus.replace(",", " ").split()]
        teacher_gpus = [int(x) for x in teacher_gpus]
        if student_idx in teacher_gpus:
            # Caught here rather than by an OOM ten minutes into the first
            # forward: a split that puts the student on a teacher card is not
            # a split, and it would fail the way v1.0 failed.
            raise ValueError(
                f"student_device {cfg.student_device} is inside teacher_gpus "
                f"{teacher_gpus} — that is co-hosting, not a split"
            )
        device_map = "auto"
        max_memory = build_teacher_split_memory(
            n_gpus,
            teacher_gpus=teacher_gpus,
            teacher_gib_per_gpu=cfg.teacher_gib_per_gpu,
        )
        print(f"[teacher] split node: teacher on GPUs "
              f"{teacher_gpus}, student on {cfg.student_device}, "
              f"max_memory={max_memory}", file=sys.stderr)
        if quant is not None:
            print("[teacher] WARNING: quantization is on in split mode — the "
                  "point of the split is to afford BF16. Set "
                  "teacher_load_in_8bit: false unless measuring the "
                  "difference on purpose.", file=sys.stderr)
    elif cfg.teacher_device_map == "auto":
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            student_idx = torch.device(cfg.student_device).index or 0
            device_map = "auto"
            max_memory = build_teacher_max_memory(
                n_gpus,
                student_gpu_index=student_idx,
                teacher_gib_per_gpu=cfg.teacher_gib_per_gpu,
                teacher_gib_on_student_gpu=cfg.teacher_gib_on_student_gpu,
            )
            print(f"[teacher] sharding across {n_gpus} GPUs, "
                  f"max_memory={max_memory}", file=sys.stderr)
        else:
            print("[teacher] teacher_device_map=auto but only 1 GPU visible "
                  "— falling back to single-device placement", file=sys.stderr)
    elif cfg.teacher_device_map != "single":
        raise ValueError(
            f"teacher_device_map must be 'single', 'auto' or 'split', "
            f"got {cfg.teacher_device_map!r}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=quant,
    )
    if cfg.teacher_adapter:
        print(f"[teacher] applying adapter {cfg.teacher_adapter}",
              file=sys.stderr)
        model = PeftModel.from_pretrained(model, cfg.teacher_adapter)
        model = model.merge_and_unload()    # collapse LoRA into base weights
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_student(cfg: DistillConfig, dtype: torch.dtype, device: str):
    print(f"[student] loading {cfg.student_model} "
          f"(finetune={cfg.student_finetune})", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.student_model,
        torch_dtype=dtype,
        device_map={"": device},
    )

    if cfg.student_finetune == "lora":
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=cfg.student_lora_rank,
            lora_alpha=cfg.student_lora_alpha,
            lora_dropout=cfg.student_lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(cfg.student_lora_target_modules),
        )
        model = get_peft_model(model, lora_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[student] LoRA r={cfg.student_lora_rank}: "
              f"trainable={trainable / 1e6:.1f}M / total={total / 1e9:.2f}B "
              f"({100 * trainable / total:.3f}%)",
              file=sys.stderr)
    elif cfg.student_finetune == "full":
        # All params trainable (default behaviour). VRAM-heavy: needs a
        # 141 GB+ GPU or multi-GPU FSDP.
        for p in model.parameters():
            p.requires_grad_(True)
        print("[student] full fine-tune: ALL params trainable",
              file=sys.stderr)
    else:
        raise ValueError(
            f"student_finetune must be 'lora' or 'full', "
            f"got {cfg.student_finetune!r}"
        )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    return model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    kl_alpha: float,
    kl_temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the distillation loss.

    Args:
        student_logits: (B, T, V) student logits.
        teacher_logits: (B, T, V) teacher logits, detached.
        labels: (B, T) ground-truth next-token ids; -100 = ignore.
        kl_alpha: weight of the soft (KL) term, complement is CE weight.
        kl_temperature: softmax temperature for both teacher and student
            logits before the KL.

    Returns:
        ``(total_loss, stats_dict)``.
    """
    # Shift for next-token: predict token t given tokens [0..t-1].
    s_logits = student_logits[..., :-1, :].contiguous()
    t_logits = teacher_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Mask padding tokens (label == -100) out of both losses.
    valid_mask = (shift_labels != -100)

    # --- Soft (KL) ---
    T = kl_temperature
    s_log_probs = F.log_softmax(s_logits / T, dim=-1)
    t_probs = F.softmax(t_logits / T, dim=-1)
    kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(-1)  # (B, T-1)
    kl = (kl * valid_mask).sum() / valid_mask.sum().clamp_min(1)
    kl = kl * (T * T)   # scale per Hinton et al. 2015

    # --- Hard (CE) ---
    ce = F.cross_entropy(
        s_logits.view(-1, s_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

    total = kl_alpha * kl + (1.0 - kl_alpha) * ce
    return total, {"loss": total.item(), "kl": kl.item(), "ce": ce.item()}


@torch.no_grad()
def evaluate(student, teacher, val_loader, cfg: DistillConfig, device,
             max_batches: int) -> dict:
    """Distillation loss on held-out data, over a fixed prefix of the val set.

    `eval_steps` was a declared config field that nothing read: the v1.1
    Phase 2 run produced 18,000 steps with no validation number at all, so
    when its training loss flattened there was nothing to say whether the
    student had stopped learning or the batches had simply got harder.

    Capped at `max_batches`, and that is the whole design. The validation
    split is 11,387 chunks and one teacher+student forward pair costs about
    0.7 s on a Booster node, so scoring all of it takes over two hours —
    against an `eval_steps` interval that is three hours of training at the
    measured rate, that is a 40 % tax on the allocation to sharpen a number
    a few hundred batches already pin down. `build_dataloaders` builds the
    validation loader with `shuffle=False`, so the prefix is the same prefix
    every time and successive evals compare like with like.

    Returns the mean of the same three components the training log prints,
    or an empty dict if the loader yielded nothing.
    """
    was_training = student.training
    student.eval()
    acc = {"loss": 0.0, "kl": 0.0, "ce": 0.0}
    seen = 0
    try:
        for batch in val_loader:
            if seen >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True)
                     for k, v in batch.items()}
            t_logits = teacher(**batch).logits.detach().to(device)
            s_logits = student(**batch).logits
            _, parts = distill_loss(
                s_logits, t_logits, batch["labels"],
                kl_alpha=cfg.kl_alpha, kl_temperature=cfg.kl_temperature,
            )
            for key in acc:
                acc[key] += parts[key]
            seen += 1
    finally:
        # Restored in `finally`: leaving the student in eval mode after an
        # interrupted eval would silently disable dropout for the rest of
        # the run, and nothing downstream would report it.
        if was_training:
            student.train()
    if seen == 0:
        return {}
    return {key: value / seen for key, value in acc.items()}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_dataloaders(cfg: DistillConfig, tokenizer):
    data_files = {
        "train": str(Path(cfg.dataset_dir) / cfg.train_file),
        "validation": str(Path(cfg.dataset_dir) / cfg.val_file),
    }
    raw = load_dataset("json", data_files=data_files)
    if cfg.max_train_samples:
        raw["train"] = raw["train"].select(range(cfg.max_train_samples))

    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.cutoff_len,
            padding=False,
            return_attention_mask=True,
        )

    cols = raw["train"].column_names
    tokenized = raw.map(
        _tok,
        batched=True,
        remove_columns=cols,
        num_proc=4,
        desc="tokenising",
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_loader = DataLoader(
        tokenized["train"], batch_size=cfg.per_device_train_batch_size,
        shuffle=True, collate_fn=collator, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        tokenized["validation"], batch_size=cfg.per_device_train_batch_size,
        shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _checkpoint_step(path: Path) -> int:
    """Step number encoded in a `checkpoint-N` directory name, -1 if absent.

    Sorting these by name is wrong and quietly so: `checkpoint-9000` sorts
    after `checkpoint-18000` lexicographically, so a resume would reload a
    checkpoint half the run old and redo nine thousand steps.
    """
    tail = path.name.split("-")[-1]
    return int(tail) if tail.isdigit() else -1


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.is_dir():
        return None
    candidates = sorted(output_dir.glob("checkpoint-*"), key=_checkpoint_step)
    return candidates[-1] if candidates else None


def prune_checkpoints(output_dir: Path, keep: int) -> list:
    """Delete all but the newest `keep` checkpoints. Returns what it removed.

    A checkpoint here is the LoRA adapter plus AdamW's two moments over the
    trainable parameters — hundreds of megabytes each. Saving often enough
    that a walltime kill costs under an hour means saving several times as
    often, and without this that multiplies straight into $WORK.

    Keeps more than one deliberately: the newest checkpoint is the one a
    kill can catch mid-write, and the run's only way back is the one before
    it. Directories whose name carries no step number are never touched.
    """
    if keep <= 0:
        return []
    numbered = [p for p in output_dir.glob("checkpoint-*")
                if p.is_dir() and _checkpoint_step(p) >= 0]
    doomed = sorted(numbered, key=_checkpoint_step)[:-keep]
    for path in doomed:
        shutil.rmtree(path, ignore_errors=True)
    return doomed


def save_checkpoint(
    student, optimizer, scheduler, scaler, step: int, output_dir: Path,
    cfg: DistillConfig,
) -> Path:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] saving to {ckpt_dir}", file=sys.stderr)
    student.save_pretrained(ckpt_dir, safe_serialization=True)
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": cfg.__dict__,
    }
    torch.save(state, ckpt_dir / "training_state.pt")
    return ckpt_dir


def load_checkpoint(ckpt_dir: Path, optimizer, scheduler, scaler) -> int:
    print(f"[resume] loading state from {ckpt_dir}", file=sys.stderr)
    state = torch.load(ckpt_dir / "training_state.pt", map_location="cpu",
                       weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return state["step"]


# ---------------------------------------------------------------------------
# Where in the data a resumed run continues
# ---------------------------------------------------------------------------
# A resume restored the weights, the optimizer, the scheduler and the step —
# and then started the data from the top. The loader was a plain
# `shuffle=True` DataLoader drawn from a global RNG that `train()` seeds with
# the same value on every start, so every job walked the SAME order from its
# first batch. A chain of links therefore trained over and over on the first
# stretch of the epoch and never reached the rest.
#
# With 24-hour links that was ~9,000 steps of data per link, seen again by
# every later link; with the 2-hour links of the grazing chains it was ~650
# steps, seen a dozen times. The split arm measured it: held-out perplexity
# on Consiglio di Stato went 5.11 at step 12,600 -> 5.29 at 18,200 -> 5.92 at
# 22,000, a student over-fitting a sliver of its corpus. The flattening of
# the control arm's transfer curve past its first link is the same effect,
# not diminishing returns.
#
# Now each epoch has a fixed order derived from (seed, epoch), independent of
# any other use of the RNG, and a resumed run skips the batches its step
# count says it has already trained on. The optimizer steps are the ledger:
# `start_step x gradient_accumulation_steps` micro-batches have been consumed.

def data_position(start_step: int, grad_accum: int,
                  batches_per_epoch: int) -> tuple[int, int]:
    """(epoch, batches into that epoch) where a run resumed at `start_step` goes on."""
    if batches_per_epoch <= 0:
        return 0, 0
    return divmod(start_step * grad_accum, batches_per_epoch)


def epoch_batches(train_loader, epoch: int, skip: int, seed: int):
    """One epoch of batches in a fixed order, from the `skip`-th batch on.

    A DataLoader is re-issued over a seeded permutation of its dataset, the
    skipped part cut off before anything is loaded. Anything else is taken
    to be batches already (the tests) and is sliced in the order given.
    """
    if not isinstance(train_loader, DataLoader):
        return list(train_loader)[skip:]
    dataset = train_loader.dataset
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed + epoch),
    ).tolist()
    return DataLoader(
        Subset(dataset, order[skip * train_loader.batch_size:]),
        batch_size=train_loader.batch_size, shuffle=False,
        collate_fn=train_loader.collate_fn,
        num_workers=train_loader.num_workers, pin_memory=train_loader.pin_memory,
    )


# ---------------------------------------------------------------------------
# Divergence guard
# ---------------------------------------------------------------------------
#
# Why this exists, from the run it would have saved. On 2026-09-24 the control
# arm's loss went from 0.6255 at step 36,600 to `nan` twenty steps later — no
# climb, no spike, a single event just after a link resumed. Nothing stopped
# it. The loop kept training a dead model for two more links and, worse, kept
# SAVING it: four NaN checkpoints pushed the healthy ones out through
# `save_total_limit: 3`, and when the NaN was found there was no checkpoint
# left to resume from. A 36,600-step run was lost to what should have cost
# one skipped update.
#
# Three separate protections, each for a different way of failing:
#
#   1. A non-finite update is SKIPPED, not applied. A NaN loss or gradient
#      that reaches `optimizer.step()` writes NaN into the weights and every
#      step after it is NaN; skipping the window leaves the weights as they
#      were. One outlier batch — which is what an instant NaN with no warning
#      climb looks like — then costs one update instead of the run.
#   2. The weights are checked before every save, and a non-finite model is
#      never written. This is the protection that keeps the healthy
#      checkpoints alive whatever the cause, because rotation only deletes old
#      checkpoints when a new one is saved.
#   3. A run that keeps producing non-finite updates is genuinely diverged, so
#      after `max_skipped_updates` in a row it writes a DIVERGED sentinel and
#      exits. Every later link of the afterany chain then refuses to start
#      in seconds, instead of loading a 61 GB teacher to train nothing.
#
# For a healthy run none of this changes anything: no finite update is ever
# skipped, so trajectories stay comparable with runs made before it existed.

DIVERGED_SENTINEL = "DIVERGED"


def all_finite(tensors) -> bool:
    """True when every tensor holds only finite values."""
    for t in tensors:
        if not torch.isfinite(t).all():
            return False
    return True


def refuse_if_diverged(output_dir: Path) -> None:
    """Stop before loading anything if an earlier link declared divergence.

    Removing the sentinel is a human decision — after deleting the bad
    checkpoints, or deciding to stop the arm — so it is never cleared here.
    """
    sentinel = Path(output_dir) / DIVERGED_SENTINEL
    if sentinel.exists():
        print(f"[guard] {sentinel} exists — this run was declared diverged:",
              file=sys.stderr)
        print(sentinel.read_text(encoding="utf-8").rstrip(), file=sys.stderr)
        print("[guard] refusing to start. Remove the non-finite checkpoints and "
              "the sentinel to resume, or leave both to keep this arm stopped.",
              file=sys.stderr)
        raise SystemExit(3)


def declare_divergence(output_dir: Path, step: int, reason: str) -> None:
    """Write the sentinel and stop, WITHOUT saving a checkpoint."""
    sentinel = Path(output_dir) / DIVERGED_SENTINEL
    record = {
        "step": step,
        "reason": reason,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "job": os.environ.get("SLURM_JOB_ID"),
    }
    sentinel.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[guard] DIVERGED at step {step}: {reason}", file=sys.stderr)
    print(f"[guard] wrote {sentinel}; no checkpoint saved, so the healthy "
          f"ones on disk are untouched.", file=sys.stderr)
    raise SystemExit(3)


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def _reload_student_from_checkpoint(
    ckpt_dir: Path, cfg: DistillConfig, dtype: torch.dtype, device: str,
):
    """Rebuild the student from a checkpoint directory.

    A LoRA checkpoint contains only the adapter (adapter_config.json +
    adapter weights), so the base student must be re-instantiated first
    and the adapter attached trainable on top. A full-FT checkpoint is a
    complete HF model directory and loads directly.
    """
    if (ckpt_dir / "adapter_config.json").is_file():
        base = AutoModelForCausalLM.from_pretrained(
            cfg.student_model, torch_dtype=dtype, device_map={"": device},
        )
        model = PeftModel.from_pretrained(base, str(ckpt_dir), is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir), torch_dtype=dtype,
        ).to(device)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    return model


def train(cfg: DistillConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = cfg.student_device
    # Require what the config asks for, not CUDA unconditionally: a student
    # placed on CPU (the tests) has no use for a GPU, and faking one with a
    # patched `torch.cuda.is_available` misleads torch itself — the optimizer
    # then probes a driver that is not there.
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"student_device={device!r} but CUDA is not available.")
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Before anything heavy: a link of a diverged chain should cost seconds,
    # not the teacher load.
    refuse_if_diverged(output_dir)

    # --- tokenizer (shared between teacher and student) ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.student_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- data ---
    train_loader, val_loader = build_dataloaders(cfg, tokenizer)
    total_micro_steps = (
        len(train_loader) * cfg.num_train_epochs
        if cfg.max_steps < 0
        else cfg.max_steps * cfg.gradient_accumulation_steps
    )
    total_optim_steps = total_micro_steps // cfg.gradient_accumulation_steps

    # --- models ---
    teacher = load_teacher(cfg, dtype, device)
    student = load_student(cfg, dtype, device)

    # --- resume: replace the student BEFORE building the optimizer, so
    # the optimizer binds to the parameters that will actually train
    # (binding it to a model that is then swapped out silently trains
    # nothing) ---
    resume_dir = (
        Path(cfg.resume_from) if cfg.resume_from
        else latest_checkpoint(output_dir)
    )
    if resume_dir:
        student = _reload_student_from_checkpoint(resume_dir, cfg, dtype, device)

    optimizer = torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_optim_steps,
    )
    scaler = None  # bf16 needs no scaler

    start_step = 0
    if resume_dir:
        start_step = load_checkpoint(resume_dir, optimizer, scheduler, scaler)
        print(f"[resume] continuing from step {start_step}", file=sys.stderr)
        # A checkpoint written non-finite — before this guard existed, or by
        # a link killed mid-write — would make every step NaN. Catching it
        # here costs one model load; not catching it costs the whole link.
        if not all_finite(p for p in student.parameters() if p.requires_grad):
            declare_divergence(
                output_dir, start_step,
                f"the checkpoint it resumed from ({resume_dir.name}) already "
                f"holds non-finite weights — delete it so the previous one is "
                f"used, if a healthy one survives",
            )

    start_epoch, skip_batches = data_position(
        start_step, cfg.gradient_accumulation_steps, len(train_loader),
    )
    if start_step:
        print(f"[resume] data: epoch {start_epoch}, skipping the "
              f"{skip_batches:,} batches of it already trained on",
              file=sys.stderr)

    # --- log ---
    print(f"[info] total optim steps: {total_optim_steps:,}",
          file=sys.stderr)
    print(f"[info] saves every {cfg.save_steps} steps to {output_dir}",
          file=sys.stderr)

    student.train()
    optimizer.zero_grad()
    micro = 0
    optim_step = start_step
    t0 = time.time()
    # Throughput was reported as `micro / (now - t0)` — an average over the
    # whole job, never reset. Thirteen hours in it printed the same 1.48 on
    # every line, which reads as reassuring stability and is really just a
    # large denominator: had the node halved in speed, the figure would have
    # taken hours to show it. The window pair below is the number that can
    # actually move; the cumulative one is kept beside it because it is the
    # one that predicts when the run ends.
    t_window = t0
    micro_window = 0
    log_loss_acc = 0.0
    log_kl_acc = 0.0
    log_ce_acc = 0.0
    log_n = 0
    # Divergence guard state; see the section above `train`.
    window_poisoned = False
    skipped_in_a_row = 0
    skipped_total = 0

    for epoch in range(start_epoch, cfg.num_train_epochs):
        skip = skip_batches if epoch == start_epoch else 0
        for batch in epoch_batches(train_loader, epoch, skip, cfg.seed):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.no_grad():
                t_out = teacher(**batch)
                # With a sharded teacher the logits land on the GPU of the
                # last pipeline stage — bring them to the student's device.
                t_logits = t_out.logits.detach().to(device)
            s_out = student(**batch)
            loss, parts = distill_loss(
                s_out.logits, t_logits, batch["labels"],
                kl_alpha=cfg.kl_alpha, kl_temperature=cfg.kl_temperature,
            )
            if torch.isfinite(loss):
                (loss / cfg.gradient_accumulation_steps).backward()
                log_loss_acc += parts["loss"]
                log_kl_acc += parts["kl"]
                log_ce_acc += parts["ce"]
                log_n += 1
            else:
                # No backward: a non-finite loss puts nan into every gradient
                # it reaches. The whole window is dropped at the boundary.
                window_poisoned = True
            micro += 1
            micro_window += 1
            if micro % cfg.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(), cfg.max_grad_norm,
                )
                # A finite loss can still overflow on the way back, and a
                # non-finite norm turns the clip coefficient into nan. Either
                # way, applying this update is what kills a run.
                if window_poisoned or not torch.isfinite(grad_norm):
                    optimizer.zero_grad()
                    skipped_in_a_row += 1
                    skipped_total += 1
                    why = ("non-finite loss" if window_poisoned
                           else f"grad norm {grad_norm.item()}")
                    print(f"[guard] skipped an update after step {optim_step} "
                          f"({why}) — {skipped_in_a_row} in a row, "
                          f"{skipped_total} in this job", file=sys.stderr)
                    window_poisoned = False
                    if skipped_in_a_row >= cfg.max_skipped_updates:
                        declare_divergence(
                            output_dir, optim_step,
                            f"{skipped_in_a_row} consecutive non-finite "
                            f"updates — not one bad batch but a run that "
                            f"no longer produces finite gradients",
                        )
                    continue
                skipped_in_a_row = 0
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                if optim_step % cfg.logging_steps == 0:
                    now = time.time()
                    rate_now = micro_window / max(now - t_window, 1e-9)
                    rate_avg = micro / max(now - t0, 1e-9)
                    avg_loss = log_loss_acc / log_n
                    avg_kl = log_kl_acc / log_n
                    avg_ce = log_ce_acc / log_n
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"[step {optim_step:>6} / {total_optim_steps}] "
                        f"loss={avg_loss:.4f}  kl={avg_kl:.4f}  ce={avg_ce:.4f}  "
                        f"lr={lr:.2e}  "
                        f"micro/s={rate_now:.2f} (avg {rate_avg:.2f})",
                        file=sys.stderr,
                    )
                    log_loss_acc = log_kl_acc = log_ce_acc = 0.0
                    log_n = 0
                    micro_window = 0
                    t_window = now

                if cfg.eval_steps > 0 and optim_step % cfg.eval_steps == 0:
                    val = evaluate(student, teacher, val_loader, cfg, device,
                                   cfg.eval_max_batches)
                    if val:
                        print(
                            f"[eval {optim_step:>6} / {total_optim_steps}] "
                            f"loss={val['loss']:.4f}  kl={val['kl']:.4f}  "
                            f"ce={val['ce']:.4f}",
                            file=sys.stderr,
                        )
                    # The eval's forward passes are not training: charging
                    # them to the window would make throughput look worse
                    # every time we measured it.
                    t_window = time.time()
                    micro_window = 0

                if optim_step % cfg.save_steps == 0:
                    # The protection that keeps healthy checkpoints alive
                    # whatever the cause: rotation deletes an old checkpoint
                    # only when a new one is written, so never writing a
                    # non-finite one is what leaves the good ones on disk.
                    if not all_finite(p for p in student.parameters()
                                      if p.requires_grad):
                        declare_divergence(
                            output_dir, optim_step,
                            "non-finite weights at save time — refusing to "
                            "write a checkpoint that would rotate a healthy "
                            "one out",
                        )
                    save_checkpoint(student, optimizer, scheduler, scaler,
                                    optim_step, output_dir, cfg)
                    prune_checkpoints(output_dir, cfg.save_total_limit)

                if optim_step >= total_optim_steps:
                    break

        if optim_step >= total_optim_steps:
            break

    # Final save — through the same gate as every other one.
    if not all_finite(p for p in student.parameters() if p.requires_grad):
        declare_divergence(output_dir, optim_step,
                           "non-finite weights at the final save")
    if skipped_total:
        print(f"[guard] {skipped_total} non-finite update(s) skipped in this "
              f"job", file=sys.stderr)
    save_checkpoint(student, optimizer, scheduler, scaler, optim_step,
                    output_dir, cfg)
    # Save tokenizer too — needed for inference / GGUF export later.
    tokenizer.save_pretrained(output_dir)

    # Phase 3 (quantize_to_gguf.sh) needs a full HF model directory, but a
    # LoRA student's checkpoints contain only the adapter — merge it into
    # the base weights and export the result alongside the checkpoints.
    if isinstance(student, PeftModel):
        merged_dir = output_dir / "merged"
        print(f"[merge] merging LoRA adapter into base → {merged_dir}",
              file=sys.stderr)
        merged = student.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        print(f"[merge] GGUF-exportable model at {merged_dir}",
              file=sys.stderr)

    print(f"[done] final student saved at {output_dir}", file=sys.stderr)


def main() -> int:
    cfg = _parse_args()
    print("[config]", json.dumps(cfg.__dict__, indent=2, default=str),
          file=sys.stderr)
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
