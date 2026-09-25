#!/usr/bin/env python3
"""Stage 3 on a distilled student: domain instruction pairs + identity, as LoRA.

Stage 2 leaves a continuation model — it writes Italian legal text, it does
not answer. This trains it on the pairs `generate_instructions.py` wrote from
the corpus, mixed with the identity examples, through the same code path the
pipeline uses (`eullm_forge.identity.fine_tune_identity`): prompt and padding
masked out of the loss, the chat template installed on the tokenizer and
carried into the adapter, so what the GGUF is prompted with at inference is
what it was trained on.

The adapter it writes is NOT the model. Merging it into the weights, the GGUF
and the measurement are CPU work and run on the serial partition afterwards
(leonardo/sbatch_stage3_package.slurm), not on this GPU.

    python forge/scripts/stage3_sft.py \\
        --model  $RUN/exports/merged/legal-it-4b-step12600 \\
        --pairs  $WORK/eullm_runs/stage3/instruct-pairs-v2.jsonl \\
        --out    $WORK/eullm_runs/stage3/sft-step12600

Resumable: a rerun with the same --out continues from its last checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.identity import IdentityConfig, fine_tune_identity  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="merged HF directory of the distilled student")
    ap.add_argument("--pairs", required=True, help="instruction pairs, JSONL")
    ap.add_argument("--out", required=True, help="checkpoints and adapter go here")
    ap.add_argument("--identity-name", default="EULLM Legal IT")
    ap.add_argument("--languages", default="it,en")
    # One pass over a few thousand pairs teaches the format; a second makes
    # it stick. More starts memorising synthetic answers word for word.
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    # Context tasks carry the passage (up to ~5,000 characters) in the prompt.
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--identity-repeat", type=int, default=20)
    ap.add_argument("--save-steps", type=int, default=100)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = IdentityConfig(
        model_path=args.model,
        identity_name=args.identity_name,
        languages=[x.strip() for x in args.languages.split(",") if x.strip()],
        lora_rank=args.rank,
        lora_alpha=2 * args.rank,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        max_length=args.max_length,
        instruction_path=args.pairs,
        identity_repeat=args.identity_repeat,
        output_dir=args.out,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        save_steps=args.save_steps,
    )
    adapter = fine_tune_identity(config)
    print(f"[stage3] adapter {adapter}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
