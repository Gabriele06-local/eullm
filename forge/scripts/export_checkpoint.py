#!/usr/bin/env python3
"""Merge a distillation checkpoint into its base model, ready for GGUF export.

**A verticalized model exists at every checkpoint, not only at the end of the
run.** `distill.py` trains a LoRA student and writes `checkpoint-N` every
`save_steps`; each of those is a complete adapter over Qwen3-4B-Base, and
merging it produces a model that loads, generates, and quantizes exactly like
the final one. It is less trained — that is the only difference.

That matters more than it sounds. `distill.py` merges the adapter only when
the whole run finishes, so the deliverable looked gated on an epoch that takes
weeks of chained 24 h jobs on a busy cluster. It is not. The pipeline can be
validated end to end, a GGUF can be put on real hardware, and evaluation can
start against a real baseline, on day three of a four-week run — and if the
allocation expires mid-run, what exists is still a model rather than a
directory of optimizer state.

Writes a full HF model directory. `forge/scripts/quantize_to_gguf.sh` takes it
from there.

Usage:
    python forge/scripts/export_checkpoint.py \\
        --checkpoint $EULLM_RUN_DIR/checkpoints/qwen3_4b_legal_it_distilled/checkpoint-8000 \\
        --output     $EULLM_RUN_DIR/exports/legal-it-4b-step8000

    # base model taken from the adapter's own metadata unless overridden:
    #   --base-model Qwen/Qwen3-4B-Base
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def read_adapter_base(checkpoint: Path) -> str | None:
    """The base model the adapter was trained on, per its own metadata.

    PEFT records it in `adapter_config.json`. Trusting that rather than a
    flag is what stops an adapter being merged into the wrong base — which
    does not error, it just produces a model whose weights are the sum of two
    unrelated things and whose output is plausible-looking noise.
    """
    cfg = checkpoint / "adapter_config.json"
    if not cfg.is_file():
        return None
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get(
            "base_model_name_or_path"
        )
    except (ValueError, OSError):
        return None


def checkpoint_step(checkpoint: Path) -> int | None:
    """Step number from a `checkpoint-N` directory name, None if absent."""
    tail = checkpoint.name.split("-")[-1]
    return int(tail) if tail.isdigit() else None


def describe(checkpoint: Path, base_model: str, output: Path) -> dict:
    """Provenance written beside the weights.

    A merged directory is indistinguishable from any other HF model once it
    leaves the cluster, so which run and which step produced it has to travel
    with it. Without this, two exports from the same run at different steps
    are two identical-looking directories.
    """
    return {
        "source_checkpoint": str(checkpoint),
        "step": checkpoint_step(checkpoint),
        "base_model": base_model,
        "export_dir": str(output),
        "note": (
            "Intermediate checkpoint merged from a distillation run in "
            "progress. Less trained than the run's final output; identical "
            "in every other respect."
        ),
    }


def resolve_base_model(explicit: str | None, checkpoint: Path) -> str:
    """Decide which base to merge into, and refuse to guess.

    Order: an explicit flag, then the adapter's own metadata. If neither is
    available there is no safe default — merging into "some Qwen" is the
    failure this function exists to prevent.
    """
    if explicit:
        recorded = read_adapter_base(checkpoint)
        if recorded and recorded != explicit:
            print(
                f"[warn] --base-model {explicit} differs from the adapter's "
                f"recorded base {recorded}. Continuing because you asked "
                f"explicitly, but a mismatch here produces a model that loads "
                f"and generates nonsense rather than one that fails.",
                file=sys.stderr,
            )
        return explicit
    recorded = read_adapter_base(checkpoint)
    if not recorded:
        raise SystemExit(
            f"[err] {checkpoint} records no base model and none was given. "
            f"Pass --base-model with the model this adapter was trained on."
        )
    return recorded


def check_output_dir(output: Path, force: bool) -> None:
    """Refuse to merge into a directory that already holds a model.

    Half-overwriting a model directory produces one that loads without
    complaint and is wrong in ways nothing downstream reports — the stale
    weight shards stay beside the new ones and `from_pretrained` picks
    whichever the index names.
    """
    if output.exists() and any(output.iterdir()) and not force:
        raise SystemExit(
            f"[err] {output} exists and is not empty. Pass --force to replace it."
        )


def publish_staging(staging: Path, output: Path) -> None:
    """Move a complete staged export into place, replacing the previous one.

    The old directory goes only once the new one is whole, so a mid-save
    crash can never leave a mix of two exports: the previous export stays
    until there is a whole new one to replace it.
    """
    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="checkpoint-N directory from a distillation run")
    p.add_argument("--output", required=True, type=Path,
                   help="directory to write the merged HF model to")
    p.add_argument("--base-model", default=None,
                   help="override the base recorded in adapter_config.json")
    p.add_argument("--dtype", default="bfloat16",
                   choices=("bfloat16", "float16", "float32"),
                   help="dtype to merge and save in (default: bfloat16, "
                        "what the student trains in)")
    p.add_argument("--force", action="store_true",
                   help="overwrite a non-empty output directory")
    p.add_argument("--keep-chat-template", action="store_true",
                   help="ship the base tokenizer's chat template (default: "
                        "drop it — see the note in main())")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    checkpoint: Path = args.checkpoint
    output: Path = args.output

    if not checkpoint.is_dir():
        raise SystemExit(f"[err] no such checkpoint directory: {checkpoint}")
    check_output_dir(output, args.force)

    base_model = resolve_base_model(args.base_model, checkpoint)
    step = checkpoint_step(checkpoint)

    # Everything is written beside the target and moved into place at the end.
    #
    # The failure this prevents has already happened: merging a 4 B model in
    # BF16 on a Leonardo login node was OOM-killed ("Ucciso") midway. That time
    # it died before any write, so the previous export survived intact — but a
    # kill a few seconds later would have left a directory holding a fresh
    # config.json, a partial shard and the old weight files, which
    # `from_pretrained` loads without a word of complaint. A model that is
    # quietly half of two exports is worse than no model.
    #
    # If this keeps being killed, the merge does not belong on a login node:
    # `forge/scripts/leonardo/sbatch_export_gguf.slurm` runs it on
    # lrd_all_serial with 30 GB, which is a request rather than a share of
    # whatever the login node has left.
    staging = output.parent / f"{output.name}.partial"
    print(f"[export] checkpoint {checkpoint}", file=sys.stderr)
    print(f"[export] step       {step if step is not None else 'unnumbered'}",
          file=sys.stderr)
    print(f"[export] base       {base_model}", file=sys.stderr)

    # Imported here, not at module scope, so --help and the argument checks
    # above work on a login node without the training stack installed.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)

    is_adapter = (checkpoint / "adapter_config.json").is_file()
    if is_adapter:
        print("[export] loading base and applying the adapter", file=sys.stderr)
        # CPU on purpose. Merging is elementwise arithmetic over the weights,
        # needs no GPU, and running it on a login node keeps a 24 h GPU
        # allocation for training rather than for a few minutes of addition.
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, device_map={"": "cpu"},
        )
        model = PeftModel.from_pretrained(model, str(checkpoint))
        print("[export] merging adapter into the weights", file=sys.stderr)
        model = model.merge_and_unload()
    else:
        # A full fine-tune checkpoint is already a complete model directory;
        # there is nothing to merge, only a copy in the requested dtype.
        print("[export] full-weights checkpoint — no adapter to merge",
              file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint), torch_dtype=dtype, device_map={"": "cpu"},
        )

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    model.save_pretrained(staging, safe_serialization=True)

    # The tokenizer travels with the weights. convert_hf_to_gguf.py needs it,
    # and a merged directory without one fails at the export step rather than
    # here, after the slow part is already done.
    tokenizer_src = checkpoint if (checkpoint / "tokenizer.json").is_file() else base_model
    print(f"[export] tokenizer from {tokenizer_src}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_src))

    # Drop the chat template, and this is not cosmetic.
    #
    # Qwen3-4B-*Base* ships the Qwen3 *Instruct* chat template in its tokenizer
    # files, and a distilled student inherits it. Recent llama.cpp switches
    # into conversation mode automatically whenever a model carries one, so it
    # wraps every prompt in <|im_start|>user ... <|im_end|><|im_start|>assistant
    # — a format this model has never seen a single token of.
    #
    # Measured on the step-8400 export: through the template the model looped,
    # echoed its own input and emitted stray subword tokens, and read as
    # broken. With the template replaced by plain passthrough, the same file
    # continued "La Corte, letti gli atti, osserva che il ricorso è" into
    # competent Italian legal prose with a correct citation of art. 365 c.p.c.
    # Same weights, same quantization; only the template differed.
    #
    # Shipping it would hand every downloader that first experience and the
    # conclusion that the model does not work. Stage 3 of the pipeline (the
    # identity LoRA) is what earns a chat template; until then the student is
    # a completion model and should present as one.
    if not args.keep_chat_template and getattr(tokenizer, "chat_template", None):
        print("[export] dropping the base tokenizer's chat template: this is a "
              "completion model, and shipping an Instruct template makes "
              "llama.cpp wrap prompts in a format it never saw",
              file=sys.stderr)
        tokenizer.chat_template = None

    tokenizer.save_pretrained(staging)
    # save_pretrained can still write the file from the source directory's
    # copy, so remove it explicitly rather than trusting the attribute.
    if not args.keep_chat_template:
        (staging / "chat_template.jinja").unlink(missing_ok=True)

    (staging / "eullm_export.json").write_text(
        json.dumps(describe(checkpoint, base_model, output), indent=2) + "\n",
        encoding="utf-8",
    )

    # Into place, now that there is a complete model to put there. The old
    # directory goes only once the new one is whole.
    publish_staging(staging, output)

    print(f"[export] done → {output}", file=sys.stderr)
    print("[export] next: forge/scripts/quantize_to_gguf.sh "
          f"{output} <gguf-out>", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
