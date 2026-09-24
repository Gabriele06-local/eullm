#!/usr/bin/env python3
"""Generate stage-3 instruction/answer pairs from the legal corpus.

Drives `Qwen/Qwen3-30B-A3B-Instruct-2507` over passages of ``train.jsonl``
and writes grounded pairs that `eullm_forge.identity` trains on. What is
generated and why, the grounding rule and the personal-data filters are in
`eullm_forge/datasets/instruct_gen.py`; this file is only the driver.

RESUMABLE BY CONSTRUCTION. Jobs are derived deterministically from the corpus
and the seed, every result — accepted or rejected — is appended with its key
the moment its batch finishes, and a rerun with the same arguments skips
every key already on disk. A walltime kill costs the batch in flight and
nothing else, which is what makes two-hour links usable for this.

    # on a login node: checks paths and shows one prompt, loads no model
    python forge/scripts/generate_instructions.py --dry-run \\
        --corpus "$EULLM_DATA_DIR/train.jsonl" --out "$OUT" --limit 60

    # inside a GPU job (see leonardo/sbatch_gen_instruct.slurm)
    python forge/scripts/generate_instructions.py \\
        --corpus "$EULLM_DATA_DIR/train.jsonl" --out "$OUT" --limit 60

Outputs, both JSONL:
    OUT                   accepted pairs: instruction, output, task, source, key
    OUT.rejected.jsonl    key, task, reason, and the raw generation, so the
                          filters can be audited instead of trusted

Neither file goes into git, in any repository: they are derived from
rulings.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.instruct_gen import (  # noqa: E402
    GenConfig,
    Job,
    Rejected,
    build_messages,
    make_jobs,
    parse_generation,
)

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def load_corpus(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def done_keys(*paths: Path) -> set[str]:
    keys: set[str] = set()
    for p in paths:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        keys.add(json.loads(line)["key"])
                    except (json.JSONDecodeError, KeyError):
                        continue  # a line cut by a walltime kill
    return keys


class TransformersGenerator:
    """Batched sampling with plain transformers, the model split across GPUs.

    Chosen because it is the path that already works on Leonardo: the
    distillation teacher is the same architecture, loaded the same way, from
    the same offline cache, with no new dependency in the environment. It is
    not the fastest engine there is. The pilot measures its throughput, and
    a faster backend only has to provide `generate(list_of_messages)`.
    """

    def __init__(self, model_id: str, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("no GPU visible — this runs inside a GPU job")
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"   # decoder-only: pad before the prompt
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        t0 = time.time()
        # "balanced" spreads the 61 GB of bf16 weights evenly over the visible
        # cards, leaving each the same room for the KV cache of a batch.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="balanced",
        )
        self.model.eval()
        print(f"[gen] loaded {model_id} on {torch.cuda.device_count()} GPUs "
              f"in {time.time() - t0:.0f}s", flush=True)

    def generate(self, conversations: list[list[dict]]) -> tuple[list[str], int]:
        prompts = [
            self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations
        ]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            # Qwen's recommended sampling for the 2507 Instruct models.
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.8, top_k=20,
                pad_token_id=self.tok.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(new, skip_special_tokens=True)
        n_tokens = int((new != self.tok.pad_token_id).sum())
        return texts, n_tokens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True, help="train.jsonl — never val")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, required=True, help="jobs to derive (pairs attempted)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--stop-after-min", type=float, default=0,
                    help="stop cleanly before this many minutes (0 = no limit)")
    ap.add_argument("--dry-run", action="store_true",
                    help="derive jobs, print one prompt, load no model")
    args = ap.parse_args()

    start = time.time()
    if "val" in args.corpus.name or "cds" in args.corpus.name.lower():
        # The evaluation sets. Generating from them puts the exam's answers
        # into the training data, and nothing downstream would notice.
        print(f"[gen] refusing {args.corpus}: that is an evaluation set", file=sys.stderr)
        return 2

    rejected_path = args.out.with_name(args.out.name + ".rejected.jsonl")
    cfg = GenConfig()
    records = load_corpus(args.corpus)
    jobs = make_jobs(records, args.limit, seed=args.seed, cfg=cfg)
    del records
    done = done_keys(args.out, rejected_path)
    todo = [j for j in jobs if j.key not in done]
    mix = collections.Counter(j.task for j in jobs)
    print(f"[gen] {len(jobs)} jobs ({dict(mix)}), {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to go", flush=True)

    if args.dry_run:
        if todo:
            for m in build_messages(todo[0]):
                print(f"--- {m['role']} ---\n{m['content'][:1500]}")
        return 0
    if not todo:
        return 0

    gen = TransformersGenerator(args.model, args.max_new_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Longest prompts first inside the run: padding is per batch, so batching
    # similar lengths together wastes less, and an OOM shows up in the first
    # batch rather than two hours in.
    todo.sort(key=lambda j: len(j.passage), reverse=True)

    reasons: collections.Counter[str] = collections.Counter()
    accepted = tokens = 0
    gen_seconds = 0.0
    last_batch = 0.0
    for i in range(0, len(todo), args.batch_size):
        elapsed_min = (time.time() - start) / 60
        if args.stop_after_min and elapsed_min + 1.5 * last_batch / 60 > args.stop_after_min:
            print(f"[gen] stopping at {elapsed_min:.0f} min, before the walltime", flush=True)
            break
        batch: list[Job] = todo[i:i + args.batch_size]
        t0 = time.time()
        texts, n = gen.generate([build_messages(j) for j in batch])
        last_batch = time.time() - t0
        gen_seconds += last_batch
        tokens += n

        with open(args.out, "a", encoding="utf-8") as ok, \
             open(rejected_path, "a", encoding="utf-8") as bad:
            for job, raw in zip(batch, texts):
                try:
                    pair = parse_generation(raw, job, cfg)
                except Rejected as exc:
                    reasons[exc.reason] += 1
                    bad.write(json.dumps({"key": job.key, "task": job.task,
                                          "reason": exc.reason, "detail": str(exc),
                                          "raw": raw[:4000]}, ensure_ascii=False) + "\n")
                    continue
                accepted += 1
                ok.write(json.dumps(pair, ensure_ascii=False) + "\n")

        seen = accepted + sum(reasons.values())
        print(f"[gen] {seen}/{len(todo)}  accepted {accepted}  "
              f"rejected {dict(reasons)}  {tokens / max(gen_seconds, 1e-9):.0f} tok/s",
              flush=True)

    seen = accepted + sum(reasons.values())
    print(f"[gen] done: {accepted}/{seen} accepted "
          f"({100 * accepted / max(seen, 1):.0f}%), {tokens} tokens in "
          f"{gen_seconds / 60:.1f} min of generation", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
