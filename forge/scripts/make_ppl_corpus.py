#!/usr/bin/env python3
"""Build a plain-text perplexity corpus from the held-out validation split.

`llama-perplexity` wants one flat text file; the corpus lives as JSONL. This
converts the second into the first, deterministically, and writes down what
it did beside the output.

Why not just `jq -r .text val.jsonl > corpus.txt` — three reasons, each of
which has already cost a measurement somewhere:

* **Provenance.** A perplexity number is only worth reporting if the corpus
  behind it can be named. A file called `val_sample.txt` sitting in a run
  directory, with nobody able to say which records it holds or how they were
  chosen, cannot be cited and cannot be reproduced. This writes a sidecar
  `.meta.json` with the source, the record count, the byte count and the
  selection rule.
* **Determinism.** Comparing a student against a base means running the same
  corpus twice. "The same" has to survive a shell history being lost, so the
  selection is by position, not by a fresh random draw, and a `--seed` sample
  is recorded when used.
* **Size.** 340 chunks took 1 h 37 m per model on the serial partition. Three
  models is most of a day for a number that settles in the first forty. This
  sizes the file to a chunk target instead of dumping the whole split.

The held-out guarantee comes from `format_pretraining.py`: val.jsonl is a 1 %
split at seed 42, disjoint from train.jsonl, over a deduplicated corpus. It
is in-domain — same courts, overlapping years — which is what makes it the
right set for "did distillation help on this domain", and not a claim about
generalizing beyond it.

Usage:
    python forge/scripts/make_ppl_corpus.py \\
        --val  $EULLM_DATA_DIR/val.jsonl \\
        --out  $WORK/eval/legal-it-val-40chunks.txt \\
        --target-chunks 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Qwen3's tokenizer on Italian legal prose runs about 3.5 characters per
# token. Only used to size the file; the true chunk count is whatever
# llama-perplexity reports, which the caller should record rather than this
# estimate.
CHARS_PER_TOKEN = 3.5


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--val", required=True, type=Path,
                   help="held-out split (val.jsonl)")
    p.add_argument("--out", required=True, type=Path,
                   help="plain-text corpus to write")
    p.add_argument("--target-chunks", type=int, default=40,
                   help="how many llama-perplexity chunks to aim for "
                        "(default: 40)")
    p.add_argument("--ctx", type=int, default=512,
                   help="context llama-perplexity will run with; a chunk is "
                        "this many tokens (default: 512)")
    p.add_argument("--text-field", default="text",
                   help="JSON field holding the text (default: text)")
    p.add_argument("--seed", type=int, default=None,
                   help="sample records at random with this seed instead of "
                        "taking them in file order")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing output file")
    return p.parse_args(argv)


def target_chars(chunks: int, ctx: int) -> int:
    """Characters needed for roughly `chunks` chunks of `ctx` tokens."""
    return int(chunks * ctx * CHARS_PER_TOKEN)


def select(records: list[str], want_chars: int, seed: int | None) -> list[str]:
    """Take records until the character budget is met.

    In file order by default, which is reproducible without recording
    anything. With a seed, shuffled first — the seed is what makes that
    reproducible instead, and it goes into the metadata.
    """
    if seed is not None:
        import random
        records = list(records)
        random.Random(seed).shuffle(records)
    out: list[str] = []
    total = 0
    for rec in records:
        out.append(rec)
        total += len(rec)
        if total >= want_chars:
            break
    return out


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.val.is_file():
        raise SystemExit(f"[err] no such file: {args.val}")
    if args.out.exists() and not args.force:
        raise SystemExit(f"[err] {args.out} exists. Pass --force to replace it.")

    want = target_chars(args.target_chunks, args.ctx)
    print(f"[corpus] source        {args.val}", file=sys.stderr)
    print(f"[corpus] target        {args.target_chunks} chunks x {args.ctx} tok "
          f"~= {want:,} chars", file=sys.stderr)

    texts: list[str] = []
    skipped = 0
    with args.val.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            text = (rec.get(args.text_field) or "").strip()
            if text:
                texts.append(text)
            else:
                skipped += 1

    if not texts:
        raise SystemExit(
            f"[err] no records with a non-empty '{args.text_field}' field in "
            f"{args.val}. Pass --text-field if the corpus uses another name."
        )

    chosen = select(texts, want, args.seed)
    body = "\n\n".join(chosen) + "\n"

    if len(body) < want:
        print(f"[warn] the whole split is {len(body):,} chars, short of the "
              f"{want:,} wanted — using all of it. Expect fewer than "
              f"{args.target_chunks} chunks.", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")

    meta = {
        "source": str(args.val),
        "records_available": len(texts),
        "records_used": len(chosen),
        "records_skipped": skipped,
        "bytes": len(body.encode("utf-8")),
        "chars": len(body),
        "selection": "shuffled" if args.seed is not None else "file order",
        "seed": args.seed,
        "target_chunks": args.target_chunks,
        "ctx": args.ctx,
        "estimated_chunks": int(len(body) / CHARS_PER_TOKEN / args.ctx),
        "note": (
            "Held-out: val.jsonl is a 1% split at seed 42, disjoint from "
            "train.jsonl, over a deduplicated corpus. In-domain by "
            "construction. estimated_chunks is an estimate at "
            f"{CHARS_PER_TOKEN} chars/token — record what llama-perplexity "
            "actually reports."
        ),
    }
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"[corpus] used          {len(chosen):,} of {len(texts):,} records",
          file=sys.stderr)
    print(f"[corpus] wrote         {args.out} ({len(body):,} chars)",
          file=sys.stderr)
    print(f"[corpus] provenance    {meta_path}", file=sys.stderr)
    print(f"[corpus] expect about  {meta['estimated_chunks']} chunks at "
          f"ctx {args.ctx}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
