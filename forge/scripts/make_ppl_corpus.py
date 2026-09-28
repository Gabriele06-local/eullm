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
  selection is a seeded shuffle, and the seed has a default: the shipped
  command names no `--seed`, and file order is not a property worth having.
  `val.jsonl` is written in source-slice order (`format_pretraining.py` writes
  the split as the corpus files are globbed, so it is grouped by court), which
  means taking a prefix of it is taking the alphabetically first court and
  nothing else. On a four-court corpus the 40-chunk corpus built that way held
  records from two of the four, and every row of `perplexity.csv` inherits
  that. Both the rule and the seed are written to the sidecar, and
  `perplexity_compare.sh` keys its cached base on the corpus size as well as
  its name, so a regenerated corpus is re-measured rather than compared
  against a number from another one.
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

# The seed the shipped command gets, so the corpus is a seeded draw and not a
# prefix of the split. It is the split's own seed, which is not a claim that
# the two draws are related — only that neither has to be typed to be
# reproducible. Written into the sidecar either way.
DEFAULT_SEED = 42


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
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="seed for the selection shuffle (default: "
                        f"{DEFAULT_SEED}). Pass a negative number to take the "
                        "records in file order, which over the split as "
                        "format_pretraining.py writes it means the first court "
                        "only.")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing output file")
    return p.parse_args(argv)


def target_chars(chunks: int, ctx: int) -> int:
    """Characters needed for roughly `chunks` chunks of `ctx` tokens."""
    return int(chunks * ctx * CHARS_PER_TOKEN)


def select(records: list[tuple[str, str | None]], want_chars: int,
           seed: int | None) -> list[tuple[str, str | None]]:
    """Take (text, kind) records until the character budget is met.

    Shuffled with ``seed`` first, so the draw does not depend on how the split
    happens to be ordered. A negative seed, or None, means file order — which
    over val.jsonl as format_pretraining.py writes it is a prefix, and so a
    single court; kept because it is what "the first N records" has to mean
    when someone asks for it.
    """
    if seed is not None and seed >= 0:
        import random
        records = list(records)
        random.Random(seed).shuffle(records)
    out: list[tuple[str, str | None]] = []
    total = 0
    for rec in records:
        out.append(rec)
        total += len(rec[0])
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

    texts: list[tuple[str, str | None]] = []
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
                # Keep the collection with the text: a perplexity number is
                # only worth citing if you can say which courts it is about.
                kind = rec.get("kind")
                texts.append((text, kind if isinstance(kind, str) else None))
            else:
                skipped += 1

    if not texts:
        raise SystemExit(
            f"[err] no records with a non-empty '{args.text_field}' field in "
            f"{args.val}. Pass --text-field if the corpus uses another name."
        )

    chosen = select(texts, want, args.seed)
    body = "\n\n".join(text for text, _ in chosen) + "\n"

    if len(body) < want:
        print(f"[warn] the whole split is {len(body):,} chars, short of the "
              f"{want:,} wanted — using all of it. Expect fewer than "
              f"{args.target_chunks} chunks.", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")

    kinds: dict[str, int] = {}
    for _, kind in chosen:
        if kind:
            kinds[kind] = kinds.get(kind, 0) + 1
    meta = {
        "source": str(args.val),
        "records_available": len(texts),
        "records_used": len(chosen),
        "records_skipped": skipped,
        "bytes": len(body.encode("utf-8")),
        "chars": len(body),
        "selection": ("file order" if args.seed is None or args.seed < 0
                      else f"shuffled at seed {args.seed}"),
        "seed": args.seed,
        "kinds_drawn": dict(sorted(kinds.items())),
        "kinds_available": dict(sorted({
            kind: sum(1 for _, k in texts if k == kind)
            for kind in {k for _, k in texts if k}
        }.items())),
        "target_chunks": args.target_chunks,
        "ctx": args.ctx,
        "estimated_chunks": int(len(body) / CHARS_PER_TOKEN / args.ctx),
        "note": (
            "Held-out: val.jsonl is a 1% split at seed 42, disjoint from "
            "train.jsonl, over a deduplicated corpus, and it is written in "
            "source-slice order — which is why the selection is shuffled "
            "rather than positional. In-domain by construction; kinds_drawn "
            "says which collections the number is actually about. "
            "estimated_chunks is an estimate at "
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
