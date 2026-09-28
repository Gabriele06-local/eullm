#!/usr/bin/env python3
"""Draw the held-out exam from the text of the law. Prints counts, never questions.

See `eullm_forge.eval.norm_exam` for what the questions are and why they come
from the legislation files rather than from anyone's memory. Run it where the
exam is to live (the cluster), and leave the file there: the point of a
held-out set is that nobody improving the models reads it.

    python forge/scripts/make_norm_exam.py $WORK/norms/legislazione_*.chunks.jsonl \\
        --out $WORK/eval/norm-exam.jsonl [--per-code 10] [--check-retrieval]

The draw uses a random seed that is not printed. ``--codes`` limits it to one
vertical, e.g. the three administrative sources.

``--exclude-pairs`` leaves out every article an open-book training file asks
about (and the made-up numbers of its absent-article pairs). A redraw after
training on such pairs must use it, or the exam partly asks what training
answered. Like the rest, it prints how many, never which.

``--exclude-exam`` leaves out the articles of another exam. It exists to draw
a DEVELOPMENT set: a second draw that may be read, to find out why answers
are wrong, while the held-out exam stays unread and shares no article with it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import NormIndex, load_eval_set, save_eval_set  # noqa: E402
from eullm_forge.eval.norm_exam import (  # noqa: E402
    build_exam,
    retrieval_hits,
    trained_articles,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("norms", nargs="+", type=Path, help="legislazione_*.chunks.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-code", type=int, default=10)
    ap.add_argument("--codes", nargs="+", help="only these codes (default: all)")
    ap.add_argument("--check-retrieval", action="store_true",
                    help="also report how often retrieval finds each item's article")
    ap.add_argument("--exclude-pairs", nargs="+", type=Path, default=[],
                    help="open-book training pairs whose articles must not be drawn")
    ap.add_argument("--exclude-exam", nargs="+", type=Path, default=[],
                    help="another exam whose articles must not be drawn (e.g. draw a "
                         "readable development set that shares nothing with the held-out one)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing exam")
    args = ap.parse_args(argv)

    if args.out.exists() and not args.force:
        print(f"[exam] {args.out} exists — an exam is drawn once; --force to redraw",
              file=sys.stderr)
        return 1
    index = NormIndex.from_files(args.norms)
    exclude: set[tuple[str, str]] = set()
    for path in args.exclude_pairs:
        with open(path, encoding="utf-8") as f:
            exclude |= trained_articles(json.loads(line) for line in f if line.strip())
    if args.exclude_pairs:
        print(f"[exam] {len(exclude)} trained articles left out")
    other: set[tuple[str, str]] = set()
    for path in args.exclude_exam:
        for it in load_eval_set(path):
            md = it.metadata
            if md.get("code") and md.get("articolo"):
                other.add((md["code"], str(md["articolo"])))
    if args.exclude_exam:
        print(f"[exam] {len(other)} articles of the other exam left out")
    exclude |= other
    items = build_exam(index.records, per_code=args.per_code,
                       codes=set(args.codes) if args.codes else None, exclude=exclude)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_eval_set(items, args.out)

    by = Counter((it.metadata["vertical"], it.metadata["tipo"]) for it in items)
    print(f"[exam] {len(items)} items -> {args.out}")
    for (vertical, kind), n in sorted(by.items()):
        print(f"[exam]   {vertical:<15} {kind:<18} {n}")
    if args.check_retrieval:
        for kind, r in retrieval_hits(items, index).items():
            print(f"[retrieval] {kind:<18} n={r['n']:<4} top1={r['top1']:.2f} "
                  f"top3={r['top3']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
