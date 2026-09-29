#!/usr/bin/env python3
"""A spreadsheet of exam items for a person to check, one row per question.

    python forge/scripts/export_exam_review.py $WORK/eval/norm-exam-dev3.jsonl \\
        --n 50 --out $WORK/eval/review-dev3.csv

The builder writes the references from the text of the law, so what a person
checks is the builder itself: is the question clear, is the reference the
right one, would a lawyer accept the rubric? That is checked on a
DEVELOPMENT draw, made by the same builder as the held-out exam: if the
builder is sound there, it is sound on the exam, and the exam stays unread.
So a file that is not a development set is refused unless ``--held-out-ok``.

The CSV opens in Excel and LibreOffice (UTF-8 with BOM, ";" separated) with
two empty columns to fill: ``giudizio`` (ok / errore / dubbio) and ``nota``.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import load_eval_set  # noqa: E402

COLUMNS = ["n", "tipo", "codice", "articolo", "domanda", "riferimento", "rubrica",
           "giudizio", "nota"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exam", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--held-out-ok", action="store_true",
                    help="allow a file that is not a development set (it stops being held out)")
    args = ap.parse_args(argv)

    if "dev" not in args.exam.name and not args.held_out_ok:
        print(f"[review] {args.exam.name} is not a development set: reading it makes it one. "
              "Draw a dev set with the same builder instead (or --held-out-ok).",
              file=sys.stderr)
        return 2
    items = load_eval_set(args.exam)
    # Every question type in proportion, so a fault in one type is seen.
    by_type: dict[str, list] = {}
    for it in items:
        by_type.setdefault(it.metadata.get("tipo", "?"), []).append(it)
    rng = random.Random(args.seed)
    chosen = []
    for kind in sorted(by_type):
        share = max(1, round(args.n * len(by_type[kind]) / len(items)))
        chosen += rng.sample(by_type[kind], min(share, len(by_type[kind])))
    chosen = chosen[:args.n]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(COLUMNS)
        for i, it in enumerate(chosen, 1):
            md = it.metadata
            w.writerow([i, md.get("tipo", ""), md.get("code", ""), md.get("articolo", ""),
                        it.question, it.reference, it.rubric, "", ""])
    kinds = {k: sum(1 for it in chosen if it.metadata.get("tipo") == k) for k in by_type}
    print(f"[review] {len(chosen)} items -> {args.out}  {kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
