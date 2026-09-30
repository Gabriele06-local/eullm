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
    # One slot per family first, if there is room for one, then what is left
    # split in proportion by largest remainder. Two things have to hold at
    # once. Every family appears whenever --n allows it, because the sheet
    # exists so a fault in one type of question is seen and a family with no row
    # is a fault nobody looks for. And the allocation adds up to exactly --n.
    # Rounding each share on its own does neither: it overshoots, and the
    # overshoot used to be trimmed off the end, which is always the
    # alphabetically last family -- on a six-family exam at --n 40 the sheet
    # held four of them, and the two missing were the last two by name, every
    # run.
    order = sorted(by_type)
    spare = max(0, args.n - len(order))
    exact = {k: spare * len(by_type[k]) / len(items) for k in order}
    shares = {k: min(len(by_type[k]), 1 + int(exact[k])) for k in order}
    for k in sorted(order, key=lambda k: exact[k] - int(exact[k]), reverse=True):
        if sum(shares.values()) >= args.n:
            break
        shares[k] = min(len(by_type[k]), shares[k] + 1)

    rng = random.Random(args.seed)
    chosen = []
    for k in order:
        chosen += rng.sample(by_type[k], shares[k])
    # Shuffled before the trim, so that when --n is smaller than the number of
    # families and something has to go, it goes at random rather than always
    # out of the same family.
    rng.shuffle(chosen)
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
