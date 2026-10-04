#!/usr/bin/env python3
"""The answers of a review sheet only, to try another judge on them in minutes.

    python forge/scripts/review_subset.py $WORK/eval/review-grades-devbig.csv \\
        --answers $WORK/eval/devbig-answers --out $WORK/eval/devreview-answers

A judge is worth what it agrees with a careful grading of the same answers.
The review sheet (export_grade_review.py) holds such a grading of forty
answers drawn from many answers files; this writes, for each model, an
answers file with just its answers that are on the sheet. A judge job pointed
at ``--out`` grades forty answers instead of thousands, and

    compare_graded.py <out or its --graded-dir> --human <sheet>

says how it agrees. The answers are copied as they are, unread: the sheet
already names which ones, through the keys file beside it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet", type=Path, help="review sheet (export_grade_review.py)")
    ap.add_argument("--answers", type=Path, required=True,
                    help="directory of the answers-<label>.jsonl the sheet was drawn from")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if "dev" not in args.out.name:
        print(f"[subset] {args.out.name}: name it as a development directory (…dev…), the "
              "answers in it are read like one", file=sys.stderr)
        return 2
    keys_file = args.sheet.with_name(args.sheet.name + ".keys.json")
    keys = json.loads(keys_file.read_text(encoding="utf-8"))
    with args.sheet.open(encoding="utf-8-sig", newline="") as f:
        codes = [r["chiave"] for r in csv.DictReader(f, delimiter=";")]
    wanted: dict[str, set[str]] = defaultdict(set)
    for code in codes:
        label, _, item = keys[code].partition("|")
        wanted[label].add(item)

    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for label, items in sorted(wanted.items()):
        src = args.answers / f"answers-{label}.jsonl"
        rows = [line for line in src.open(encoding="utf-8")
                if line.strip() and json.loads(line)["id"] in items]
        (args.out / src.name).write_text("".join(rows), encoding="utf-8")
        n += len(rows)
    print(f"[subset] {n} answers of {len(codes)} on the sheet, {len(wanted)} models -> {args.out}")
    return 0 if n == len(codes) else 1


if __name__ == "__main__":
    sys.exit(main())
