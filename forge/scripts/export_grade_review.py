#!/usr/bin/env python3
"""A sheet of graded answers for a person to grade too, without seeing the judge.

    python forge/scripts/export_grade_review.py $WORK/eval/devbig-answers \\
        --n 40 --out $WORK/eval/review-grades-devbig.csv

Every decision on the development set rests on the LLM judge. Whether the
judge is right is checked here, on a sample of answers a person grades
blind: the sheet shows the question, the reference and the answer, never the
judge's verdict, and compare_graded.py --human puts the two side by side.

Three quarters of the sample are answers to questions on which the models
disagree, because that is where a judge error changes a comparison; the rest
are drawn from all answers, so agreement on the easy ones is measured too.

Like export_exam_review.py, it refuses a directory that is not a development
set: answers to the held-out exam are not read.

The CSV opens in Excel and LibreOffice (UTF-8 with BOM, ";" separated). Fill
``giudizio`` with corretto / parziale / sbagliato and, if useful, ``nota``.
Which model wrote an answer is not shown either: ``chiave`` is an opaque code,
and the file beside the sheet (``<out>.keys.json``) maps it back for
compare_graded.py. Leave ``chiave`` as it is.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval.paired import kind_of, label_of  # noqa: E402


def keys_path(sheet: Path) -> Path:
    """Where the codes of a sheet are mapped back to model and item."""
    return sheet.with_name(sheet.name + ".keys.json")


COLUMNS = ["n", "chiave", "tipo", "domanda", "riferimento", "risposta", "giudizio", "nota"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graded_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--labels", nargs="+", help="only these models (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if "dev" not in args.graded_dir.resolve().name:
        print(f"[review] {args.graded_dir} is not a development set: its answers are not "
              "read. Point this at the answers of a dev draw.", file=sys.stderr)
        return 2
    rows: dict[tuple[str, str], dict] = {}
    for path in sorted(args.graded_dir.glob("*.graded.jsonl")):
        label = label_of(path)
        if args.labels and label not in args.labels:
            continue
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("grade") in ("correct", "partial", "wrong"):
                    rows[(label, r["id"])] = r
    if not rows:
        print(f"[review] no graded answers in {args.graded_dir}", file=sys.stderr)
        return 1

    verdicts: dict[str, set[bool]] = defaultdict(set)
    for (_, item), r in rows.items():
        verdicts[item].add(r["grade"] == "correct")
    split = [k for k in rows if len(verdicts[k[1]]) > 1]
    rng = random.Random(args.seed)
    want_split = min(len(split), args.n * 3 // 4)
    picked = rng.sample(split, want_split)
    rest = [k for k in rows if k not in set(picked)]
    picked += rng.sample(rest, min(len(rest), args.n - len(picked)))
    rng.shuffle(picked)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = {}
    with args.out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(COLUMNS)
        for n, (label, item) in enumerate(picked, 1):
            r = rows[(label, item)]
            code = f"r{rng.randrange(16**6):06x}"
            while code in keys:
                code = f"r{rng.randrange(16**6):06x}"
            keys[code] = f"{label}|{item}"
            w.writerow([n, code, kind_of(item), r.get("question", ""),
                        r.get("reference", ""), r.get("answer", ""), "", ""])
    keys_path(args.out).write_text(json.dumps(keys, indent=1), encoding="utf-8")
    print(f"[review] {len(picked)} answers ({want_split} where the models disagree) "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
