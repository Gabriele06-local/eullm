#!/usr/bin/env python3
"""Paired comparison of graded answers: which differences are real.

    python forge/scripts/compare_graded.py $WORK/eval/devbig-answers \\
        --baseline ministral-8b-open [--csv paired.csv] [--human review.csv]

Reads every ``*.graded.jsonl`` in the directory (judge_answers.py output) and
prints, per model: its grades and mean answer length; then, against
``--baseline``, the questions only it got right, the questions only the
baseline got right, the exact McNemar p-value, the same with "partial"
counted as right, the split by question kind, and among the questions where
the two disagree how often the answer judged right was the longer one (see
`eullm_forge.eval.paired`).

A difference is called real at p < 0.05 on BOTH the strict and the lenient
count: a result that flips with how "partial" is read is not a result.

The ``no-judge`` column and line score the deadline and absent-article
questions by a check that needs no model at all (`eullm_forge.eval.paired.
verifiable`): the same comparison, on the part of the exam where the judge
cannot be the reason for a difference.

``--human`` adds how the judge agrees with a person on the answers they
labelled (export_grade_review.py writes the sheet).

Prints labels and counts only, never an item, so it is safe on the held-out
exam's answers too.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval.paired import compare, human_agreement, load_graded  # noqa: E402

CSV_HEADER = ["model", "baseline", "n", "model_only", "baseline_only", "diff", "p",
              "diff_lenient", "p_lenient", "verdict", "longer_wins", "mean_chars",
              "baseline_mean_chars"]


def verdict(c) -> str:
    if c.p < 0.05 and c.p_lenient < 0.05 and (c.diff > 0) == (
            c.a_only_lenient > c.base_only_lenient):
        return "better" if c.diff > 0 else "worse"
    return "no difference shown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graded", nargs="+", type=Path,
                    help="a directory of *.graded.jsonl, or the files themselves")
    ap.add_argument("--baseline", help="label to compare every model against (without it, "
                    "only the per-model table and --human)")
    ap.add_argument("--csv", type=Path, help="write one row per comparison here")
    ap.add_argument("--human", type=Path, help="filled review sheet (export_grade_review.py)")
    args = ap.parse_args(argv)

    files = []
    for p in args.graded:
        files.extend(sorted(p.glob("*.graded.jsonl")) if p.is_dir() else [p])
    models = {g.label: g for g in map(load_graded, files)}
    if args.baseline and args.baseline not in models:
        print(f"[paired] no graded answers for baseline {args.baseline!r}; have: "
              f"{', '.join(sorted(models)) or 'none'}", file=sys.stderr)
        return 1

    print(f"{'model':<28} {'n':>5} {'correct':>8} {'partial':>8} {'wrong':>6} "
          f"{'unparsed':>9} {'chars':>7}  {'no-judge':>9}")
    for label in sorted(models):
        g = models[label]
        c = g.counts()
        ok, nv = g.verif_counts()
        print(f"{label:<28} {len(g.grades):>5} {c['correct']:>8} {c['partial']:>8} "
              f"{c['wrong']:>6} {c['unparsed']:>9} {g.mean_length():>7.0f}  {ok:>4}/{nv:<4}")

    rows = []
    if args.baseline:
        base = models[args.baseline]
        print(f"\nagainst {args.baseline} "
              "(only-model : only-baseline, p; lenient = partial counts)")
        for label in sorted(models):
            if label == args.baseline:
                continue
            c = compare(models[label], base)
            v = verdict(c)
            lw = "-" if c.longer_wins is None else f"{c.longer_wins:.2f}"
            kinds = ", ".join(f"{k} {x}:{y}" for k, (x, y) in c.by_kind.items())
            print(f"{label:<28} n={c.n:<5} {c.a_only:>3}:{c.base_only:<3} diff {c.diff:+4d} "
                  f"p={c.p:.3f}  lenient {c.a_only_lenient}:{c.base_only_lenient} "
                  f"p={c.p_lenient:.3f}  longer-right {lw}  -> {v}")
            if kinds:
                print(f"{'':<28} by kind: {kinds}")
            if c.verif_n:
                print(f"{'':<28} no judge (deadlines, absent articles; n={c.verif_n}): "
                      f"{c.verif_a_only}:{c.verif_base_only} p={c.p_verif:.3f}")
            rows.append([label, args.baseline, c.n, c.a_only, c.base_only, c.diff,
                         f"{c.p:.4f}", c.a_only_lenient - c.base_only_lenient,
                         f"{c.p_lenient:.4f}", v, lw, f"{models[label].mean_length():.0f}",
                         f"{base.mean_length():.0f}"])

    if args.csv and rows:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            w.writerows(rows)

    if args.human:
        with args.human.open(encoding="utf-8-sig", newline="") as f:
            sheet = list(csv.DictReader(f, delimiter=";"))
        keys = args.human.with_name(args.human.name + ".keys.json")
        if keys.is_file():
            codes = json.loads(keys.read_text(encoding="utf-8"))
            for r in sheet:
                r["chiave"] = codes.get(r.get("chiave", ""), r.get("chiave", ""))
        h = human_agreement(sheet, models)
        if not h["n"]:
            print("\n[paired] the review sheet has no labelled rows yet")
        else:
            print(f"\njudge vs person on {h['n']} answers: same grade {h['same_label']:.2f}, "
                  f"same right/not-right {h['same_right_or_not']:.2f}; judge passed "
                  f"{h['judge_too_kind']} the person failed, failed {h['judge_too_harsh']} "
                  f"the person passed")
            print("  " + ", ".join(f"{k} {v}" for k, v in h["confusion"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
