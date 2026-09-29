#!/usr/bin/env python3
"""How often retrieval finds each question's own article, at several settings.

    python forge/scripts/check_retrieval.py $WORK/eval/norm-exam-dev.jsonl \\
        $WORK/eval/norm-exam-dev2.jsonl \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl --boost 0 3 6

Prints, per question type and per ``heading_boost``, the share of questions
whose article is first (top1) and among the first three (top3). Counts only,
never a question — but choose settings on DEVELOPMENT sets: tuning retrieval
against the held-out exam turns it into one more development set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import NormIndex, load_eval_set  # noqa: E402
from eullm_forge.eval.norm_exam import retrieval_hits  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exams", nargs="+", type=Path, help="exam JSONL files, pooled")
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--boost", nargs="+", type=int, default=[0, 3])
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args(argv)

    items = [it for path in args.exams for it in load_eval_set(path)]
    records = NormIndex.from_files(args.norms).records
    print(f"[retrieval] {len(items)} items from {len(args.exams)} file(s)")
    for boost in args.boost:
        index = NormIndex(records, heading_boost=boost)
        for kind, r in retrieval_hits(items, index, k=args.k).items():
            print(f"[retrieval] boost={boost:<3} {kind:<18} n={r['n']:<4} "
                  f"top1={r['top1']:.2f} top{args.k}={r[f'top{args.k}']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
