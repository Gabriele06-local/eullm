#!/usr/bin/env python3
"""How often retrieval finds each question's own article, at several settings.

    python forge/scripts/check_retrieval.py $WORK/eval/norm-exam-dev.jsonl \\
        $WORK/eval/norm-exam-dev2.jsonl \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl --boost 0 3 6

Prints, per question type and per ``heading_boost``, the share of questions
whose article is first (top1) and among the first three (top3). Counts only,
never a question — but choose settings on DEVELOPMENT sets: tuning retrieval
against the held-out exam turns it into one more development set.

``--pairs`` measures on the open-book training pairs instead: the teacher's
questions that ask by topic in its own words (``named`` false), each with the
article it was written from. The exam's topic questions quote the rubrica
verbatim, so a heavy rubrica weight wins there by construction; the teacher
paraphrases, the way people do, and is the fairer test of a weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import EvalItem, NormIndex, load_eval_set  # noqa: E402
from eullm_forge.eval.norm_exam import retrieval_hits  # noqa: E402


def topic_questions(path: Path) -> list[EvalItem]:
    """The teacher's by-topic questions of an open-book pairs file, as items
    that name the article they were written from."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            key = str(p.get("key", ""))
            if p.get("task") != "openbook_grounded" or p.get("named", True) \
                    or not key.startswith("ob-g-"):
                continue
            code, _, number = key[len("ob-g-"):].partition("-")
            question = p["instruction"].rsplit("Domanda: ", 1)[-1].strip()
            items.append(EvalItem(id=key, domain="legal", lang="it", question=question,
                                  metadata={"tipo": "argomento_insegnante", "code": code,
                                            "articolo": number}))
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exams", nargs="*", type=Path, help="exam JSONL files, pooled")
    ap.add_argument("--pairs", nargs="+", type=Path, default=[],
                    help="open-book pairs: measure on their topic questions")
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--boost", nargs="+", type=int, default=[0, 3])
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args(argv)

    items = [it for path in args.exams for it in load_eval_set(path)]
    for path in args.pairs:
        items.extend(topic_questions(path))
    if not items:
        ap.error("give exam files or --pairs")
    records = NormIndex.from_files(args.norms).records
    print(f"[retrieval] {len(items)} items from {len(args.exams) + len(args.pairs)} file(s)")
    for boost in args.boost:
        index = NormIndex(records, heading_boost=boost)
        for kind, r in retrieval_hits(items, index, k=args.k).items():
            print(f"[retrieval] boost={boost:<3} {kind:<18} n={r['n']:<4} "
                  f"top1={r['top1']:.2f} top{args.k}={r[f'top{args.k}']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
