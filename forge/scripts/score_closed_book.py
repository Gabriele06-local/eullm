#!/usr/bin/env python3
"""Score answers to the closed-book suite by program, by kind.

    python forge/scripts/score_closed_book.py $WORK/eval/closed-book.jsonl \\
        $WORK/eval/closed-book-answers/answers-*.jsonl

One row per model: the share right of every kind, the total, and the wrong
answers given with no doubt expressed ("sure-wrong"), which count against a
model more than an honest "I do not know". Counts only, never a question.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval.closed_book import KINDS, score  # noqa: E402

SHORT = {"esiste_vera": "esist", "esiste_falsa": "falso", "numero_da_rubrica": "n<-rub",
         "rubrica_da_numero": "rub<-n", "termine": "termi", "pena": "pena",
         "procedibilita": "proc", "abrogato": "abrog", "sentenza_falsa": "sent?"}


def label_of(path: Path) -> str:
    name = path.name.removesuffix(".jsonl")
    return name.removeprefix("answers-")


def summarize(items: dict[str, dict], answers: dict[str, str]) -> dict:
    by = defaultdict(Counter)
    sure_wrong = 0
    for i, ans in answers.items():
        if i not in items:
            continue
        grade, confident = score(items[i], ans)
        by[items[i]["metadata"]["tipo"]][grade] += 1
        sure_wrong += grade == "wrong" and confident
    total = sum(sum(c.values()) for c in by.values())
    right = sum(c["right"] for c in by.values())
    return {"by": by, "n": total, "right": right, "sure_wrong": sure_wrong}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", type=Path)
    ap.add_argument("answers", nargs="+", type=Path)
    args = ap.parse_args(argv)

    with args.suite.open(encoding="utf-8") as f:
        items = {d["id"]: d for d in map(json.loads, filter(str.strip, f))}
    kinds = [k for k in KINDS if any(it["metadata"]["tipo"] == k for it in items.values())]
    print(f"{'model':<30} {'right':>11} {'sure-wrong':>10}  "
          + " ".join(f"{SHORT[k]:>6}" for k in kinds))
    for path in args.answers:
        with path.open(encoding="utf-8") as f:
            answers = {r["id"]: r.get("answer", "") for r in map(json.loads, filter(str.strip, f))}
        s = summarize(items, answers)
        if not s["n"]:
            print(f"{label_of(path):<30} no answers to this suite")
            continue
        cells = []
        for k in kinds:
            c = s["by"].get(k, Counter())
            n = sum(c.values())
            cells.append(f"{100 * c['right'] / n:5.0f}%" if n else f"{'-':>6}")
        print(f"{label_of(path):<30} {s['right']:>4}/{s['n']:<4} "
              f"{100 * s['right'] / s['n']:3.0f}% {s['sure_wrong']:>9}  " + " ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
