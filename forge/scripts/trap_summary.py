#!/usr/bin/env python3
"""The outside test's traps, per model: right of 36, and of 33 without the off-topic three.

    python forge/scripts/trap_summary.py --traps $WORK/eval/lexia/lexia-trappole.jsonl \\
        --graded $WORK/eval/lexia/graded r2l-steps-8b-quote r2q-steps-8b-p7

Each label is matched as ``*<label>-q4-hybnn.graded.jsonl`` in ``--graded``
(the trap questions are asked without the note); without labels every
graded file there is summed up.

The three off-topic traps ("Fuori perimetro") assume a service limited to
criminal law, and their judge is a coin: on 2026-10-25 the same bare refusal
was graded right on two and wrong on the third. So the count of 33 without
them is printed beside the official one of 36, which still decides a
promotion. Counts only, never a question: the test is private.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED = ("correzione", "astensione", "risposta")


def off_topic(trap: dict) -> bool:
    """One of the off-topic traps, by category or by id."""
    return (trap.get("category", "").lower().startswith("fuori")
            or "-fuori-" in trap.get("id", ""))


def summary(traps: dict[str, dict], grades: dict[str, str]) -> dict:
    """Right answers of all, of all but the off-topic ones, by expected behaviour."""
    right = {i for i, g in grades.items() if g == "correct" and i in traps}
    kept = {i for i in traps if not off_topic(traps[i])}
    return {"n": len(traps), "right": len(right),
            "n_kept": len(kept), "right_kept": len(right & kept),
            "by_expected": {e: sum(traps[i]["metadata"].get("expected") == e for i in right)
                            for e in EXPECTED},
            "off_topic": {i.rsplit("-", 1)[-1]: i in right
                          for i in sorted(traps) if off_topic(traps[i])}}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traps", type=Path, required=True)
    ap.add_argument("--graded", type=Path, required=True, help="directory of *.graded.jsonl")
    ap.add_argument("labels", nargs="*")
    args = ap.parse_args(argv)

    traps = {t["id"]: t for t in _read(args.traps)}
    if args.labels:
        files = [(lab, sorted(args.graded.glob(f"*{lab}-q4-hybnn.graded.jsonl")))
                 for lab in args.labels]
    else:
        files = [(p.name.removeprefix("answers-").removesuffix(".graded.jsonl"), [p])
                 for p in sorted(args.graded.glob("*.graded.jsonl"))]
    for label, found in files:
        if not found:
            print(f"{label:<26} no graded answers")
            continue
        s = summary(traps, {r["id"]: r.get("grade") for r in _read(found[0])})
        by = " ".join(f"{e[:5]} {s['by_expected'][e]}" for e in EXPECTED)
        off = " ".join(f"{k}:{'✓' if v else '·'}" for k, v in s["off_topic"].items())
        print(f"{label:<26} {s['right']:>2}/{s['n']}  {s['right_kept']:>2}/{s['n_kept']}"
              f"  | {by} | off-topic {off}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
