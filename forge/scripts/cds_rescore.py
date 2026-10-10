#!/usr/bin/env python3
"""The citation checks of cds_answer.py again, on answers already written.

    python forge/scripts/cds_rescore.py $WORK/eval/cds-exam-pc/answers-*.jsonl

When `cds_answer.cited_numbers` changes, the answers do not need asking
again: each row keeps the answer and the rulings it was given (``context``).
This recomputes ``cited_ok`` and ``source_cited`` with the current reading
and prints, per file, the old figure and the new one. Counts only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _cds_answer():
    spec = importlib.util.spec_from_file_location(
        "cds_answer", Path(__file__).resolve().parent / "cds_answer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rescore(rows: list[dict], cited_numbers) -> dict[str, float]:
    """Old and new share of answers citing only given rulings, and citing the source."""
    n = len(rows) or 1
    ok = src = 0
    for r in rows:
        cited = cited_numbers(r.get("answer", ""))
        in_ctx = {c.split("/", 1)[-1] for c in r.get("context", [])}
        ok += cited <= in_ctx
        src += str(r.get("ruling", "")).split("/", 1)[-1] in cited
    return {"cited_ok_before": sum(bool(r.get("cited_ok")) for r in rows) / n,
            "cited_ok": ok / n,
            "source_cited_before": sum(bool(r.get("source_cited")) for r in rows) / n,
            "source_cited": src / n}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="+", type=Path, help="answers-*.jsonl of cds_answer.py")
    args = ap.parse_args(argv)
    cited_numbers = _cds_answer().cited_numbers
    for p in args.answers:
        rows = [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]
        if not rows or "context" not in rows[0]:
            print(f"{p.name}: not a cds_answer.py answers file, skipped")
            continue
        s = rescore(rows, cited_numbers)
        print(f"{p.name}: {len(rows)} answers | citing only rulings given "
              f"{s['cited_ok_before']:.3f} -> {s['cited_ok']:.3f} | citing the source "
              f"{s['source_cited_before']:.3f} -> {s['source_cited']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
