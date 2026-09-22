#!/usr/bin/env python3
"""Refuse to train on the evaluation set.

Consiglio di Stato 2025-2026 is the project's held-out corpus. It is the
only clean number we have: the 23 % improvement measured at step 28,000 means
something precisely because it was measured on a court the training never
touched, and every point of the transfer curve inherits that property. One
formatting run that quietly swept 2025 into ``train.jsonl`` would not fail —
it would produce a *better-looking* number that means nothing, and it would
retroactively invalidate the points already measured, because they would no
longer be comparable with the ones after it.

Nothing else in the pipeline can catch that. The formatter does not know
which years are sacred, the trainer reads whatever it is given, and the loss
curve looks fine either way. So the check is here, and the launchers run it
before spending a node on the corpus.

Two questions, because a corpus can fail in both directions:

  * **Is the held-out material absent?** Any record from the held-out source
    at or after the held-out year is a hard failure, and the script names the
    offending years and counts rather than just saying no.
  * **Is the new material present at all?** An arm whose whole purpose is the
    larger corpus, pointed by accident at the old data directory, trains
    happily and produces a result that answers a different question. So
    ``--require-source`` asserts the new material is actually there.

Both checks read the *formatted* ``train.jsonl`` rather than the corpus that
produced it, since that file is what the trainer will actually see.

Usage:
    python3 check_corpus_holdout.py DATA_DIR
    python3 check_corpus_holdout.py DATA_DIR --require-source cds
    python3 check_corpus_holdout.py DATA_DIR --holdout-source cds \\
        --holdout-from-year 2025 --files train.jsonl

Exit status is 0 when the corpus is safe to train on and 1 when it is not,
so it can gate an sbatch without any parsing on the caller's side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Where a record says which court it came from. Checked in order; the first
# one present on the record wins. `kind` is what format_pretraining.py infers
# from the corpus filename, so it is set even when the fetcher wrote neither
# of the others.
SOURCE_FIELDS = ("source_id", "source", "kind")


def record_source(rec: dict) -> str:
    for field in SOURCE_FIELDS:
        val = rec.get(field)
        if isinstance(val, str) and val:
            return val.lower()
    return ""


def record_year(rec: dict) -> int | None:
    """The record's year, from the field or from the front of an id.

    `year` is backfilled by the formatter, but a corpus built by hand may not
    carry it, and returning None there would silently pass the check. So an
    id of the shape `snciv/2023/12345` is read too, and a record with no year
    at all is reported as unknown rather than treated as safe.
    """
    year = rec.get("year")
    if isinstance(year, int):
        return year
    if isinstance(year, str) and year.isdigit():
        return int(year)
    for field in ("source_id", "sentence_id"):
        val = rec.get(field)
        if isinstance(val, str):
            for part in val.replace("-", "/").split("/"):
                if len(part) == 4 and part.isdigit() and 1900 < int(part) < 2100:
                    return int(part)
    return None


def scan(path: Path, holdout_source: str, holdout_year: int,
         require_source: str | None) -> tuple[dict[int, int], int, int, int]:
    """Return (violations by year, records from the required source,
    records with no year, total records)."""
    violations: dict[int, int] = {}
    required = 0
    undated = 0
    total = 0
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                # Malformed lines are the formatter's problem, not this
                # script's; it warns about them already. Skipping one here
                # cannot turn a violation into a pass, because a line that
                # does not parse is a line the trainer will not read either.
                print(f"[warn] {path.name}:{lineno} malformed JSON", file=sys.stderr)
                continue
            total += 1
            src = record_source(rec)
            year = record_year(rec)
            if require_source and require_source in src:
                required += 1
            if holdout_source in src:
                if year is None:
                    undated += 1
                elif year >= holdout_year:
                    violations[year] = violations.get(year, 0) + 1
    return violations, required, undated, total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("data_dir", type=Path, help="directory holding train.jsonl")
    p.add_argument("--files", nargs="+", default=["train.jsonl"],
                   help="files to scan (default: train.jsonl)")
    p.add_argument("--holdout-source", default="cds",
                   help="substring identifying the held-out court "
                        "(default: cds)")
    p.add_argument("--holdout-from-year", type=int, default=2025,
                   help="first year that must not appear (default: 2025)")
    p.add_argument("--require-source", default=None,
                   help="fail unless at least one record matches this "
                        "source substring — use it on arms whose point is "
                        "the larger corpus")
    args = p.parse_args(argv)

    if not args.data_dir.is_dir():
        print(f"[err] no such data directory: {args.data_dir}", file=sys.stderr)
        return 1

    failed = False
    for name in args.files:
        path = args.data_dir / name
        if not path.is_file():
            print(f"[err] {path} does not exist", file=sys.stderr)
            failed = True
            continue

        violations, required, undated, total = scan(
            path, args.holdout_source.lower(), args.holdout_from_year,
            args.require_source.lower() if args.require_source else None,
        )
        print(f"[..] {name}: {total:,} records")

        if violations:
            failed = True
            n = sum(violations.values())
            print(f"[ERR] {name} contains {n:,} record(s) from "
                  f"'{args.holdout_source}' at or after "
                  f"{args.holdout_from_year} — this is the held-out "
                  f"evaluation set:", file=sys.stderr)
            for year in sorted(violations):
                print(f"[ERR]   {year}: {violations[year]:,}", file=sys.stderr)
            print("[ERR] Rebuild the corpus excluding those years. Training "
                  "on them destroys every transfer-curve point already "
                  "measured, not just the next one.", file=sys.stderr)

        if undated:
            failed = True
            print(f"[ERR] {name}: {undated:,} record(s) from "
                  f"'{args.holdout_source}' carry no year, so they cannot be "
                  f"shown to be outside the held-out range.", file=sys.stderr)

        if args.require_source is not None:
            if required == 0:
                failed = True
                print(f"[ERR] {name} contains no record matching "
                      f"'{args.require_source}'. This arm exists to train on "
                      f"the larger corpus; pointed at the old data directory "
                      f"it would answer a different question and look fine "
                      f"doing it.", file=sys.stderr)
            else:
                print(f"[ok] {name}: {required:,} record(s) from "
                      f"'{args.require_source}'")

    if failed:
        return 1
    print("[ok] corpus respects the held-out partition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
