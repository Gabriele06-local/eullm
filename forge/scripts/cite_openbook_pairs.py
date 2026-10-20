#!/usr/bin/env python3
"""Open-book pairs whose answers name their passage ``[n]`` and quote it.

    python forge/scripts/cite_openbook_pairs.py $WORK/eullm_runs/stage3/openbook-v05.jsonl \\
        --out $WORK/eullm_runs/stage3/openbook-v06.jsonl

Every grounded pair gets the passage number after its first citation of the
article and a closing literal quotation of that passage
(`eullm_forge.datasets.openbook_cite`); every other row is copied unchanged.
A grounded pair that cannot be cited by program is kept as it was and
counted. Counts only, never a pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.openbook_cite import cite  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", type=Path, help="stage-3 SFT files")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    stats: Counter = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as out:
        for p in args.pairs:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("task") == "openbook_grounded":
                        cited = cite(row)
                        stats["grounded:cited" if cited else "grounded:left_as_is"] += 1
                        row = cited or row
                    else:
                        stats["other"] += 1
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print("[cite] " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items()))
          + f" -> {args.out}")
    return 0 if stats["grounded:cited"] else 1


if __name__ == "__main__":
    sys.exit(main())
