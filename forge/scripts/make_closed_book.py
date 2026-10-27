#!/usr/bin/env python3
"""Build the closed-book suite (`eullm_forge.eval.closed_book`) from the legislation.

    python forge/scripts/make_closed_book.py --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --out $WORK/eval/closed-book.jsonl

Asked with no texts (sbatch_norm_exam.slurm without EXAM_NORMS) and scored by
score_closed_book.py. Private like the outside test: the file stays on the
cluster, never in this repository or in a training mix, and this script
prints counts only, never a question. An existing file is kept, so every
model is asked the same items; ``--force`` draws again.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval.closed_book import build_suite  # noqa: E402
from eullm_forge.eval.dataset import save_eval_set  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--per-kind", type=int, default=15, help="items of each kind per code")
    ap.add_argument("--rulings", type=int, default=40, help="made-up rulings asked about")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.out.is_file() and not args.force:
        print(f"[closed-book] {args.out} exists, kept")
        return 0
    records = []
    for p in args.norms:
        with p.open(encoding="utf-8") as f:
            records += [json.loads(line) for line in f if line.strip()]
    items = build_suite(records, per_kind=args.per_kind, n_rulings=args.rulings, seed=args.seed)
    save_eval_set(items, args.out)
    kinds = Counter(it.metadata["tipo"] for it in items)
    print(f"[closed-book] {len(items)} items -> {args.out}: "
          + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())), flush=True)
    return 0 if items else 1


if __name__ == "__main__":
    sys.exit(main())
