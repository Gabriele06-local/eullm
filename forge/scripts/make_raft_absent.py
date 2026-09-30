#!/usr/bin/env python3
"""Add RAFT "absent context" pairs to an open-book training set.

    python forge/scripts/make_raft_absent.py $WORK/eullm_runs/stage3/openbook-v04.jsonl \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl --share 0.2 \\
        --out $WORK/eullm_runs/stage3/openbook-v05.jsonl

Takes ``--share`` of the by-topic grounded pairs, removes their own article
from the retrieved texts and answers "the texts in front of me do not hold
it" (see `openbook_gen.absent_context_pair`). The originals are all kept: the
same question then appears once with its article and once without, which is
what teaches the difference. No model is loaded; it runs on a login node in
a minute.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.openbook_gen import absent_context_pair  # noqa: E402
from eullm_forge.eval import NormIndex  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", type=Path)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--share", type=float, default=0.2,
                    help="fraction of the by-topic grounded pairs to turn (default 0.2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.out.resolve() == args.pairs.resolve():
        print("[raft] --out must differ from the input", file=sys.stderr)
        return 2
    pairs = [json.loads(line) for line in args.pairs.open(encoding="utf-8") if line.strip()]
    topic = [p for p in pairs if p.get("task") == "openbook_grounded"
             and not p.get("named", True)]
    rng = random.Random(args.seed)
    chosen = rng.sample(topic, int(round(len(topic) * args.share)))
    index = NormIndex.from_files(args.norms)
    absent = [a for p in chosen if (a := absent_context_pair(p, index))]
    out = pairs + absent
    rng.shuffle(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[raft] {len(pairs)} pairs, {len(topic)} by topic, {len(absent)} absent-context "
          f"added -> {len(out)} in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
