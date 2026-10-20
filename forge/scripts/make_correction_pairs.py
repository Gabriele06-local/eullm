#!/usr/bin/env python3
"""Correction exercises and DPO pairs from a model's own sampled answers.

    python forge/scripts/make_correction_pairs.py $WORK/eullm_runs/premise/samples-8b.jsonl \\
        --out-sft $WORK/eullm_runs/premise/sft-correzione.jsonl \\
        --out-dpo $WORK/eullm_runs/premise/dpo.jsonl

Reads sample_answers.py output (`eullm_forge.datasets.correction_gen`): for
each verifiable question answered both right and wrong, one DPO pair (its
right answer over its wrong one) and one correction exercise (check the
proposed answer against the texts, say what is wrong, write the right one).
``--right-share`` of the exercises propose a right answer, to be confirmed.

Counts only, never a question or an answer.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.correction_gen import build  # noqa: E402


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", type=Path, help="sample_answers.py output")
    ap.add_argument("--right-share", type=float, default=0.3,
                    help="share of exercises whose proposed answer is right")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-sft", type=Path)
    ap.add_argument("--out-dpo", type=Path)
    args = ap.parse_args(argv)
    if not args.out_sft and not args.out_dpo:
        ap.error("give --out-sft, --out-dpo or both")

    rows = []
    for p in args.samples:
        with p.open(encoding="utf-8") as f:
            rows += [json.loads(ln) for ln in f if ln.strip()]
    rng = random.Random(args.seed)
    corrections, prefs, stats = build(rows, rng, right_share=args.right_share)
    rng.shuffle(corrections)
    rng.shuffle(prefs)
    if args.out_sft:
        write(args.out_sft, corrections)
    if args.out_dpo:
        write(args.out_dpo, prefs)
    print(f"[corr] {len(rows)} questions: {len(corrections)} correction exercises, "
          f"{len(prefs)} DPO pairs; " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
    return 0 if corrections or prefs else 1


if __name__ == "__main__":
    sys.exit(main())
