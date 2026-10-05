#!/usr/bin/env python3
"""Inventory of the Consiglio di Stato corpus and its development split.

    python forge/scripts/cds_split.py \\
        --chunks $WORK/datasets/legal_it_amm/train.jsonl $WORK/datasets/legal_it_amm/val.jsonl \\
        --openga $WORK/eval/openga/cds-sentenze-20*.csv \\
        --out-dir $WORK/eval/cds

Step 0 of the case-law plan (research report of 2026-10-05). Two things:

1. **Inventory**, printed: rulings, chunks, characters and an estimate of
   tokens, rulings per year and per section, how many match the OpenGA
   index. Counts only, never a ruling.
2. **The development split**: about ``--dev`` rulings of 2017-2024 that no
   training step may use, written as ``cds-dev-ids.txt`` with every other
   ruling of those years in ``cds-train-ids.txt``. Chosen by whole appeal
   (NRG): rulings on the same appeal restate the same facts, so they go to
   the same side. Stratified by year and section, so the dev set looks like
   the corpus. ``cds-split.json`` records the counts and the SHA-256 of both
   lists: a dev set is frozen when its hash is written down.

2025-2026 rulings are the final test and never belong to either list; one
found in the chunk files is counted and reported, since the training corpus
should hold none (check_corpus_holdout.py).

A second run with the same inputs and seed writes the same lists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.caselaw import attach_meta, group_key, load_openga, load_rulings  # noqa: E402

LAST_TRAINING_YEAR = 2024


def choose_dev(rulings: dict, target: int, seed: int) -> set[str]:
    """Whole appeals, stratified by (year, section), until ``target`` rulings."""
    groups: dict[str, list[str]] = defaultdict(list)
    for r in rulings.values():
        groups[group_key(r)].append(r.id)
    strata: dict[tuple, list[str]] = defaultdict(list)
    for key, ids in sorted(groups.items()):
        first = rulings[min(ids)]
        strata[(first.year or 0, first.section)].append(key)
    rng = random.Random(seed)
    for keys in strata.values():
        rng.shuffle(keys)
    total = sum(len(v) for v in groups.values())
    dev: set[str] = set()
    # Each stratum gives its share of the target, then a round-robin tops up.
    for _, keys in sorted(strata.items()):
        share = sum(len(groups[k]) for k in keys) / total
        quota = round(target * share)
        while keys and quota > 0:
            k = keys.pop()
            dev.update(groups[k])
            quota -= len(groups[k])
    pool = [k for s in sorted(strata) for k in strata[s]]
    rng.shuffle(pool)
    while len(dev) < target and pool:
        dev.update(groups[pool.pop()])
    return dev


def digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", nargs="+", type=Path, required=True,
                    help="chunk files holding the rulings (kind cds)")
    ap.add_argument("--openga", nargs="*", type=Path, default=[],
                    help="OpenGA cds-sentenze CSVs: appeal number, section, outcome")
    ap.add_argument("--dev", type=int, default=1300, help="development rulings to set aside")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    rulings = load_rulings(args.chunks)
    if not rulings:
        print("[cds] no cds rulings in the chunk files", file=sys.stderr)
        return 1
    matched = attach_meta(rulings, load_openga(args.openga)) if args.openga else 0
    chars = sum(len(r.text) for r in rulings.values())
    print(f"[cds] {len(rulings):,} rulings from {sum(r.n_chunks for r in rulings.values()):,} "
          f"chunks, {chars / 1e6:,.0f} M characters (~{chars / 3.5 / 1e6:,.0f} M tokens)")
    print(f"[cds] OpenGA metadata for {matched:,} of {len(rulings):,}"
          + ("" if args.openga else " (no --openga: grouping by ruling, not by appeal)"))
    by_year = Counter(r.year for r in rulings.values())
    print("[cds] by year: " + ", ".join(f"{y}: {n:,}" for y, n in sorted(
        by_year.items(), key=lambda x: (x[0] is None, x[0] or 0))))
    sections = Counter(r.section or "?" for r in rulings.values())
    print("[cds] by section: " + ", ".join(f"{s}: {n:,}" for s, n in sections.most_common(10)))
    late = sum(n for y, n in by_year.items() if y and y > LAST_TRAINING_YEAR)
    if late:
        print(f"[cds] WARNING: {late:,} rulings after {LAST_TRAINING_YEAR} in the chunk files: "
              "left out of both lists, and the training corpus should not hold them",
              file=sys.stderr)
    eligible = {k: r for k, r in rulings.items() if r.year and r.year <= LAST_TRAINING_YEAR}
    groups = {group_key(r) for r in eligible.values()}
    print(f"[cds] {len(eligible):,} rulings of 2017-{LAST_TRAINING_YEAR} in {len(groups):,} "
          "appeal groups")

    dev = sorted(choose_dev(eligible, args.dev, args.seed))
    train = sorted(set(eligible) - set(dev))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cds-dev-ids.txt").write_text("\n".join(dev) + "\n", encoding="utf-8")
    (args.out_dir / "cds-train-ids.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    manifest = {"dev": len(dev), "train": len(train), "seed": args.seed,
                "dev_sha256": digest(dev), "train_sha256": digest(train),
                "openga_matched": matched, "rulings": len(rulings),
                "dev_by_year": dict(sorted(Counter(str(rulings[i].year) for i in dev).items()))}
    (args.out_dir / "cds-split.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                                 encoding="utf-8")
    print(f"[cds] dev {len(dev):,} rulings, train {len(train):,} -> {args.out_dir} "
          f"(dev sha256 {manifest['dev_sha256'][:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
