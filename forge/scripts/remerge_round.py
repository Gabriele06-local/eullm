#!/usr/bin/env python3
"""Merge a GRPO round's weights again, and those of the rounds it started from.

    python forge/scripts/remerge_round.py $WORK/eullm_runs/grpo/r2l-steps-8b-quote

A round's merged weights are its START plus its adapter, and START is the
merged weights of the round before. When those are gone too, the chain is
walked back through each adapter's own record of the model it was trained on
(``adapter_config.json``, ``base_model_name_or_path``) to the first one still
on disk, then merged forward: each intermediate into a hidden directory
beside the round's merged weights, removed once the next one is written, the
last into the round's ``merged`` (behind its symlink, if it had one).

Written on 2026-10-25, when the scratch cleanup of submit_grpo_round.sh
removed the merged weights of r2l, the best 8B, whose ancestors had been
removed the day before: eleven rounds to merge again, nothing lost but time.
``--dry-run`` prints the chain without merging.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _present(model: Path) -> bool:
    return (model / "config.json").is_file()


def chain_of(round_dir: Path, max_depth: int = 50) -> tuple[Path, list[Path]]:
    """The first start still on disk, and the rounds to merge from it, oldest first."""
    rounds: list[Path] = []
    cur = round_dir
    for _ in range(max_depth):
        cfg = cur / "adapter" / "adapter_config.json"
        if not cfg.is_file():
            raise SystemExit(f"[remerge] no adapter at {cfg.parent}: the chain cannot go on")
        rounds.append(cur)
        base = Path(json.loads(cfg.read_text(encoding="utf-8"))["base_model_name_or_path"])
        if _present(base):
            return base, rounds[::-1]
        # NAME/merged, or the scratch directory it pointed to (.../grpo-merged/NAME)
        if base.name == "merged":
            cur = base.parent
        elif (round_dir.parent / base.name / "adapter").is_dir():
            cur = round_dir.parent / base.name
        else:
            raise SystemExit(f"[remerge] {cur.name} started from {base}, which is gone and is "
                             "not a round's merged weights")
    raise SystemExit(f"[remerge] more than {max_depth} rounds back from {round_dir}")


def _merge(start: str, adapter: str, out: str) -> None:
    from eullm_forge.identity import merge_identity_adapter
    merge_identity_adapter(start, adapter, out)


def main(argv: list[str] | None = None,
         merge: Callable[[str, str, str], None] = _merge) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("round", type=Path, help="the round's directory ($WORK/eullm_runs/grpo/NAME)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    target = args.round / "merged"
    if _present(target):
        print(f"[remerge] {target} is there: nothing to do")
        return 0
    start, rounds = chain_of(args.round)
    print(f"[remerge] from {start}: " + " -> ".join(r.name for r in rounds), flush=True)
    if args.dry_run:
        return 0
    # Through the symlink: the weights go back where they were (the scratch).
    final = Path(os.readlink(target)) if target.is_symlink() else target
    final.mkdir(parents=True, exist_ok=True)
    previous = None
    for i, rnd in enumerate(rounds):
        last = i == len(rounds) - 1
        out = final if last else final.parent / f".remerge-{rnd.name}"
        print(f"[remerge] {i + 1}/{len(rounds)} {rnd.name}", flush=True)
        merge(str(start), str(rnd / "adapter"), str(out))
        if previous is not None:
            shutil.rmtree(previous, ignore_errors=True)
        previous, start = (None if last else out), out
    print(f"[remerge] {target} written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
