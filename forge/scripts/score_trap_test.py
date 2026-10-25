#!/usr/bin/env python3
"""The trap test graded without a judge, by kind (make_trap_test.py's set).

    python forge/scripts/score_trap_test.py $DATA/eval/trap-test.jsonl \\
        $RUNS/teacher-eval/answers-trap-test-*.jsonl

For every answers file `legal_eval.py` wrote on the trap test, each answer is
checked by `eullm_forge.rl.rewards.score_answer`, the check the GRPO rewards
use: did it correct the false premise, keep the true one, say the article
does not exist. Prints one row per model, the share right of each kind and
in all, the false premises and their true twins apart (a model that refuses
everything passes the first and fails the second). The judge grades the same
answers (sbatch_judge.slurm); where the two disagree, the answers are read
before either decides. Counts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.rl import score_answer  # noqa: E402

TRUE_KINDS = {"premessa_vera", "premessa_sentenza_vera"}


def label_of(path: Path) -> str:
    name = path.name
    return name[len("answers-"):-len(".jsonl")] if name.startswith("answers-") else path.stem


def score_file(path: Path, items: dict[str, dict]) -> dict[str, list[float]]:
    by_kind: dict[str, list[float]] = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        md = items.get(row.get("id"), {})
        if not md:
            continue
        try:
            s = score_answer(row.get("answer", ""), md["tipo"], md.get("keywords") or [])
        except ValueError:
            continue
        by_kind[md["tipo"]].append(s)
    return by_kind


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", type=Path, help="the trap test (make_trap_test.py --out)")
    ap.add_argument("answers", nargs="+", type=Path, help="answers-*.jsonl of legal_eval.py")
    args = ap.parse_args(argv)
    items = {}
    for line in args.items.open(encoding="utf-8"):
        it = json.loads(line)
        items[it["id"]] = it["metadata"]
    rows = []
    for path in args.answers:
        if path.name.endswith(".graded.jsonl"):
            continue
        by_kind = score_file(path, items)
        if by_kind:
            rows.append((label_of(path), by_kind))
    if not rows:
        print("[trap-test] no answers on the trap test")
        return 1
    kinds = sorted({k for _, bk in rows for k in bk})
    short = {k: k.replace("premessa_", "") for k in kinds}
    print(f"{'model':<34} {'all':>6} {'false':>6} {'true':>6}  " +
          " ".join(f"{short[k][:10]:>10}" for k in kinds))
    for label, bk in sorted(rows):
        every = [s for v in bk.values() for s in v]
        false = [s for k, v in bk.items() if k not in TRUE_KINDS for s in v]
        true = [s for k, v in bk.items() if k in TRUE_KINDS for s in v]
        pct = lambda v: f"{100 * sum(v) / len(v):5.1f}%" if v else "     -"  # noqa: E731
        print(f"{label:<34} {pct(every):>6} {pct(false):>6} {pct(true):>6}  " +
              " ".join(f"{pct(bk.get(k, [])):>10}" for k in kinds))
    n = defaultdict(int)
    for md in items.values():
        n[md["tipo"]] += 1
    print("questions: " + ", ".join(f"{short.get(k, k)} {n[k]}" for k in sorted(n)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
