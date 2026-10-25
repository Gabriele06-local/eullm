#!/usr/bin/env python3
"""GRPO prompts made elsewhere, kept off the articles of this site's tests.

    python forge/scripts/filter_grpo_prompts.py $DATA/grpo/leonardo/*.jsonl \\
        --exclude-exam $DATA/eval/trap-test.jsonl $DATA/eval/norm-exam*.jsonl \\
            $P/practical-dev.jsonl $P/lexia-trappole.jsonl \\
        --out $RUNS/grpo/<name>/prompts.jsonl

The GRPO prompts of Leonardo's rounds (make_grpo_prompts.py,
make_practical_prompts.py, make_premise_pairs.py --out-grpo) leave out the
articles of Leonardo's exams, not those of the tests drawn on JUPITER: the
390-question trap test (make_trap_test.py) was drawn there, on articles the
JUPITER premise pairs had not trained on, and nothing kept Leonardo's
prompts off them. Here every prompt whose article is one of an exam's, by
its metadata or by what its question names (`premise_gen.excluded_articles`,
as make_trap_test.py does), is dropped; prompts read from more than one file
are kept once, by id, and written shuffled (seed 0), so the first N of the
file (grpo_train.py --limit, the smoke test) hold every kind.

The missing-article note is taken out of ``--note-share`` of the prompts that
open with one, as drop_prompt_notes.py does on Leonardo (same hash, same
share, so a file it already went through comes out the same): the model is
asked without the note, as the product asks.

Counts only, never a prompt: the exams may be private.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drop_prompt_notes import dropped, without_note  # noqa: E402

from eullm_forge.datasets.premise_gen import excluded_articles  # noqa: E402
from eullm_forge.eval import load_eval_set  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", nargs="+", type=Path, help="GRPO prompts files")
    ap.add_argument("--exclude-exam", nargs="*", type=Path, default=[],
                    help="eval sets whose articles no prompt may be on")
    ap.add_argument("--note-share", type=float, default=0.7,
                    help="share of the noted prompts whose note is taken out")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    held_out = excluded_articles([it for p in args.exclude_exam for it in load_eval_set(p)])
    seen: set[str] = set()
    kept: list[dict] = []
    dropped_exam = duplicates = noted = taken = 0
    for path in args.prompts:
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("id", ""))
            if rid in seen:
                duplicates += 1
                continue
            seen.add(rid)
            if (row.get("code"), str(row.get("articolo"))) in held_out:
                dropped_exam += 1
                continue
            msg = (row.get("prompt") or [{}])[-1]
            bare = without_note(msg.get("content", ""))
            if bare is not None:
                noted += 1
                if dropped(rid, args.note_share):
                    msg["content"] = bare
                    taken += 1
            kept.append(row)
    if not kept:
        print("[grpo-prompts] no prompt left", file=sys.stderr)
        return 1
    random.Random(0).shuffle(kept)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    kinds = ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["tipo"] for r in kept).items()))
    print(f"[grpo-prompts] {len(kept):,} kept ({kinds}); {dropped_exam:,} on an exam's "
          f"article and {duplicates:,} repeated dropped ({len(held_out):,} exam articles); "
          f"note taken out of {taken:,} of {noted:,} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
