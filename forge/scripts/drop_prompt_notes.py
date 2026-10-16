#!/usr/bin/env python3
"""GRPO prompts with the missing-article note taken out of a share of them.

    python forge/scripts/drop_prompt_notes.py $WORK/eullm_runs/grpo/prompts-practical-abst.jsonl \\
        --share 0.7 --out $WORK/eullm_runs/grpo/prompts-practical-abst-nn.jsonl

The exam's retrieval and the GRPO prompts open a question about an article
the collection does not hold with "Nota: nella raccolta normativa non è
presente art. N" (`NormIndex.missing_article_note`). RAG Enterprise writes no
such note. On 2026-10-16 a 9B trained on these prompts answered every absent
article of the exam right with the note in front (0:0 against the published
8B) and lost 43 of them inside RAG Enterprise, where the published model,
trained to see from the texts that the article is not there, lost none. The
note had become the signal.

This takes the note out of ``--share`` of the prompts that carry one (which
ones by a hash of the row id, so a second run takes out the same), leaving
the texts and the question as they are: the model has to notice the article
missing from the texts themselves. Rows without a note are written unchanged.
Counts only, never a prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HEAD = "Testi normativi di riferimento:"


def without_note(content: str) -> str | None:
    """The prompt without the note it opens with, or None if it opens with none."""
    if not content.startswith("Nota:") or HEAD not in content:
        return None
    return content[content.index(HEAD):]


def dropped(row_id: str, share: float) -> bool:
    h = int(hashlib.sha256(str(row_id).encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < share


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", type=Path)
    ap.add_argument("--share", type=float, default=0.7,
                    help="share of the noted prompts whose note is taken out")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    noted = taken = total = 0
    tmp = args.out.with_name(args.out.name + ".partial")
    with args.prompts.open(encoding="utf-8") as f, tmp.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            msg = (row.get("prompt") or [{}])[-1]
            bare = without_note(msg.get("content", ""))
            if bare is not None:
                noted += 1
                if dropped(row.get("id", ""), args.share):
                    msg["content"] = bare
                    taken += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print(f"[notes] {total:,} prompts, {noted:,} with a note, taken out of {taken:,} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
