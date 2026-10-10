#!/usr/bin/env python3
"""Abstention and articles cited from memory, per answers file and per kind of item.

    python forge/scripts/abstain_summary.py $WORK/eval/abstain-exam/answers-*.jsonl

Reads the answers files of `legal_eval.py`. Those it wrote since the
abstention exam carry ``abstained`` and ``unsourced_articles``. For an older
file abstaining is worked out here, but citing is shown as n/a: the texts it
was asked with are not on file, and an article a text refers to ("ai sensi
dell'art. 1176") is a citation from the text, not from memory. Counted
without them, every by-topic answer that names its article looked invented
(100% on 2026-10-09).

One line per file, then one per kind of item (the ``tipo`` in the exam's
ids, ``norm-<tipo>-<code>-<article>``): how many answers abstain, and how many
cite an article that neither the question nor the texts given name. Which is
good depends on the run: on the open-book exam an abstention is a lost
answer; with ``--absent`` or closed book it is the right one, and so is
citing nothing from memory.

Counts only, never a question or an answer: the files are the held-out exam's.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval.abstain import abstained, unsourced_articles  # noqa: E402


def kind_of(item_id: str) -> str:
    """The exam's ``tipo`` from an item id, "?" for ids of another shape."""
    parts = item_id.split("-")
    return parts[1] if len(parts) > 2 and parts[0] == "norm" else "?"


def summarize(rows: list[dict]) -> dict[str, list[int]]:
    """[items, abstained, citing articles not in hand] per kind, and over all;
    the last is None for a file written before the check."""
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        answer = r.get("answer", "")
        # Both read again with the current checks wherever the file allows,
        # so runs answered before a fix of the checks compare with runs after:
        # abstaining needs only the answer; citing, with no texts given
        # (closed book), only the answer and the question.
        abst = abstained(answer)
        unsourced = r.get("unsourced_articles")
        if unsourced is not None and r.get("context") is None:
            unsourced = unsourced_articles(answer, r.get("question", ""))
        for key in ("all", kind_of(str(r.get("id", "")))):
            t = tally[key]
            t[0] += 1
            t[1] += bool(abst)
            t[2] += bool(unsourced)
    return dict(tally)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="+", type=Path, help="answers-*.jsonl of legal_eval.py")
    args = ap.parse_args(argv)
    for p in args.answers:
        rows = [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]
        if not rows:
            print(f"{p.name}: empty")
            continue
        t = summarize(rows)
        mode = ("absent" if rows[0].get("absent")
                else "closed" if rows[0].get("context") is None else "open")
        for key in ["all"] + sorted(k for k in t if k != "all"):
            n, a, u = t[key]
            head = f"{p.name} [{mode}]" if key == "all" else f"    {key}"
            cited = (f"{u} ({u / n:.1%})" if "unsourced_articles" in rows[0]
                     else "n/a (written before the check)")
            print(f"{head}: {n} items | abstained {a} ({a / n:.1%}) | "
                  f"citing articles not in hand {cited}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
