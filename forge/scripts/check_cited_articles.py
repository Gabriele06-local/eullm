#!/usr/bin/env python3
"""Does every article an answer cites exist? Checked against the statute corpus, no judge.

    python forge/scripts/check_cited_articles.py \\
        $WORK/eval/devbig-answers/answers-*-hyb.jsonl \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl

The answers come first: --norms takes every path after it.

Every "art. N" an answer cites is given a code -- the one named in the 60
characters after the citation ("art. 360 c.p.c."), else the one the question
names -- and looked up among the articles the corpus holds for that code. An
article the question itself names is left out: an answer saying "art. 3434 does
not exist" names it to deny it. A citation whose code cannot be told is not
counted either way.

This is the check LegalHalBench makes programmatically for fabricated
provisions (Hu et al., COLING 2025), and the one the judge cannot be trusted
with: it reads the answer, not the code. Prints counts only, never an item, so
it is safe on the held-out exams' answers.

--show N prints, per file, the first N flagged citations with the text that
follows them: the way to tell an invented article from an act the corpus does
not hold that the check failed to recognise, before a rate is quoted. It prints
answer text, so it refuses any file whose name says held-out.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import NormIndex  # noqa: E402
from eullm_forge.eval.abstain import _CITE, _MORE, _key, cited_articles  # noqa: E402
from eullm_forge.eval.retrieval import named_code  # noqa: E402

WINDOW = 60
# An act the corpus does not hold, named right after the citation ("art. 5 del
# d.lgs. 165/2001"): the citation is about that act, not the question's code.
_OTHER_ACT = re.compile(r"\b(?:l\.|legge|d\.\s?lgs|d\.\s?p\.\s?r|d\.\s?l\.|decreto|regolamento|"
                        r"direttiva|trattato|tfue|cedu|convenzione)", re.IGNORECASE)


def corpus_articles(index: NormIndex) -> dict[str, set[str]]:
    """Every article the corpus holds, per code."""
    out: dict[str, set[str]] = {}
    for r in index.records:
        code = r.get("code")
        if code:
            out.setdefault(code, set()).update(index.articles_of(r))
    return out


def invented(answer: str, question: str, known: dict[str, set[str]]) -> list[str]:
    """"code art. N" for each article the answer cites that its code does not hold."""
    return [label for label, _ in flagged(answer, question, known)]


def flagged(answer: str, question: str,
            known: dict[str, set[str]]) -> list[tuple[str, str]]:
    """("code art. N", the citation and the text after it) for each invented article."""
    asked = cited_articles(question)
    default = named_code(question)
    out = []
    for m in _CITE.finditer(answer or ""):
        after = answer[m.end():m.end() + WINDOW]
        code = named_code(after)
        if code is None:
            if _OTHER_ACT.search(after[:40]):
                continue
            code = default
        if code not in known:
            continue
        nums = [_key(m.group(1), m.group(2))] + [_key(n, s) for n, s in _MORE.findall(m.group(3))]
        span = " ".join(answer[m.start():m.end() + WINDOW].split())
        out.extend((f"{code} art. {n}", span) for n in nums
                   if n not in asked and n not in known[code])
    return out


def summarize(rows: list[dict], known: dict[str, set[str]]) -> dict:
    flagged = [invented(r.get("answer", ""), r.get("question", ""), known) for r in rows]
    return {"answers": len(rows), "with_invented": sum(bool(f) for f in flagged),
            "invented": sum(len(f) for f in flagged)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="+", type=Path, help="answers files (JSONL)")
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help="print the first N flagged citations of each file (not on held-out)")
    args = ap.parse_args(argv)
    if args.show:
        held = [p.name for p in args.answers if re.search(r"held.?out", p.name, re.IGNORECASE)]
        if held:
            ap.error(f"--show prints answer text; not on held-out answers: {', '.join(held)}")
    known = corpus_articles(NormIndex.from_files(args.norms))
    for p in args.answers:
        rows = [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]
        s = summarize(rows, known)
        share = s["with_invented"] / s["answers"] if s["answers"] else 0.0
        print(f"{p.name}: {s['answers']} answers | citing an article that does not exist "
              f"{s['with_invented']} ({share:.1%}) | such citations {s['invented']}")
        shown = 0
        for r in rows:
            for label, span in flagged(r.get("answer", ""), r.get("question", ""), known):
                if shown >= args.show:
                    break
                print(f"    {label} | {span}")
                shown += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
