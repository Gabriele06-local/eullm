#!/usr/bin/env python3
"""Does every article an answer cites exist? Checked against the statute corpus, no judge.

    python forge/scripts/check_cited_articles.py \\
        $WORK/eval/devbig-answers/answers-*-hyb.jsonl \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl

The answers come first: --norms takes every path after it.

Every "art. N" an answer cites is looked up among the articles the corpus
holds for its code. An article whose code is named next to it ("art. 360
c.p.c.", "Codice civile, art. 2043") counts if that code does not hold it; one
cited without its code counts only if no code in the corpus holds it (see
`flagged`). A citation followed by an act the corpus does not hold ("R.D.
267/1942", "d.lgs.", "disp. att.", "codice della nautica") is about that act and
is left out, and so is a retrieved file's name ("art. 392-bis.txt"). An article
the question itself names is left out too: an answer saying "art. 3434 does not
exist" names it to deny it.

The first version read the code from the window after the citation, by name
priority, and counted "costituzione in mora" as the Constitution. On the
general models in RAG Enterprise most of a 45-citation sample was its own
error (2026-10-14): a source list "art. 2383] e [artt. 576-583 Codice Penale]"
filed art. 2383 c.c. under the criminal code. The second still took a code
named anywhere in the next 60 characters as the citation's own -- "artt. 1129,
1130-bis) e del Codice di procedura penale" -- and read "La costituzione oltre i
termini" as the Constitution.

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
from eullm_forge.eval.metrics import normalize_text  # noqa: E402
from eullm_forge.eval.retrieval import CODE_NAMES, named_code  # noqa: E402

WINDOW = 60
# Where the code a citation belongs to is named, earliest first. Retrieval's
# names, without "costituzione": lowercase, the word is almost always civil
# procedure -- "costituzione in mora", "la costituzione oltre i termini",
# "mancata costituzione", a company's "(Costituzione)" -- so the Constitution is
# read only as it is cited (_CONSTITUTION).
_NAMES = sorted([(n, c) for n, c in CODE_NAMES if c != "costituzione"]
                + [("cpp", "codice_procedura_penale"), ("cpc", "codice_procedura_civile"),
                   ("cod civ", "codice_civile"), ("cod pen", "codice_penale")],
                key=lambda nc: -len(nc[0]))
_CODE_OF = dict(_NAMES)
# "della Costituzione", "Cost.", "Costituzione, art. 120", "Costituzione italiana".
_CONSTITUTION = re.compile(r"\bdella\s+Costituzione\b(?!\s+(?:in|del|dei|delle|di|entro|oltre)\b)"
                           r"|\bCost\.|\bCostituzione(?=,?\s+art|\s+italiana|\s+della\s+Repubblica)")
_CONST_MARK = " costrep "
# An act the corpus does not hold ("art. 5 del d.lgs. 165/2001", "art. 186-bis
# R.D. 267/1942", "art. 161-quater disp. att.", "art. 53-bis del codice della
# nautica"): the citation is about that act. Words the codes use themselves are
# not read as another act: "decreto ingiuntivo", "regolamento di competenza",
# "statuto della società", "convenzioni matrimoniali".
_OTHER_ACT = (r"legge|l \d+|d ?lgs|dpr|d p r|d ?l \d+|decreto(?! ingiuntiv)|"
              r"regolamento(?! di (?:competenza|giurisdizione|condominio))|direttiva|trattato|"
              r"tfue|cedu|convenzione(?! matrimonial)|r d|regio decreto|"
              r"disp(?:osizioni)? (?:di |per l )?(?:att|coord|trans)\w*|l fall|testo unico|t u|"
              r"statuto(?! (?:della|dell|sociale))|preleggi|"
              r"codice (?:della|del|dei|delle|dell) \w+|ccii|cod nav")
# Code names come first in the alternation, so at the same place "codice del
# consumo" is the code and not "codice del <something>", another act.
_WHERE = re.compile(r"\b(?:(?P<code>" + "|".join(re.escape(n) for n, _ in _NAMES) + r")"
                    rf"|(?P<cost>{_CONST_MARK.strip()})|(?P<other>{_OTHER_ACT}))\b")
OTHER = "other act"
# The retrieved sources' own spelling, "[Codice civile, art. 2447-octies.txt",
# a list's heading, "Codice penale e norme correlate (artt. 131-bis", and
# "Costituzione art. 120": a code
# named right BEFORE the citation, which is also where the window after the
# previous citation ends.
_PREFIX = re.compile(r"((?:codice|costituzione|cod\.)[^\[\]();:*\n,]{0,40}?)\s*[,(]?\s*$",
                     re.IGNORECASE)
# The window after a citation ends at the end of its item: "art. 2341-bis;
# *Codice penale*" names the code of the next item, not of this one.
_ITEM_END = re.compile(r"[;\]\n]")
# What may stand between a citation and its own code: "art. 360 c.p.c.",
# "art. 521 del Codice Penale", "art. 99 della Costituzione".
_LEAD = re.compile(r"^[\s*_)]*(?:,\s*)?(?:(?:del|dello|della|dell['’])\s*)?[\s*_\[]*",
                   re.IGNORECASE)


def _first(text: str) -> tuple[str | None, int]:
    """The code, or OTHER, that the text names first, and where; (None, -1) if neither."""
    m = _WHERE.search(f" {normalize_text(_CONSTITUTION.sub(_CONST_MARK, text))} ")
    if m is None:
        return None, -1
    if m.group("code"):
        return _CODE_OF[m.group("code")], m.start()
    return ("costituzione" if m.group("cost") else OTHER), m.start()


def code_in(text: str) -> str | None:
    """The code, or OTHER, that the text names first; None if it names neither."""
    return _first(text)[0]


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
    """("code art. N", the citation and the text after it) for each invented article.

    A citation's code is explicit when it is named right before it ("Codice
    civile, art. N") or right after it ("art. N c.c.", "art. N del codice
    penale"). An article with an explicit code is invented if that code does not
    hold it -- a real article under the wrong code is a wrong citation too. Any
    other article is checked against the first code named later in its item, else
    the question's, and counted only if no code holds it: "art. 131-bis" in a
    civil-law answer is a citation without its code, not an invented article. A
    citation spelt as a retrieved file, "art. 392-bis.txt", is a document the
    retrieval returned and is left out.
    """
    answer = answer or ""
    asked = cited_articles(question)
    default = named_code(question)
    cites = list(_CITE.finditer(answer))
    out = []
    for i, m in enumerate(cites):
        if answer.startswith(".txt", m.end()):
            continue
        nxt = cites[i + 1].start() if i + 1 < len(cites) else len(answer)
        p = _PREFIX.search(answer[max(0, m.start() - 50):m.start()])
        code = code_in(p.group(1) + ", art.") if p else None
        explicit = code is not None
        if code is None:
            after = answer[m.end():min(m.end() + WINDOW, nxt)]
            if i + 1 < len(cites) and nxt < m.end() + WINDOW:
                after = _PREFIX.sub("", after)   # the next citation's own code
            after = _CONSTITUTION.sub(_CONST_MARK, _ITEM_END.split(after, 1)[0])
            after = after[_LEAD.match(after).end():]
            code, at = _first(after)
            explicit = code is not None and at <= 1
        code = code or default
        if code == OTHER or code not in known:
            continue
        nums = [_key(m.group(1), m.group(2))] + [_key(n, s) for n, s in _MORE.findall(m.group(3))]
        span = " ".join(answer[m.start():m.end() + WINDOW].split())
        out.extend((f"{code} art. {n}", span) for n in nums
                   if n not in asked and n not in known[code]
                   and (explicit or not any(n in arts for arts in known.values())))
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
