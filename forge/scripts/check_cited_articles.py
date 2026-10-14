#!/usr/bin/env python3
"""Does every article an answer cites exist? Checked against the statute corpus, no judge.

    python forge/scripts/check_cited_articles.py \\
        $WORK/eval/devbig-answers/answers-*-hyb.jsonl \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl

The answers come first: --norms takes every path after it.

Every "art. N" an answer cites is given a code and looked up among the
articles the corpus holds for it. The code is the one named right before the
citation, as the retrieved sources spell it ("[Codice civile, art. 2043.txt"),
else the first one named in the 60 characters after it and before the next
citation ("art. 360 c.p.c."), else the one the question names. A citation
followed by an act the corpus does not hold ("R.D. 267/1942", "d.lgs.",
"disp. att.") is about that act and is left out. So is one cited without its
code that another code holds ("art. 131-bis" in a civil-law answer): that is a
citation without its code, not an invented article. An article the question
itself names is left out too: an answer saying "art. 3434 does not exist" names
it to deny it.

The first version read the code from the window after the citation, by name
priority, and counted "costituzione in mora" as the Constitution. On the
general models in RAG Enterprise most of a 45-citation sample was its own
error (2026-10-14): a source list "art. 2383] e [artt. 576-583 Codice Penale]"
filed art. 2383 c.c. under the criminal code.

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
# names, without "costituzione": "costituzione in mora", "mancata costituzione
# entro i termini" and "costituzione del debitore" are civil procedure, not the
# Constitution, and were 3 of the first 45 citations sampled (2026-10-14).
_NAMES = sorted([(n, c) for n, c in CODE_NAMES if c != "costituzione"]
                + [("cpp", "codice_procedura_penale"), ("cpc", "codice_procedura_civile"),
                   ("cod civ", "codice_civile"), ("cod pen", "codice_penale")],
                key=lambda nc: -len(nc[0]))
_CODE_OF = dict(_NAMES)
_CONSTITUTION = r"costituzione(?! (?:in|del|delle|dei|di|entro|della (?!repubblica)))|cost"
# An act the corpus does not hold ("art. 5 del d.lgs. 165/2001", "art. 186-bis
# R.D. 267/1942", "art. 161-quater disp. att."): the citation is about that act.
# Words the codes use themselves are not read as another act: "decreto
# ingiuntivo", "regolamento di competenza", "statuto della società",
# "convenzioni matrimoniali".
_OTHER_ACT = (r"legge|l \d|d ?lgs|dpr|d p r|d ?l \d|decreto(?! ingiuntiv)|"
              r"regolamento(?! di (?:competenza|giurisdizione|condominio))|direttiva|trattato|"
              r"tfue|cedu|convenzione(?! matrimonial)|r d|regio decreto|"
              r"disp(?:osizioni)? (?:di |per l )?att\w*|l fall|testo unico|t u|"
              r"statuto(?! (?:della|dell|sociale))|preleggi|"
              r"codice della (?:strada|crisi|navigazione)|ccii|cod nav")
_WHERE = re.compile(r"\b(?:(?P<code>" + "|".join(re.escape(n) for n, _ in _NAMES) + r")"
                    rf"|(?P<cost>{_CONSTITUTION})|(?P<other>{_OTHER_ACT}))\b")
OTHER = "other act"
# The retrieved sources' own spelling, "[Codice civile, art. 2447-octies.txt": a
# code named right BEFORE the citation, which is also where the window after the
# previous citation ends.
_PREFIX = re.compile(r"((?:codice|costituzione|cod\.)[^\[\]();:*\n,]{0,40}?)\s*,\s*$",
                     re.IGNORECASE)


def code_in(text: str) -> str | None:
    """The code, or OTHER, that the text names first; None if it names neither."""
    m = _WHERE.search(f" {normalize_text(text)} ")
    if m is None:
        return None
    if m.group("code"):
        return _CODE_OF[m.group("code")]
    return "costituzione" if m.group("cost") else OTHER


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

    The code is the one named just before the citation ("Codice civile, art. N"),
    else the first one named after it and before the next citation, else the
    question's. An article cited without its code, which the question's code does
    not hold but another does ("art. 131-bis" in a civil-law answer), is not
    counted: it is a citation without its code, not an invented article.
    """
    answer = answer or ""
    asked = cited_articles(question)
    default = named_code(question)
    cites = list(_CITE.finditer(answer))
    out = []
    for i, m in enumerate(cites):
        nxt = cites[i + 1].start() if i + 1 < len(cites) else len(answer)
        p = _PREFIX.search(answer[max(0, m.start() - 50):m.start()])
        code = code_in(p.group(1)) if p else None
        if code is None:
            after = answer[m.end():min(m.end() + WINDOW, nxt)]
            if nxt < m.end() + WINDOW:
                after = _PREFIX.sub("", after)
            code = code_in(after)
        explicit = code is not None
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
