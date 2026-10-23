"""Open-book answers that say which passage they come from, and quote it.

The LexIA test of 2026-10-18 (a report received on reddit.com) found two
things in legal-it-8b's statute answers: none named the passage it came from
with the ``[n]`` the prompt numbers them by, so a reader could not check one;
and what they cited was reworded, not the text. The pairs taught exactly
that: the teacher wrote "Secondo l'art. 1418 del codice civile, ..." and was
never asked for the passage number or for the article's words.

`cite` adds both to an existing grounded pair, by program, with nothing
invented:

* ``[n]`` right after the answer's first citation of its article, where n is
  the passage that is the article, read from the prompt's own labels;
* a closing quotation, «...», of the sentence of that passage that shares
  the most words with the answer -- a literal span of the text shown, so
  the student learns to quote and not to paraphrase.

A pair whose article is not among its passages by label or header, or whose
answer never cites it, is left as it was: nothing is added that cannot be
checked against the prompt.
"""

from __future__ import annotations

import re

from ..eval.norm_exam import CODE_LABELS, split_sentences
from .openbook_gen import _article_number_pattern
from .premise_gen import quote_span

_BLOCK = re.compile(r"^\[(\d+)\] ([^\n]*)\n(.*?)(?=\n\n\[\d+\] |\n\nRispondi alla domanda|\Z)",
                    re.MULTILINE | re.DOTALL)
_WORD = re.compile(r"[a-zà-ù]{4,}")
MIN_SHARED_WORDS = 4


def passages(instruction: str) -> list[tuple[int, str, str]]:
    """(number, label, text) of each passage of an open-book prompt."""
    return [(int(m.group(1)), m.group(2), m.group(3)) for m in _BLOCK.finditer(instruction)]


def article_passage(instruction: str, code: str, number: str) -> tuple[int, str] | None:
    """The passage that is the article: by its label, else by its header."""
    label = code.replace("_", " ")
    numbered = _article_number_pattern(number)
    candidates = [(pos, head, text) for pos, head, text in passages(instruction)
                  if head.startswith(label)]
    # Label first, over every passage: a c.p.a. index chunk also carries
    # "Art. 120 - ..." lines, so the header fallback found the TOC before the
    # article whose label names it, and the pair quoted the index.
    for pos, head, text in candidates:
        if re.search(rf",\s*art\.\s*{numbered}\s*$", head):
            return pos, text
    for pos, _head, text in candidates:
        if re.search(rf"(?im)^\s*(?:art\.\s*)?art(?:icolo)?\.?\s*{numbered}", text):
            return pos, text
    return None


def best_sentence(text: str, answer: str) -> str | None:
    """The sentence of the text sharing the most words with the answer, if any shares enough."""
    said = set(_WORD.findall(answer.lower()))
    best, shared = None, MIN_SHARED_WORDS - 1
    for _, s in split_sentences(text, marks=".;:"):
        flat = " ".join(s.split())
        # a heading is not a sentence, and a passage cut for length ends in […]
        if len(flat) < 40 or "[…]" in flat or re.match(r"(?i)^(?:art\.|articolo)\s*\d", flat):
            continue
        n = len(said & set(_WORD.findall(flat.lower())))
        if n > shared:
            best, shared = flat, n
    return best


def cite(pair: dict) -> dict | None:
    """The pair with ``[n]`` and a quotation added, or None when it cannot be done."""
    key = str(pair.get("key", ""))
    if pair.get("task") != "openbook_grounded" or not key.startswith("ob-g-"):
        return None
    code, _, number = key[len("ob-g-"):].partition("-")
    number = re.sub(r"-v\d+$", "", number)
    found = article_passage(pair["instruction"], code, number)
    answer = pair["output"]
    if not found or re.search(r"\[\d+\]", answer):
        return None
    pos, text = found
    of = CODE_LABELS.get(code, ("", ""))[0]
    m = re.search(rf"\bart(?:icolo|\.)?\s*{_article_number_pattern(number)}"
                  rf"(?:,?\s+{re.escape(of)})?", answer, re.IGNORECASE)
    sentence = best_sentence(text, answer)
    if not m or not sentence:
        return None
    quote = quote_span(sentence)
    if quote not in " ".join(text.split()):
        return None
    out = answer[:m.end()] + f" [{pos}]" + answer[m.end():]
    out = out.rstrip() + f" Il testo [{pos}] dice: «{quote}»."
    return {**pair, "output": out, "cited": pos}
