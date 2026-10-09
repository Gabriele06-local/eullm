"""Did the model say the texts do not hold the answer, and what did it cite?

Two programmatic checks for the abstention exam (`legal_eval.py --absent`,
closed book, and `scripts/abstain_summary.py`), no judge needed:

* `abstained`: the answer says it cannot answer from the texts in hand --
  the way the absent-article and no-text OPD rows teach it to
  (make_abstain_prompts.py) and the way RAG Enterprise's own prompt says it;
* `unsourced_articles`: article numbers the answer cites that are neither in
  the question nor among the articles it was given. With no texts, every
  article cited beyond the one asked is cited from memory: legal-it-8b,
  asked in a plain chat on 2026-10-08, cited artt. 625, 345, 288 and 49,
  wrongly.

`ABSTAIN` is copied verbatim into `scripts/rag_enterprise_eval.py`, which is
standard library only so it runs where RAG Enterprise runs; a test keeps the
two the same.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Narrow on purpose: on the normal exam an answer read as an abstention is a
# lost answer, so a phrase that a right answer also uses ("se il convenuto
# non è presente", "ove non risulta diversamente") must not match.
ABSTAIN = re.compile(
    r"non (?:ho trovato|trovo)|non contengono|non dispongo|"
    r"non (?:è|sono) present[ei] (?:nei|tra i|nella raccolta)|"
    r"non risulta(?:no)? (?:nei|dai|tra i) (?:testi|documenti)|"
    r"non (?:è|sono) (?:indicat|previst|riportat)[oaie] nei (?:testi|documenti)|"
    r"nei (?:testi|documenti) (?:forniti|disponibili|riportati) non|"
    r"non posso (?:quindi )?(?:dar\w*|fornir\w*|rispondert?\w*)[^.]{0,40}"
    r"(?:certezza|sicur[oa])|"
    r"senza (?:il|i) test[oi] dell[ae] norm|"
    r"no relevant information|informazioni rilevanti", re.IGNORECASE)

_SUFFIXES = "bis|ter|quater|quinquies|sexies|septies|octies|novies|decies"
_NUM = rf"(\d+)(?:[\s-]*({_SUFFIXES}))?\b"
# "art. 54", "articolo 54-bis", and every number of "artt. 1176 e 1375".
_CITE = re.compile(rf"\bart(?:icol[oi]|t)?\.?\s*{_NUM}((?:\s*(?:,|e|ed)\s*\d+"
                   rf"(?:[\s-]*(?:{_SUFFIXES}))?\b)*)", re.IGNORECASE)
_MORE = re.compile(rf"(?:,|\be|\bed)\s*{_NUM}", re.IGNORECASE)


def _key(num: str, suffix: str | None) -> str:
    """As `retrieval._article_key` spells an article: "54", "54-bis"."""
    return f"{int(num)}-{suffix.lower()}" if suffix else str(int(num))


def abstained(answer: str) -> bool:
    """Whether the answer says it cannot answer from the texts it has."""
    return bool(ABSTAIN.search(answer or ""))


def cited_articles(text: str) -> set[str]:
    """Article numbers cited in a text, lists of articles included."""
    out = set()
    for m in _CITE.finditer(text or ""):
        out.add(_key(m.group(1), m.group(2)))
        out.update(_key(n, s) for n, s in _MORE.findall(m.group(3)))
    return out


def unsourced_articles(answer: str, question: str, in_hand: Iterable[str] = ()) -> list[str]:
    """Articles the answer cites that neither the question nor the texts given name."""
    return sorted(cited_articles(answer) - cited_articles(question) - set(in_hand))
