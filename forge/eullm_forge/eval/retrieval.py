"""Find the text of the norm a legal question is about, so the model can read it.

legal-it-4b v0.2, asked closed-book, put the TAR appeal deadline at 30 days
(it is 60) and the ricorso straordinario at 30 (it is 120). Qwen's own 4B
instruct model said article 2043 of the civil code does not exist. A model
of this size does not reliably remember what a given article says, and in
law the number is the answer. The remedy everyone uses is to hand the model
the text instead of hoping it remembers it; this is the retrieval half.

Two ways in, tried in order:

* **By article.** A question that names an article ("art. 2043 c.c.",
  "l'articolo 27 della Costituzione") gets that article, looked up by code
  and number in the records `prepare_legislation.py` writes. Asking by
  number is how lawyers ask, and ranking cannot beat an exact lookup.
* **By words.** Everything else goes to BM25 over the same records. Plain,
  dependency-free, and good enough to answer the question this exists for:
  does reading the norm fix the answers?

Records are the ``legislazione_*.chunks.jsonl`` lines: ``text``, ``code``
(e.g. ``codice_civile``), ``article_num``. Nothing here imports torch.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import normalize_text

# Words that say nothing about which norm is meant. Kept short on purpose:
# BM25's idf already discounts what is common, this only drops the noise.
_STOP = set("""
il lo la i gli le un uno una di a da in con su per tra fra e o ma se che chi
cui non piu del dello della dei degli delle al allo alla ai agli alle dal
dallo dalla dai dagli dalle nel nello nella nei negli nelle sul sullo sulla
sui sugli sulle come quale quali quando dove cosa sono essere ha hanno deve
devono puo possono viene questo questa questi queste quello quella
art artt articolo articoli comma
""".split())

# How a question names a code, normalised, mapped to the ``code`` field.
# Longest names first, so "codice di procedura civile" is not read as
# "codice civile".
CODE_NAMES: list[tuple[str, str]] = [
    ("codice di procedura civile", "codice_procedura_civile"),
    ("codice di procedura penale", "codice_procedura_penale"),
    ("procedura civile", "codice_procedura_civile"),
    ("procedura penale", "codice_procedura_penale"),
    ("codice del consumo", "codice_consumo"),
    ("codice civile", "codice_civile"),
    ("codice penale", "codice_penale"),
    ("costituzione", "costituzione"),
    ("c p c", "codice_procedura_civile"),
    ("c p p", "codice_procedura_penale"),
    ("c c", "codice_civile"),
    ("c p", "codice_penale"),
    ("gdpr", "gdpr"),
    ("2016 679", "gdpr"),
    ("ai act", "ai_act"),
    ("2024 1689", "ai_act"),
]

_ARTICLE = re.compile(r"\bart(?:icolo|icoli|t)?\s+(\d+)(?:\s*(bis|ter|quater|quinquies))?\b")


def tokens(text: str) -> list[str]:
    """Normalised words worth matching on."""
    return [w for w in normalize_text(text).split()
            if w not in _STOP and (len(w) > 2 or w.isdigit())]


def named_code(text: str) -> str | None:
    """The code a question names, if it names one."""
    norm = f" {normalize_text(text)} "
    toks = norm.split()
    for name, code in CODE_NAMES:
        parts = name.split()
        for i in range(len(toks) - len(parts) + 1):
            if toks[i : i + len(parts)] != parts:
                continue
            if all(len(p) == 1 for p in parts):
                # An all-initials abbreviation must not match as the prefix
                # of a longer initial run: "c p" in "c p a" names nothing
                # (there is no amministrativa entry to fall back to, so the
                # lookup refuses and search falls back to honest BM25).
                # One trailing initial is not enough to tell "c p a" from
                # "c p e seguenti", so it still matches.
                following = toks[i + len(parts) : i + len(parts) + 2]
                if (
                    following
                    and len(following[0]) == 1
                    and (len(following) == 1 or len(following[1]) == 1)
                ):
                    continue
            return code
    return None


def named_articles(text: str) -> list[str]:
    """Article numbers a question names, as the records spell them."""
    out = []
    for num, suffix in _ARTICLE.findall(normalize_text(text)):
        out.append(f"{num}-{suffix}" if suffix else num)
    return out


@dataclass
class NormIndex:
    """Article lookup plus BM25 over legislation records."""

    records: list[dict]
    k1: float = 1.5
    b: float = 0.75
    _docs: list[Counter] = field(default_factory=list, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _avg: float = 0.0

    def __post_init__(self) -> None:
        self._docs = [Counter(tokens(r.get("text", ""))) for r in self.records]
        for d in self._docs:
            self._df.update(d.keys())
        self._avg = sum(sum(d.values()) for d in self._docs) / max(1, len(self._docs))

    @classmethod
    def from_files(cls, paths: list[str | Path]) -> NormIndex:
        """Load every record of the given JSONL files."""
        records = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                records.extend(json.loads(line) for line in f if line.strip())
        if not records:
            raise ValueError(f"no legislation records in {[str(p) for p in paths]}")
        return cls(records)

    def by_article(self, question: str) -> list[dict]:
        """Records of the article(s) the question names in the code it names."""
        code = named_code(question)
        nums = named_articles(question)
        if not code or not nums:
            return []
        hits = [r for r in self.records
                if r.get("code") == code and str(r.get("article_num", "")).lower() in nums]
        return sorted(hits, key=lambda r: r.get("chunk_index", 0))

    def bm25(self, question: str, k: int) -> list[dict]:
        """The k records BM25 ranks highest for the question."""
        q = tokens(question)
        n = len(self._docs)
        scores = []
        for i, d in enumerate(self._docs):
            length = sum(d.values())
            s = 0.0
            for w in q:
                tf = d.get(w, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self._df[w] + 0.5) / (self._df[w] + 0.5))
                s += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * length / (self._avg or 1)))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [self.records[i] for _, i in scores[:k]]

    def search(self, question: str, k: int = 3) -> list[dict]:
        """Named article first, then BM25 to fill up to k, without repeats."""
        found = self.by_article(question)[:k]
        seen = {id(r) for r in found}
        for r in self.bm25(question, k):
            if len(found) >= k:
                break
            if id(r) not in seen:
                found.append(r)
                seen.add(id(r))
        return found


def label(record: dict) -> str:
    """How a record is cited in the prompt: code and article."""
    code = (record.get("code") or "").replace("_", " ")
    num = record.get("article_num")
    return f"{code}, art. {num}" if num else code


def open_book_prompt(question: str, records: list[dict], max_chars: int = 1500) -> str:
    """The question with the retrieved texts in front of it.

    Worded like the context tasks of stage 3 — a text, then what to do with
    it — so the model meets the format it was trained on.
    """
    if not records:
        return question
    blocks = [f"[{i}] {label(r)}\n{r.get('text', '')[:max_chars].strip()}"
              for i, r in enumerate(records, 1)]
    return ("Testi normativi di riferimento:\n\n" + "\n\n".join(blocks)
            + "\n\nRispondi alla domanda basandoti sui testi sopra, se sono "
              "pertinenti.\n\nDomanda: " + question)
