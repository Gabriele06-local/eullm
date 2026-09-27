"""An exam built from the text of the law, which nobody working on the models reads.

The ten seed questions found real faults and became the development set: every
fix since has been checked on them, so their score no longer says how a model
does on questions it was not tuned against. A held-out set written by a lawyer
is expensive; one written by ChatGPT has the same failure the models have —
it misremembers article numbers and post-reform deadlines — and a wrong
reference rewards the model that is wrong the same way.

This builds the questions from the legislation files themselves, so every
reference IS the text of an article and cannot be misremembered:

* ``contenuto``   — "Che cosa prevede l'art. N del <codice>?"; reference: the
  article.
* ``termine``     — only articles stating exactly one deadline ("entro
  sessanta giorni"), so the question has one right answer; reference: the
  sentence that states it; keyword: the deadline, in digits or words.
* ``termine_argomento`` — the same deadline asked by the article's heading
  instead of its number, which is how people ask and what retrieval by words
  has to handle.
* ``inesistente`` — an article number past the end of the code. The right
  answer is that it does not exist; a model that describes it is inventing.

The articles are drawn at random with a seed the builder does not print, and
the script that runs it reports counts only, so the exam can live on the
cluster without anyone improving the models against it by reading it.
Nothing here imports torch.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass

from .dataset import EvalItem
from .metrics import normalize_text
from .retrieval import _HEADER, _article_key

# How each code is named in a question, as "of" and "in" — worded so
# `named_code` recognises it.
CODE_LABELS: dict[str, tuple[str, str]] = {
    "codice_civile": ("del codice civile", "nel codice civile"),
    "codice_penale": ("del codice penale", "nel codice penale"),
    "codice_procedura_civile": ("del codice di procedura civile",
                                "nel codice di procedura civile"),
    "codice_procedura_penale": ("del codice di procedura penale",
                                "nel codice di procedura penale"),
    "codice_consumo": ("del codice del consumo", "nel codice del consumo"),
    "costituzione": ("della Costituzione", "nella Costituzione"),
    "codice_processo_amministrativo": ("del codice del processo amministrativo",
                                       "nel codice del processo amministrativo"),
    "legge_procedimento_amministrativo": ("della legge n. 241/1990",
                                          "nella legge n. 241/1990"),
    "ricorsi_amministrativi": ("del d.P.R. n. 1199/1971", "nel d.P.R. n. 1199/1971"),
}
AMMINISTRATIVO = {"codice_processo_amministrativo", "legge_procedimento_amministrativo",
                  "ricorsi_amministrativi"}

_NUMBER_WORDS = {
    "un": 1, "uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6,
    "sette": 7, "otto": 8, "nove": 9, "dieci": 10, "undici": 11, "dodici": 12,
    "quindici": 15, "venti": 20, "ventiquattro": 24, "trenta": 30, "quaranta": 40,
    "quarantacinque": 45, "cinquanta": 50, "sessanta": 60, "settanta": 70,
    "novanta": 90, "centoventi": 120, "centocinquanta": 150, "centottanta": 180,
    "trecentosessantacinque": 365,
}
_WORD_FOR = {v: k for k, v in _NUMBER_WORDS.items() if k not in ("un", "una")}
_UNITS = {"giorno": "giorni", "giorni": "giorni", "mese": "mesi", "mesi": "mesi",
          "anno": "anni", "anni": "anni", "ora": "ore", "ore": "ore"}
_DEADLINE = re.compile(
    r"\b(?:entro|nel termine(?: perentorio| di decadenza)? di|non oltre|decorsi|"
    r"nei|trascorsi)\s+(?:il termine (?:perentorio |di decadenza )?di\s+)?"
    r"(\d+|[a-z]+)\s+(giorni|giorno|mesi|mese|anni|anno|ore)\b", re.IGNORECASE)


@dataclass
class Article:
    """One article, reassembled from however the file chunked it."""

    code: str
    number: str
    text: str

    @property
    def heading(self) -> str:
        """The rubrica, when the text carries one right after the header."""
        lines = [ln.strip() for ln in self.text.splitlines() if ln.strip()]
        for ln in lines[:3]:
            m = re.fullmatch(r"\(+\s*(.+?)\s*\)+\.?", ln)
            if m and len(m.group(1)) < 120:
                return m.group(1)
            m = re.search(r"\(\(\s*(.+?)\s*\)\)", ln)
            if m and len(m.group(1)) < 120:
                return m.group(1)
        return ""


def articles_from_records(records: list[dict]) -> dict[tuple[str, str], Article]:
    """Split legislation records into whole articles, keyed by (code, number).

    A record may hold one article, the continuation of the previous one, or
    several (the c.p.a. chunks). Text before the first header of a record
    continues the article before it. A number that turns up twice in one
    code, not as a continuation — the allegati of the c.p.a. restart their
    numbering — is ambiguous and dropped: a question about it has no single
    right answer.
    """
    parts: dict[tuple[str, str], list[str]] = defaultdict(list)
    ambiguous: set[tuple[str, str]] = set()
    last: dict[str, str] = {}
    for r in records:
        code, text = r.get("code") or "", r.get("text", "")
        marks = list(_HEADER.finditer(text))
        lead = text[: marks[0].start()] if marks else text
        if lead.strip() and code in last and r.get("chunk_index", 0):
            parts[(code, last[code])].append(lead)
        for i, m in enumerate(marks):
            key = (code, _article_key(m.group(1), m.group(2) or ""))
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            if parts.get(key) and last.get(code) != key[1]:
                ambiguous.add(key)
            parts[key].append(text[m.start():end])
            last[code] = key[1]
    return {k: Article(k[0], k[1], "\n".join(v).strip())
            for k, v in parts.items() if k not in ambiguous}


def _deadlines(text: str) -> set[tuple[int, str]]:
    found = set()
    for num, unit in _DEADLINE.findall(text):
        n = int(num) if num.isdigit() else _NUMBER_WORDS.get(num.lower())
        if n:
            found.add((n, _UNITS[unit.lower()]))
    return found


def _deadline_keyword(n: int, unit: str) -> str:
    alts = [f"{n} {unit}"]
    if n in _WORD_FOR:
        alts.append(f"{_WORD_FOR[n]} {unit}")
    return "|".join(alts)


def _sentence_with(text: str, n: int) -> str:
    """The sentence of the article that states the deadline."""
    alts = [str(n)] + ([_WORD_FOR[n]] if n in _WORD_FOR else [])
    pat = re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.IGNORECASE)
    for sentence in re.split(r"(?<=[.;])\s+", " ".join(text.split())):
        if pat.search(sentence):
            return sentence.strip()
    return " ".join(text.split())[:400]


def _usable(a: Article) -> bool:
    head = normalize_text(a.text[:300])
    return len(a.text) >= 150 and "abrogat" not in head


def build_exam(records: list[dict], per_code: int = 10, seed: int | None = None,
               codes: set[str] | None = None) -> list[EvalItem]:
    """Draw the exam: for each code, up to ``per_code`` items of each kind.

    ``seed`` None means a random one, which is the point: the builder must
    not be able to reproduce the draw from anything it can see.
    """
    rng = random.Random(seed if seed is not None else random.SystemRandom().random())
    arts = articles_from_records(records)
    by_code: dict[str, list[Article]] = defaultdict(list)
    for a in arts.values():
        if a.code in CODE_LABELS and (codes is None or a.code in codes) and _usable(a):
            by_code[a.code].append(a)

    items: list[EvalItem] = []
    for code in sorted(by_code):
        pool = by_code[code]
        of, in_ = CODE_LABELS[code]
        vertical = "amministrativo" if code in AMMINISTRATIVO else "civile_penale"

        def item(kind, a_num, question, reference, keywords, rubric, _code=code,
                 _vertical=vertical):
            return EvalItem(
                id=f"norm-{kind}-{_code}-{a_num}", domain="legal", lang="it",
                category=_code, question=question, reference=reference, rubric=rubric,
                keywords=keywords,
                metadata={"tipo": kind, "code": _code, "articolo": a_num,
                          "vertical": _vertical, "fonte": "testo di legge"})

        for a in rng.sample(pool, min(per_code, len(pool))):
            ref = " ".join(a.text.split())[:800]
            items.append(item(
                "contenuto", a.number, f"Che cosa prevede l'art. {a.number} {of}?", ref, [],
                "Corretto se riporta il contenuto essenziale dell'articolo di riferimento; "
                "sbagliato se descrive un altro articolo o ne inventa il contenuto."))

        timed = [(a, next(iter(d))) for a in pool if len(d := _deadlines(a.text)) == 1]
        for a, (n, unit) in rng.sample(timed, min(per_code, len(timed))):
            kw = [_deadline_keyword(n, unit)]
            ref = _sentence_with(a.text, n)
            rub = f"Corretto solo se indica il termine di {n} {unit}."
            items.append(item("termine", a.number,
                              f"Quale termine prevede l'art. {a.number} {of}?",
                              ref, kw, rub))
            if a.heading:
                items.append(item("termine_argomento", a.number,
                                  f"{in_[0].upper()}{in_[1:]}, in materia di "
                                  f"«{a.heading}», qual è il termine previsto?",
                                  ref, kw, rub))

        last = max(int(re.match(r"\d+", a.number).group()) for a in pool)
        for _ in range(max(1, per_code // 5)):
            fake = last + rng.randint(50, 900)
            it = item(
                "inesistente", str(fake), f"Che cosa prevede l'art. {fake} {of}?",
                f"Non esiste l'art. {fake} {of}.",
                # Only ways of saying the article is not there. "non contiene"
                # and "non prevede un" are satisfied by an answer that invents
                # the article's content and then hedges, which the rubric calls
                # wrong.
                ["non esiste|inesistente|non è previsto"],
                "Corretto solo se dice che l'articolo non esiste; sbagliato se ne "
                "descrive un contenuto.")
            # What the draw assumed, kept in the metadata so the assumption is
            # auditable: `last` is the end of the corpus this exam was built
            # from, which is the end of the code only if the corpus is whole.
            # The reference does not state it, because the builder cannot know
            # it -- a corpus that stops at art. 120 once produced "l'art. 969
            # non esiste: la numerazione arriva all'art. 120", and art. 969
            # exists.
            it.metadata["last_article"] = last
            items.append(it)
    unique: dict[str, EvalItem] = {}
    for it in items:
        unique.setdefault(it.id, it)
    return list(unique.values())


def retrieval_hits(items: list[EvalItem], index, k: int = 3) -> dict[str, dict[str, float]]:
    """How often the retrieval puts the item's own article in the top 1 and top k.

    Measured on the drawn exam rather than on the ten development questions,
    so an improvement to retrieval has to hold on articles nobody picked.
    ``inesistente`` items have no article to find and are skipped.
    """
    from .retrieval import record_articles

    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for it in items:
        kind, code, art = (it.metadata.get(x) for x in ("tipo", "code", "articolo"))
        if kind == "inesistente":
            continue
        found = index.search(it.question, k)
        ok = [r.get("code") == code and art in record_articles(r) for r in found]
        t = tally[kind]
        t[0] += 1
        t[1] += bool(ok[:1] and ok[0])
        t[2] += any(ok)
    return {kind: {"n": n, "top1": a / n, f"top{k}": b / n}
            for kind, (n, a, b) in sorted(tally.items())}
