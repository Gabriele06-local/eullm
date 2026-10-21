"""Questions built on a false premise, and answers that correct it first.

The LexIA test of 2026-10-18 (a lawyer's 36 trap questions, a report
received on reddit.com) found what our exams had not: legal-it-8b corrected
6 of 18 false premises, and no teacher we measured did much better -- the
best, Qwen3.6-27B, corrected 6 too. Distilling a teacher cannot teach what
the teacher does not do. These need no teacher: the premise is made false by
construction, from the statute itself, so the right answer is known and a
program can check it.

Three kinds, each in the exact prompt the exam and the engine use
(`open_book_prompt`, same retrieval), and with no note saying an article is
missing, as RAG Enterprise asks:

* ``premessa_termine`` -- the question states a deadline the article does
  not: "L'art. 2 della legge n. 241/1990 prevede un termine di 60 giorni:
  ...". The article is among the texts; the answer says the premise is
  wrong, quotes the sentence that states the real deadline and gives it.
* ``premessa_inesistente`` -- the question takes for granted an article that
  is not in the collection: "-bis" of an article that has none, or a number
  past the last. The answer says it is not there and attributes it nothing.
* ``premessa_abrogato`` -- the question treats a repealed article as in
  force. Its text, among the others, says it was repealed; the answer says
  so, quoting it.
* ``premessa_vera`` -- the same deadline question with the RIGHT deadline in
  it. The answer confirms it, quoting the text. Without these a model taught
  only false premises learns to correct everything: on 2026-10-21 one that
  had been through the false premises answered 1 of the 9 LexIA questions
  whose premise was sound. Asked of the same article as its false twin, it
  is the article that decides, not the shape of the question.

The answer reasons in fixed steps before it answers (``fmt="steps"``):
"Verifica della premessa", "Norma applicabile", "Risposta". Each step can be
checked against the texts shown -- the passage number ``[n]``, the quoted
words, the deadline -- which a free reasoning cannot: on 2026-10-19 thinking
cost Qwen3.6-27B 6 of its 16 right trap answers. ``fmt="plain"`` is the same
content in one paragraph, for the ablation that decides between the two.

Every quotation is a literal span of the text the prompt shows, checked when
the pair is built: a pair whose quote is not there is not written.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..eval.norm_exam import (
    CODE_LABELS,
    _deadline_keyword,
    _deadlines,
    _sentence_with,
    all_deadlines,
    articles_from_records,
)
from ..eval.retrieval import named_articles, named_code, open_book_prompt

KINDS = ("premessa_termine", "premessa_inesistente", "premessa_abrogato", "premessa_vera")
FORMATS = ("steps", "plain")

# Deadlines a person might wrongly believe, by unit: the usual ones of
# Italian procedure. The wrong one is drawn from these, never from the
# deadlines the article itself states.
USUAL = {"giorni": [3, 5, 8, 10, 15, 20, 30, 40, 45, 60, 90, 120, 180],
         "mesi": [1, 2, 3, 4, 6, 12, 18, 24, 36],
         "anni": [1, 2, 3, 5, 10, 20],
         "ore": [12, 24, 48, 72]}
_SINGULAR = {"giorni": "giorno", "mesi": "mese", "anni": "anno", "ore": "ora"}

# A quotation longer than this is cut to the words around what it is quoted for.
MAX_QUOTE_CHARS = 300

TERMINE_NAMED = [
    "L'art. {n} {of} prevede un termine di {wrong}: entro quando devo quindi agire?",
    "Mi confermi che il termine previsto dall'art. {n} {of} è di {wrong}?",
    "Il mio avvocato dice che l'art. {n} {of} dà {wrong} di tempo. È così?",
]
TERMINE_TOPIC = [
    "{In}, in materia di «{heading}», il termine è di {wrong}, giusto?",
    "So che {in_}, in materia di «{heading}», ci sono {wrong} di tempo: me lo confermi?",
]
INESISTENTE = [
    "Quali obblighi impone l'art. {n} {of}?",
    "Ho letto che l'art. {n} {of} è stato introdotto di recente: che cosa prevede?",
    "Secondo l'art. {n} {of}, chi è competente a decidere?",
    "Come si applica in concreto l'art. {n} {of}?",
]
ABROGATO = [
    "Che cosa prevede oggi l'art. {n} {of}?",
    "Posso invocare l'art. {n} {of} nel mio caso?",
    "Come si applica l'art. {n} {of}?",
]


@dataclass
class PremisePair:
    """One question on a false premise, its prompt and its answer."""

    key: str
    kind: str
    code: str
    number: str
    question: str
    instruction: str
    output: str
    keywords: list[str] = field(default_factory=list)

    def sft_row(self) -> dict:
        return {"instruction": self.instruction, "output": self.output, "task": self.kind,
                "source": self.code, "key": self.key}

    def grpo_row(self) -> dict:
        """A GRPO prompt, in make_grpo_prompts.py's shape; graded by
        `eullm_forge.rl.rewards` on ``tipo`` and ``keywords``."""
        return {"id": self.key, "prompt": [{"role": "user", "content": self.instruction}],
                "tipo": self.kind, "keywords": self.keywords, "code": self.code,
                "articolo": self.number}


def say(n: int, unit: str) -> str:
    """A deadline as a sentence writes it: "1 anno", "30 giorni"."""
    return f"{n} {_SINGULAR.get(unit, unit) if n == 1 else unit}"


def _flat(text: str) -> str:
    return " ".join(text.split())


def quote_span(sentence: str, anchor: str = "", max_chars: int = MAX_QUOTE_CHARS) -> str:
    """The sentence, or the words around ``anchor`` in it when it is too long."""
    sentence = _flat(sentence).strip(" .;")
    if len(sentence) <= max_chars:
        return sentence
    words = sentence.split()
    at = next((i for i, w in enumerate(words) if anchor and anchor in w), 0)
    lo, hi = max(0, at - 12), min(len(words), at + 13)
    return " ".join(words[lo:hi]).strip(" .;,")


def shown(records: list[dict], max_chars: int = 3000) -> list[str]:
    """The texts the prompt shows, flattened, cut where `open_book_prompt` cuts them."""
    return [_flat(r.get("text", "")[:max_chars]) for r in records]


def wrong_deadline(n: int, unit: str, stated: set[tuple[int, str]],
                   rng: random.Random) -> int | None:
    """A usual deadline of the same unit that the article does not state."""
    pool = [m for m in USUAL.get(unit, []) if m != n and (m, unit) not in stated]
    return rng.choice(pool) if pool else None


def excluded_articles(items) -> set[tuple[str, str]]:
    """(code, article) of every exam item, by metadata and by what its question names.

    The metadata is how our exams record it (``code``/``articolo``, and the
    group of a practical item); the question is the only record an outside
    set has (the LexIA traps name "art. 640 c.p." in the text), so the
    articles a question names in the code it names are left out too.
    """
    out = set()
    for it in items:
        md = it.metadata if hasattr(it, "metadata") else it.get("metadata", {})
        if md.get("code") and md.get("articolo"):
            out.add((md["code"], str(md["articolo"])))
        for ref in md.get("gruppo") or []:
            c, _, a = str(ref).partition(":")
            if c and a:
                out.add((c, a))
        question = it.question if hasattr(it, "question") else it.get("question", "")
        code = named_code(question)
        if code:
            out.update((code, n) for n in named_articles(question))
    return out


def _with_article(index, question: str, code: str, number: str, k: int,
                  rng: random.Random) -> tuple[list[dict], int | None]:
    """What retrieval returns for the question, the article put in if it missed
    it, and the 1-based position of the article's first text."""
    found = index.search(question, k)
    mine = [r for r in index.records if r.get("code") == code
            and number in index.articles_of(r)]
    if mine and not any(r in found for r in mine):
        found = found[:k - 1]
        found.insert(rng.randrange(len(found) + 1), mine[0])
    pos = next((i for i, r in enumerate(found, 1) if r.get("code") == code
                and number in index.articles_of(r)), None)
    return found, pos


def _termine_answer(fmt: str, n: str, of: str, right: str, wrong: str, pos: int,
                    quote: str) -> str:
    if fmt == "plain":
        return (f"La premessa non è corretta: l'art. {n} {of} prevede un termine di {right}, "
                f"non di {wrong} [{pos}]. Il testo dice: «{quote}».")
    return (f"Verifica della premessa: il testo [{pos}] è l'art. {n} {of}, e prevede un "
            f"termine di {right}, non di {wrong}. La premessa della domanda è errata.\n"
            f"Norma applicabile: art. {n} {of} [{pos}]: «{quote}».\n"
            f"Risposta: il termine previsto è di {right}.")


def _vera_answer(fmt: str, n: str, of: str, right: str, pos: int, quote: str) -> str:
    if fmt == "plain":
        return (f"Sì: l'art. {n} {of} prevede proprio un termine di {right} [{pos}]. "
                f"Il testo dice: «{quote}».")
    return (f"Verifica della premessa: il testo [{pos}] è l'art. {n} {of}, e prevede proprio "
            f"un termine di {right}. La premessa della domanda è corretta.\n"
            f"Norma applicabile: art. {n} {of} [{pos}]: «{quote}».\n"
            f"Risposta: sì, il termine previsto è di {right}.")


def _inesistente_answer(fmt: str, n: str, of: str) -> str:
    if fmt == "plain":
        return (f"La domanda parte da un riferimento errato: tra i testi riportati non c'è "
                f"l'art. {n} {of}, e non mi risulta che esista, quindi non posso dirti che "
                "cosa prevede. Verifica il numero dell'articolo e la fonte; se mi indichi "
                "l'argomento, posso cercare la norma pertinente.")
    return (f"Verifica della premessa: tra i testi riportati non c'è l'art. {n} {of}, e non "
            "mi risulta che esista: la domanda parte da un riferimento errato.\n"
            "Norma applicabile: nessuno dei testi riportati è l'articolo citato, quindi non "
            "posso attribuirgli un contenuto.\n"
            f"Risposta: non posso dirti che cosa prevede l'art. {n} {of}. Verifica il numero "
            "dell'articolo e la fonte; se mi indichi l'argomento, posso cercare la norma "
            "pertinente.")


def _abrogato_answer(fmt: str, n: str, of: str, pos: int, quote: str) -> str:
    if fmt == "plain":
        return (f"La premessa non è corretta: l'art. {n} {of} è stato abrogato, come riporta "
                f"il testo [{pos}]: «{quote}». Non si applica più; per la disciplina vigente "
                "occorre la norma che lo ha sostituito, se esiste.")
    return (f"Verifica della premessa: il testo [{pos}] riporta che l'art. {n} {of} è stato "
            f"abrogato: «{quote}». La domanda lo presuppone in vigore.\n"
            f"Norma applicabile: l'art. {n} {of} non è più in vigore; la disciplina vigente è "
            "quella della norma che lo ha sostituito, se esiste.\n"
            f"Risposta: l'art. {n} {of} è stato abrogato, quindi non si applica. Se mi indichi "
            "l'argomento, posso cercare la norma oggi in vigore.")


def _repealed_sentence(article) -> str:
    """The sentence that says THIS article is repealed, or "".

    "Abrogat" alone is not enough: a codifying decree's final article
    ("Sono abrogati: a) ...") is in force and repeals other acts. The
    sentence must name the article, the norm or a disposition as the thing
    repealed, or say it was repealed by a later act.
    """
    pattern = re.compile(
        r"\b(?:art(?:icol[oi])?\.?|norm[ae]|disposizion[ei])\s+"
        r"(?:\d+[\w-]*\s+)?(?:(?:è|e'|sono)\s+stat[oaie]\s+|è|e'|sono|risulta(?:no)?|viene|vengono)?"
        r"\s*abrogat[oaie]\b"
        r"|\babrogat[oaie]\s+(?:dalla\b|dal\b|dall['’]|con\b|ai sensi\b)",
        re.IGNORECASE,
    )
    for s in re.split(r"(?<=[.;])\s+|\n+", article.text):
        if pattern.search(s):
            return s
    if article.heading.strip("()[] ").lower() in ("abrogato", "abrogata"):
        return next((s for s in re.split(r"(?<=[.;])\s+|\n+", article.text)
                     if "abrogat" in s.lower()), "")
    return ""


def _quoted_in(quote: str, records: list[dict], pos: int) -> bool:
    return bool(quote) and quote in shown(records)[pos - 1]


def build_pairs(index, *, per_code: int, rng: random.Random, fmt: str = "steps",
                exclude: set[tuple[str, str]] = frozenset(), k: int = 3,
                bis_share: float = 0.5, true_share: float = 0.5,
                stats: dict | None = None) -> list[PremisePair]:
    """Up to ``per_code`` pairs of each kind for each code, none on an excluded article.

    ``true_share`` of the wrong-deadline questions also get their twin with
    the right deadline (``premessa_vera``).
    """
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}")
    stats = stats if stats is not None else {}

    def skip(reason: str) -> None:
        stats[reason] = stats.get(reason, 0) + 1

    arts = articles_from_records(index.records)
    by_code: dict[str, list] = {}
    for (code, num), a in sorted(arts.items()):
        if code in CODE_LABELS and (code, num) not in exclude:
            by_code.setdefault(code, []).append(a)
    pairs: list[PremisePair] = []
    for code in sorted(by_code):
        of, in_ = CODE_LABELS[code]
        pool = by_code[code]
        live = [a for a in pool if not _repealed_sentence(a) and len(a.text) >= 150]

        timed = [(a, next(iter(d))) for a in live
                 if len(d := _deadlines(a.text)) == 1 and all_deadlines(a.text) == d]
        for a, (n, unit) in rng.sample(timed, min(per_code, len(timed))):
            wrong = wrong_deadline(n, unit, all_deadlines(a.text), rng)
            if wrong is None:
                skip("termine:no_wrong_deadline")
                continue
            values = {"n": a.number, "of": of, "wrong": say(wrong, unit)}
            if a.heading and rng.random() < 0.5:
                q = rng.choice(TERMINE_TOPIC).format(
                    In=in_[0].upper() + in_[1:], in_=in_, heading=a.heading, **values)
            else:
                q = rng.choice(TERMINE_NAMED).format(**values)
            found, pos = _with_article(index, q, code, a.number, k, rng)
            quote = quote_span(_sentence_with(a.text, n, unit), anchor=str(n))
            if pos is None or not _quoted_in(quote, found, pos):
                skip("termine:quote_not_shown")
                continue
            pairs.append(PremisePair(
                key=f"pf-t-{code}-{a.number}", kind="premessa_termine", code=code,
                number=a.number, question=q, instruction=open_book_prompt(q, found),
                output=_termine_answer(fmt, a.number, of, say(n, unit), say(wrong, unit),
                                       pos, quote),
                keywords=[_deadline_keyword(n, unit)]))
            if rng.random() < true_share:
                # The twin, with the deadline the article states: the same
                # template, so only the number tells the two apart.
                tq = q.replace(say(wrong, unit), say(n, unit))
                tfound, tpos = _with_article(index, tq, code, a.number, k, rng)
                if tpos is None or not _quoted_in(quote, tfound, tpos):
                    skip("vera:quote_not_shown")
                    continue
                pairs.append(PremisePair(
                    key=f"pf-v-{code}-{a.number}", kind="premessa_vera", code=code,
                    number=a.number, question=tq, instruction=open_book_prompt(tq, tfound),
                    output=_vera_answer(fmt, a.number, of, say(n, unit), tpos, quote),
                    keywords=[_deadline_keyword(n, unit)]))

        taken = {a.number for a in pool}
        last = max(int(re.match(r"\d+", a.number).group()) for a in pool)
        plain = [a for a in live if a.number.isdigit() and f"{a.number}-bis" not in taken]
        made: set[str] = set()
        for _ in range(per_code):
            if plain and rng.random() < bis_share:
                fake = f"{rng.choice(plain).number}-bis"
            else:
                fake = str(last + rng.randint(30, 900))
            if fake in made or (code, fake) in exclude:
                skip("inesistente:repeat")
                continue
            made.add(fake)
            q = rng.choice(INESISTENTE).format(n=fake, of=of)
            # The same test as the note the exam gives (missing_article_note),
            # without giving the note: is the article named in the collection?
            if not index.missing_article_note(q):
                skip("inesistente:exists")
                continue
            found = index.search(q, k)
            pairs.append(PremisePair(
                key=f"pf-i-{code}-{fake}", kind="premessa_inesistente", code=code,
                number=fake, question=q, instruction=open_book_prompt(q, found),
                output=_inesistente_answer(fmt, fake, of)))

        repealed = [a for a in pool if _repealed_sentence(a)]
        for a in rng.sample(repealed, min(per_code, len(repealed))):
            line = _repealed_sentence(a)
            q = rng.choice(ABROGATO).format(n=a.number, of=of)
            found, pos = _with_article(index, q, code, a.number, k, rng)
            quote = quote_span(line, anchor="brogat")
            if pos is None or not _quoted_in(quote, found, pos):
                skip("abrogato:quote_not_shown")
                continue
            pairs.append(PremisePair(
                key=f"pf-a-{code}-{a.number}", kind="premessa_abrogato", code=code,
                number=a.number, question=q, instruction=open_book_prompt(q, found),
                output=_abrogato_answer(fmt, a.number, of, pos, quote)))
    return pairs
