"""Practical questions: what a lawyer asks, answered from more than one article.

Step 1.1 of the legal-it roadmap (2026-10-14). Every question of the statute
exam (`norm_exam`) is about one article, asked by its number or its heading.
People do not ask that way: they ask "how do I object to...", "what time does
the defendant have to...", and the answer takes two or three provisions, often
of different codes. This builds that exam from the text of the law:

* a **group** is an article and up to ``max_refs`` of the articles its own
  text refers to ("ai sensi dell'articolo 1453", "dell'art. 2043 del codice
  civile") that the corpus holds -- related by the law itself, not by a
  similarity score;
* a teacher model reads the group and writes one practical question that the
  group answers, the articles a correct answer must cite, a reference answer
  that cites them and a one-line rubric (`messages`);
* `parse_practical` refuses what is not usable rather than repairing it: a
  question that names an article (it would be asked by number again), an
  answer that does not cite every article it says is needed, a placeholder;
* `to_item` writes the exam item, ``norm-pratica-<code>-<article>``, so the
  judge, the paired comparison and the reports read it like every other kind.

Which items are held out is decided by a hash of the group's first article,
not by the order of generation, so a second run splits the same way.
"""

from __future__ import annotations

import hashlib
import json
import random
import re

from .abstain import _CITE, _MORE, _key, cited_articles
from .dataset import EvalItem
from .norm_exam import CODE_LABELS, Article, _usable
from .retrieval import named_code

KIND = "pratica"
MAX_ARTICLE_CHARS = 2500

# How each code is named to the teacher, and in the answers it writes.
CODE_NAMES = {
    "codice_civile": ("Codice civile", "c.c."),
    "codice_penale": ("Codice penale", "c.p."),
    "codice_procedura_civile": ("Codice di procedura civile", "c.p.c."),
    "codice_procedura_penale": ("Codice di procedura penale", "c.p.p."),
    "codice_consumo": ("Codice del consumo", "cod. consumo"),
    "costituzione": ("Costituzione", "Cost."),
    "codice_processo_amministrativo": ("Codice del processo amministrativo", "c.p.a."),
    "legge_procedimento_amministrativo": ("Legge n. 241/1990", "l. n. 241/1990"),
    "ricorsi_amministrativi": ("D.P.R. n. 1199/1971", "d.P.R. n. 1199/1971"),
}

SYSTEM = (
    "Sei un avvocato italiano esperto che prepara un esame per un assistente giuridico. "
    "Scrivi domande pratiche come le farebbe un cliente o un collega di studio: una "
    "situazione concreta ma generale, senza nomi di persone, società o luoghi, e senza "
    "citare numeri di articolo. Rispondi solo con un oggetto JSON."
)

TASK = (
    "Testi normativi:\n\n{texts}\n\n"
    "Scrivi UNA domanda pratica a cui si risponde correttamente solo usando questi testi, "
    "possibilmente combinandone due o più. Restituisci un oggetto JSON con questi campi:\n"
    "- \"domanda\": la domanda, 1-3 frasi, senza numeri di articolo né nomi delle norme;\n"
    "- \"articoli\": i numeri tra parentesi quadre ([1], [2], ...) dei testi che una risposta "
    "corretta deve citare, come lista di interi;\n"
    "- \"risposta\": la risposta corretta in 3-6 frasi, che cita ciascuno di quegli articoli "
    "nella forma \"art. N c.c.\" (o il codice giusto);\n"
    "- \"rubrica\": una frase che dice che cosa una risposta deve contenere per essere "
    "corretta (le regole, i termini, le condizioni).\n"
    "Se con questi testi non si può scrivere una domanda pratica sensata, restituisci "
    "{{\"domanda\": \"\"}}."
)

_NAMES_ARTICLE = re.compile(r"\bart(?:icol[oi]|t)?\.?\s*\d", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"\[[A-Z_]+(?:_\d+)?\]")


class PracticalRejected(ValueError):
    """A teacher reply that is not a usable item; ``reason`` is a short code."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def references(a: Article, known: dict[tuple[str, str], Article]) -> list[tuple[str, str]]:
    """(code, article) of the articles ``a``'s own text refers to, that the corpus holds.

    The code is the one named in the 60 characters after the citation, else
    ``a``'s own. In the order they are cited, without repeats or ``a`` itself.
    """
    out: list[tuple[str, str]] = []
    for m in _CITE.finditer(a.text):
        code = named_code(a.text[m.end():m.end() + 60]) or a.code
        nums = [_key(m.group(1), m.group(2))] + [_key(n, s) for n, s in _MORE.findall(m.group(3))]
        for n in nums:
            k = (code, n)
            if k in known and k != (a.code, a.number) and k not in out:
                out.append(k)
    return out


def groups(articles: dict[tuple[str, str], Article], per_code: int, rng: random.Random,
           max_refs: int = 2,
           exclude: frozenset[tuple[str, str]] = frozenset()) -> list[list[Article]]:
    """Up to ``per_code`` groups a code: an article and the articles it refers to.

    Only usable articles (`norm_exam._usable`) with at least one reference to
    another usable article take part; none of them may be in ``exclude`` (the
    articles a model was trained on, `norm_exam.trained_articles`).
    """
    ok = {k: a for k, a in articles.items()
          if a.code in CODE_LABELS and _usable(a) and k not in exclude}
    by_code: dict[str, list[list[Article]]] = {}
    for k in sorted(ok):
        refs = [r for r in references(ok[k], ok)][:max_refs]
        if refs:
            by_code.setdefault(k[0], []).append([ok[k]] + [ok[r] for r in refs])
    out = []
    for code in sorted(by_code):
        pool = by_code[code]
        rng.shuffle(pool)
        out.extend(pool[:per_code])
    return out


def _label(a: Article) -> str:
    return f"{CODE_NAMES[a.code][0]}, art. {a.number}"


def messages(group: list[Article]) -> list[dict]:
    """The chat messages that ask for one practical item about ``group``."""
    texts = "\n\n".join(f"[{i}] {_label(a)}\n{a.text[:MAX_ARTICLE_CHARS]}"
                        for i, a in enumerate(group, 1))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TASK.format(texts=texts)}]


def parse_practical(raw: str, group: list[Article]) -> dict:
    """The item in a teacher reply, checked; raises PracticalRejected."""
    text = re.sub(r"(?s)<think>.*?</think>", "", (raw or "").strip()).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise PracticalRejected("no_json")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise PracticalRejected("bad_json", str(exc)) from None
    if not isinstance(obj, dict):
        raise PracticalRejected("bad_json", "not an object")
    question = str(obj.get("domanda") or "").strip()
    if not question:
        raise PracticalRejected("no_question")
    answer = str(obj.get("risposta") or "").strip()
    rubric = str(obj.get("rubrica") or "").strip()
    if not 30 <= len(question) <= 600:
        raise PracticalRejected("length", "domanda")
    if not 100 <= len(answer) <= 2500 or not 20 <= len(rubric) <= 600:
        raise PracticalRejected("length", "risposta/rubrica")
    if _NAMES_ARTICLE.search(question):
        raise PracticalRejected("names_article")
    if any(_PLACEHOLDER.search(t) for t in (question, answer, rubric)):
        raise PracticalRejected("placeholder")
    idx = obj.get("articoli")
    if (not isinstance(idx, list) or not idx
            or not all(isinstance(i, int) and 1 <= i <= len(group) for i in idx)):
        raise PracticalRejected("bad_articles")
    needed = [group[i - 1] for i in dict.fromkeys(idx)]
    cited = cited_articles(answer)
    missing = [a for a in needed if a.number not in cited]
    if missing:
        raise PracticalRejected("answer_misses_article")
    return {"domanda": question, "risposta": answer, "rubrica": rubric, "articoli": needed}


def heldout(a: Article, share: float) -> bool:
    """Whether the group that starts at ``a`` is held out: a hash, not the draw order."""
    h = int(hashlib.sha256(f"{a.code}-{a.number}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < share


def to_item(group: list[Article], parsed: dict, split: str, teacher: str = "") -> EvalItem:
    """The exam item: graded by its rubric, against the reference and the articles' text."""
    head = group[0]
    needed = parsed["articoli"]
    names = "; ".join(f"art. {a.number} {CODE_NAMES[a.code][1]}" for a in needed)
    rubric = (f"{parsed['rubrica']} Una risposta corretta indica le norme applicabili ({names}): "
              "attribuire a un articolo un contenuto che non ha, o citare un articolo diverso "
              "o inesistente come fondamento, la rende sbagliata; ometterne uno ma dare la regola "
              "giusta la rende al più parziale.")
    texts = "\n\n".join(f"{_label(a)}\n{a.text[:MAX_ARTICLE_CHARS]}" for a in needed)
    reference = f"{parsed['risposta']}\n\nTesti degli articoli necessari:\n{texts}"
    return EvalItem(
        id=f"norm-{KIND}-{head.code}-{head.number}", domain="legal", lang="it",
        question=parsed["domanda"], reference=reference, rubric=rubric,
        category=head.code,
        metadata={"tipo": KIND, "code": head.code, "articolo": head.number,
                  "articoli": [f"{a.code}:{a.number}" for a in needed],
                  "gruppo": [f"{a.code}:{a.number}" for a in group],
                  "split": split, "teacher": teacher})
