"""Are the retrieved texts enough to answer? Labelled examples for a context gate.

The gate is a small model in front of the legal one: it reads the question
and the texts retrieval found for it, and says whether those texts hold what
the answer needs. When they do not, the product can search again, ask the
user what is missing, or say the matter is not covered, instead of letting
the model answer from memory. It is the shape of most of the outside traps
no model solved on 2026-10-22: a question outside the collection, a
provision the texts do not contain, a reference that cannot be checked.

Every label is known by construction, never from a judge:

* ``presente`` (sufficient): a grounded open-book question, written from an
  article, with that article among its texts (put in if retrieval missed it);
* ``articolo_assente`` (insufficient): the same question with its article
  taken out of the texts -- the same question on both sides, so the gate
  learns to read the texts and not the wording;
* ``articolo_inesistente`` (insufficient): a question naming an article that
  does not exist (the false-premise generator's);
* ``fuori_ambito`` (insufficient): a question on a matter the collection does
  not cover (`make_offtopic_questions.py`).

The gate scores (question, texts) the way Qwen3-Reranker scores (question,
text), so it starts from the reranker the product already runs, with its own
instruction (`GATE_TASK`), and is fine-tuned on these examples.
"""

from __future__ import annotations

import random
import re

from ..eval.retrieval import label

GATE_TASK = ("Given a question about Italian law and the legal texts retrieved for it, judge "
             "whether the texts contain the provision needed to answer the question")
REASONS = ("presente", "articolo_assente", "articolo_inesistente", "fuori_ambito")
#: Characters of each text the gate reads: three texts fit its context, and
#: the provision that answers is rarely past the first part of an article.
DOC_CHARS = 1500


def gate_document(records: list[dict], max_chars: int = DOC_CHARS) -> str:
    """The retrieved texts as the gate reads them: numbered, labelled, cut."""
    if not records:
        return "(nessun testo trovato)"
    blocks = []
    for i, r in enumerate(records, 1):
        text = " ".join(r.get("text", "").split())
        blocks.append(f"[{i}] {label(r)}\n{text[:max_chars]}")
    return "\n\n".join(blocks)


def example(question: str, records: list[dict], reason: str, key: str) -> dict:
    """One labelled example; ``reason`` says why it is (in)sufficient."""
    if reason not in REASONS:
        raise ValueError(f"reason must be one of {REASONS}")
    return {"question": question, "document": gate_document(records),
            "label": int(reason == "presente"), "reason": reason, "key": key}


def openbook_article(pair: dict) -> tuple[str, str] | None:
    """(code, article) of a grounded open-book pair, from its key ``ob-g-<code>-<n>``."""
    key = str(pair.get("key", ""))
    if pair.get("task") != "openbook_grounded" or not key.startswith("ob-g-"):
        return None
    code, _, number = key[len("ob-g-"):].partition("-")
    number = re.sub(r"-v\d+$", "", number)
    return (code, number) if code and number else None


def question_of(instruction: str) -> str:
    """The question of an open-book prompt (`open_book_prompt`)."""
    return instruction.rsplit("Domanda: ", 1)[-1].strip()


def _is(index, record: dict, code: str, number: str) -> bool:
    # articles_of, not record_articles: a chunk continuing the article has no
    # header of its own and is the article all the same (absent_context_pair).
    return record.get("code") == code and number in index.articles_of(record)


def from_openbook(pair: dict, index, rng: random.Random, k: int = 3) -> list[dict]:
    """The pair's question with its article in the texts, and without it."""
    ref = openbook_article(pair)
    if ref is None:
        return []
    code, number = ref
    question = question_of(pair.get("instruction", ""))
    mine = [r for r in index.records if _is(index, r, code, number)]
    if not question or not mine:
        return []
    pool = index.search(question, k + 6)
    found = pool[:k]
    if not any(_is(index, r, code, number) for r in found):
        found = found[:k - 1]
        found.insert(rng.randrange(len(found) + 1), mine[0])
    without = [r for r in pool if not _is(index, r, code, number)][:k]
    key = str(pair["key"])
    return [example(question, found, "presente", key + "-gate-in"),
            example(question, without, "articolo_assente", key + "-gate-out")]


def from_question(question: str, index, reason: str, key: str, k: int = 3) -> dict:
    """A question whose texts cannot answer it, with what retrieval finds for it."""
    return example(question, index.search(question, k), reason, key)
