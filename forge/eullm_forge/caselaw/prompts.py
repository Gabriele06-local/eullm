"""The prompts of case-law questions: what the student sees, and what the teacher sees.

The student is asked the way it will be used: the question and the texts
retrieval found. The teacher of the privileged-context distillation (step 4
of the research report of 2026-10-05) is asked the same, with the ruling the
question was written from put in front: it answers knowing what the right
source says, and the student learns to answer the same from what it is
shown. Same wording as the statute prompt (`eval.retrieval.open_book_prompt`)
where the two overlap, so the format the models know does not change.
"""

from __future__ import annotations

PASSAGE_CHARS = 3000


def ruling_label(meta: dict) -> str:
    """How a ruling is cited in a prompt: "Cons. Stato, Sezione V, n. 202301234"."""
    parts = ["Cons. Stato", meta.get("sezione") or "", f"n. {meta['numero']}" if meta.get("numero")
             else ""]
    return ", ".join(p for p in parts if p)


def caselaw_prompt(question: str, passages: list[tuple[str, str]]) -> str:
    """The question with the retrieved passages, as (label, text) pairs, in front."""
    if not passages:
        return question
    blocks = []
    for i, (label, text) in enumerate(passages, 1):
        body = text[:PASSAGE_CHARS].rstrip() + (" […]" if len(text) > PASSAGE_CHARS else "")
        blocks.append(f"[{i}] {label}\n{body}")
    return ("Testi di riferimento (sentenze del Consiglio di Stato):\n\n" + "\n\n".join(blocks)
            + "\n\nRispondi alla domanda basandoti sui testi sopra, se sono pertinenti, "
              "citando le sentenze che usi.\n\nDomanda: " + question)


def privileged_prompt(question: str, passages: list[tuple[str, str]], ruling: tuple[str, str],
                      max_chars: int = 12000, citable: bool = True) -> str:
    """The student's prompt with the source ruling in front: the teacher's view.

    ``citable=False`` when retrieval did not find the source: the teacher is
    shown the ruling without its number and told not to cite it. A teacher
    that cites a ruling the student was never shown teaches the student to
    cite from memory -- to invent. On 2026-10-06 the 8B taught by Ministral-3-14B
    went from 84.9% to 65.7% of answers citing only rulings they were given.
    """
    label, text = ruling
    body = text[:max_chars].rstrip() + (" […]" if len(text) > max_chars else "")
    if citable:
        head = f"Sentenza da cui proviene la domanda ({label}):"
    else:
        head = ("Sentenza da cui proviene la domanda (non è tra i testi di riferimento: usala per "
                "sapere che cosa è giusto, ma non citarla):")
    return f"{head}\n{body}\n\n---\n\n" + caselaw_prompt(question, passages)
