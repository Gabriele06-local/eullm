"""The courts whose rulings the case-law tools read, and what differs between them.

The cards, the split and the case-law prompts were written for the
Consiglio di Stato. The Corte di cassazione corpus (ItalGiure 2021-2026,
`docs/corpus-acquisition.md`) arrives in the same chunk format, with
``kind`` "snciv" or "snpen" and ``sentence_id`` "<kind>/<year>/<number>",
so the same tools serve it once they stop saying "Consiglio di Stato":
who writes the card, what a citation looks like, which outcomes a ruling
can have.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Court:
    kind: str            # the chunks' ``kind`` and the sentence_id prefix
    of_name: str         # "del Consiglio di Stato": "Sentenza <of_name>" in a card request
    cite: str            # a citation's head, "Cons. Stato"
    massimario: str      # who the teacher is asked to be
    plural: str          # "sentenze del Consiglio di Stato", in a case-law prompt
    outcomes: tuple[str, ...]
    norm_examples: str   # two citations in the shape wanted in a card
    materia_example: str


CDS = Court(
    kind="cds", of_name="del Consiglio di Stato", cite="Cons. Stato",
    massimario="dell'Ufficio del Massimario della giustizia amministrativa",
    plural="sentenze del Consiglio di Stato",
    outcomes=("accoglimento", "rigetto", "inammissibilità", "improcedibilità",
              "accoglimento parziale", "cessata materia del contendere", "altro"),
    norm_examples="\"art. 21-octies, l. n. 241/1990\" o \"art. 120 c.p.a.\"",
    materia_example="appalti pubblici - esclusione dalla gara",
)

# The Cassazione decides on the law, not the facts: it rejects the appeal,
# declares it inadmissible, or quashes the ruling below, sending the case
# back to a judge of the merits or deciding it itself.
_CASS_OUTCOMES = ("rigetto", "inammissibilità", "cassazione con rinvio",
                  "cassazione senza rinvio", "accoglimento parziale", "estinzione", "altro")

SNCIV = Court(
    kind="snciv", of_name="della Corte di cassazione, sezioni civili", cite="Cass. civ.",
    massimario="dell'Ufficio del Massimario della Corte di cassazione",
    plural="sentenze della Corte di cassazione",
    outcomes=_CASS_OUTCOMES,
    norm_examples="\"art. 2043 c.c.\" o \"art. 360, primo comma, n. 3, c.p.c.\"",
    materia_example="responsabilità contrattuale - onere della prova",
)

SNPEN = Court(
    kind="snpen", of_name="della Corte di cassazione, sezioni penali", cite="Cass. pen.",
    massimario="dell'Ufficio del Massimario della Corte di cassazione",
    plural="sentenze della Corte di cassazione",
    outcomes=_CASS_OUTCOMES,
    norm_examples="\"art. 640 c.p.\" o \"art. 606, comma 1, lett. e), c.p.p.\"",
    materia_example="truffa - elemento soggettivo",
)

COURTS = {c.kind: c for c in (CDS, SNCIV, SNPEN)}


def court(kind: str) -> Court:
    """The court of a chunk ``kind``; a ValueError names the known ones."""
    try:
        return COURTS[kind]
    except KeyError:
        raise ValueError(f"unknown court kind {kind!r}: one of {', '.join(COURTS)}") from None
