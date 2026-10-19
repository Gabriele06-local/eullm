"""Case law for the legal-it models: rulings, split, cards.

Consiglio di Stato and Corte di cassazione (`courts`).

The rulings are pseudonymised, not anonymous, so they do not go into the
published weights (research report of 2026-10-05). What they give is an
index and the material for training on skills: one structured card per
ruling (`schede`) -- abstract principles, norms applied, outcome, questions
-- written by a teacher model, checked, and free of the facts of the case.
"""

from __future__ import annotations

from .courts import COURTS, Court, court
from .rulings import (
    Ruling,
    attach_meta,
    chunk_kind,
    chunk_ruling_id,
    citation_key,
    cited_number,
    group_key,
    join_chunks,
    load_openga,
    load_rulings,
    ruling_view,
)

__all__ = ["COURTS", "Court", "Ruling", "attach_meta", "chunk_kind", "chunk_ruling_id",
           "citation_key", "cited_number", "court", "group_key", "join_chunks",
           "load_openga", "load_rulings", "ruling_view"]
