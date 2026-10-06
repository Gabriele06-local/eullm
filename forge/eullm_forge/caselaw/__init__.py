"""Case law (Consiglio di Stato) for the legal-it models: rulings, split, cards.

The rulings are pseudonymised, not anonymous, so they do not go into the
published weights (research report of 2026-10-05). What they give is an
index and the material for training on skills: one structured card per
ruling (`schede`) -- abstract principles, norms applied, outcome, questions
-- written by a teacher model, checked, and free of the facts of the case.
"""

from __future__ import annotations

from .rulings import (
    Ruling,
    attach_meta,
    group_key,
    join_chunks,
    load_openga,
    load_rulings,
    ruling_view,
)

__all__ = ["Ruling", "attach_meta", "group_key", "join_chunks", "load_openga", "load_rulings",
           "ruling_view"]
