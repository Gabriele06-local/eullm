"""The context gate at answer time: are the retrieved texts enough to answer?

Trained by train_context_gate.py on `eullm_forge.datasets.context_gate`
examples, and read through `dense.Reranker` with the gate's instruction, the
way it was trained. ``prob`` is the probability that the texts hold what
the answer needs; below ``threshold`` (from ``gate.json``, chosen on the
development set) the question is gated.

What a gated question gets is the caller's choice (legal_eval.py
``--gate-mode``): `GATE_ANSWER`, which says the texts do not hold the answer
and what would be needed, without asking the model; or `GATE_NOTE` in front
of the texts, leaving the answer to the model.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

from ..datasets.context_gate import gate_document

GATE_ANSWER = (
    "I testi normativi che ho davanti non contengono la disposizione che serve per rispondere "
    "a questa domanda, quindi non posso darti una risposta affidabile. Per rispondere mi "
    "servirebbe il testo della norma pertinente: se mi indichi l'articolo e la legge, o il "
    "testo della sentenza a cui ti riferisci, posso verificarlo.")
GATE_NOTE = (
    "Nota: i testi riportati potrebbero non contenere la disposizione che risponde alla "
    "domanda. Se non la contengono, dillo e indica che cosa servirebbe per rispondere.")


class ContextGate:
    """A trained gate directory (weights and ``gate.json``)."""

    def __init__(self, model_dir: str | Path, threshold: float | None = None):
        from .dense import Reranker

        meta = json.loads((Path(model_dir) / "gate.json").read_text(encoding="utf-8"))
        self.threshold = float(meta["threshold"] if threshold is None else threshold)
        self.scorer = Reranker(str(model_dir), task=meta["task"], batch_size=1,
                               max_length=int(meta.get("max_length", 3072)))

    def prob(self, question: str, records: list[dict]) -> float:
        """The probability that ``records`` hold what the answer to ``question`` needs."""
        return self.scorer.scores(question, [gate_document(records)])[0]

    def release(self) -> None:
        """Give the GPU back before the model that answers is loaded."""
        self.scorer = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
