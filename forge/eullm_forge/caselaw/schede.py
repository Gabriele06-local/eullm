"""One card per ruling: what a teacher model distils out of a Consiglio di Stato ruling.

The card is the unit everything downstream uses (research report of
2026-10-05, step 1): its one-line form prefixes the ruling's chunks in the
index, it is a second retrieval unit, its questions are the prompts of the
on-policy training, and its principles are the only thing derived from the
rulings that may ever reach the published weights. So it must hold the law
and not the case: principles stated in the abstract, with no party, place
or fact that identifies the dispute.

What the model is NOT asked for is what the index already knows exactly:
number, section, date and appeal number come from OpenGA, so they are right
by construction instead of 99% of the time.

`parse_card` refuses a card rather than repairing it: a pseudonym
placeholder, a tax code or an IBAN in a card means the case leaked into it.
"""

from __future__ import annotations

import json
import re

from ..datasets.anonymize import RE_CF, RE_IBAN

OUTCOMES = ("accoglimento", "rigetto", "inammissibilità", "improcedibilità",
            "accoglimento parziale", "cessata materia del contendere", "altro")

SYSTEM = (
    "Sei un magistrato dell'Ufficio del Massimario della giustizia amministrativa. "
    "Estrai dalle sentenze i principi di diritto in forma astratta, come una massima: "
    "mai nomi di parti, persone, società, enti locali specifici, luoghi o fatti che "
    "identifichino la controversia. Scrivi in italiano giuridico corretto. Rispondi "
    "solo con un oggetto JSON."
)

TASK = (
    "Sentenza del Consiglio di Stato{where}:\n\n--- SENTENZA ---\n{text}\n--- FINE ---\n\n"
    "Restituisci un oggetto JSON con questi campi:\n"
    "- \"principi\": da 1 a 4 principi di diritto affermati dalla sentenza, ciascuno in 1-3 "
    "frasi, in forma astratta come una massima ufficiale (niente parti, luoghi, date dei fatti, "
    "importi).\n"
    "- \"norme\": le disposizioni applicate o interpretate, nella forma \"art. 21-octies, l. n. "
    "241/1990\" o \"art. 120 c.p.a.\" (al massimo 8).\n"
    "- \"esito\": uno tra {outcomes}.\n"
    "- \"materia\": la materia in poche parole (per esempio \"appalti pubblici - esclusione "
    "dalla gara\").\n"
    "- \"domande_ricerca\": da 3 a 5 domande che un avvocato potrebbe porre e a cui questa "
    "sentenza risponde, in termini generali e senza riferimenti al caso.\n"
    "- \"domande_esame\": da 2 a 3 oggetti {{\"domanda\", \"risposta\", \"rubrica\"}}: la "
    "domanda è generale, la risposta (2-4 frasi) espone il principio della sentenza, la rubrica "
    "dice in una frase che cosa una risposta deve contenere per essere corretta.\n"
)

_NORM = re.compile(
    r"\bart[t]?\.|\bd\.?\s*lgs|\bl(?:egge|\.)\s*n?\.?\s*\d|\bd\.?p\.?r|c\.p\.a|cost", re.I)
_PLACEHOLDER = re.compile(r"\[[A-Z_]+(?:_\d+)?\]")


class CardRejected(ValueError):
    """A teacher reply that is not a usable card; ``reason`` is a short code."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def messages(text: str, *, section: str = "", year: int | None = None) -> list[dict]:
    """The chat messages that ask for one card."""
    where = ", " + ", ".join(p for p in (section, str(year) if year else "") if p) \
        if (section or year) else ""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TASK.format(where=where, text=text,
                                                    outcomes=", ".join(OUTCOMES))}]


def _strings(obj, key: str, lo: int, hi: int, min_len: int, max_len: int) -> list[str]:
    vals = obj.get(key)
    if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
        raise CardRejected("bad_field", key)
    vals = [v.strip() for v in vals if v.strip()]
    if not lo <= len(vals) <= hi:
        raise CardRejected("count", f"{key}={len(vals)}")
    if any(not min_len <= len(v) <= max_len for v in vals):
        raise CardRejected("length", key)
    return vals


def _leaks(text: str) -> str | None:
    if _PLACEHOLDER.search(text):
        return "placeholder"
    if RE_CF.search(text) or RE_IBAN.search(text):
        return "structured_pii"
    return None


def parse_card(raw: str) -> dict:
    """The card in a teacher reply, checked; raises CardRejected."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise CardRejected("no_json")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise CardRejected("bad_json", str(exc)) from None
    if not isinstance(obj, dict):
        raise CardRejected("bad_json", "not an object")

    card = {
        "principi": _strings(obj, "principi", 1, 4, 40, 800),
        "norme": _strings(obj, "norme", 0, 8, 4, 200),
        "domande_ricerca": _strings(obj, "domande_ricerca", 3, 5, 15, 400),
    }
    if card["norme"] and not all(_NORM.search(n) for n in card["norme"]):
        raise CardRejected("bad_norm")
    esito = str(obj.get("esito") or "").strip().lower()
    card["esito"] = esito if esito in OUTCOMES else "altro"
    card["materia"] = str(obj.get("materia") or "").strip()[:200]
    exam = obj.get("domande_esame")
    if not isinstance(exam, list) or not 2 <= len(exam) <= 3:
        raise CardRejected("count", "domande_esame")
    card["domande_esame"] = []
    for q in exam:
        if not isinstance(q, dict) or not all(isinstance(q.get(k), str) and q[k].strip()
                                              for k in ("domanda", "risposta", "rubrica")):
            raise CardRejected("bad_field", "domande_esame")
        card["domande_esame"].append({k: q[k].strip() for k in ("domanda", "risposta", "rubrica")})

    flat = json.dumps(card, ensure_ascii=False)
    leak = _leaks(flat)
    if leak:
        raise CardRejected(leak)
    return card


def prefix(card: dict, ruling_meta: dict) -> str:
    """The one-line form of a card that goes in front of the ruling's chunks.

    Short and generic on purpose: a plain summary beat one written around
    legal elements in the only study of it (Summary-Augmented Chunking,
    NLLP 2025).
    """
    head = ", ".join(p for p in (
        "Cons. Stato", ruling_meta.get("sezione") or "",
        f"n. {ruling_meta.get('numero')}" if ruling_meta.get("numero") else "",
        str(ruling_meta.get("data") or ruling_meta.get("year") or "")) if p)
    norms = "; ".join(card.get("norme", [])[:3])
    return f"[{head} - {card.get('materia', '')} - {card.get('esito', '')}" + \
        (f" - {norms}]" if norms else "]")
