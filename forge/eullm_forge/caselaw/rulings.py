"""Whole rulings back from the training chunks, with the metadata OpenGA knows.

The Consiglio di Stato corpus reached Leonardo as continued-pretraining
chunks (format_pretraining.py: ``text``, ``sentence_id``, ``chunk_index``,
``year``), about 2,000 tokens each with a 200-character overlap. A ruling
card (`schede`) and a dev split need the ruling, not the chunk, so the
chunks are put back together in order with the overlap dropped.

What the chunks lost -- the appeal number (NRG), section, outcome -- is in
the OpenGA index (CC BY 4.0, one CSV per year), keyed by the ruling number
that the fetcher made into ``source_id = "cds/<NUMERO_PROVVEDIMENTO>"``.
The NRG matters for the split: an appeal restates the first-instance facts
and related rulings on the same appeal repeat them, so a dev ruling whose
sibling is in training would be a dev ruling the model has read.

Nothing here prints a ruling's text.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Ruling:
    """One ruling, reassembled."""

    id: str                      # sentence_id, e.g. "cds/202301234"
    year: int | None
    text: str
    n_chunks: int
    meta: dict = field(default_factory=dict)   # OpenGA row fields, when known

    @property
    def number(self) -> str:
        return self.id.split("/", 1)[-1]

    @property
    def nrg(self) -> str:
        return str(self.meta.get("NUMERO_RICORSO") or "").strip()

    @property
    def section(self) -> str:
        return str(self.meta.get("NOME_SEZIONE") or "").strip()


def join_chunks(chunks: list[str], max_overlap: int = 400) -> str:
    """Chunks in order into one text, the overlap each carries dropped.

    chunk_corpus.py starts every chunk after the first with the tail of the
    previous one, snapped to a word boundary, so the overlap is found rather
    than assumed: the longest prefix of the next chunk that ends the text so
    far, up to ``max_overlap`` characters.
    """
    if not chunks:
        return ""
    out = chunks[0]
    for nxt in chunks[1:]:
        cut = 0
        for k in range(min(len(nxt), max_overlap, len(out)), 19, -1):
            if out.endswith(nxt[:k]):
                cut = k
                break
        out += nxt[cut:] if cut else ("\n" + nxt)
    return out


def load_rulings(paths: Iterable[str | Path], kind: str = "cds") -> dict[str, Ruling]:
    """Every ruling of ``kind`` in the given chunk files, by sentence_id."""
    parts: dict[str, list[tuple[int, str]]] = defaultdict(list)
    years: dict[str, int | None] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                sid = str(r.get("sentence_id") or r.get("source_id") or "")
                if not sid or (r.get("kind") or sid.split("/", 1)[0]) != kind:
                    continue
                parts[sid].append((int(r.get("chunk_index") or 0), r.get("text") or ""))
                years.setdefault(sid, r.get("year"))
    out = {}
    for sid, chunks in parts.items():
        chunks.sort()
        out[sid] = Ruling(sid, years.get(sid), join_chunks([t for _, t in chunks]), len(chunks))
    return out


def load_openga(paths: Iterable[str | Path]) -> dict[str, dict]:
    """OpenGA index rows by ruling number (NUMERO_PROVVEDIMENTO)."""
    rows: dict[str, dict] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            delim = ";" if sample.count(";") > sample.count(",") else ","
            for row in csv.DictReader(f, delimiter=delim):
                num = (row.get("NUMERO_PROVVEDIMENTO") or "").strip()
                if num:
                    rows[num] = row
    return rows


def attach_meta(rulings: dict[str, Ruling], index: dict[str, dict]) -> int:
    """Copy each ruling's OpenGA row into ``meta``; returns how many matched."""
    n = 0
    for r in rulings.values():
        row = index.get(r.number)
        if row:
            r.meta = dict(row)
            n += 1
    return n


def group_key(r: Ruling) -> str:
    """Rulings on the same appeal share a key: they repeat the same facts."""
    return f"nrg:{r.nrg}" if r.nrg else f"id:{r.id}"


_REASONS = re.compile(
    r"\n\s*(?:CONSIDERATO\s+IN\s+)?(?:DIRITTO|MOTIVI\s+DELLA\s+DECISIONE)\s*\n", re.IGNORECASE)


def ruling_view(text: str, max_chars: int = 24000) -> str:
    """The ruling, or as much of it as matters when it is too long.

    The reasons ("DIRITTO") carry the principles a card is about; the head
    carries the parties' claims and what was appealed. A long ruling keeps a
    quarter of the budget for the head and the rest for the reasons, from
    their heading on, marked where it was cut.
    """
    if len(text) <= max_chars:
        return text
    m = _REASONS.search(text)
    head = text[:max_chars // 4].rstrip()
    if m and m.start() > len(head):
        rest = text[m.start():m.start() + max_chars - len(head)].rstrip()
        return head + "\n[…]\n" + rest + (" […]" if m.start() + len(rest) < len(text) else "")
    return text[:max_chars].rstrip() + " […]"
