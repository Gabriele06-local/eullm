"""Evaluation dataset — held-out items used to rank verticalized models.

Items are stored as JSONL (one JSON object per line). Each item is parametric
by ``domain`` and ``lang`` so a single harness serves every vertical. The seed
set for the Italian legal domain lives at
``data/legal_it_heldout.seed.jsonl`` and is meant to be expanded and validated
by a domain expert (F0-A in the Forge R&D roadmap).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

SEED_PATH = Path(__file__).parent / "data" / "legal_it_heldout.seed.jsonl"


@dataclass
class EvalItem:
    """A single held-out evaluation item.

    Attributes:
        id: Stable unique identifier.
        domain: Vertical domain (e.g. ``"legal"``, ``"medical"``, ``"finance"``).
        lang: ISO language code (e.g. ``"it"``, ``"de"``, ``"fr"``).
        question: The prompt shown to the model.
        reference: A reference answer (optional; used by exact-match / judge).
        rubric: Scoring guidance for the LLM-as-judge (optional).
        keywords: Terms that a correct answer should contain (keyword coverage).
        category: Sub-domain tag (e.g. ``"civile"``, ``"gdpr"``).
        metadata: Free-form extra fields (source note, article, ...).
    """

    id: str
    domain: str
    lang: str
    question: str
    reference: str = ""
    rubric: str = ""
    keywords: list[str] = field(default_factory=list)
    category: str = ""
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> EvalItem:
        """Build an item from a plain dict, ignoring unknown keys."""
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_dict(self) -> dict:
        return asdict(self)


def load_eval_set(path: str | Path) -> list[EvalItem]:
    """Load a JSONL eval set. Blank lines and ``#`` comment lines are skipped."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")
    items: list[EvalItem] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not data.get("id"):
                raise ValueError(f"{path}:{lineno}: item is missing 'id'")
            try:
                item = EvalItem.from_dict(data)
            except TypeError as exc:  # missing required field (domain/lang/question)
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if item.id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id '{item.id}'")
            seen.add(item.id)
            items.append(item)
    return items


def save_eval_set(items: list[EvalItem], path: str | Path) -> Path:
    """Write items to JSONL, creating parent directories as needed.

    Serialised in full before the destination is touched, and moved into
    place with os.replace. Opening the target with "w" truncated it first, so
    a value json cannot represent -- a Path left in an item's metadata is the
    easy one -- aborted on that line and left the exam that was already on
    disk shortened by exactly one item, with nothing in the file to say so
    and no count anywhere to notice against. load_eval_set read the result
    without complaint.

    The error now names the item and the file it was going into.
    """
    path = Path(path)
    lines = []
    for i, item in enumerate(items):
        try:
            lines.append(json.dumps(item.to_dict(), ensure_ascii=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: item {i} ({item.id!r}) is not JSON serialisable: {exc}"
            ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with open(partial, "w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return path


def filter_items(
    items: list[EvalItem],
    *,
    domain: str | None = None,
    lang: str | None = None,
    category: str | None = None,
) -> list[EvalItem]:
    """Filter items by domain / language / category (case-insensitive)."""

    def keep(it: EvalItem) -> bool:
        if domain and it.domain.lower() != domain.lower():
            return False
        if lang and it.lang.lower() != lang.lower():
            return False
        if category and it.category.lower() != category.lower():
            return False
        return True

    return [it for it in items if keep(it)]


def load_seed() -> list[EvalItem]:
    """Load the bundled Italian-legal seed held-out set."""
    return load_eval_set(SEED_PATH)
