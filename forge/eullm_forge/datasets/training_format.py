"""Final-stage formatter: dedup'd chunks → continued-pretraining JSONL.

Reads ``italgiure_*.dedup.jsonl`` from the corpus directory, mixes the
records across files, splits into train/val, and writes two files in
HuggingFace-compatible format:

    {"text": "...", "source_id": "snciv/2023/12345", "year": 2023,
     "kind": "snciv", "chunk_index": 2, "chunk_total": 5}

The output drops heavy fields (the per-record audit trail produced by
the anonymiser) and keeps only what the trainer / dataloader needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional

# Fields kept on every output record. ``text`` is the only one the
# trainer looks at; the others are useful for sampling, weighting,
# debugging and audit.
DEFAULT_KEEP_FIELDS: tuple[str, ...] = (
    "text",
    "source_id",
    "year",
    "kind",
    "chunk_index",
    "chunk_total",
    "sentence_id",
)


@dataclass
class FormatStats:
    seen: int = 0
    written_train: int = 0
    written_val: int = 0
    skipped_empty: int = 0


def slim_record(
    rec: dict[str, Any],
    *,
    keep_fields: Iterable[str] = DEFAULT_KEEP_FIELDS,
    text_field: str = "text",
) -> Optional[dict[str, Any]]:
    """Strip a record down to ``keep_fields``. Returns None if the text
    field is empty (caller should skip).
    """
    text = rec.get(text_field) or ""
    if not text.strip():
        return None
    out = {k: rec[k] for k in keep_fields if k in rec}
    out[text_field] = text
    return out


def split_indices(
    n: int,
    val_ratio: float,
    seed: int,
    groups: Optional[list[Any]] = None,
) -> tuple[list[int], list[int]]:
    """Deterministic train/val index split with shuffling.

    Uses Python's stdlib ``random`` (Mersenne Twister) seeded with
    ``seed`` so the same corpus + same ratio + same seed always produce
    the same split — important for reproducibility of training runs.

    ``groups`` holds one key per record; records sharing a key go to the
    same side. **Pass it whenever records are fragments of a larger
    document**, or the split measures the wrong thing.

    Why this argument exists, from a result it spoiled. A record here is a
    ~2,048-token chunk of a court ruling, not a ruling. Splitting over
    records scattered the chunks of one ruling across both sides — chunk 3
    trained on, chunk 4 held out — so a model was scored on passages whose
    immediate neighbours it had read: same parties, same cited articles,
    same recurring formulas. The base model it was compared against had
    seen none of it, and the measured gap was inflated by an amount nothing
    in the measurement could bound. A validation set has to be held out at
    the unit a reader will assume, which for a corpus of documents is the
    document.

    Grouping makes the ratio approximate: whole groups are taken until the
    target is reached, so the realised ratio depends on group sizes and the
    caller should report what it got rather than what it asked for.
    """
    import random
    rng = random.Random(seed)

    if groups is None:
        indices = list(range(n))
        rng.shuffle(indices)
        n_val = max(1, int(n * val_ratio))
        return sorted(indices[n_val:]), sorted(indices[:n_val])

    if len(groups) != n:
        raise ValueError(
            f"groups has {len(groups)} keys for {n} records — one key per "
            f"record is required, or the split silently misassigns them"
        )

    # A record whose key is missing is its own group rather than sharing a
    # bucket with every other keyless record: lumping them together would
    # put an arbitrary slice of the corpus on one side, which is a worse
    # failure than the one this function is fixing.
    by_group: dict[Any, list[int]] = {}
    for i, key in enumerate(groups):
        k = key if key is not None and key != "" else ("__ungrouped__", i)
        by_group.setdefault(k, []).append(i)

    keys = sorted(by_group, key=repr)
    rng.shuffle(keys)

    target = max(1, int(n * val_ratio))
    val_idx: list[int] = []
    taken = 0
    for k in keys:
        if taken >= target:
            break
        members = by_group[k]
        val_idx.extend(members)
        taken += len(members)

    val_set = set(val_idx)
    train_idx = [i for i in range(n) if i not in val_set]
    return train_idx, sorted(val_idx)


def iter_slimmed(
    records: Iterable[dict[str, Any]],
    *,
    keep_fields: Iterable[str] = DEFAULT_KEEP_FIELDS,
    stats: Optional[FormatStats] = None,
) -> Iterator[dict[str, Any]]:
    s = stats if stats is not None else FormatStats()
    for rec in records:
        s.seen += 1
        slim = slim_record(rec, keep_fields=keep_fields)
        if slim is None:
            s.skipped_empty += 1
            continue
        yield slim
