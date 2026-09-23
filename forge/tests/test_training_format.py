"""Tests for the training_format module."""

from __future__ import annotations

from eullm_forge.datasets.training_format import (
    DEFAULT_KEEP_FIELDS,
    FormatStats,
    iter_slimmed,
    slim_record,
    split_indices,
)


def test_slim_record_keeps_only_default_fields():
    rec = {
        "text": "Una sentenza di Cassazione.",
        "source_id": "snciv/2023/1",
        "year": 2023,
        "kind": "snciv",
        "chunk_index": 0,
        "chunk_total": 1,
        "metadata": {"anonymization": {"person_ner": 5}},
        "internal_blob": "x" * 5000,
    }
    out = slim_record(rec)
    assert out is not None
    assert set(out.keys()) <= set(DEFAULT_KEEP_FIELDS)
    assert "metadata" not in out
    assert "internal_blob" not in out
    assert out["text"] == rec["text"]


def test_slim_record_returns_none_for_empty_text():
    assert slim_record({"text": ""}) is None
    assert slim_record({"text": "   \n  "}) is None
    assert slim_record({}) is None


def test_slim_record_preserves_text_when_field_missing_from_keep_list():
    """The text field is always preserved, even with a custom keep list."""
    rec = {"text": "ciao", "source_id": "x"}
    out = slim_record(rec, keep_fields=("source_id",))
    assert out is not None
    assert out["text"] == "ciao"
    assert out["source_id"] == "x"


def test_split_indices_is_deterministic():
    a_train, a_val = split_indices(1000, val_ratio=0.05, seed=42)
    b_train, b_val = split_indices(1000, val_ratio=0.05, seed=42)
    assert a_train == b_train
    assert a_val == b_val


def test_split_indices_different_seeds_produce_different_splits():
    _, v1 = split_indices(10000, val_ratio=0.05, seed=1)
    _, v2 = split_indices(10000, val_ratio=0.05, seed=2)
    assert v1 != v2


def test_split_indices_no_overlap_and_full_coverage():
    n = 1000
    train, val = split_indices(n, val_ratio=0.05, seed=7)
    assert set(train).isdisjoint(set(val))
    assert sorted(train + val) == list(range(n))


def test_split_indices_at_least_one_val():
    """Even with 1 record and val_ratio=0.001, we get one val record."""
    train, val = split_indices(1, val_ratio=0.001, seed=0)
    assert len(val) == 1
    assert len(train) == 0


def test_iter_slimmed_increments_stats():
    recs = [
        {"text": "alpha", "source_id": "a"},
        {"text": "", "source_id": "b"},        # skipped
        {"text": "  \n  ", "source_id": "c"},  # skipped
        {"text": "beta", "source_id": "d"},
    ]
    stats = FormatStats()
    out = list(iter_slimmed(recs, stats=stats))
    assert len(out) == 2
    assert stats.seen == 4
    assert stats.skipped_empty == 2


# --- document-level split ---------------------------------------------------
#
# These pin the property a per-chunk split silently broke: a ruling's chunks
# must never straddle train and val, or the held-out set shares parties,
# citations and formulas with text the model trained on and every number taken
# against it is inflated.


def _groups_of(n_docs: int, chunks_per_doc: int) -> list[str]:
    return [f"doc-{d}" for d in range(n_docs) for _ in range(chunks_per_doc)]


def test_grouped_split_never_straddles_a_document():
    groups = _groups_of(200, 5)
    train, val = split_indices(len(groups), val_ratio=0.1, seed=42, groups=groups)
    train_docs = {groups[i] for i in train}
    val_docs = {groups[i] for i in val}
    assert train_docs & val_docs == set()


def test_grouped_split_still_covers_every_record_exactly_once():
    groups = _groups_of(100, 7)
    n = len(groups)
    train, val = split_indices(n, val_ratio=0.1, seed=3, groups=groups)
    assert sorted(train + val) == list(range(n))
    assert set(train).isdisjoint(val)


def test_grouped_split_is_deterministic():
    groups = _groups_of(50, 4)
    a = split_indices(len(groups), val_ratio=0.2, seed=11, groups=groups)
    b = split_indices(len(groups), val_ratio=0.2, seed=11, groups=groups)
    assert a == b


def test_ungrouped_records_are_each_their_own_document():
    """A missing key must not lump every keyless record into one bucket.

    Sharing a bucket would put an arbitrary slice of the corpus wholly on one
    side — a worse failure than the one grouping fixes.
    """
    groups = [None, "", "doc-a", "doc-a", None, "doc-b"]
    train, val = split_indices(len(groups), val_ratio=0.5, seed=1, groups=groups)
    assert sorted(train + val) == list(range(len(groups)))
    # doc-a's two chunks stay together wherever they land.
    a_sides = {i in set(val) for i in (2, 3)}
    assert len(a_sides) == 1


def test_groups_length_must_match_record_count():
    import pytest
    with pytest.raises(ValueError, match="one key per record"):
        split_indices(10, val_ratio=0.1, seed=1, groups=["a", "b"])


def test_out_of_range_val_ratio_is_rejected_not_silently_applied():
    """A ratio outside (0, 1) cannot mean anything: 1.5 would empty train,
    0 would silently hold out one record instead of none. Fail before
    writing an empty train.jsonl discovered at GPU time."""
    import pytest

    groups = [f"doc-{i // 5}" for i in range(100)]
    for bad in (-0.1, 0, 0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="val_ratio"):
            split_indices(100, bad, seed=42)
        with pytest.raises(ValueError, match="val_ratio"):
            split_indices(100, bad, seed=42, groups=groups)


def test_ungrouped_split_is_unchanged_when_groups_is_none():
    """The old behaviour must survive exactly, for reproducing old corpora."""
    a = split_indices(1000, val_ratio=0.05, seed=42)
    b = split_indices(1000, val_ratio=0.05, seed=42, groups=None)
    assert a == b


def test_per_chunk_split_does_straddle_documents():
    """The bug, pinned, so the grouped tests above are known to discriminate.

    Without grouping almost every multi-chunk document lands on both sides —
    which is why no slice of the 2026-09 corpus could be salvaged as a clean
    held-out set and a fresh one had to be built.
    """
    groups = _groups_of(200, 5)
    _, val = split_indices(len(groups), val_ratio=0.1, seed=42)
    val_docs = {groups[i] for i in val}
    train_docs = {groups[i] for i in range(len(groups)) if i not in set(val)}
    assert val_docs & train_docs, "expected the per-chunk split to straddle"


def test_grouped_split_realised_ratio_is_close_to_requested():
    """Whole documents are indivisible, so the ratio is approximate — but not
    arbitrary. Callers report the realised value; this bounds how far off it
    can drift on evenly sized documents."""
    groups = _groups_of(400, 5)
    n = len(groups)
    _, val = split_indices(n, val_ratio=0.1, seed=5, groups=groups)
    assert 0.09 <= len(val) / n <= 0.12
