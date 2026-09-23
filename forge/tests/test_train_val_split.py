"""Tests for train_val_split.

The function had no tests and an unguarded `max(1, int(n * ratio))`, which
fails silently in both directions: a ratio above 1 returns an empty training
set as a success, a ratio of 0 or below holds out one record instead of none.
Silence is what makes it expensive — an empty `train.jsonl` is discovered on
a compute node, hours after the corpus was built.

The same arithmetic in `split_indices` is fixed separately; these tests pin
the sibling so the two cannot drift apart.
"""

from __future__ import annotations

import pytest

from eullm_forge.datasets.base import train_val_split


def recs(n: int) -> list[dict]:
    return [{"text": f"record {i}", "source_id": f"doc/{i}"} for i in range(n)]


def test_splits_the_last_fraction_into_validation():
    train, val = train_val_split(recs(100), val_ratio=0.05)
    assert len(train) == 95
    assert len(val) == 5
    # Deterministic and contiguous: the tail goes to validation, in order.
    assert val[0]["source_id"] == "doc/95"
    assert val[-1]["source_id"] == "doc/99"


def test_every_record_lands_on_exactly_one_side():
    train, val = train_val_split(recs(37), val_ratio=0.1)
    assert len(train) + len(val) == 37
    ids = [r["source_id"] for r in train] + [r["source_id"] for r in val]
    assert sorted(ids) == sorted(r["source_id"] for r in recs(37))


def test_a_tiny_ratio_still_holds_out_one_record():
    """max(1, ...) is deliberate: asking for 0.1% of 100 means one, not zero."""
    train, val = train_val_split(recs(100), val_ratio=0.001)
    assert len(val) == 1
    assert len(train) == 99


@pytest.mark.parametrize("bad", [1.5, 1.0, 5, 0, 0.0, -0.1])
def test_ratio_outside_the_unit_interval_is_rejected(bad):
    """1.5 used to empty training and report success — the failure this
    guard exists for. 0 and negatives used to hold out one record through
    the max(), which is not what either value asks for."""
    with pytest.raises(ValueError, match="val_ratio"):
        train_val_split(recs(100), val_ratio=bad)


def test_the_rejected_ratio_would_have_emptied_training():
    """Pins WHY 1.5 is rejected rather than clamped, so the guard cannot be
    softened later without this failing: the old arithmetic really did
    return an empty training set, not a small one."""
    n_val = max(1, int(100 * 1.5))
    assert recs(100)[:-n_val] == []
    assert len(recs(100)[-n_val:]) == 100


@pytest.mark.parametrize("n", [0, 1])
def test_too_few_records_to_split(n):
    """One record cannot be split: it would go to validation and leave
    training empty, by the same arithmetic and with the same silence."""
    with pytest.raises(ValueError, match="at least"):
        train_val_split(recs(n), val_ratio=0.05)


def test_two_records_split_one_each():
    train, val = train_val_split(recs(2), val_ratio=0.05)
    assert len(train) == 1
    assert len(val) == 1


@pytest.mark.parametrize("ratio", [0.001, 0.05, 0.5, 0.9, 0.99])
def test_training_is_never_empty_for_any_accepted_ratio(ratio):
    """The property the guard is really protecting, across the whole
    accepted range and a spread of corpus sizes."""
    for n in (2, 3, 10, 99, 100, 1000):
        train, val = train_val_split(recs(n), val_ratio=ratio)
        assert train, f"empty train for n={n}, ratio={ratio}"
        assert val, f"empty val for n={n}, ratio={ratio}"
        assert len(train) + len(val) == n
