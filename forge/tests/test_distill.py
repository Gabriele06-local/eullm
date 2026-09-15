"""Tests for the distillation cost estimator and teacher-sharding helper."""

import pytest

from eullm_forge.distill import (
    build_teacher_max_memory,
    build_teacher_split_memory,
    estimate_distillation_cost,
)


def test_estimate_14b_to_7b():
    cost = estimate_distillation_cost(
        teacher_params_b=14.0,
        student_params_b=7.0,
        num_tokens_b=50.0,
    )
    assert cost["gpu_hours"] > 0
    assert cost["num_gpus"] >= 1
    assert cost["wall_hours"] > 0
    assert cost["estimated_cost"] > 0


def test_estimate_70b_to_14b():
    cost = estimate_distillation_cost(
        teacher_params_b=70.0,
        student_params_b=14.0,
        num_tokens_b=50.0,
    )
    # 70B needs more GPUs than 14B
    assert cost["num_gpus"] >= 3


def test_estimate_custom_gpu_cost():
    cost_cheap = estimate_distillation_cost(14.0, 7.0, 50.0, gpu_cost_per_hour=1.0)
    cost_expensive = estimate_distillation_cost(14.0, 7.0, 50.0, gpu_cost_per_hour=5.0)
    assert cost_expensive["estimated_cost"] > cost_cheap["estimated_cost"]


def test_teacher_max_memory_leonardo_node():
    # Leonardo Booster node: 4x A100 64 GB, student on GPU 0.
    mm = build_teacher_max_memory(4)
    assert mm == {0: "8GiB", 1: "58GiB", 2: "58GiB", 3: "58GiB"}
    # The teacher budget must fit a 32B BF16 teacher (~64 GiB).
    total = sum(int(v.removesuffix("GiB")) for v in mm.values())
    assert total >= 64


def test_teacher_max_memory_custom_student_gpu():
    mm = build_teacher_max_memory(
        4, student_gpu_index=2, teacher_gib_per_gpu=50, teacher_gib_on_student_gpu=4
    )
    assert mm == {0: "50GiB", 1: "50GiB", 2: "4GiB", 3: "50GiB"}


def test_teacher_max_memory_rejects_single_gpu():
    with pytest.raises(ValueError):
        build_teacher_max_memory(1)


def test_teacher_max_memory_rejects_bad_student_index():
    with pytest.raises(ValueError):
        build_teacher_max_memory(4, student_gpu_index=4)


def test_unknown_dataset_raises_instead_of_silent_fallback(monkeypatch):
    """An explicitly requested dataset that fails to load must fail loudly.

    Falling back to wikitext here would train the student for days on the
    wrong corpus while looking healthy, so the loader refuses instead.
    """
    import sys
    import types
    from unittest.mock import MagicMock

    from eullm_forge import distill as distill_module

    def _raise(*args, **kwargs):
        raise FileNotFoundError("Dataset 'no-such-dataset' doesn't exist")

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = _raise
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="no-such-dataset"):
        distill_module._load_distillation_dataset("no-such-dataset", tokenizer=MagicMock())


# ── design B: the teacher gets whole GPUs, never the student's ───────────
# Co-hosting is what forced the teacher to 8-bit in v1.0. These guard the
# map that stops it happening again, and every failure mode here is one that
# would otherwise surface as an OOM minutes into a multi-day job.

def test_split_gives_the_teacher_its_gpus_and_zero_elsewhere():
    got = build_teacher_split_memory(4, [0, 1], teacher_gib_per_gpu=58)
    assert got == {0: "58GiB", 1: "58GiB", 2: "0GiB", 3: "0GiB"}


def test_split_names_every_device_including_the_forbidden_ones():
    """accelerate treats an absent device as unconstrained.

    Omitting GPUs 2-3 instead of pinning them to 0GiB would hand the teacher
    the whole node — exactly the co-hosting the split exists to remove, and
    silently, because the run would still start.
    """
    got = build_teacher_split_memory(4, [0])
    assert sorted(got) == [0, 1, 2, 3]
    assert [got[i] for i in (1, 2, 3)] == ["0GiB", "0GiB", "0GiB"]


def test_split_refuses_to_take_the_whole_node():
    with pytest.raises(ValueError, match="at least one left"):
        build_teacher_split_memory(4, [0, 1, 2, 3])


def test_split_rejects_a_device_that_does_not_exist():
    with pytest.raises(ValueError, match="out of range"):
        build_teacher_split_memory(4, [0, 4])


def test_split_rejects_a_repeated_device():
    with pytest.raises(ValueError, match="repeats"):
        build_teacher_split_memory(4, [0, 0])


def test_split_rejects_an_empty_teacher():
    with pytest.raises(ValueError, match="needs a GPU"):
        build_teacher_split_memory(4, [])


def test_split_needs_more_than_one_gpu():
    with pytest.raises(ValueError, match=">= 2 GPUs"):
        build_teacher_split_memory(1, [0])


def test_split_and_shared_maps_differ_exactly_where_it_matters():
    """The shared map lets the teacher onto the student's card; split does not."""
    shared = build_teacher_max_memory(4, student_gpu_index=2)
    split = build_teacher_split_memory(4, [0, 1])
    assert shared[2] != "0GiB"     # v1.0: a teacher shard sits with the student
    assert split[2] == "0GiB"      # design B: it cannot
