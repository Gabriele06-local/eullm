"""Tests for the distillation cost estimator and teacher-sharding helper."""

import pytest

from eullm_forge.distill import build_teacher_max_memory, estimate_distillation_cost


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
