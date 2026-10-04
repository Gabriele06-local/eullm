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


def test_default_wikitext_is_requested_with_its_config(monkeypatch):
    """The default dataset must be asked for by name AND config.

    `wikitext` declares four configs and marks none of them default, so
    `load_dataset("wikitext")` raises rather than picking one. Before the
    fallback was removed that error was caught and the reload supplied the
    config, which is the only reason the default ever worked; afterwards
    `run_distillation` could not start without an explicit --dataset. This
    pins the call, so removing the branch fails here instead of in a job.
    """
    import sys
    import types
    from unittest.mock import MagicMock

    from eullm_forge import distill as distill_module

    seen: list[tuple] = []

    class _FakeDS:
        column_names = ["text"]

        def __len__(self):
            return 1

        def select(self, _):
            return self

        def filter(self, _):
            return self

        def map(self, *a, **k):
            return self

        def set_format(self, _):
            pass

    def _record(*args, **kwargs):
        seen.append(args)
        return _FakeDS()

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = _record
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    distill_module._load_distillation_dataset(
        distill_module.WIKITEXT_DEFAULT, tokenizer=MagicMock()
    )

    assert seen, "load_dataset was never called"
    assert seen[0][:2] == (
        distill_module.WIKITEXT_DEFAULT,
        distill_module.WIKITEXT_DEFAULT_CONFIG,
    ), f"the default must carry its config, got {seen[0]!r}"


def _trainer_with_stub_models(vocab, length):
    """A DistillationTrainer whose teacher and student are fixed tensors.

    Enough for train_step: it only calls each model with the batch and reads
    .logits, so the loss it computes is the real one over the real labels.
    """
    import torch

    from eullm_forge.distill import DistillationTrainer, DistillConfig

    class Fixed(torch.nn.Module):
        """Returns the same logits every call.

        The student's are a Parameter so train_step's backward() has something
        to differentiate; the teacher's are under no_grad in the real trainer
        and are read only.
        """

        def __init__(self, logits, *, trainable=False):
            super().__init__()
            if trainable:
                self.logits = torch.nn.Parameter(logits)
            else:
                self.register_buffer("logits", logits)

        def forward(self, input_ids=None, attention_mask=None):
            class Out:
                pass
            out = Out()
            out.logits = self.logits
            return out

    trainer = DistillationTrainer.__new__(DistillationTrainer)
    trainer.config = DistillConfig(temperature=2.0, alpha=0.5)
    trainer.device = torch.device("cpu")
    torch.manual_seed(1)
    trainer.teacher = Fixed(torch.randn(2, length, vocab))
    trainer.student = Fixed(torch.randn(2, length, vocab), trainable=True)
    trainer.optimizer = torch.optim.SGD(trainer.student.parameters(), lr=1e-9)
    return trainer


def test_the_kd_loss_does_not_read_the_padding():
    """Two batches that differ only where attention_mask says to ignore must
    give the same loss.

    The loader pads every sample to max_length, and the labels used to be the
    raw shifted input_ids -- nothing ever set a label to -100 -- so most of
    what the loss read was padding. Changing one token at a padded position
    moved it.
    """
    import torch

    vocab, length = 50, 64
    trainer = _trainer_with_stub_models(vocab, length)

    def loss_of(input_ids, attention_mask):
        return trainer.train_step({"input_ids": input_ids,
                                  "attention_mask": attention_mask})

    ids = torch.zeros(2, length, dtype=torch.long)    # 0 = PAD
    ids[0, :8] = torch.randint(1, vocab, (8,))
    ids[1, :40] = torch.randint(1, vocab, (40,))
    mask = torch.zeros(2, length, dtype=torch.long)
    mask[0, :8] = 1
    mask[1, :40] = 1

    base = loss_of(ids, mask)

    ignored = ids.clone()
    ignored[0, 50] = 7                               # a padded position
    assert mask[0, 50] == 0
    assert abs(loss_of(ignored, mask) - base) < 1e-6, "the loss read a padded token"

    kept = ids.clone()
    kept[0, 3] = 7                                    # a real token
    assert mask[0, 3] == 1
    assert abs(loss_of(kept, mask) - base) > 1e-6, \
        "the loss stopped depending on the text at all"


def test_a_batch_of_nothing_but_padding_is_zero_and_not_nan():
    import math

    import torch

    vocab, length = 50, 16
    trainer = _trainer_with_stub_models(vocab, length)
    value = trainer.train_step({"input_ids": torch.zeros(2, length, dtype=torch.long),
                                "attention_mask": torch.zeros(2, length,
                                                               dtype=torch.long)})
    assert math.isfinite(value), f"a padded batch gave {value}"
    assert value == 0.0
