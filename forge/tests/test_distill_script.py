"""Tests for the Leonardo distillation script's bookkeeping.

`forge/scripts/distill.py` is a hand-written training loop, so the parts that
are not the loop itself carry no framework behind them and get no framework's
testing either. Two of them can destroy a run:

  * `prune_checkpoints` deletes directories. A run's only way back from a
    walltime kill is the checkpoint it last wrote, so an off-by-one here does
    not produce a wrong number, it produces no way to resume.
  * `_checkpoint_step` decides which checkpoint is newest. Sorted as strings,
    `checkpoint-9000` comes after `checkpoint-18000`, and a resume would
    silently redo nine thousand steps — three days of Booster time.

Both are pure filesystem logic, so they are tested against real directories.
`evaluate` is tested against stand-in models, since the real pair is 34 B
parameters and needs four GPUs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "distill.py"


def _load():
    spec = importlib.util.spec_from_file_location("distill_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves `cls.__module__` through
    # sys.modules, and without this the config class fails to build.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


distill = _load()


def make_checkpoints(root: Path, steps) -> None:
    for step in steps:
        d = root / f"checkpoint-{step}"
        d.mkdir(parents=True)
        (d / "training_state.pt").write_bytes(b"x")


# ── which checkpoint is the newest ───────────────────────────────────────

def test_step_is_read_as_a_number_not_a_string(tmp_path):
    """The bug this guards: '9000' > '18000' lexicographically."""
    make_checkpoints(tmp_path, [9000, 18000])
    assert distill.latest_checkpoint(tmp_path).name == "checkpoint-18000"


def test_unnumbered_directories_never_win(tmp_path):
    make_checkpoints(tmp_path, [500])
    (tmp_path / "checkpoint-final").mkdir()
    assert distill.latest_checkpoint(tmp_path).name == "checkpoint-500"


def test_no_checkpoints_is_none_not_an_error(tmp_path):
    assert distill.latest_checkpoint(tmp_path) is None
    assert distill.latest_checkpoint(tmp_path / "absent") is None


# ── pruning ──────────────────────────────────────────────────────────────

def test_keeps_the_newest_n_by_step(tmp_path):
    make_checkpoints(tmp_path, [300, 600, 900, 1200, 1500])
    distill.prune_checkpoints(tmp_path, keep=3)
    left = sorted(p.name for p in tmp_path.glob("checkpoint-*"))
    assert left == ["checkpoint-1200", "checkpoint-1500", "checkpoint-900"]


def test_pruning_ranks_by_number_too(tmp_path):
    """Keeping the three 'largest' as strings would drop checkpoint-18000."""
    make_checkpoints(tmp_path, [9000, 12000, 15000, 18000])
    distill.prune_checkpoints(tmp_path, keep=2)
    left = {p.name for p in tmp_path.glob("checkpoint-*")}
    assert left == {"checkpoint-15000", "checkpoint-18000"}


def test_fewer_checkpoints_than_the_limit_deletes_nothing(tmp_path):
    make_checkpoints(tmp_path, [300, 600])
    assert distill.prune_checkpoints(tmp_path, keep=3) == []
    assert len(list(tmp_path.glob("checkpoint-*"))) == 2


def test_keep_zero_or_less_disables_pruning(tmp_path):
    """A limit of 0 must mean 'keep everything', never 'delete everything'."""
    make_checkpoints(tmp_path, [300, 600, 900])
    assert distill.prune_checkpoints(tmp_path, keep=0) == []
    assert distill.prune_checkpoints(tmp_path, keep=-1) == []
    assert len(list(tmp_path.glob("checkpoint-*"))) == 3


def test_unnumbered_directories_are_never_deleted(tmp_path):
    """`output_dir` also holds the merged model and the tokenizer."""
    make_checkpoints(tmp_path, [300, 600, 900, 1200])
    keepsake = tmp_path / "checkpoint-merged"
    keepsake.mkdir()
    distill.prune_checkpoints(tmp_path, keep=1)
    assert keepsake.is_dir()
    assert (tmp_path / "checkpoint-1200").is_dir()


def test_pruning_survives_a_directory_that_vanished(tmp_path):
    """Two links of a chain can overlap for a moment at the handover."""
    make_checkpoints(tmp_path, [300, 600, 900])
    (tmp_path / "checkpoint-300" / "training_state.pt").unlink()
    (tmp_path / "checkpoint-300").rmdir()
    distill.prune_checkpoints(tmp_path, keep=1)
    assert {p.name for p in tmp_path.glob("checkpoint-*")} == {"checkpoint-900"}


def test_a_checkpoint_file_is_not_mistaken_for_a_checkpoint_dir(tmp_path):
    make_checkpoints(tmp_path, [300, 600])
    (tmp_path / "checkpoint-notes.txt").write_text("stray", encoding="utf-8")
    distill.prune_checkpoints(tmp_path, keep=1)
    assert (tmp_path / "checkpoint-notes.txt").is_file()


# ── validation ───────────────────────────────────────────────────────────
# The real pair is a 30 B MoE teacher and a 4 B student across four GPUs.
# What is worth pinning here is the contract around the forward passes:
# the cap is honoured, train mode comes back, and an empty loader is not a
# crash halfway through a 24 h job.

VOCAB = 8
SEQ = 4


class FakeOut:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """Returns fixed logits and counts how many times it was called."""

    def __init__(self, fill: float):
        self.fill = fill
        self.calls = 0
        self.training = True

    def __call__(self, **batch):
        self.calls += 1
        n = batch["labels"].shape[0]
        return FakeOut(torch.full((n, SEQ, VOCAB), self.fill))

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


def batches(n: int):
    for _ in range(n):
        yield {"labels": torch.zeros(1, SEQ, dtype=torch.long),
               "input_ids": torch.zeros(1, SEQ, dtype=torch.long)}


def cfg():
    return distill.DistillConfig(kl_alpha=0.7, kl_temperature=2.0)


def test_evaluate_stops_at_the_cap():
    """The reason the cap exists: the full split is over two hours per eval."""
    student, teacher = FakeModel(0.5), FakeModel(0.4)
    distill.evaluate(student, teacher, list(batches(50)), cfg(), "cpu",
                     max_batches=7)
    assert student.calls == 7
    assert teacher.calls == 7


def test_evaluate_returns_the_three_components():
    student, teacher = FakeModel(0.5), FakeModel(0.4)
    got = distill.evaluate(student, teacher, list(batches(3)), cfg(), "cpu",
                           max_batches=3)
    assert set(got) == {"loss", "kl", "ce"}
    assert all(isinstance(v, float) for v in got.values())


def test_evaluate_restores_train_mode():
    """Left in eval mode, the rest of the run trains with dropout off."""
    student, teacher = FakeModel(0.5), FakeModel(0.4)
    student.train()
    distill.evaluate(student, teacher, list(batches(2)), cfg(), "cpu",
                     max_batches=2)
    assert student.training is True


def test_evaluate_restores_train_mode_even_when_a_batch_raises():
    class Exploding(FakeModel):
        def __call__(self, **batch):
            raise RuntimeError("CUDA out of memory")

    student, teacher = FakeModel(0.5), Exploding(0.4)
    student.train()
    with pytest.raises(RuntimeError):
        distill.evaluate(student, teacher, list(batches(2)), cfg(), "cpu",
                         max_batches=2)
    assert student.training is True


def test_evaluate_leaves_a_model_that_was_not_training_alone():
    student, teacher = FakeModel(0.5), FakeModel(0.4)
    student.eval()
    distill.evaluate(student, teacher, list(batches(2)), cfg(), "cpu",
                     max_batches=2)
    assert student.training is False


def test_an_empty_loader_reports_nothing_rather_than_dividing_by_zero():
    student, teacher = FakeModel(0.5), FakeModel(0.4)
    assert distill.evaluate(student, teacher, [], cfg(), "cpu",
                            max_batches=10) == {}


def test_a_zero_cap_scores_nothing_and_does_not_hang():
    student, teacher = FakeModel(0.5), FakeModel(0.4)
    assert distill.evaluate(student, teacher, list(batches(5)), cfg(), "cpu",
                            max_batches=0) == {}
    assert student.calls == 0
