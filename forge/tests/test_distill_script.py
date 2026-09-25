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
import json
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


# ── divergence guard ─────────────────────────────────────────────────────
# On 2026-09-24 the control arm went from loss 0.6255 to nan in twenty steps
# and nothing stopped it: two more links trained a dead model and saved it
# four times, and save_total_limit rotated every healthy checkpoint away. A
# 36,600-step run was lost to what should have cost one skipped update.
#
# The helpers are tested directly, and then `train()` itself is run end to
# end on CPU with stand-in models, because a guard that exists as a helper
# but is not wired into the loop is the failure mode worth ruling out.


def test_all_finite_accepts_ordinary_tensors():
    assert distill.all_finite([torch.zeros(3), torch.ones(2, 2)])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_all_finite_rejects_any_non_finite_value(bad):
    t = torch.zeros(4)
    t[2] = bad
    assert not distill.all_finite([torch.zeros(3), t])


def test_all_finite_of_nothing_is_true():
    assert distill.all_finite([])


def test_no_sentinel_lets_a_run_start(tmp_path):
    assert distill.refuse_if_diverged(tmp_path) is None


def test_a_sentinel_stops_a_run_before_it_starts(tmp_path):
    (tmp_path / distill.DIVERGED_SENTINEL).write_text("{}\n")
    with pytest.raises(SystemExit) as exc:
        distill.refuse_if_diverged(tmp_path)
    assert exc.value.code == 3


def test_declaring_divergence_records_why_and_saves_nothing(tmp_path):
    with pytest.raises(SystemExit) as exc:
        distill.declare_divergence(tmp_path, 36620, "loss went to nan")
    assert exc.value.code == 3
    record = json.loads((tmp_path / distill.DIVERGED_SENTINEL).read_text())
    assert record["step"] == 36620
    assert "nan" in record["reason"]
    assert not list(tmp_path.glob("checkpoint-*"))


class TinyStudent(torch.nn.Module):
    """One trainable vector, so there is a real backward and a real update."""

    def __init__(self, fill: float = 0.0):
        super().__init__()
        self.w = torch.nn.Parameter(torch.full((VOCAB,), fill))

    def forward(self, **batch):
        n = batch["labels"].shape[0]
        return FakeOut(self.w.expand(n, SEQ, VOCAB))

    def save_pretrained(self, path, safe_serialization=True):
        torch.save(self.state_dict(), Path(path) / "student.pt")


class PoisonTeacher:
    """Uniform logits, except nan on the calls listed in `poison`."""

    def __init__(self, poison=()):
        self.poison = set(poison)
        self.calls = 0

    def __call__(self, **batch):
        i = self.calls
        self.calls += 1
        n = batch["labels"].shape[0]
        fill = float("nan") if i in self.poison else 0.1
        return FakeOut(torch.full((n, SEQ, VOCAB), fill))


class StubTokenizer:
    pad_token_id = 0
    eos_token = "</s>"

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def save_pretrained(self, *a, **k):
        pass


def run_train(monkeypatch, tmp_path, *, n_batches=12, poison=(),
              max_skipped=3, student=None):
    """Run the real `train()` on CPU with stand-ins for everything heavy."""
    teacher = PoisonTeacher(poison)
    student = student if student is not None else TinyStudent()
    loaded = {"teacher": False}

    def fake_teacher(*a, **k):
        loaded["teacher"] = True
        return teacher

    monkeypatch.setattr(distill, "AutoTokenizer", StubTokenizer)
    monkeypatch.setattr(distill, "build_dataloaders",
                        lambda cfg, tok: (list(batches(n_batches)), []))
    monkeypatch.setattr(distill, "load_teacher", fake_teacher)
    monkeypatch.setattr(distill, "load_student", lambda *a, **k: student)

    c = distill.DistillConfig(
        output_dir=str(tmp_path), student_device="cpu", bf16=False,
        gradient_checkpointing=False, gradient_accumulation_steps=2,
        logging_steps=1, eval_steps=0, save_steps=2, save_total_limit=3,
        warmup_steps=0, num_train_epochs=1, max_skipped_updates=max_skipped,
    )
    distill.train(c)
    return student, teacher, loaded


def steps_on_disk(tmp_path):
    return sorted(int(p.name.split("-")[1]) for p in tmp_path.glob("checkpoint-*"))


def test_a_clean_run_skips_nothing(monkeypatch, tmp_path, capsys):
    """For a healthy run the guard must change nothing."""
    student, _, _ = run_train(monkeypatch, tmp_path)
    assert "[guard]" not in capsys.readouterr().err
    assert steps_on_disk(tmp_path) == [2, 4, 6]       # 12 batches / 2 = 6
    assert not (tmp_path / distill.DIVERGED_SENTINEL).exists()
    assert distill.all_finite(student.parameters())


def test_one_poisoned_batch_costs_one_update_not_the_run(monkeypatch, tmp_path,
                                                         capsys):
    """The case that killed the control arm: a single bad batch."""
    student, _, _ = run_train(monkeypatch, tmp_path, poison={5})
    err = capsys.readouterr().err
    assert "skipped an update" in err
    # Window 3 (batches 4 and 5) is dropped: five updates instead of six.
    assert steps_on_disk(tmp_path)[-1] == 5
    assert distill.all_finite(student.parameters())
    assert not (tmp_path / distill.DIVERGED_SENTINEL).exists()


def test_a_run_that_stays_nan_stops_and_keeps_the_healthy_checkpoints(
        monkeypatch, tmp_path):
    """The other half: a real divergence is stopped, and stopped WITHOUT
    saving — so the checkpoint written before it survives the rotation that
    destroyed the control arm's."""
    with pytest.raises(SystemExit) as exc:
        run_train(monkeypatch, tmp_path, poison=set(range(4, 100)),
                  max_skipped=3)
    assert exc.value.code == 3
    record = json.loads((tmp_path / distill.DIVERGED_SENTINEL).read_text())
    assert record["step"] == 2
    assert steps_on_disk(tmp_path) == [2]
    saved = torch.load(tmp_path / "checkpoint-2" / "student.pt")
    assert distill.all_finite(saved.values())


def test_a_diverged_chain_refuses_before_loading_the_teacher(monkeypatch,
                                                             tmp_path):
    """Every later link of the afterany chain must cost seconds, not a
    61 GB teacher load."""
    (tmp_path / distill.DIVERGED_SENTINEL).write_text('{"step": 36620}\n')
    loaded = {}
    with pytest.raises(SystemExit) as exc:
        _, _, loaded = run_train(monkeypatch, tmp_path)
    assert exc.value.code == 3
    assert loaded == {}          # run_train never returned: nothing loaded


def test_resuming_from_a_non_finite_checkpoint_is_caught_at_load(monkeypatch,
                                                                 tmp_path):
    """A checkpoint already holding nan would make every step nan; catching
    it at load costs one model load instead of the whole link."""
    (tmp_path / "checkpoint-4").mkdir()
    monkeypatch.setattr(distill, "_reload_student_from_checkpoint",
                        lambda *a, **k: TinyStudent(float("nan")))
    monkeypatch.setattr(distill, "load_checkpoint", lambda *a, **k: 4)
    with pytest.raises(SystemExit) as exc:
        run_train(monkeypatch, tmp_path)
    assert exc.value.code == 3
    record = json.loads((tmp_path / distill.DIVERGED_SENTINEL).read_text())
    assert record["step"] == 4
    assert "checkpoint-4" in record["reason"]


# --- a resumed run continues the data where it stopped ---------------------------
#
# Regression for the split arm, 25 September: every resume started the data
# from the top, so a chain of 2-hour links retrained the same ~650 steps of
# corpus a dozen times and held-out perplexity went 5.11 -> 5.92.

class RecordingTeacher:
    """Uniform logits; remembers which batch (by its id) it was shown."""

    def __init__(self):
        self.seen = []

    def __call__(self, **batch):
        self.seen.append(int(batch["input_ids"][0, 0]))
        n = batch["labels"].shape[0]
        return FakeOut(torch.full((n, SEQ, VOCAB), 0.1))


def numbered_batches(n: int):
    return [{"labels": torch.zeros(1, SEQ, dtype=torch.long),
             "input_ids": torch.full((1, SEQ), i, dtype=torch.long)}
            for i in range(n)]


def run_link(monkeypatch, tmp_path, *, n_batches=12, max_steps=-1):
    """One link of a chain: train() on CPU, resuming from tmp_path if it can."""
    teacher = RecordingTeacher()

    def reload(ckpt_dir, cfg, dtype, device):
        student = TinyStudent()
        student.load_state_dict(torch.load(Path(ckpt_dir) / "student.pt"))
        return student

    monkeypatch.setattr(distill, "AutoTokenizer", StubTokenizer)
    monkeypatch.setattr(distill, "build_dataloaders",
                        lambda cfg, tok: (numbered_batches(n_batches), []))
    monkeypatch.setattr(distill, "load_teacher", lambda *a, **k: teacher)
    monkeypatch.setattr(distill, "load_student", lambda *a, **k: TinyStudent())
    monkeypatch.setattr(distill, "_reload_student_from_checkpoint", reload)

    distill.train(distill.DistillConfig(
        output_dir=str(tmp_path), student_device="cpu", bf16=False,
        gradient_checkpointing=False, gradient_accumulation_steps=2,
        logging_steps=1, eval_steps=0, save_steps=1, save_total_limit=10,
        warmup_steps=0, num_train_epochs=1, max_steps=max_steps,
    ))
    return teacher.seen


def test_a_resumed_link_continues_the_data_instead_of_restarting_it(monkeypatch, tmp_path):
    # First link stops after 2 optimizer steps = 4 micro-batches (0-3).
    first = run_link(monkeypatch, tmp_path, max_steps=2)
    assert first == [0, 1, 2, 3]

    # The next link resumes at step 2 and must go on from batch 4. Before the
    # fix it saw [0, 1, 2, …] again — the same data as the link before it.
    second = run_link(monkeypatch, tmp_path)
    assert second == list(range(4, 12))


def test_a_resumed_run_stops_at_its_schedule_not_after_another_epoch(monkeypatch, tmp_path):
    run_link(monkeypatch, tmp_path, max_steps=2)
    run_link(monkeypatch, tmp_path)
    # 12 batches / 2 per step = 6 steps in the epoch, and not one more.
    assert max(steps_on_disk(tmp_path)) == 6

    # A link submitted after the run is complete trains on nothing.
    assert run_link(monkeypatch, tmp_path) == []


def test_data_position_counts_micro_batches_across_epochs():
    assert distill.data_position(0, 8, 100) == (0, 0)
    assert distill.data_position(10, 8, 100) == (0, 80)
    assert distill.data_position(25, 8, 100) == (2, 0)
    assert distill.data_position(26, 8, 100) == (2, 8)


def test_a_real_loader_has_one_fixed_order_per_epoch_and_skips_by_batch():
    """The permutation depends on (seed, epoch) only, so every link agrees on it."""
    loader = torch.utils.data.DataLoader(list(range(20)), batch_size=4, shuffle=True)
    a = [b.tolist() for b in distill.epoch_batches(loader, epoch=0, skip=0, seed=7)]
    torch.manual_seed(12345)          # whatever else consumed the global RNG
    b = [b.tolist() for b in distill.epoch_batches(loader, epoch=0, skip=0, seed=7)]
    assert a == b
    assert sorted(x for batch in a for x in batch) == list(range(20))

    rest = [b.tolist() for b in distill.epoch_batches(loader, epoch=0, skip=2, seed=7)]
    assert rest == a[2:]

    other = [b.tolist() for b in distill.epoch_batches(loader, epoch=1, skip=0, seed=7)]
    assert other != a
