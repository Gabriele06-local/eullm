"""The launcher wrapper must not decide which checkpoint a resume uses.

`distill.py` sorts checkpoints by step number and logs the one it loads. The
wrapper used to look for `checkpoint-*` itself, take the one with the newest
mtime and pass it as `--resume-from`, which is how every phase-2 job starts.
mtime is not the step number: anything that touches a checkpoint directory
without writing a checkpoint reorders that list — an `rsync` of the run
directory between links is exactly that — and a resume then reloads a
checkpoint half the run old and redoes thousands of steps.

These are static assertions on the wrapper, and they run everywhere, including
on a machine with no bash. Running the wrapper for real needs bash, and the
guard is worth having in both places: a stubbed `python` on PATH would prove
the same thing more directly, but that test could only be run on a Linux CI
runner, so it would not have been run at all while this change was written.
The behaviour of the choice itself is covered in test_distill_script.py, where
`latest_checkpoint` is asserted to pick the highest step.
"""

from __future__ import annotations

from pathlib import Path

WRAPPER = Path(__file__).resolve().parents[1] / "scripts" / "distill.sh"


def _text() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def _code() -> str:
    """The wrapper without its comments, which discuss the very flags below.

    Every comment in distill.sh is a whole line — nothing trails a command
    with a `#` — so dropping those lines leaves the code that is executed.
    """
    return "\n".join(ln for ln in _text().splitlines()
                     if not ln.strip().startswith("#"))


def test_the_wrapper_lets_distill_py_choose_the_checkpoint():
    assert "--resume-from" not in _code()


def test_the_wrapper_does_not_rank_checkpoints_by_mtime():
    # `-printf '%T@` is what made mtime the ordering, and `sort -rn` on it.
    for fragment in ("%T@", "sort -rn", "-name 'checkpoint-*'"):
        assert fragment not in _code(), fragment


def test_the_wrapper_still_passes_the_config_and_the_data_dir():
    text = _code()
    assert '--config "$CONFIG"' in text
    assert '--dataset-dir "$DATA_DIR"' in text
    # ...and still refuses a run it cannot satisfy, rather than letting the
    # trainer discover it 40 minutes in.
    assert 'err "missing $DATA_DIR/train.jsonl"' in text
    assert 'err "missing $DATA_DIR/val.jsonl"' in text
