"""sbatch_norm_exam.slurm as one link of a chain, with legal_eval.py stubbed.

What is pinned: an answers file with one line per item is a finished run and
is not asked again, a shorter one is asked again, and EXAM_OPEN_ONLY drops the
closed-book pass. The model is never loaded: `python` is a stub that writes a
full answers file and records what it was asked.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "forge" / "scripts" / "leonardo" / "sbatch_norm_exam.slurm"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash (on Windows, bash is WSL's)")

# Writes as many lines as the exam has to the --answers path and logs the label.
STUB = r'''#!/usr/bin/env bash
items=""; answers=""; label=""
while [ $# -gt 0 ]; do
    case "$1" in
        --items) items="$2"; shift ;;
        --answers) answers="$2"; shift ;;
        --label) label="$2"; shift ;;
    esac
    shift
done
echo "$label" >> "$CALLS"
[ -n "$answers" ] && sed 's/.*/{}/' "$items" > "$answers"
exit 0
'''


@pytest.fixture
def job(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    items = work / "exam-dev.jsonl"
    items.write_text("{}\n{}\n{}\n")
    model = work / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    out = work / "answers"
    out.mkdir()
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    py = bin_ / "python"
    py.write_text(STUB)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    calls = tmp_path / "calls"
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "WORK": str(work),
           "CALLS": str(calls), "EXAM_REPO": str(REPO), "EXAM_ITEMS": str(items),
           "EXAM_OUT": str(out), "EXAM_MODELS": f"a={model} b={model}",
           "EXAM_NORMS": str(work / "norms.jsonl"), "EULLM_VENV": str(tmp_path / "nov"),
           # the heartbeat's sleep holds the output pipe until it wakes
           "HEARTBEAT_INTERVAL": "1"}
    env.pop("SLURM_JOB_ID", None)

    def run(**extra):
        r = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env={**env, **extra},
                           capture_output=True, text=True, timeout=60)
        asked = calls.read_text().split() if calls.exists() else []
        calls.unlink(missing_ok=True)
        return r, asked

    return run, out


def test_a_second_link_asks_only_what_the_first_did_not_finish(job):
    run, out = job
    (out / "answers-a.jsonl").write_text("{}\n{}\n{}\n")        # whole
    (out / "answers-a-open.jsonl").write_text("{}\n")           # cut short
    r, asked = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert asked == ["a-open", "b", "b-open"]
    assert "a: answered already, skipped" in r.stdout
    r, asked = run()
    assert r.returncode == 0 and asked == []


def test_open_only_skips_the_closed_book_pass(job):
    run, _ = job
    r, asked = run(EXAM_OPEN_ONLY="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert asked == ["a-open", "b-open"]
