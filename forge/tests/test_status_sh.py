"""`status.sh` on the situations it has actually met, with Slurm stubbed.

A morning check that cries wolf gets ignored, and one that misses the real
problem is worse than none. Its first run (2026-09-28) raised three alarms and
all three were noise: the last generation link exiting in seconds because
nothing was left, a watcher whose file had arrived minutes after its last
wake, and the package watcher waiting — correctly — for a training run. The
cases below pin both sides: the noise stays quiet, the real problems do not.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leonardo" / "status.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def run_status(tmp_path: Path, queue: list[tuple[str, str, str]],
               ended: list[tuple[str, str, str, str]]) -> str:
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    q_reason = "\\n".join("|".join(j) for j in queue)
    q_state = "\\n".join(f"{n} {s}" for n, s, _ in queue)
    q_name = "\\n".join(n for n, _, _ in queue)
    _exe(bin_ / "squeue", f"""#!/usr/bin/env bash
case "$*" in
  *"%j|%T|%r"*) printf '{q_reason}\\n' ;;
  *"%j %T"*) printf '{q_state}\\n' ;;
  *) printf '{q_name}\\n' ;;
esac
""")
    rows = "\\n".join("|".join(e) for e in ended)
    short = "\\n".join(f"{n}|{s}" for _, n, s, _ in ended)
    _exe(bin_ / "sacct", f"""#!/usr/bin/env bash
case "$*" in
  *JobID*) printf '{rows}\\n' ;;
  *) printf '{short}\\n' ;;
esac
""")
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}",
           "EULLM_RUNS": str(tmp_path / "runs")}
    out = subprocess.run(["bash", str(SCRIPT), "2026-09-27T21:00"], env=env,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.fixture
def runs(tmp_path):
    (tmp_path / "runs" / "stage3" / "logs").mkdir(parents=True)
    return tmp_path / "runs" / "stage3"


def watcher_log(runs: Path, name: str, need: Path, left: int = 100) -> None:
    (runs / "logs" / f"{name}-9.out").write_text(
        f"[wait] 2026-09-28 12:34:09 not yet: {need}\n"
        f"[wait] 2026-09-28 12:34:09 will look again in 20 minutes (job 9, {left} tries left)\n")


def test_this_mornings_three_false_alarms_stay_quiet(tmp_path, runs):
    (runs / "logs" / "eullm-gen-openbook-30.out").write_text(
        "[ob] nothing left to do -> x.done\n")
    done = runs / "openbook-pairs.jsonl.done"
    done.write_text("done\n")                      # arrived after the last wake
    watcher_log(runs, "wait-eullm-stage3", done)
    (runs / "logs" / "wait-eullm-stage3-pkg-8.out").write_text(
        f"[wait] 2026-09-28 12:34:09 not yet: {runs / 'sft' / 'adapter_config.json'}\n"
        "[wait] 2026-09-28 12:34:09 will look again in 20 minutes (job 8, 100 tries left)\n")
    out = run_status(
        tmp_path,
        queue=[("wait-eullm-stage3", "PENDING", "BeginTime"),
               ("wait-eullm-stage3-pkg", "PENDING", "BeginTime"),
               ("eullm-stage3", "RUNNING", "None")],
        ended=[("30", "eullm-gen-openbook", "COMPLETED", "00:00:07"),
               ("1", "eullm-p2-amm", "TIMEOUT", "02:00:10")])
    assert "[!!]" not in out, out
    assert "nothing wrong found" in out
    assert "submits at its next wake" in out


def test_the_real_problems_are_flagged(tmp_path, runs):
    empty = runs / "openbook-pairs.jsonl.done"
    empty.touch()                                   # last night's bug
    watcher_log(runs, "wait-eullm-stage3", empty)
    out = run_status(
        tmp_path,
        queue=[("wait-eullm-stage3", "PENDING", "BeginTime"),
               ("eullm-p2-8b", "PENDING", "DependencyNeverSatisfied")],
        ended=[("2", "eullm-p2-amm", "COMPLETED", "00:00:03"),   # quota, gate, missing file
               ("3", "eullm-stage3", "FAILED", "00:12:00"),
               ("4", "eullm-p2-split", "OUT_OF_MEMORY", "00:40:00")])
    assert "exists but is EMPTY" in out
    assert "DependencyNeverSatisfied" in out
    assert "a link that short did no work" in out
    assert "eullm-stage3 ended FAILED" in out
    assert "ended OUT_OF_MEMORY" in out
    assert "5 thing(s) above need a look" in out
