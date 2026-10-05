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
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leonardo" / "status.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash (on Windows, bash is WSL's)")


def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def run_status(tmp_path: Path, queue: list[tuple[str, str, str]],
               ended: list[tuple[str, str, str, str]],
               gpu_queue: list[str] | None = None, disk_pct: int = 50) -> str:
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    q_reason = "\\n".join("|".join(j) for j in queue)
    q_state = "\\n".join(f"{n} {s}" for n, s, _ in queue)
    q_name = "\\n".join(n for n, _, _ in queue)
    # what is on the GPU partition: by default one job, so that the tests of
    # other situations are not also an idle allocation
    q_gpu = "\\n".join(["1"] if gpu_queue is None else gpu_queue)
    _exe(bin_ / "squeue", f"""#!/usr/bin/env bash
case "$*" in
  *boost_usr_prod*) [ -n "{q_gpu}" ] && printf '{q_gpu}\\n' ;;
  *"%j|%T|%r"*) printf '{q_reason}\\n' ;;
  *"%j %T"*) printf '{q_state}\\n' ;;
  *) printf '{q_name}\\n' ;;
esac
""")
    rows = "\\n".join("|".join(e) for e in ended)
    short = "\\n".join(f"{n}|{s}" for _, n, s, _ in ended)
    _exe(bin_ / "sacct", f"""#!/usr/bin/env bash
case "$*" in
  *ElapsedRaw*) printf '6480 1\\n' ;;
  *JobID*) printf '{rows}\\n' ;;
  *) printf '{short}\\n' ;;
esac
""")
    _exe(bin_ / "df", f"""#!/usr/bin/env bash
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'
printf 'lustre 1000 {disk_pct * 10} 0 {disk_pct}%% /work\\n'
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


def test_a_stage3_link_after_training_finished_is_not_an_alarm(tmp_path, runs):
    """2026-09-30: the v0.4 chain's spare link resumed from the final
    checkpoint, saved the adapter again and ended in 2:43 — flagged as a
    link that did no work, when the work had been done by the link before."""
    (runs / "logs" / "eullm-stage3-59054377.out").write_text(
        "[s3] out    /w/sft-v04\n[stage3] adapter /w/sft-v04/adapter\n")
    (runs / "logs" / "eullm-s3-it4b-v05-60.out").write_text(
        "[stage3] adapter already at /w/sft-it4b-v05/adapter: training finished, "
        "nothing left to do (move it away to train again)\n")
    out = run_status(tmp_path, queue=[],
                     ended=[("59054377", "eullm-stage3", "COMPLETED", "00:02:43"),
                            ("60", "eullm-s3-it4b-v05", "COMPLETED", "00:00:40")])
    assert "[!!]" not in out, out
    assert "training finished (fine)" in out
    assert "60 eullm-s3-it4b-v05: ended in 00:00:40 with nothing left to do" in out


def test_a_grpo_link_after_training_finished_is_not_an_alarm(tmp_path, runs):
    """A GRPO link that resumed at the last step saved and ended, which is
    what grpo_train.py does with no steps left. Same case as the stage-3 link
    above, and it was raising the alarm: eullm-grpo* had been added to the list
    of names checked, but the escape only matched the stage-3 prefix."""
    (runs / "logs" / "eullm-grpo-77.out").write_text(
        "[grpo] resuming from /w/eullm_runs/grpo/v03/checkpoint-9000\n"
        "[grpo] adapter /w/eullm_runs/grpo/v03/adapter\n")
    (runs / "logs" / "eullm-grpo-78.out").write_text(
        "[grpo] adapter already at /w/eullm_runs/grpo/v03/adapter: nothing "
        "left to do (move it away to train again)\n")
    out = run_status(tmp_path, queue=[],
                     ended=[("77", "eullm-grpo", "COMPLETED", "00:04:31"),
                            ("78", "eullm-grpo", "COMPLETED", "00:00:12")])
    assert "[!!]" not in out, out
    assert "77 eullm-grpo: ended in 00:04:31, training finished (fine)" in out
    assert "78 eullm-grpo: ended in 00:00:12 with nothing left to do" in out


def test_a_named_stage3_link_that_did_nothing_is_flagged(tmp_path, runs):
    """A chain renamed with -J is still a training chain: a three-minute link
    that never reached the end of training needs a human."""
    (runs / "logs" / "eullm-s3-q35-9b-v04-61.out").write_text(
        "[s3] base   /w/Qwen3.5-9B\nTraceback (most recent call last):\n")
    out = run_status(tmp_path, queue=[],
                     ended=[("61", "eullm-s3-q35-9b-v04", "COMPLETED", "00:03:00")])
    assert "61 eullm-s3-q35-9b-v04 ended COMPLETED after only 00:03:00" in out


def test_jobs_that_are_short_on_purpose_are_not_stalled_links(tmp_path, runs):
    """Two jobs the eullm-p* glob took, and both are short on purpose.

    sbatch_quantize.slurm submits eullm-p3-gguf on lrd_all_serial with no
    --gres at all -- a CPU quantize, done in minutes. sbatch_backfill_probe's
    own header says eullm-probe "does nothing but report where it landed and
    exit", so it ends in seconds every time it runs. Neither resumes a chain,
    so neither has work it could have failed to do, and a morning check that
    cries wolf is one that gets ignored. The real chain link beside them keeps
    its alarm.
    """
    (runs / "logs" / "eullm-p3-gguf-62.out").write_text(
        "[quant] llama-quantize model-f16.gguf model-Q4_K_M.gguf\n")
    (runs / "logs" / "eullm-probe-61.out").write_text("[probe] backfill window found\n")
    (runs / "logs" / "eullm-p2-8b-63.out").write_text("[distill] nothing at all happened\n")
    out = run_status(tmp_path, queue=[],
                     ended=[("61", "eullm-probe", "COMPLETED", "00:00:21"),
                            ("62", "eullm-p3-gguf", "COMPLETED", "00:04:30"),
                            ("63", "eullm-p2-8b", "COMPLETED", "00:00:21")])
    assert "eullm-probe ended COMPLETED" not in out, out
    assert "eullm-p3-gguf ended COMPLETED" not in out, out
    assert "63 eullm-p2-8b ended COMPLETED after only 00:00:21" in out
    assert "[!!] 1 thing(s) above need a look" in out, out


def test_grpo_progress_is_shown_and_a_stop_is_flagged(tmp_path, runs):
    logs = runs.parent / "grpo" / "logs"
    logs.mkdir(parents=True)
    (logs / "eullm-grpo-70.out").write_text(
        "[grpo] step 10/250 reward 0.712 (std 0.301) zero-std 0.45 kl 0.0010 len 88 31s/step\n"
        " 8%|##  | 20/250 [10:00<1:55:00, 30.0s/it]"     # tqdm's bar, no newline
        "[grpo] step 20/250 reward 0.744 (std 0.288) zero-std 0.48 kl 0.0021 len 85 30s/step\n")
    (logs / "eullm-grpo-71.out").write_text(
        "[grpo] step 10/250 reward 0.990 (std 0.010) zero-std 0.98 kl 0.0001 len 60 30s/step\n"
        "[grpo] STOP: 20 steps in a row with 98% of groups all-equal (step 30): "
        "nothing left to learn from these prompts\n")
    out = run_status(tmp_path, queue=[], ended=[])
    assert "     [grpo] step 20/250 reward 0.744" in out
    assert "[!!] eullm-grpo-71.out: STOP: 20 steps in a row" in out
    assert "1 thing(s) above need a look" in out


def test_an_idle_gpu_queue_is_flagged_and_a_queued_chain_is_not(tmp_path, runs):
    """2026-10-04: a day at 1.8 node-hours, noticed by the user, not by this."""
    out = run_status(tmp_path, queue=[("eullm-gguf-grpo-v04", "RUNNING", "None")],
                     ended=[], gpu_queue=[])
    assert "the allocation is idle" in out
    assert "1.8 node-hours since" in out
    out = run_status(tmp_path, queue=[("eullm-grpo-r2-v04", "PENDING", "Dependency")],
                     ended=[], gpu_queue=["59332673"])
    assert "idle" not in out and "nothing wrong found" in out


def test_a_full_disk_is_flagged(tmp_path, runs):
    """2026-10-05: $WORK at 109% of its quota, found by a conversion failing."""
    out = run_status(tmp_path, queue=[("eullm-grpo", "RUNNING", "None")], ended=[],
                     disk_pct=94)
    assert "[!!] $WORK is 94% full" in out
    out = run_status(tmp_path, queue=[("eullm-grpo", "RUNNING", "None")], ended=[],
                     disk_pct=78)
    assert "78% full" in out and "nothing wrong found" in out
