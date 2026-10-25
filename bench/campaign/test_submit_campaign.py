"""submit_campaign.sh asks Slurm for the time the label's waiting points
need (campaign.py walltime), not sbatch_campaign.slurm's 48 hours, which kept
jobs of a few hours waiting on priority (24-10-2026); --time among the sbatch
args overrides it, and with nothing waiting for the label nothing is
submitted. sbatch and squeue are stand-ins that record what they were asked.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "lumi" / "submit_campaign.sh"
CAMPAIGN = ROOT / "bench" / "campaign" / "campaign.py"
FAKE = ROOT / "bench" / "campaign" / "fake_eullm.py"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash")


@pytest.fixture()
def lumi(tmp_path):
    spec = tmp_path / "c-mtp.json"
    spec.write_text(json.dumps({
        "campaign": "c-mtp", "defaults": {"gcds": 1},
        "groups": [{"name": "mtp-ab", "priority": 95, "est_s": 1500,
                    "set": {"kind": "workload", "sets": ["gsm8k"], "limit": 100},
                    "axes": {"model": ["qwen3-8b", "qwen3-4b"],
                             "extra_args": [[], ["--mtp", "1"], ["--mtp", "2"],
                                            ["--mtp", "3"]]}}]}))
    queue = tmp_path / "campaign"
    subprocess.run([sys.executable, str(CAMPAIGN), "plan", "--queue", str(queue),
                    "--engine-label", "next26", "--engine", str(FAKE), str(spec)],
                   check=True, capture_output=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "sbatch.log"
    (bin_dir / "sbatch").write_text(f'#!/bin/bash\necho "$*" >> {log}\necho 4242\n')
    (bin_dir / "squeue").write_text("#!/bin/bash\nexit 0\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
               CAMPAIGN_DIR=str(queue), EULLM_BIN=str(FAKE), SBATCH_ACCOUNT="project_test",
               PY=sys.executable, EULLM_ENGINE_LABEL="next26", USER="tester")
    env.pop("EULLM_REPO", None)

    def submit(*args, **extra):
        p = subprocess.run(["bash", str(SCRIPT), *args], env=dict(env, **extra),
                           capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, p.stdout + p.stderr
        return p.stdout, log.read_text().splitlines() if log.exists() else []

    return submit


def test_a_job_asks_for_the_time_its_label_needs(lumi):
    out, calls = lumi("1", "1")
    assert "time per job: 2:00:00" in out
    assert len(calls) == 1 and "--time=2:00:00" in calls[0].split()


def test_a_time_given_to_sbatch_is_kept(lumi):
    _, calls = lumi("1", "1", "--time=6:00:00")
    assert len(calls) == 1 and "--time=6:00:00" in calls[0].split()
    assert "--time=2:00:00" not in calls[0].split()


def test_nothing_is_submitted_for_a_label_with_nothing_waiting(lumi):
    out, calls = lumi("1", "1", EULLM_ENGINE_LABEL="next99")
    assert "nothing submitted" in out and calls == []


def test_a_job_with_fewer_gcds_asks_for_longer(lumi):
    # Eight 25-minute points on four GCDs of small-g run in two waves.
    out, calls = lumi("1", "1", "--partition=small-g", "--gpus-per-node=4")
    assert "time per job: 3:00:00" in out
    assert len(calls) == 1 and "--time=3:00:00" in calls[0].split()
    _, calls = lumi("1", "1", "--gpus-per-node", "2")
    assert "--time=4:00:00" in calls[-1].split()
