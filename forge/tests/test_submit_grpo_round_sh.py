"""submit_grpo_round.sh: one GRPO round queued whole, each job on the one it needs."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leonardo" / "submit_grpo_round.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash")


def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _setup(tmp_path: Path, queued: str = "") -> tuple[dict, Path]:
    work = tmp_path / "work"
    (work / "eullm_runs" / "grpo").mkdir(parents=True)
    (work / "eullm_runs" / "grpo" / "prompts-judge.jsonl").write_text("{}\n")
    judge = work / "gguf" / "qwen3-30b-a3b-instruct-2507" / "qwen3-30b-a3b-instruct-2507-q8_0.gguf"
    judge.parent.mkdir(parents=True)
    judge.write_text("gguf")
    start = work / "start"
    start.mkdir()
    (start / "config.json").write_text("{}")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "sbatch.log"
    # each call: its arguments and the variables the slurm scripts read, one line;
    # job ids 100, 101, ...
    _exe(bin_ / "sbatch", f"""#!/usr/bin/env bash
n=$(( $(wc -l < "{log}" 2>/dev/null || echo 0) + 100 ))
{{ printf '%s|%s|GRPO_MODEL=%s|GRPO_ARGS=%s' "$n" "$*" "${{GRPO_MODEL:-}}" "${{GRPO_ARGS:-}}"
  printf '|CE_LABEL=%s|ROUND_MODELS=%s' "${{CE_LABEL:-}}" "${{ROUND_MODELS:-}}"
  printf '|ROUND_GGUF=%s|ROUND_REPORT_FILE=%s\\n' "${{ROUND_GGUF:-}}" "${{ROUND_REPORT_FILE:-}}"
}} >> "{log}"
echo "$n"
""")
    _exe(bin_ / "squeue", f"#!/usr/bin/env bash\nprintf '%s' '{queued}'\n")
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "WORK": str(work)}
    return env, log


def _run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True,
                          text=True, timeout=30)


def test_the_round_is_queued_whole_and_in_order(tmp_path):
    env, log = _setup(tmp_path)
    start = Path(env["WORK"]) / "start"
    r = _run(env, "jg-x", str(start), "3", "750")
    assert r.returncode == 0, r.stderr + r.stdout
    calls = [ln.split("|") for ln in log.read_text().splitlines()]
    names = [next(a for a in c[1].split() if a.startswith("eullm-")) for c in calls]
    assert names == ["eullm-grpo-jg-x"] * 3 + ["eullm-merge-jg-x", "eullm-gguf-jg-x",
                                               "eullm-cds-exam", "eullm-exam-round-jg-x"]
    ids = [c[0] for c in calls]
    # the first link waits on nothing, the next ones on the link before
    assert "--dependency" not in calls[0][1]
    assert f"--dependency=afterany:{ids[0]}" in calls[1][1]
    assert f"--dependency=afterany:{ids[1]}" in calls[2][1]
    assert f"--dependency=afterany:{ids[2]}" in calls[3][1]          # merge after the last link
    assert f"--dependency=afterok:{ids[3]}" in calls[4][1]           # GGUF after the merge
    assert f"--dependency=afterok:{ids[3]}" in calls[5][1]           # case law after the merge
    assert f"--dependency=afterok:{ids[4]}" in calls[6][1]           # statute exam after the GGUF
    assert calls[0][2] == f"GRPO_MODEL={start}" and "--max-steps 750" in calls[0][3]
    assert calls[5][4] == "CE_LABEL=jg-x-bf16"
    assert calls[6][5].startswith("ROUND_MODELS=jg-x-q4=")
    assert calls[6][6].endswith("/grpo/jg-x/gguf/jg-x-q4_k_m.gguf")
    assert calls[6][7] == "ROUND_REPORT_FILE=report-jg-x.txt"
    assert "rm -f" in calls[4][1] and "jg-x-f16.gguf" in calls[4][1]


def test_a_round_already_queued_is_not_submitted_again(tmp_path):
    env, log = _setup(tmp_path, queued="123")
    r = _run(env, "jg-x", str(Path(env["WORK"]) / "start"), "3", "750")
    assert r.returncode == 0 and "already queued" in r.stdout
    assert not log.exists()


def test_after_chains_on_a_previous_round_and_skips_the_start_check(tmp_path):
    env, log = _setup(tmp_path)
    r = _run({**env, "AFTER": "77", "NO_CDS": "1"}, "jg-y", "/not/yet/merged", "1", "10")
    assert r.returncode == 0, r.stderr
    calls = [ln.split("|") for ln in log.read_text().splitlines()]
    assert "--dependency=afterok:77" in calls[0][1]
    assert not any("eullm-cds-exam" in c[1] for c in calls)


def test_missing_inputs_stop_before_anything_is_queued(tmp_path):
    env, log = _setup(tmp_path)
    r = _run(env, "jg-x", "/no/such/model", "3", "750")
    assert r.returncode == 1 and "missing start model" in r.stderr
    assert not log.exists()
