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


def _setup(tmp_path: Path, queued: str = "", serial: int = 0,
           packaging: str = "") -> tuple[dict, Path]:
    work = tmp_path / "work"
    (work / "eullm_runs" / "grpo").mkdir(parents=True)
    (work / "eullm_runs" / "grpo" / "prompts-judge.jsonl").write_text("{}\n")
    judge = work / "gguf" / "qwen3-30b-a3b-instruct-2507" / "qwen3-30b-a3b-instruct-2507-q8_0.gguf"
    judge.parent.mkdir(parents=True)
    judge.write_text("gguf")
    (work / "eval").mkdir()
    (work / "eval" / "practical-dev.jsonl").write_text("{}\n")
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
  printf '|CE_LABEL=%s|EXAM_MODELS=%s' "${{CE_LABEL:-}}" "${{EXAM_MODELS:-}}"
  printf '|EXAM_GGUF=%s|EXAM_ITEMS=%s\\n' "${{EXAM_GGUF:-}}" "${{EXAM_ITEMS:-}}"
}} >> "{log}"
echo "$n"
""")
    # -p lrd_all_serial: that many jobs; -n <name>: the queued GRPO chain or
    # packaging, if any
    _exe(bin_ / "squeue", f"""#!/usr/bin/env bash
case "$*" in
  *lrd_all_serial*) for i in $(seq 1 {serial}); do echo "$i"; done ;;
  *eullm-package-*) printf '%s' '{packaging}' ;;
  *) printf '%s' '{queued}' ;;
esac
""")
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
    names = [next((a for a in c[1].split() if a.startswith("eullm-")), "judge") for c in calls]
    assert names == ["eullm-grpo-jg-x"] * 3 + ["eullm-package-jg-x", "eullm-cds-exam",
                                               "eullm-exam-jg-x", "eullm-exam-jg-x", "judge",
                                               "eullm-exam-jg-x", "eullm-exam-jg-x", "judge",
                                               "eullm-exam-jg-x", "judge"]
    ids = [c[0] for c in calls]
    # the first link waits on nothing, the next ones on the link before
    assert "--dependency" not in calls[0][1]
    assert f"--dependency=afterany:{ids[0]}" in calls[1][1]
    assert f"--dependency=afterany:{ids[1]}" in calls[2][1]
    assert f"--dependency=afterany:{ids[2]}" in calls[3][1]       # packaging after the last link
    assert f"--dependency=afterok:{ids[3]}" in calls[4][1]        # case law after the packaging
    assert f"--dependency=afterok:{ids[3]}" in calls[5][1]        # statute exam after it too
    assert f"--dependency=afterany:{ids[5]}" in calls[6][1]       # its second link
    assert f"--dependency=afterany:{ids[6]}" in calls[7][1]       # then its judge
    assert "JUDGE_GLOB=answers-jg-x-q4-hyb.jsonl" in calls[7][1]
    assert "--graded-dir" in calls[7][1] and "devbig-graded-v3" in calls[7][1]
    # the same exam without the note, as RAG Enterprise asks it
    assert "JUDGE_GLOB=answers-jg-x-q4-hybnn.jsonl" in calls[10][1]
    assert calls[11][7].endswith("practical-dev.jsonl")           # the practical exam
    assert "practical-answers" in calls[12][1]
    assert "lrd_all_serial" in calls[3][1] and "merge_identity_adapter" in calls[3][1]
    assert "quantize_to_gguf.sh" in calls[3][1] and "jg-x-f16.gguf" in calls[3][1]
    assert sum("lrd_all_serial" in c[1] for c in calls) == 1     # one serial place a round
    assert calls[0][2] == f"GRPO_MODEL={start}" and "--max-steps 750" in calls[0][3]
    assert calls[4][4] == "CE_LABEL=jg-x-bf16"
    assert calls[5][5] == f"EXAM_MODELS=jg-x-q4={Path(env['WORK'])}/eullm_runs/grpo/jg-x/merged"
    assert calls[5][6].endswith("/grpo/jg-x/gguf/jg-x-q4_k_m.gguf")
    assert calls[5][7].endswith("norm-exam-devbig.jsonl")


def test_a_full_serial_partition_refuses_the_round_before_anything_is_queued(tmp_path):
    env, log = _setup(tmp_path, serial=10)
    r = _run(env, "jg-x", str(Path(env["WORK"]) / "start"), "3", "750")
    assert r.returncode == 1 and "lrd_all_serial" in r.stderr
    assert not log.exists()


def test_links_zero_packages_and_examines_a_chain_already_queued(tmp_path):
    env, log = _setup(tmp_path, queued="555")
    r = _run({**env, "AFTER": "555"}, "jg-x", str(Path(env["WORK"]) / "start"), "0", "750")
    assert r.returncode == 0, r.stderr
    calls = [ln.split("|") for ln in log.read_text().splitlines()]
    assert "eullm-package-jg-x" in calls[0][1] and "--dependency=afterany:555" in calls[0][1]
    assert len(calls) == 10 and not any("eullm-grpo" in c[1] for c in calls)


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


def test_packaged_queues_only_the_exams_and_takes_no_serial_place(tmp_path):
    env, log = _setup(tmp_path, serial=10)
    r = _run({**env, "PACKAGED": "4242", "NO_CDS": "1"}, "jg-x", "/x", "0", "750")
    assert r.returncode == 0, r.stderr
    calls = [ln.split("|") for ln in log.read_text().splitlines()]
    assert not any("lrd_all_serial" in c[1] for c in calls)
    assert not any("eullm-grpo" in c[1] or "eullm-package" in c[1] for c in calls)
    assert "--dependency=afterok:4242" in calls[0][1]          # the statute exam
    assert len(calls) == 8                     # 2 links + judge, twice; 1 + judge


def test_a_round_already_packaged_is_not_submitted_again(tmp_path):
    # 2026-10-16: a rerun over a finished round (adapter written, GRPO no
    # longer queued) queued its packaging and exams a second time
    env, log = _setup(tmp_path)
    gguf = Path(env["WORK"]) / "eullm_runs" / "grpo" / "jg-x" / "gguf" / "jg-x-q4_k_m.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_text("gguf")
    r = _run(env, "jg-x", str(Path(env["WORK"]) / "start"), "3", "750")
    assert r.returncode == 0 and "already packaged" in r.stdout
    assert not log.exists()


def test_a_packaging_already_queued_is_not_submitted_again(tmp_path):
    env, log = _setup(tmp_path, packaging="321")
    r = _run(env, "jg-x", str(Path(env["WORK"]) / "start"), "3", "750")
    assert r.returncode == 0 and "packaging already queued" in r.stdout
    assert not log.exists()


def test_the_merged_weights_go_to_the_scratch_behind_a_symlink(tmp_path):
    env, log = _setup(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    r = _run({**env, "GRPO_MERGED_ROOT": str(scratch / "grpo-merged")},
             "jg-x", str(Path(env["WORK"]) / "start"), "1", "750")
    assert r.returncode == 0, r.stderr
    merged = Path(env["WORK"]) / "eullm_runs" / "grpo" / "jg-x" / "merged"
    assert merged.is_symlink() and merged.resolve() == (scratch / "grpo-merged" / "jg-x")
    assert (scratch / "grpo-merged" / "jg-x").is_dir()


def test_without_the_scratch_the_merged_weights_stay_on_work(tmp_path):
    env, log = _setup(tmp_path)
    r = _run({**env, "GRPO_MERGED_ROOT": str(tmp_path / "no" / "scratch" / "grpo-merged")},
             "jg-x", str(Path(env["WORK"]) / "start"), "1", "750")
    assert r.returncode == 0, r.stderr
    assert not (Path(env["WORK"]) / "eullm_runs" / "grpo" / "jg-x" / "merged").exists()
