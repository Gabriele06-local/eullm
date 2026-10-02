"""Every Leonardo GPU job asks for the shape that gets placed.

The rules in forge/CLAUDE.md ("Leonardo: the shape of a job that starts")
were each measured once and then broken again by the next new script:
24 h jobs and whole-node jobs were found to wait for days on 2026-09-21/23,
and on 2026-10-01 the new GRPO script asked for 24 h and four GPUs anyway and
sat in the queue all afternoon. A rule that lives only in prose is
re-learnt at the cost of a day each time; this one fails CI instead.

Static checks on the #SBATCH headers:

* ``boost_usr_prod`` jobs ask for at most two hours — chains of 2 h links,
  never one long job;
* at most three GPUs, with at most 8 cores and 123 GB of memory per GPU
  (a quarter of a node). Asking for a fourth GPU, 32 cores or more than
  three quarters of the memory makes the job need an idle whole node.

The scripts that already ask for a whole node belong to finished phases and
are listed below with the reason; a new script is not added to that list
without the same kind of reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

LEONARDO = Path(__file__).resolve().parents[1] / "scripts" / "leonardo"
GPU_PARTITION = "boost_usr_prod"
MAX_SECONDS = 2 * 3600
MAX_GPUS = 3
CORES_PER_GPU = 8
MEM_GB_PER_GPU = 123  # 494 GB of a node / 4

#: Whole-node scripts kept as they are. Each needs all four GPUs of one node
#: by construction (a sharded 30B teacher, or a measurement of the 4-GPU
#: case itself). Phase 1 and phase 2 ended on 2026-09-29.
WHOLE_NODE = {
    "sbatch_backfill_probe.slurm": "measures how long a whole-node request waits",
    "sbatch_p1_logprob_probe.slurm": "30B teacher sharded over four GPUs",
    "sbatch_p3_truncation.slurm": "30B teacher sharded over four GPUs",
    "sbatch_phase1.slurm": "30B MoE continued pretraining, ZeRO-3 over four GPUs",
    "sbatch_phase2.slurm": "30B teacher + student, four GPUs",
    "sbatch_phase2_cds.slurm": "30B teacher + student, four GPUs",
    "sbatch_phase2_split_graze.slurm": "split-node teacher/student, four GPUs",
}


def _headers(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"#SBATCH\s+--([\w-]+)=(\S+)", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _seconds(walltime: str) -> int:
    days = 0
    if "-" in walltime:
        d, walltime = walltime.split("-", 1)
        days = int(d)
    parts = [int(p) for p in walltime.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def _mem_gb(mem: str) -> float:
    m = re.fullmatch(r"(\d+)([KMGT]?)B?", mem.upper())
    assert m, f"unreadable --mem={mem}"
    scale = {"K": 1 / 1024**2, "M": 1 / 1024, "": 1 / 1024, "G": 1, "T": 1024}
    return int(m.group(1)) * scale[m.group(2)]


def _gpus(headers: dict[str, str]) -> int:
    m = re.search(r"gpu:(\d+)", headers.get("gres", ""))
    return int(m.group(1)) if m else 0


GPU_SCRIPTS = sorted(p for p in LEONARDO.glob("*.slurm")
                     if _headers(p).get("partition") == GPU_PARTITION)


def test_there_are_gpu_scripts_to_check():
    assert len(GPU_SCRIPTS) >= 10


@pytest.mark.parametrize("script", GPU_SCRIPTS, ids=lambda p: p.name)
def test_two_hours_at_most(script):
    h = _headers(script)
    assert "time" in h, f"{script.name}: no --time, the partition default is 24 h"
    assert _seconds(h["time"]) <= MAX_SECONDS, (
        f"{script.name} asks for {h['time']}: chain 2 h links with "
        "submit_chain.sh instead of one long job")


@pytest.mark.parametrize("script", GPU_SCRIPTS, ids=lambda p: p.name)
def test_a_quarter_node_per_gpu_and_never_the_whole_node(script):
    if script.name in WHOLE_NODE:
        pytest.skip(WHOLE_NODE[script.name])
    h = _headers(script)
    gpus = _gpus(h)
    assert 1 <= gpus <= MAX_GPUS, (
        f"{script.name} asks for {gpus} GPUs: four is a whole node, which waits "
        "for an idle node; three is placed within the hour")
    cpus = int(h.get("cpus-per-task", "1")) * int(h.get("ntasks-per-node", "1"))
    assert cpus <= CORES_PER_GPU * gpus, (
        f"{script.name}: {cpus} cores for {gpus} GPUs makes the node exclusive")
    assert "mem" in h, f"{script.name}: no --mem, the default may be the whole node"
    assert _mem_gb(h["mem"]) <= MEM_GB_PER_GPU * gpus, (
        f"{script.name}: --mem={h['mem']} for {gpus} GPUs")


def test_the_whole_node_list_only_shrinks():
    """An entry for a script that is gone, or no longer whole-node, is removed."""
    for name in WHOLE_NODE:
        path = LEONARDO / name
        assert path.is_file(), f"{name} is gone: drop it from WHOLE_NODE"
        assert _gpus(_headers(path)) == 4, f"{name} no longer asks for 4 GPUs"


def test_the_checks_catch_the_2026_10_01_grpo_script(tmp_path):
    bad = tmp_path / "sbatch_bad.slurm"
    bad.write_text("#!/bin/bash\n#SBATCH --partition=boost_usr_prod\n"
                   "#SBATCH --time=24:00:00\n#SBATCH --gres=gpu:4\n"
                   "#SBATCH --cpus-per-task=32\n#SBATCH --mem=450G\n")
    with pytest.raises(AssertionError):
        test_two_hours_at_most(bad)
    with pytest.raises(AssertionError):
        test_a_quarter_node_per_gpu_and_never_the_whole_node(bad)
