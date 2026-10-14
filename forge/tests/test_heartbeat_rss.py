"""The heartbeat's RSS column must see every long trainer.

`_heartbeat_rss_gib` finds the job's workers with `pgrep -f` and a pattern
naming the trainer entry points. It named only llamafactory and distill, so
for stage 3, OPD, GRPO and phase 1 the column read `rss ? GiB` for the whole
run -- a checkpoint load indistinguishable from a hang, which is exactly
what the heartbeat exists to tell apart.

These tests read the pattern out of the script, so they pin what it matches
without duplicating it, and they run everywhere: the pattern is ERE, and
everything it uses (`[^ ]`, `.*`, `(a|b)`) reads the same under Python `re`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leonardo" / "heartbeat.sh"


def _pattern() -> str:
    m = re.search(r"pgrep .* -f '([^']+)'", SCRIPT.read_text(encoding="utf-8"))
    assert m, "the pgrep pattern moved; update this test with it"
    return m.group(1)


@pytest.mark.parametrize("cmd", [
    # phase 2, via distill.sh through sbatch_phase2.slurm
    "python /repo/forge/scripts/distill.py --config c.yaml --dataset-dir /d",
    # stage 3, sbatch_stage3.slurm
    'python "$REPO/forge/scripts/stage3_sft.py" --model /m --pairs p --out /o',
    # OPD, sbatch_opd.slurm
    'python "$REPO/forge/scripts/opd_train.py" --student /s --teacher /t',
    # GRPO, sbatch_grpo.slurm: accelerate, not python, leads the line
    'accelerate launch --num_processes 3 --mixed_precision bf16 /repo/forge/scripts/grpo_train.py',
    # phase 1, train.sh through sbatch_phase1.slurm: no python prefix at all
    "/work/eullm_venv/bin/llamafactory-cli train /tmp/x.yaml",
])
def test_the_rss_pattern_sees_every_trainer(cmd):
    assert re.search(_pattern(), cmd), cmd


@pytest.mark.parametrize("cmd", [
    "tail -f /work/slurm-123.out",                                  # a log tail
    "bash /repo/forge/scripts/leonardo/heartbeat.sh",               # this script
    "bash /repo/forge/scripts/leonardo/sbatch_grpo.slurm",          # a launcher shell
    "python /repo/forge/scripts/compare_graded.py /rdir",           # a short tool
])
def test_the_rss_pattern_ignores_what_is_not_a_trainer(cmd):
    assert not re.search(_pattern(), cmd), cmd
