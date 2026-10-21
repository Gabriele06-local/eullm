"""A second submission of the sampling job must re-grade the GRPO prompts too.

The slurm skips sampling when the answers file exists and only reruns
make_correction_pairs.py. Since #996 that script re-grades every answer from
the stored samples, so a later reward fix needs no new sampling -- but the
slurm did not pass `--informative-out`, so the GRPO prompt file stayed as the
first sampling wrote it: stale under the old reward, and missing the
questions that became mixed.
"""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leonardo" / \
    "sbatch_sample_answers.slurm"


def _repair_command() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text.rsplit("make_correction_pairs.py", 1)[1]


def test_the_repair_path_writes_the_informative_file_too():
    repair = _repair_command()
    assert '--out-sft "$SA_OUT-correzione.jsonl"' in repair
    assert '--out-dpo "$SA_OUT-dpo.jsonl"' in repair
    assert '--informative-out "$SA_OUT-informative.jsonl"' in repair
