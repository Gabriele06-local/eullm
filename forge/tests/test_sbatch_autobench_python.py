"""The AutoBench Slurm job must use LUMI's selected interpreter.

LUMI's /usr/bin/python3 is 3.6, and `ab_data.py` needs 3.8+: its walrus at
line 242 is a parse-time SyntaxError, before any set runs. The job used bare
python3 for both the benchmark and the report, after loading the models.
`campaign_env.sh` picks the newest usable interpreter (then cray-python);
the sibling LUMI jobs source it and use `$PY`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "tools" / "lumi" / "sbatch_autobench.slurm"
AB_DATA = ROOT / "bench" / "reflexbench" / "ab_data.py"


def test_ab_data_really_needs_python_38():
    with pytest.raises(SyntaxError):
        ast.parse(AB_DATA.read_text(encoding="utf-8"), feature_version=(3, 7))


def test_the_job_uses_the_interpreter_campaign_env_selects():
    text = JOB.read_text(encoding="utf-8")
    assert "campaign_env.sh" in text
    assert re.search(r'"\$PY"\s+"\$BENCH"', text)
    assert re.search(r'"\$PY"\s+-\s+"\$PREFIX\.json"', text)
    assert 'python3 "$BENCH"' not in text
    assert 'python3 - "$PREFIX.json"' not in text
