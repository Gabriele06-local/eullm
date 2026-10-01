"""`quantize_to_gguf.sh` must not upgrade the environment it runs in.

On Leonardo it runs inside the venv every training and exam job shares. On
2026-09-30 its `pip install --upgrade "torch>=2.1"` pulled the newest torch
from PyPI — a CUDA 13 build — into that venv during a package job, and from
then on every job died at `import torch` ("undefined symbol:
ncclCommResume"). These are static checks on the commands it executes.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "quantize_to_gguf.sh"


def _pip_commands() -> list[str]:
    """Every executed `pip install`, continuation lines joined, comments dropped."""
    code = [ln for ln in SCRIPT.read_text(encoding="utf-8").splitlines()
            if not ln.strip().startswith("#")]
    joined = re.sub(r"\\\n\s*", " ", "\n".join(code))
    return [ln for ln in joined.splitlines() if re.search(r"\bpip\b.*\binstall\b", ln)]


def test_nothing_is_upgraded():
    for cmd in _pip_commands():
        assert "--upgrade" not in cmd and " -U " not in f"{cmd} ", cmd


def test_torch_and_transformers_are_never_installed():
    for cmd in _pip_commands():
        assert not re.search(r"\b(torch|transformers)\b", cmd), cmd


def test_a_missing_torch_is_an_error_not_an_install():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'python3 -c "import torch, transformers"' in text
