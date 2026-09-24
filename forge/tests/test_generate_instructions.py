"""Tests for the instruction-generation driver.

Generating from an evaluation set puts the exam's answers into the training
data with nothing downstream noticing, so the refusal must not depend on the
filename's capitalisation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_instructions.py"


def _load():
    spec = importlib.util.spec_from_file_location("generate_instructions", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gen_mod = _load()


def test_eval_corpus_refused_regardless_of_case(tmp_path, monkeypatch, capsys):
    for name in ("val.jsonl", "VAL.jsonl", "Val.jsonl", "cds_eval.jsonl", "CDS_eval.jsonl"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "generate_instructions.py",
                "--corpus",
                str(tmp_path / name),
                "--out",
                str(tmp_path / "out.jsonl"),
                "--limit",
                "1",
                "--dry-run",
            ],
        )
        assert gen_mod.main() == 2
        assert "evaluation set" in capsys.readouterr().err
