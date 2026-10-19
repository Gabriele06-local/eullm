"""legal_eval.split_reasoning: only the answer after a reasoning block is graded."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "legal_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("legal_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_both_reasoning_formats_are_taken_apart():
    mod = _load()
    out = mod.split_reasoning("<think>\nla premessa è falsa\n</think>\n\n"
                              "No: da sei mesi a tre anni.")
    assert out == ("No: da sei mesi a tre anni.", "la premessa è falsa", True)
    out = mod.split_reasoning("[THINK]art. 640[/THINK]Risposta.")
    assert out == ("Risposta.", "art. 640", True)


def test_an_answer_without_reasoning_is_left_alone_and_a_cut_one_has_no_answer():
    mod = _load()
    assert mod.split_reasoning("Risposta diretta.") == ("Risposta diretta.", "", True)
    assert mod.split_reasoning("[THINK]sto ancora ragionando") == ("", "sto ancora ragionando",
                                                                   False)
