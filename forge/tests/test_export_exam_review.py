"""The review spreadsheet: every question type, the columns to fill, and the
held-out exam refused so that reading it stays a choice, not an accident."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from eullm_forge.eval import EvalItem, save_eval_set

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_exam_review.py"


def _mod():
    spec = importlib.util.spec_from_file_location("export_exam_review", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _exam(path, n=40):
    kinds = ["contenuto", "termine", "termine_argomento", "inesistente"]
    save_eval_set([EvalItem(id=f"norm-{kinds[i % 4]}-codice_civile-{i}", domain="legal",
                            lang="it", question=f"Domanda {i}?", reference=f"Riferimento {i}",
                            rubric="Corretto se ...",
                            metadata={"tipo": kinds[i % 4], "code": "codice_civile",
                                      "articolo": str(i)}) for i in range(n)], path)
    return path


def test_the_review_file_covers_every_type_and_has_columns_to_fill(tmp_path):
    exam = _exam(tmp_path / "norm-exam-dev3.jsonl")
    out = tmp_path / "review.csv"
    assert _mod().main([str(exam), "--out", str(out), "--n", "12"]) == 0
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                 # Excel reads the accents
    rows = list(csv.DictReader(out.read_text(encoding="utf-8-sig").splitlines(),
                               delimiter=";"))
    assert len(rows) == 12
    assert {r["tipo"] for r in rows} == {"contenuto", "termine", "termine_argomento",
                                         "inesistente"}
    assert all(r["giudizio"] == "" and r["nota"] == "" for r in rows)


def test_the_held_out_exam_is_refused(tmp_path, capsys):
    exam = _exam(tmp_path / "norm-exam-v4.jsonl")
    out = tmp_path / "review.csv"
    assert _mod().main([str(exam), "--out", str(out)]) == 2
    assert not out.exists()
    assert "not a development set" in capsys.readouterr().err
