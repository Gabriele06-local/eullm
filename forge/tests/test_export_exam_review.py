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


def _skewed_exam(path, sizes):
    """The shape the builder actually produces: one contenuto per article drawn,
    one inesistente per per_code//5, so the families are very unequal."""
    items = []
    for kind, n in sizes.items():
        items += [EvalItem(id=f"norm-{kind}-codice_civile-{i}", domain="legal", lang="it",
                           question=f"Domanda {kind} {i}?", reference=f"Riferimento {i}",
                           rubric="Corretto se ...",
                           metadata={"tipo": kind, "code": "codice_civile",
                                     "articolo": str(i)}) for i in range(n)]
    save_eval_set(items, path)
    return path


def test_every_type_survives_a_skewed_exam(tmp_path):
    """Rounding each family's share on its own overshoots, and the overshoot
    was trimmed off the end — which is always the alphabetically last family.
    On this exam at --n 40 the sheet held four of the six types, and the two
    missing were the last two by name, on every run."""
    exam = _skewed_exam(tmp_path / "norm-exam-dev3.jsonl",
                        {"contenuto": 90, "inesistente": 5, "legislazione": 2,
                         "termine": 1, "termine_argomento": 1, "trattato": 1})
    for n in ("50", "40", "20"):
        out = tmp_path / f"review-{n}.csv"
        assert _mod().main([str(exam), "--out", str(out), "--n", n]) == 0
        rows = list(csv.DictReader(out.read_text(encoding="utf-8-sig").splitlines(),
                                   delimiter=";"))
        assert len(rows) == int(n)
        assert {r["tipo"] for r in rows} == set(exam_six), f"a type is missing at --n {n}"


def test_fewer_slots_than_types_does_not_always_drop_the_same_one(tmp_path):
    """With --n below the number of types something has to go; it must not be
    the same type every run, or the review quietly stops covering it."""
    exam = _skewed_exam(tmp_path / "norm-exam-dev3.jsonl",
                        {"a_famiglia": 5, "b_famiglia": 5, "c_famiglia": 5,
                         "d_famiglia": 5, "e_famiglia": 5})
    picked = set()
    for seed in ("0", "1", "2", "3"):
        out = tmp_path / f"tiny-{seed}.csv"
        assert _mod().main([str(exam), "--out", str(out), "--n", "2",
                            "--seed", seed]) == 0
        rows = list(csv.DictReader(out.read_text(encoding="utf-8-sig").splitlines(),
                                   delimiter=";"))
        picked.add(tuple(sorted(r["tipo"] for r in rows)))
    assert len(picked) > 1, "the same types are dropped whatever the seed"


exam_six = {"contenuto", "inesistente", "legislazione", "termine",
            "termine_argomento", "trattato"}


def test_the_held_out_exam_is_refused(tmp_path, capsys):
    exam = _exam(tmp_path / "norm-exam-v4.jsonl")
    out = tmp_path / "review.csv"
    assert _mod().main([str(exam), "--out", str(out)]) == 2
    assert not out.exists()
    assert "not a development set" in capsys.readouterr().err
