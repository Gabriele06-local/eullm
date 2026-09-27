"""The release gate's scoring: v0.1c's answer to art. 2043 must fail it."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from eullm_forge.eval import evaluate_qa, load_seed

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "legal_eval.py"
spec = importlib.util.spec_from_file_location("legal_eval", SCRIPT)
legal_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legal_eval)

# Verbatim from the v0.1c smoke test of 25 September.
V01C_2043 = (
    "L'articolo 2043 del codice civile stabilisce che chiunque ha agito in modo "
    "contrario ai doveri di buona fede, come previsto dagli articoli 1176 e 1375, "
    "è tenuto a risarcire il danno cagionato, a meno che non provi di non aver "
    "agito con colpa."
)
RIGHT_2043 = (
    "Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, "
    "obbliga colui che ha commesso il fatto a risarcire il danno."
)


def item(item_id):
    return next(it for it in load_seed() if it.id == item_id)


def test_the_v01c_answer_scores_below_a_correct_one():
    it = item("legal-it-civ-002")
    wrong = evaluate_qa([it], {it.id: V01C_2043})["per_item"][0]["keyword_coverage"]
    right = evaluate_qa([it], {it.id: RIGHT_2043})["per_item"][0]["keyword_coverage"]
    assert right == 1.0
    assert wrong <= 0.25


def test_the_summary_row_counts_full_marks_and_endings():
    items = [item("legal-it-civ-002"), item("legal-it-amm-002")]
    report = evaluate_qa(items, {"legal-it-civ-002": RIGHT_2043,
                                 "legal-it-amm-002": "Non lo so."})
    row = legal_eval.summary_row("x", "m", report, ended=2)
    assert dict(zip(legal_eval.CSV_HEADER, row))["fully_covered"] == 1
    # By name, not by position: row[-1] is not ended_turn any more.
    fields = dict(zip(legal_eval.CSV_HEADER, row))
    assert fields["items"] == 2 and fields["keyword_coverage"] == "0.500"
    assert fields["ended_turn"] == 2 and fields["keyword_items"] == 2


def test_quiet_grading_prints_no_item(tmp_path, capsys):
    """Held-out exam ids name the article asked; --quiet must not print them."""
    import csv
    import json

    spec = importlib.util.spec_from_file_location(
        "judge_answers", SCRIPT.parent / "judge_answers.py")
    ja = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ja)
    from eullm_forge.eval import ReferenceGrader

    src = tmp_path / "answers-m.jsonl"
    src.write_text("\n".join(json.dumps({"id": f"norm-termine-codice_civile-{n}",
                                         "question": "Q?", "answer": "A.",
                                         "reference": "R."}) for n in (1, 2, 3)) + "\n")

    class Batched:
        calls = 0

        def __call__(self, p):
            return self.batch([p])[0]

        def batch(self, ps):
            Batched.calls += 1
            return ["Grade: correct\nok"] * len(ps)

    grades = ja.grade_file(src, ReferenceGrader(Batched()), batch_size=2, quiet=True)
    assert [g.label for g in grades] == ["correct"] * 3
    assert Batched.calls == 2                       # 3 prompts in batches of 2
    assert "codice_civile" not in capsys.readouterr().out

    # A run killed while the first model was being graded leaves a 0-byte CSV:
    # the file exists, so a header written on existence alone is skipped and the
    # next row lands in its place, leaving DictReader with nothing.
    out = tmp_path / "graded.csv"
    for module in (ja, legal_eval):
        out.write_text("")
        module.append_csv_row(out, ["2026-09-27T00:00:00", "v0.3", "m", 3, 1, 0.5, 3, 3])
        module.append_csv_row(out, ["2026-09-27T00:01:00", "v0.3", "m2", 3, 1, 0.5, 3, 3])
        with out.open(encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == module.CSV_HEADER, module.CSV_HEADER[:2]
        assert len(rows) == 3, module.CSV_HEADER[:2]
        assert {r[1] for r in rows[1:]} == {"v0.3"}    # no row read as a header
        out.unlink()

    # And on a file that already has a header, nothing is repeated.
    out = tmp_path / "graded2.csv"
    for module in (ja, legal_eval):
        out.unlink(missing_ok=True)
        for label in ("a", "b"):
            module.append_csv_row(out, ["t", label, "m", 1, 1, "1.0", 1, 1])
        with out.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["label"] for r in rows] == ["a", "b"], module.CSV_HEADER[:2]
