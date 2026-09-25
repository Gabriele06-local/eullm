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
    assert row[3] == 2 and row[4] == "0.500" and row[-1] == 2
