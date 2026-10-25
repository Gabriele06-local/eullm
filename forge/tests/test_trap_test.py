"""The trap test: drawn on articles no premise pair trained on, graded without a judge."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from eullm_forge.eval import load_eval_set

FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3


def rec(code, text):
    return {"code": code, "article_num": "", "chunk_index": 0, "text": text}


# As in test_premise_gen.py: two deadlines, a repealed article, two without.
RECORDS = [
    rec("codice_civile", "Art. 1. \n \n (Fonti). \n \n Sono fonti le leggi." + FILLER),
    rec("codice_civile", "Art. 2. \n \n (Termine di prova). \n \n La domanda si propone "
        "entro sessanta giorni dalla notificazione." + FILLER),
    rec("codice_civile", "Art. 3. \n \n (Opposizione). \n \n L'opposizione si propone "
        "entro dieci giorni dalla comunicazione." + FILLER),
    rec("codice_civile", "Art. 4. \n \n (Abrogato). \n \n Articolo abrogato dalla legge "
        "n. 10 del 2000." + FILLER),
    rec("codice_civile", "Art. 5. \n \n (Contratti). \n \n Il contratto è nullo se manca "
        "la causa." + FILLER),
]

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norms(tmp_path):
    norms = tmp_path / "legislazione_test.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    return norms


def test_no_question_is_on_an_article_the_premise_pairs_trained_on(tmp_path, capsys):
    norms = _norms(tmp_path)
    # The real producer: the training pairs as make_premise_pairs.py writes them.
    sft = tmp_path / "sft.jsonl"
    assert _load("make_premise_pairs").main(
        ["--norms", str(norms), "--per-code", "1", "--grpo-share", "0", "--out-sft", str(sft)]) == 0
    trained = {(r["source"], r["key"].split("-", 3)[3])
               for r in map(json.loads, sft.read_text().splitlines())}
    assert trained
    out = tmp_path / "trap-test.jsonl"
    assert _load("make_trap_test").main(
        ["--norms", str(norms), "--per-code", "10", "--exclude-trained", str(sft),
         "--out", str(out)]) == 0
    items = load_eval_set(out)                       # legal_eval.py reads it as it is
    assert items
    assert not {(i.metadata["code"], i.metadata["articolo"]) for i in items} & trained
    assert all(i.rubric and i.reference and i.metadata["tipo"] == i.category for i in items)
    printed = capsys.readouterr().out
    assert "[trap-test]" in printed and "giorni" not in printed and "Verifica" not in printed


def test_answers_are_graded_by_kind_and_a_model_that_refuses_everything_fails_the_true_ones(
        tmp_path, capsys):
    norms = _norms(tmp_path)
    out = tmp_path / "trap-test.jsonl"
    assert _load("make_trap_test").main(["--norms", str(norms), "--per-code", "10",
                                         "--out", str(out)]) == 0
    items = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert {"premessa_termine", "premessa_vera"} <= {i["category"] for i in items}
    # Answers files as legal_eval.py writes them: the reference itself, and a refusal.
    right = tmp_path / "answers-trap-test-right.jsonl"
    right.write_text("".join(json.dumps({"id": i["id"], "answer": i["reference"]}) + "\n"
                             for i in items))
    doubt = tmp_path / "answers-trap-test-doubt.jsonl"
    doubt.write_text("".join(json.dumps({"id": i["id"], "answer": "La premessa è errata."}) + "\n"
                             for i in items))
    assert _load("score_trap_test").main([str(out), str(right), str(doubt)]) == 0
    rows = {ln.split()[0]: ln.split() for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("trap-test-")}
    assert rows["trap-test-right"][1] == "100.0%"
    assert rows["trap-test-doubt"][3] == "0.0%"     # every true premise refused
