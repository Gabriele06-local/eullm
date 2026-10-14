"""check_cited_articles.py: an article cited that its code does not hold."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from eullm_forge.eval import NormIndex

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_cited_articles.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_cited_articles", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(code, num):
    return {"text": f"Art. {num}. Testo.", "code": code, "article_num": str(num),
            "chunk_index": 0}


def _known(mod):
    return mod.corpus_articles(NormIndex([
        _rec("codice_civile", 1453), _rec("codice_civile", 2043), _rec("codice_civile", 2969),
        _rec("codice_procedura_civile", 360), _rec("codice_penale", 624)]))


def test_an_article_past_the_code_is_flagged_and_a_real_one_is_not():
    mod = _load()
    known = _known(mod)
    q = "Che cosa prevede l'art. 2043 del codice civile?"
    assert mod.invented("Ai sensi dell'art. 2043 e dell'art. 1453 c.c. ...", q, known) == []
    assert mod.invented("Si veda anche l'art. 3500 c.c.", q, known) == \
        ["codice_civile art. 3500"]
    # the code named right after the citation wins over the question's
    assert mod.invented("e l'art. 360 c.p.c. sul ricorso", q, known) == []
    assert mod.invented("e l'art. 361 c.p.c. sul ricorso", q, known) == \
        ["codice_procedura_civile art. 361"]


def test_the_article_asked_about_is_not_counted_when_the_answer_denies_it():
    mod = _load()
    known = _known(mod)
    q = "Che cosa prevede l'art. 3434 del codice civile?"
    assert mod.invented("L'art. 3434 del codice civile non esiste.", q, known) == []


def test_a_citation_whose_code_cannot_be_told_is_left_out():
    mod = _load()
    assert mod.invented("Vedi l'art. 99999.", "Una domanda senza codice.", _known(mod)) == []
    # an act outside the corpus, named after the article, is not the question's code
    q = "Che cosa prevede l'art. 2043 del codice civile?"
    assert mod.invented("come l'art. 5 del d.lgs. 165/2001", q, _known(mod)) == []
    assert mod.invented("e l'art. 7 della legge n. 300/1970", q, _known(mod)) == []


def test_the_script_prints_counts_only(tmp_path, capsys):
    mod = _load()
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("".join(json.dumps(_rec("codice_civile", n)) + "\n" for n in (1, 2, 3)))
    ans = tmp_path / "answers-m.jsonl"
    ans.write_text("".join(json.dumps(r) + "\n" for r in [
        {"question": "Che cosa prevede l'art. 2 del codice civile?",
         "answer": "Art. 2 e art. 9 c.c."},
        {"question": "Che cosa prevede l'art. 3 del codice civile?",
         "answer": "L'art. 3 c.c. ..."}]))
    assert mod.main([str(ans), "--norms", str(norms)]) == 0
    out = capsys.readouterr().out
    assert "answers-m.jsonl: 2 answers | citing an article that does not exist 1 (50.0%) | " \
           "such citations 1" in out
    assert "Art. 2 e art. 9" not in out


def test_show_prints_the_flagged_citation_with_its_context(tmp_path, capsys):
    mod = _load()
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("".join(json.dumps(_rec("codice_civile", n)) + "\n" for n in (1, 2, 3)))
    ans = tmp_path / "answers-m.jsonl"
    ans.write_text(json.dumps({"question": "Che cosa prevede l'art. 2 del codice civile?",
                               "answer": "Art. 2 e art. 9 c.c. sul punto."}) + "\n")
    assert mod.main([str(ans), "--norms", str(norms), "--show", "5"]) == 0
    assert "    codice_civile art. 9 | art. 9 c.c. sul punto." in capsys.readouterr().out


def test_show_refuses_held_out_answers(tmp_path):
    mod = _load()
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text(json.dumps(_rec("codice_civile", 1)) + "\n")
    ans = tmp_path / "answers-m-heldout.jsonl"
    ans.write_text("")
    with pytest.raises(SystemExit):
        mod.main([str(ans), "--norms", str(norms), "--show", "3"])
