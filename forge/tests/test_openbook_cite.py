"""openbook_cite: grounded answers that name their passage [n] and quote it literally."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from eullm_forge.datasets.openbook_cite import article_passage, cite
from eullm_forge.eval.retrieval import open_book_prompt

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cite_openbook_pairs.py"
OTHER = {"code": "codice_civile", "article_num": "", "chunk_index": 0,
         "text": "Art. 1418. \n (Cause di nullità del contratto). \n Il contratto è nullo "
                 "quando è contrario a norme imperative, salvo che la legge disponga "
                 "diversamente."}
MINE = {"code": "codice_civile", "article_num": "", "chunk_index": 0,
        "text": "Art. 1419. \n (Nullità parziale). \n La nullità parziale di un contratto o "
                "la nullità di singole clausole importa la nullità dell'intero contratto, se "
                "risulta che i contraenti non lo avrebbero concluso senza quella parte del suo "
                "contenuto che è colpita dalla nullità."}
QUESTION = "Che cosa succede se una sola clausola del contratto è nulla?"


def _pair(answer, records=(OTHER, MINE), key="ob-g-codice_civile-1419-v2"):
    return {"instruction": open_book_prompt(QUESTION, list(records)), "output": answer,
            "task": "openbook_grounded", "source": "codice_civile", "key": key}


ANSWER = ("Secondo l'art. 1419 del codice civile, la nullità di singole clausole importa la "
          "nullità dell'intero contratto solo se i contraenti non lo avrebbero concluso senza "
          "quella parte.")


def test_the_article_is_found_by_its_label():
    pos, text = article_passage(_pair(ANSWER)["instruction"], "codice_civile", "1419")
    assert pos == 2 and "Nullità parziale" in text
    assert article_passage(_pair(ANSWER)["instruction"], "codice_civile", "141") is None


def test_the_passage_number_follows_the_citation_and_the_quote_is_literal():
    out = cite(_pair(ANSWER))["output"]
    assert out.startswith("Secondo l'art. 1419 del codice civile [2], la nullità")
    quote = out.split("«", 1)[1].split("»", 1)[0]
    assert quote.startswith("La nullità parziale di un contratto")
    assert quote in " ".join(MINE["text"].split())
    assert out.endswith("».") and "Il testo [2] dice:" in out


def test_a_pair_that_cannot_be_cited_by_program_is_left_alone():
    assert cite(_pair(ANSWER, records=(OTHER,))) is None              # article not shown
    assert cite(_pair("La clausola nulla travolge il contratto, di solito.")) is None
    assert cite(_pair(ANSWER + " [2]")) is None                       # already cited
    assert cite({**_pair(ANSWER), "task": "openbook_missing"}) is None


def test_the_script_copies_every_row_and_prints_counts_only(tmp_path, capsys):
    src = tmp_path / "v04.jsonl"
    rows = [_pair(ANSWER), {"instruction": "x", "output": "y", "task": "openbook_missing",
                            "key": "ob-m-codice_civile-9999-0"}]
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "v05.jsonl"
    spec = importlib.util.spec_from_file_location("cite_openbook_pairs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([str(src), "--out", str(out)]) == 0
    got = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(got) == 2 and "[2]" in got[0]["output"] and got[1] == rows[1]
    printed = capsys.readouterr().out
    assert "grounded:cited 1" in printed and "nullità" not in printed
