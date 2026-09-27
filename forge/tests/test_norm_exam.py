"""The held-out exam drawn from the text of the law.

Built on made-up articles in the three formats the real files hold (see
test_eval_retrieval.py): an empty article_num with the header in the text,
"Art. N." with the XML parser's doubled prefix, and a chunk holding several
articles. The real draw is never in the repository.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from eullm_forge.eval import NormIndex, keyword_coverage
from eullm_forge.eval.norm_exam import articles_from_records, build_exam, retrieval_hits
from eullm_forge.eval.retrieval import named_code

FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3


def rec(code, text, article_num="", chunk_index=0):
    return {"code": code, "article_num": article_num, "chunk_index": chunk_index,
            "text": text}


RECORDS = [
    rec("codice_civile", "DISPOSIZIONI GENERALI \n \n Art. 1. \n \n (Fonti). \n \n Sono fonti "
        "le leggi." + FILLER),
    rec("codice_civile", "Art. 2. \n \n (Termine di prova). \n \n La domanda si propone "
        "entro sessanta giorni dalla notificazione." + FILLER),
    rec("codice_civile", "Art. 3. \n \n (Due termini). \n \n Entro dieci giorni si "
        "comunica, entro trenta giorni si decide." + FILLER),
    rec("codice_civile", "Art. 4. \n \n (Abrogato). \n \n Articolo abrogato." + FILLER),
    rec("codice_civile", "Art. 5. \n \n (Lungo). \n \n Prima parte." + FILLER),
    rec("codice_civile", "seconda parte dell'articolo cinque, nel termine di 90 giorni.",
        chunk_index=1),
    rec("legge_procedimento_amministrativo",
        "Art. Art. 2. ((Conclusione del procedimento))\nIl procedimento si conclude "
        "entro trenta giorni." + FILLER, article_num="Art. 2."),
    rec("codice_processo_amministrativo",
        "Art. 29 \n Azione di annullamento \n 1. L'azione si propone nel termine di "
        "decadenza di sessanta giorni." + FILLER + "\n Art. 30 \n Azione di condanna \n "
        "1. Si propone entro centoventi giorni." + FILLER),
    rec("codice_processo_amministrativo",
        "ALLEGATO 2 \n Art. 29 \n Altra norma con lo stesso numero." + FILLER),
]


def test_articles_are_reassembled_from_every_format():
    arts = articles_from_records(RECORDS)
    assert ("codice_civile", "1") in arts and ("codice_civile", "2") in arts
    assert "seconda parte" in arts[("codice_civile", "5")].text
    assert ("legge_procedimento_amministrativo", "2") in arts
    assert ("codice_processo_amministrativo", "30") in arts


def test_a_number_used_twice_in_one_code_is_dropped_as_ambiguous():
    arts = articles_from_records(RECORDS)
    assert ("codice_processo_amministrativo", "29") not in arts


def test_the_heading_is_read_in_both_styles():
    arts = articles_from_records(RECORDS)
    assert arts[("codice_civile", "2")].heading == "Termine di prova"
    assert arts[("legge_procedimento_amministrativo", "2")].heading == \
        "Conclusione del procedimento"


@pytest.fixture
def exam():
    return build_exam(RECORDS, per_code=10, seed=1)


def of_kind(exam, kind):
    return [it for it in exam if it.metadata["tipo"] == kind]


def test_only_articles_with_exactly_one_deadline_become_deadline_questions(exam):
    arts = {(it.metadata["code"], it.metadata["articolo"]) for it in of_kind(exam, "termine")}
    assert ("codice_civile", "2") in arts            # sixty days, once
    assert ("codice_civile", "3") not in arts        # two deadlines: ambiguous
    assert ("codice_civile", "5") in arts            # found in the continuation chunk


def test_a_deadline_question_is_scored_on_the_deadline_in_digits_or_words(exam):
    # `i`, not `it`: naming the variable being assigned in the generator
    # expression made it a free variable of that expression, and the test died
    # with a NameError on every run instead of asserting anything.
    it = next(i for i in of_kind(exam, "termine") if i.metadata["articolo"] == "2"
              and i.metadata["code"] == "codice_civile")
    assert "sessanta giorni" in it.reference
    assert keyword_coverage("Entro 60 giorni.", it.keywords) == 1.0
    assert keyword_coverage("Entro sessanta giorni.", it.keywords) == 1.0
    assert keyword_coverage("Entro trenta giorni.", it.keywords) == 0.0


RATE_RECORD = rec("codice_civile",
                  "Art. 1224. \n \n (Interessi legali) \n \n Gli interessi legali sono "
                  "calcolati al tasso del 6 per cento, salvo quanto disposto per le "
                  "obbligazioni in valuta estera. L'azione giudiziale si esercita entro "
                  "sei mesi dalla maturazione della domanda." + FILLER,
                  article_num="1224")


def test_the_deadline_reference_is_the_sentence_that_states_the_deadline():
    """The number can turn up earlier in the article for another reason — here
    an interest rate — and looking the sentence up by the bare number hands the
    grader that one instead, under a rubric that still asks for the deadline."""
    items = build_exam([RATE_RECORD], per_code=4, seed=1)
    it = next(i for i in items if i.metadata["tipo"] == "termine")
    assert "sei mesi" in it.reference
    assert "per cento" not in it.reference
    assert keyword_coverage("Entro sei mesi dalla domanda.", it.keywords) == 1.0


def test_a_cross_reference_is_not_the_deadline_answer_key():
    cross = rec("codice_procedura_civile",
                "Art. 750. \n \n (Interpretazione) \n \n Quando la legge rinvia ad altre "
                "disposizioni si applicano le regole dell'articolo 30 del codice "
                "penale. L'istanza si propone entro trenta giorni dalla notifica."
                + FILLER,
                article_num="750")
    items = build_exam([cross], per_code=4, seed=2)
    it = next(i for i in items if i.metadata["tipo"] == "termine")
    assert "trenta giorni" in it.reference
    assert "articolo 30" not in it.reference


def test_repealed_articles_are_left_out(exam):
    assert not [it for it in exam if it.metadata["articolo"] == "4"
                and it.metadata["code"] == "codice_civile"]


def test_a_nonexistent_article_is_past_the_end_and_must_be_refused(exam):
    fake = [it for it in of_kind(exam, "inesistente") if it.metadata["code"] == "codice_civile"]
    assert fake and all(int(it.metadata["articolo"]) > 5 for it in fake)
    assert keyword_coverage("L'articolo non esiste.", fake[0].keywords) == 1.0
    assert keyword_coverage("Prevede il risarcimento del danno.", fake[0].keywords) == 0.0


def test_every_question_names_its_code_so_retrieval_and_readers_can_tell(exam):
    for it in exam:
        assert named_code(it.question) == it.metadata["code"], it.question


def test_the_verticals_are_tagged(exam):
    v = {it.metadata["code"]: it.metadata["vertical"] for it in exam}
    assert v["codice_civile"] == "civile_penale"
    assert v["codice_processo_amministrativo"] == "amministrativo"


def test_the_draw_is_random_unless_seeded():
    a = [it.id for it in build_exam(RECORDS, per_code=1, seed=3)]
    assert a == [it.id for it in build_exam(RECORDS, per_code=1, seed=3)]


def test_retrieval_hits_find_named_articles(exam):
    hits = retrieval_hits(exam, NormIndex(RECORDS))
    assert hits["contenuto"]["top1"] == 1.0
    assert "inesistente" not in hits


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_norm_exam.py"


def test_the_script_prints_counts_and_refuses_to_redraw(tmp_path, capsys):
    import json

    spec = importlib.util.spec_from_file_location("make_norm_exam", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    out = tmp_path / "exam.jsonl"
    assert mod.main([str(norms), "--out", str(out), "--check-retrieval"]) == 0
    printed = capsys.readouterr().out
    assert "[exam]" in printed and "[retrieval]" in printed
    assert "Che cosa prevede" not in printed          # counts only, never questions
    assert mod.main([str(norms), "--out", str(out)]) == 1
