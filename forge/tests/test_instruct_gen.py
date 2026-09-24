"""Tests for stage-3 pair generation: job selection, prompts and the filters."""

from __future__ import annotations

import json
import random

import pytest

from eullm_forge.datasets.instruct_gen import (
    GenConfig,
    Job,
    Rejected,
    build_messages,
    italian_ratio,
    make_jobs,
    parse_generation,
    passage_window,
)

ARTICLE = (
    "Art. 2043. Risarcimento per fatto illecito. Qualunque fatto doloso o "
    "colposo, che cagiona ad altri un danno ingiusto, obbliga colui che ha "
    "commesso il fatto a risarcire il danno. " * 6
)

GOOD_ANSWER = (
    "Sì. Secondo l'articolo 2043 del codice civile, chi con un fatto doloso o "
    "colposo cagiona ad altri un danno ingiusto è obbligato a risarcirlo. "
    "Occorrono quindi un fatto, la colpa o il dolo, un danno ingiusto e il "
    "nesso di causalità tra il fatto e il danno."
)


def qa_job():
    return Job(key="k1", task="qa", passage=ARTICLE, source="codice_civile")


def ctx_job(task="riassunto"):
    return Job(key="k2", task=task, passage=ARTICLE, source="codice_civile",
               instruction_prefix="Riassumi il seguente testo:")


def gen(**fields):
    return json.dumps(fields, ensure_ascii=False)


# --- job selection -------------------------------------------------------------

def records(n=50):
    return [{"text": f"Documento {i}.\n\n" + ARTICLE, "source": f"s{i}"} for i in range(n)]


def test_jobs_are_deterministic_for_a_seed_and_differ_across_seeds():
    a = make_jobs(records(), 20, seed=1)
    b = make_jobs(records(), 20, seed=1)
    c = make_jobs(records(), 20, seed=2)
    assert [j.key for j in a] == [j.key for j in b]
    assert [j.key for j in a] != [j.key for j in c]


def test_a_pilot_is_a_prefix_of_the_full_run():
    """So the pilot's pairs are reused, not paid for twice, when the limit is raised."""
    pilot = make_jobs(records(), 5, seed=0)
    full = make_jobs(records(), 30, seed=0)
    assert [j.key for j in full[:5]] == [j.key for j in pilot]


def test_short_records_are_skipped():
    recs = [{"text": "Troppo corto.", "source": "x"}] * 10 + records(3)
    jobs = make_jobs(recs, 10)
    assert len(jobs) == 3 and all(j.source.startswith("s") for j in jobs)


def test_passages_are_pseudonymised_before_the_generator_sees_them():
    rec = {"text": ARTICLE + " Il ricorrente, C.F. RSSMRA70C03H501Z, propone ricorso.",
           "source": "x"}
    (job,) = make_jobs([rec], 1)
    assert "RSSMRA70C03H501Z" not in job.passage
    assert "RSSMRA70C03H501Z" not in build_messages(job)[1]["content"]


def test_context_tasks_carry_a_user_instruction_and_qa_does_not():
    jobs = make_jobs(records(200), 200, seed=3)
    assert {j.task for j in jobs} == {"qa", "riassunto", "spiegazione"}
    for j in jobs:
        assert bool(j.instruction_prefix) == (j.task != "qa")


def test_a_long_document_is_windowed_at_paragraph_breaks():
    paras = [f"Paragrafo {i}. " + "testo " * 60 for i in range(40)]
    text = "\n\n".join(paras)
    w = passage_window(text, 2000, random.Random(0))
    assert len(w) <= 2000
    assert w.startswith("Paragrafo")          # starts at a paragraph
    assert w.rstrip().endswith("testo")       # and does not end mid-word


# --- parsing and filters -------------------------------------------------------

def test_a_good_qa_pair_is_accepted_with_provenance():
    pair = parse_generation(gen(domanda="Chi causa un danno ingiusto deve risarcirlo?",
                                risposta=GOOD_ANSWER), qa_job())
    assert pair["instruction"] == "Chi causa un danno ingiusto deve risarcirlo?"
    assert pair["output"] == GOOD_ANSWER
    assert (pair["task"], pair["source"], pair["key"]) == ("qa", "codice_civile", "k1")


def test_code_fences_and_stray_prose_are_tolerated():
    raw = "Ecco il JSON:\n```json\n" + gen(risposta=GOOD_ANSWER) + "\n```"
    assert parse_generation(raw, ctx_job())["output"] == GOOD_ANSWER


def test_a_context_task_puts_the_passage_in_the_instruction():
    pair = parse_generation(gen(risposta=GOOD_ANSWER), ctx_job())
    assert pair["instruction"].startswith("Riassumi il seguente testo:\n\n")
    assert ARTICLE.strip()[:50] in pair["instruction"]


@pytest.mark.parametrize("raw, reason", [
    ("nessun json qui", "no_json"),
    ('{"risposta": "non chiuso', "no_json"),
    ('{"risposta": "a" "b"}', "bad_json"),
    (gen(risposta=""), "empty_answer"),
    (gen(risposta="Troppo breve."), "answer_length"),
    (gen(risposta="This answer is written in English and it is long enough to pass "
                   "the length check without any trouble at all, but it is not "
                   "Italian, which is what the student must learn to write."),
     "not_italian"),
    (gen(risposta=GOOD_ANSWER + " Il ricorrente [PERSONA_1] ha proposto appello."),
     "placeholder"),
    (gen(risposta=GOOD_ANSWER + " Il ricorrente [persona_1] ha proposto appello."),
     "placeholder"),
    (gen(risposta=GOOD_ANSWER + " Il ricorrente [Persona_1] ha proposto appello."),
     "placeholder"),
    (gen(risposta=GOOD_ANSWER + " Contatti: mario.rossi@example.it"), "personal_data"),
    ("<think>ragiono</think>" + gen(risposta=GOOD_ANSWER), "thinking"),
])
def test_bad_generations_are_rejected_with_a_reason(raw, reason):
    with pytest.raises(Rejected) as exc:
        parse_generation(raw, ctx_job())
    assert exc.value.reason == reason


def test_a_closed_book_question_may_not_point_at_a_text_the_user_never_gave():
    with pytest.raises(Rejected) as exc:
        parse_generation(gen(domanda="Cosa prevede il testo sul risarcimento?",
                             risposta=GOOD_ANSWER), qa_job())
    assert exc.value.reason == "refers_to_text"


def test_testo_unico_is_not_a_reference_to_the_passage():
    pair = parse_generation(
        gen(domanda="Cosa prevede il testo unico sull'edilizia per la SCIA?",
            risposta=GOOD_ANSWER), qa_job())
    assert "testo unico" in pair["instruction"]


def test_a_qa_generation_without_a_question_is_rejected():
    with pytest.raises(Rejected) as exc:
        parse_generation(gen(risposta=GOOD_ANSWER), qa_job())
    assert exc.value.reason == "question_length"


def test_italian_prose_clears_the_ratio_by_a_margin():
    assert italian_ratio(GOOD_ANSWER) > GenConfig().min_italian_ratio * 1.5
    assert italian_ratio("") == 0.0


# --- findings of the 24 September pilot ----------------------------------------

@pytest.mark.parametrize("question", [
    "Il ricorso contro la sentenza della Corte d'Appello di Bari del 15 gennaio "
    "2024 è stato dichiarato ammissibile?",
    "Con la sentenza n. 1234 il TAR ha accolto la domanda?",
    "Il ricorso n. 2019/4455 è stato respinto?",
])
def test_a_question_about_one_case_is_rejected(question):
    """Its answer can only be invented; training on it teaches inventing outcomes."""
    with pytest.raises(Rejected) as exc:
        parse_generation(gen(domanda=question, risposta=GOOD_ANSWER), qa_job())
    assert exc.value.reason == "case_specific"


@pytest.mark.parametrize("question", [
    "Cosa prevede la legge 7 agosto 1990, n. 241 sul silenzio assenso?",
    "Il d.lgs. 30 giugno 2003, n. 196 si applica ancora dopo il GDPR?",
    "Cosa stabilisce il d.P.R. 6 giugno 2001, n. 380 sulla SCIA?",
])
def test_a_statute_named_by_its_date_is_not_a_case(question):
    pair = parse_generation(gen(domanda=question, risposta=GOOD_ANSWER), qa_job())
    assert pair["instruction"] == question


def test_the_surname_in_a_case_citation_is_dropped_and_the_citation_kept():
    answer = (GOOD_ANSWER + " Così Cass. pen., Sez. 6, n. 25273 del 23 maggio 2018, "
              "Zidane, Rv. 273392; e n. 1111 del 02/03/2019, De Luca Rossi, Rv. 275000-01.")
    out = parse_generation(gen(risposta=answer), ctx_job())["output"]
    assert "Zidane" not in out and "De Luca" not in out
    assert "n. 25273 del 23 maggio 2018, Rv. 273392" in out
    assert "02/03/2019, Rv. 275000-01" in out


def test_a_chunk_that_starts_mid_sentence_is_trimmed_to_a_whole_one():
    chunk = ("oggetto dello scorporo catastale. 2.2 Con il quarto motivo la società si "
             "duole della violazione dell'art. 1. " + "Il motivo è fondato. " * 40
             + "Resta da stabilire se la")
    w = passage_window(chunk, 5000, random.Random(0))
    assert w.startswith("2.2 Con il quarto motivo")
    assert w.endswith("Il motivo è fondato.")


def test_a_text_that_starts_properly_is_left_alone():
    text = "Art. 2043. " + "Qualunque fatto doloso cagiona un danno. " * 20
    assert passage_window(text, 5000, random.Random(0)).startswith("Art. 2043.")


def test_the_source_falls_back_to_the_corpus_own_keys():
    rec = {"text": ARTICLE, "kind": "cassazione_penale"}
    (job,) = make_jobs([rec], 1)
    assert job.source == "cassazione_penale"
