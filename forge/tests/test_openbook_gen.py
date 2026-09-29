"""Open-book stage-3 pairs: grounded answers, absent articles, exam kept out."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from eullm_forge.datasets.instruct_gen import Rejected
from eullm_forge.datasets.openbook_gen import (
    OpenBookJob,
    exam_exclusions,
    make_openbook_jobs,
    missing_pair,
    parse_openbook,
)
from eullm_forge.eval import EvalItem, NormIndex

FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3


def rec(code, num, body):
    return {"code": code, "article_num": "", "chunk_index": 0,
            "text": f"Art. {num}. \n \n (Rubrica {num}). \n \n {body}{FILLER}"}


RECORDS = [rec("codice_civile", n, f"Disciplina numero {n} della materia civile.")
           for n in range(1, 41)]
RECORDS.append(rec("codice_civile", 41, "Il creditore agisce entro sessanta giorni."))


@pytest.fixture
def index():
    return NormIndex(RECORDS)


def exam_item(code, art, kind="contenuto"):
    return EvalItem(id=f"norm-{kind}-{code}-{art}", domain="legal", lang="it",
                    question="q", metadata={"code": code, "articolo": art, "tipo": kind})


def test_the_exams_articles_are_never_drawn(index):
    exclude = exam_exclusions([exam_item("codice_civile", str(n)) for n in range(1, 21)])
    jobs = make_openbook_jobs(index, 30, seed=1, exclude=exclude)
    grounded = {j.number for j in jobs if j.kind == "grounded"}
    assert grounded and not grounded & {str(n) for n in range(1, 21)}


def test_a_share_of_jobs_asks_about_absent_articles_past_the_end(index):
    jobs = make_openbook_jobs(index, 20, seed=2, missing_share=0.25)
    missing = [j for j in jobs if j.kind == "missing"]
    assert len(missing) == 5
    assert all(int(j.number) > 41 for j in missing)


def test_an_absent_article_is_answered_by_saying_so(index):
    job = OpenBookJob(key="m1", kind="missing", code="codice_civile", number="3500")
    pair = missing_pair(job, index)
    assert "non è presente art. 3500" in pair["instruction"]
    assert "non compare l'art. 3500" in pair["output"]
    assert pair["instruction"].endswith("?")


def test_an_existing_article_cannot_become_a_missing_pair(index):
    job = OpenBookJob(key="m2", kind="missing", code="codice_civile", number="5")
    with pytest.raises(Rejected):
        missing_pair(job, index)


def grounded_job(named=True, number="41"):
    return OpenBookJob(key=f"g-{number}", kind="grounded", code="codice_civile",
                       number=number, text=RECORDS[int(number) - 1]["text"], named=named)


def gen(q, a):
    return json.dumps({"domanda": q, "risposta": a}, ensure_ascii=False)


ANSWER = ("Secondo l'art. 41 del codice civile, il creditore deve agire entro "
          "sessanta giorni; il termine è stabilito espressamente dal testo della norma.")


def test_a_grounded_pair_carries_the_article_in_the_prompt(index):
    pair = parse_openbook(gen("Entro quanto agisce il creditore secondo l'art. 41 del "
                              "codice civile?", ANSWER), grounded_job(), index)
    assert "entro sessanta giorni" in pair["instruction"]
    assert pair["output"] == ANSWER and pair["task"] == "openbook_grounded"
    assert pair["instruction"].startswith("Testi normativi di riferimento")


def test_the_article_is_put_in_when_retrieval_misses_it(index):
    # Asked by topic in words that match other articles better, the target
    # may not be retrieved; the training prompt must still hold it.
    pair = parse_openbook(gen("Nel codice civile, quale disciplina della materia civile "
                              "vale per il creditore?", ANSWER),
                          grounded_job(named=False), index)
    assert "entro sessanta giorni" in pair["instruction"]


@pytest.mark.parametrize("q, a, named, reason", [
    ("Che cosa dice il codice civile sul creditore?", ANSWER, True, "question_not_naming"),
    ("Cosa prevede l'art. 41 del codice civile?", ANSWER, False, "question_names_article"),
    ("Cosa prevede l'art. 41 del codice civile?",
     "Il creditore deve agire entro sessanta giorni, come prevede la norma del codice.",
     True, "answer_not_citing"),
    ("Cosa prevede l'art. 41 del codice civile?", "Troppo breve.", True, "answer_length"),
])
def test_generations_that_break_the_format_are_rejected(index, q, a, named, reason):
    with pytest.raises(Rejected) as exc:
        parse_openbook(gen(q, a), grounded_job(named=named), index)
    assert exc.value.reason == reason


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_openbook_pairs.py"


def load_script():
    spec = importlib.util.spec_from_file_location("generate_openbook_pairs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_driver_refuses_to_run_without_an_exam_to_exclude(tmp_path):
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    assert load_script().main(["--norms", str(norms), "--out", str(tmp_path / "o.jsonl"),
                               "--limit", "5", "--dry-run"]) == 2


def test_a_dry_run_writes_the_teacher_free_pairs_and_says_how_many_it_excluded(tmp_path,
                                                                              capsys):
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    exam = tmp_path / "exam.jsonl"
    exam.write_text(json.dumps(exam_item("codice_civile", "3").to_dict()) + "\n")
    out = tmp_path / "o.jsonl"
    mod = load_script()
    assert mod.main(["--norms", str(norms), "--exclude-exam", str(exam), "--out", str(out),
                     "--limit", "20", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "1 exam articles left out" in printed
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert lines and {p["task"] for p in lines} == {"openbook_missing"}
    # grounded jobs are still to do: a dry run never claims the data is complete
    assert not out.with_name("o.jsonl.done").exists()


def test_the_done_marker_appears_only_when_no_job_is_left(tmp_path):
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    out = tmp_path / "o.jsonl"
    mod = load_script()
    args = ["--norms", str(norms), "--no-exam", "--out", str(out), "--limit", "10"]
    jobs = make_openbook_jobs(NormIndex(RECORDS), 10, seed=0)
    # every grounded job already answered by an earlier link
    with open(out.with_name("o.jsonl.rejected.jsonl"), "w") as f:
        for j in jobs:
            if j.kind == "grounded":
                f.write(json.dumps({"key": j.key, "reason": "x"}) + "\n")
    assert mod.main(args) == 0
    # submit_when_ready.sh tests with -s: an empty marker counts as missing
    assert out.with_name("o.jsonl.done").stat().st_size > 0


# --- more than one question per article, and jobs in parallel ----------------

def test_one_pair_per_article_is_the_original_draw(index):
    old = make_openbook_jobs(index, 20, seed=5)
    same = make_openbook_jobs(index, 20, seed=5, per_article=1)
    assert [j.key for j in old] == [j.key for j in same]


def test_several_questions_per_article_have_their_own_keys_and_focus(index):
    from eullm_forge.datasets.openbook_gen import build_messages
    jobs = [j for j in make_openbook_jobs(index, 24, seed=1, per_article=3,
                                          missing_share=0) if j.kind == "grounded"]
    keys = [j.key for j in jobs]
    assert len(keys) == len(set(keys)) == 24
    by_article = {}
    for j in jobs:
        by_article.setdefault(j.number, []).append(j)
    assert all(len(v) == 3 for v in by_article.values())
    prompts = {build_messages(j)[1]["content"] for j in by_article[next(iter(by_article))]}
    assert len(prompts) == 3                     # greedy teacher, three different asks


def test_shards_split_one_draw_without_overlap(index):
    whole = make_openbook_jobs(index, 30, seed=2, per_article=2)
    parts = [make_openbook_jobs(index, 30, seed=2, per_article=2, shard=(k, 3))
             for k in range(3)]
    keys = [j.key for p in parts for j in p]
    assert sorted(keys) == sorted(j.key for j in whole) and len(keys) == len(set(keys))


def test_a_second_question_still_counts_as_the_same_trained_article():
    from eullm_forge.eval.norm_exam import trained_articles
    assert trained_articles([{"key": "ob-g-codice_civile-2-bis-v2"},
                             {"key": "ob-g-codice_civile-41"}]) == {
        ("codice_civile", "2-bis"), ("codice_civile", "41")}


def test_the_driver_excludes_every_exam_it_is_given(tmp_path, capsys):
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    a, b = tmp_path / "v3.jsonl", tmp_path / "dev.jsonl"
    a.write_text(json.dumps(exam_item("codice_civile", "3").to_dict()) + "\n")
    b.write_text(json.dumps(exam_item("codice_civile", "4").to_dict()) + "\n")
    assert load_script().main(["--norms", str(norms), "--exclude-exam", str(a), str(b),
                               "--out", str(tmp_path / "o.jsonl"), "--limit", "10",
                               "--per-article", "2", "--shard", "1/2", "--dry-run"]) == 0
    assert "2 exam articles left out" in capsys.readouterr().out
