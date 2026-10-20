"""premise_gen: questions on a false premise, answered by correcting it from the texts shown."""

from __future__ import annotations

import importlib.util
import json
import random
import re
from pathlib import Path

import pytest

from eullm_forge.datasets.premise_gen import (
    build_pairs,
    excluded_articles,
    quote_span,
    say,
    wrong_deadline,
)
from eullm_forge.eval import EvalItem, NormIndex
from eullm_forge.rl import corrects, score_answer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_premise_pairs.py"
FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3


def rec(code, text, chunk_index=0):
    return {"code": code, "article_num": "", "chunk_index": chunk_index, "text": text}


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


def _index():
    return NormIndex(RECORDS)


def _pairs(fmt="steps", exclude=frozenset(), seed=0, stats=None):
    return build_pairs(_index(), per_code=10, rng=random.Random(seed), fmt=fmt,
                       exclude=exclude, stats=stats)


def _passage(instruction: str, pos: int) -> str:
    return " ".join(instruction.split(f"[{pos}] ", 1)[1].split("\n[")[0].split())


def test_a_wrong_deadline_is_corrected_with_the_article_quoted_where_it_is_shown():
    pairs = [p for p in _pairs() if p.kind == "premessa_termine"]
    assert {p.number for p in pairs} == {"2", "3"}
    for p in pairs:
        right = "60 giorni" if p.number == "2" else "10 giorni"
        assert right not in p.question                       # the premise is wrong
        pos = int(re.search(r"\[(\d)\]", p.output).group(1))
        assert f"art. {p.number}" in _passage(p.instruction, pos).lower()
        quote = re.search(r"«(.+?)»", p.output).group(1)
        assert quote in _passage(p.instruction, pos)         # a literal span of the text shown
        assert p.output.startswith("Verifica della premessa:")
        assert "Norma applicabile:" in p.output and f"Risposta: il termine previsto è di {right}" \
            in p.output
        assert score_answer(p.output, p.kind, p.keywords) == 1.0
        assert "Nota:" not in p.instruction


def test_the_wrong_deadline_is_one_the_article_does_not_state():
    rng = random.Random(1)
    for _ in range(50):
        w = wrong_deadline(30, "giorni", {(30, "giorni"), (60, "giorni")}, rng)
        assert w not in (30, 60)
    assert say(1, "anni") == "1 anno" and say(3, "mesi") == "3 mesi"


def test_an_article_not_in_the_collection_is_attributed_nothing():
    pairs = [p for p in _pairs() if p.kind == "premessa_inesistente"]
    assert pairs
    taken = {"1", "2", "3", "4", "5"}
    for p in pairs:
        assert p.number not in taken
        assert "Nota:" not in p.instruction                  # as RAG Enterprise asks it
        assert f"non c'è l'art. {p.number}" in p.output
        assert score_answer(p.output, p.kind, p.keywords) == 1.0
    assert any(p.number.endswith("-bis") for p in pairs)


def test_a_repealed_article_is_said_to_be_repealed_quoting_its_text():
    [p] = [p for p in _pairs() if p.kind == "premessa_abrogato"]
    assert p.number == "4"
    pos = int(re.search(r"\[(\d)\]", p.output).group(1))
    assert re.search(r"«(.+?)»", p.output).group(1) in _passage(p.instruction, pos)
    assert score_answer(p.output, p.kind, []) == 1.0
    assert score_answer("L'art. 4 prevede che il contratto sia nullo.", p.kind, []) == 0.0


def test_the_plain_format_says_the_same_in_one_paragraph():
    for p in _pairs(fmt="plain"):
        assert "\n" not in p.output and "Verifica della premessa" not in p.output
        assert score_answer(p.output, p.kind, p.keywords) == 1.0


def test_excluded_articles_are_never_used_and_an_outside_set_is_read_from_its_questions():
    lexia = EvalItem(id="lexia-1", domain="legal", lang="it", category="x",
                     question="Ai sensi dell'art. 2 del codice civile, entro quando?",
                     reference="", metadata={})
    exam = EvalItem(id="norm-termine-codice_civile-3", domain="legal", lang="it",
                    category="codice_civile", question="Quale termine prevede l'art. 3?",
                    reference="", metadata={"code": "codice_civile", "articolo": "3"})
    exclude = excluded_articles([lexia, exam])
    assert exclude == {("codice_civile", "2"), ("codice_civile", "3")}
    assert not any(p.number in ("2", "3") for p in _pairs(exclude=exclude))


def test_a_long_sentence_is_quoted_around_the_deadline():
    sentence = " ".join(["parola"] * 80) + " entro 60 giorni " + " ".join(["altro"] * 80)
    q = quote_span(sentence, anchor="60")
    assert "60 giorni" in q and len(q) < len(sentence) and q in sentence


def test_the_reward_wants_the_premise_corrected_not_only_the_right_number():
    kw = ["60 giorni|sessanta giorni"]
    assert score_answer("Il termine è di 60 giorni.", "premessa_termine", kw) == 0.0
    assert score_answer("No: il termine è di 60 giorni, non di 30 giorni.",
                        "premessa_termine", kw) == 1.0
    assert score_answer("La premessa è errata: il termine è di 30 giorni.",
                        "premessa_termine", kw) == 0.0
    assert corrects("In realtà l'articolo prevede altro") and not corrects("Non so.")


def _load():
    spec = importlib.util.spec_from_file_location("make_premise_pairs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_script_splits_sft_and_grpo_by_article_and_prints_counts_only(tmp_path, capsys):
    norms = tmp_path / "legislazione_test.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    sft, grpo = tmp_path / "sft.jsonl", tmp_path / "grpo.jsonl"
    assert _load().main(["--norms", str(norms), "--per-code", "10", "--out-sft", str(sft),
                         "--out-grpo", str(grpo)]) == 0
    s = [json.loads(ln) for ln in sft.read_text().splitlines()]
    g = [json.loads(ln) for ln in grpo.read_text().splitlines()]
    assert s and g
    assert not {(r["source"], r["key"].split("-", 3)[3]) for r in s} & \
        {(r["code"], r["articolo"]) for r in g}
    assert all(r["prompt"][0]["role"] == "user" and r["tipo"].startswith("premessa_") for r in g)
    out = capsys.readouterr().out
    assert "[premise]" in out and "Verifica" not in out and "giorni" not in out


def test_the_plain_run_writes_the_same_pairs_as_the_steps_run(tmp_path):
    norms = tmp_path / "legislazione_test.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    steps, plain = tmp_path / "steps.jsonl", tmp_path / "plain.jsonl"
    common = ["--norms", str(norms), "--per-code", "10"]
    assert _load().main(common + ["--out-sft", str(steps), "--out-grpo",
                                  str(tmp_path / "g.jsonl")]) == 0
    assert _load().main(common + ["--format", "plain", "--out-sft", str(plain)]) == 0
    a = [json.loads(ln) for ln in steps.read_text().splitlines()]
    b = [json.loads(ln) for ln in plain.read_text().splitlines()]
    assert [r["key"] for r in a] == [r["key"] for r in b]
    assert [r["instruction"] for r in a] == [r["instruction"] for r in b]
    assert all("Verifica della premessa" not in r["output"] for r in b)


def test_the_script_needs_an_output(tmp_path):
    with pytest.raises(SystemExit):
        _load().main(["--norms", str(tmp_path / "x.jsonl")])
