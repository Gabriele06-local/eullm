"""make_caselaw_sft.py: case-law SFT pairs from clean cards, cited by their made-up references."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_caselaw_sft.py"


def _load():
    spec = importlib.util.spec_from_file_location("make_caselaw_sft", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOPICS = ["inadempimento contrattuale onere della prova", "clausola penale riduzione",
          "locazione recesso conduttore", "appalto vizi dell'opera", "mutuo interessi usurari",
          "condominio delibera assembleare"]


def _cards(tmp_path):
    rows = []
    for n, t in enumerate(TOPICS):
        rows.append({"id": f"card-{n:06d}",
                     "estremi": {"autorita": "Cass. civ.", "sezione": "Sez. III",
                                 "numero": 51000 + n, "anno": 2023, "data": "2023-03-01"},
                     "principi": [f"In tema di {t}, il principio numero {n} si applica sempre."],
                     "norme": ["art. 1218 c.c."], "materia": t,
                     "domande_esame": [{"domanda": f"Che cosa vale in tema di {t}?",
                                        "risposta": f"Vale il principio numero {n}",
                                        "rubrica": "x"}]})
    path = tmp_path / "clean.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_grounded_pairs_show_the_card_among_others_and_cite_it(tmp_path, capsys):
    mod = _load()
    out = tmp_path / "sft.jsonl"
    mix = tmp_path / "statute.jsonl"
    mix.write_text(json.dumps({"instruction": "Testi normativi...", "output": "Secondo l'art."})
                   + "\n")
    assert mod.main([str(_cards(tmp_path)), "--absent-share", "0", "--mix", str(mix),
                     "--out", str(out)]) == 0
    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(rows) == len(TOPICS) + 1
    pairs = [r for r in rows if r.get("task") == "caselaw_grounded"]
    for r in pairs:
        n = int(r["key"].split("-")[1])
        assert r["instruction"].startswith("Testi di riferimento (sentenze della Corte di "
                                           "cassazione)")
        assert f"Cass. civ., Sez. III, n. {51000 + n}/2023" in r["instruction"]
        assert r["instruction"].count("Cass. civ.") == 3
        assert r["output"] == f"Vale il principio numero {n}. In questo senso Cass. civ., " \
                              f"Sez. III, n. {51000 + n}/2023."
    printed = capsys.readouterr().out
    assert "principio" not in printed and f"{len(TOPICS)} grounded" in printed


def test_absent_pairs_leave_the_card_out_and_cite_nothing(tmp_path):
    mod = _load()
    out = tmp_path / "sft.jsonl"
    assert mod.main([str(_cards(tmp_path)), "--absent-share", "1", "--out", str(out)]) == 0
    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert rows and all(r["task"] == "caselaw_absent" for r in rows)
    for r in rows:
        n = int(r["key"].split("-")[1])
        assert f"n. {51000 + n}/2023" not in r["instruction"]
        assert r["output"] == mod.ABSENT_ANSWER and "Cass." not in r["output"]


def test_cards_of_the_rulings_themselves_are_refused(tmp_path):
    mod = _load()
    raw = tmp_path / "schede.jsonl"
    raw.write_text(json.dumps({"id": "snciv/2023/12345", "principi": ["p"],
                               "domande_esame": []}) + "\n")
    with pytest.raises(SystemExit):
        mod.main([str(raw), "--out", str(tmp_path / "o.jsonl")])
