"""The perplexity corpus is a measurement someone will cite, so what it holds
has to be knowable: which records, chosen how, and — the part that used to be
unknowable — which collections.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_ppl_corpus.py"


def _load():
    spec = importlib.util.spec_from_file_location("make_ppl_corpus", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mpc = _load()

COURTS = ["cassazione_civile", "cassazione_penale", "snciv", "sncpen"]
TEXT = "Dispositivo: la corte ha ritenuto fondato il ricorso. " * 12


def _val(path: Path, per_court: int = 40) -> Path:
    """A val.jsonl as format_pretraining.py writes one: grouped by the slice
    file each record came from, so the courts are contiguous blocks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for court in COURTS:
        for i in range(per_court):
            lines.append(json.dumps(
                {"text": f"{TEXT} {court} record {i}", "kind": court},
                ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _meta(out: Path) -> dict:
    return json.loads(out.with_suffix(out.suffix + ".meta.json").read_text(encoding="utf-8"))


def test_the_shipped_command_does_not_take_a_prefix_of_the_split(tmp_path):
    """val.jsonl is written in source-slice order, so a positional draw is the
    alphabetically first court and nothing else: on a four-court corpus the
    40-chunk corpus held records from two of them, and every row of
    perplexity.csv inherits that. The shipped command names no --seed."""
    val = _val(tmp_path / "val.jsonl")
    out = tmp_path / "eval.txt"
    assert mpc.main(["--val", str(val), "--out", str(out), "--target-chunks", "40"]) == 0
    meta = _meta(out)
    assert set(meta["kinds_drawn"]) == set(COURTS)
    assert set(meta["kinds_available"]) == set(COURTS)
    assert meta["selection"] == "shuffled at seed 42"
    assert meta["seed"] == mpc.DEFAULT_SEED


def test_the_same_seed_gives_the_same_corpus(tmp_path):
    """Determinism is the reason the rule is a seeded shuffle and not a fresh
    draw: the number has to survive a lost shell history."""
    val = _val(tmp_path / "val.jsonl")
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    explicit = tmp_path / "c.txt"
    assert mpc.main(["--val", str(val), "--out", str(first)]) == 0
    assert mpc.main(["--val", str(val), "--out", str(second)]) == 0
    assert mpc.main(["--val", str(val), "--out", str(explicit),
                     "--seed", str(mpc.DEFAULT_SEED)]) == 0
    assert first.read_bytes() == second.read_bytes() == explicit.read_bytes()


def test_file_order_is_still_asked_for_explicitly(tmp_path):
    """A negative seed keeps "the first N records" meaning that, which over a
    slice-ordered split is one court — the sidecar has to say so rather than
    present it as the whole held-out set."""
    val = _val(tmp_path / "val.jsonl")
    out = tmp_path / "eval.txt"
    # A budget well under one court per slice, so file order cannot reach past
    # the first one: a 5-chunk corpus taken positionally is one court.
    assert mpc.main(["--val", str(val), "--out", str(out), "--target-chunks", "5",
                     "--seed", "-1"]) == 0
    meta = _meta(out)
    assert meta["selection"] == "file order"
    assert set(meta["kinds_drawn"]) == {COURTS[0]}
    assert len(meta["kinds_drawn"]) < len(meta["kinds_available"])


def test_records_without_a_kind_still_count_towards_the_budget(tmp_path):
    val = tmp_path / "val.jsonl"
    lines = [json.dumps({"text": TEXT * 2}) for _ in range(80)]
    val.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "eval.txt"
    assert mpc.main(["--val", str(val), "--out", str(out), "--target-chunks", "5"]) == 0
    meta = _meta(out)
    assert meta["kinds_drawn"] == {}
    assert meta["records_used"] > 0
    assert out.read_text(encoding="utf-8").strip()


def test_an_existing_corpus_is_not_silently_replaced(tmp_path):
    val = _val(tmp_path / "val.jsonl")
    out = tmp_path / "eval.txt"
    assert mpc.main(["--val", str(val), "--out", str(out)]) == 0
    before = out.read_bytes()
    try:
        mpc.main(["--val", str(val), "--out", str(out)])
    except SystemExit as e:
        assert "exists" in str(e)
    else:
        raise AssertionError("an existing corpus must not be replaced without --force")
    assert out.read_bytes() == before
