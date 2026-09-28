"""Tests for the pretraining formatter's train/val split.

Grouping is what makes the held-out set a validation set rather than a
decorated training set: a typo in --group-by must fail loudly, not degrade
to the per-chunk split the project removed for inflating held-out scores.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "format_pretraining.py"


def _load():
    spec = importlib.util.spec_from_file_location("format_pretraining", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


fmt = _load()


def _corpus(root: Path, n_docs: int = 2, chunks_per_doc: int = 5) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for d in range(n_docs):
        for c in range(chunks_per_doc):
            lines.append(
                json.dumps(
                    {
                        "text": f"document {d} chunk {c} with enough words to survive",
                        "sentence_id": f"doc-{d}",
                    }
                )
            )
    (root / "italgiure_snciv_2023.dedup.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return root


def _read_docs(path: Path) -> set[str]:
    return {
        json.loads(line)["sentence_id"] for line in path.read_text(encoding="utf-8").splitlines()
    }


def test_unknown_group_by_field_is_refused_not_silently_ungrouped(tmp_path):
    """A --group-by typo must fail, not produce a per-chunk split."""
    _corpus(tmp_path)
    with pytest.raises(SystemExit) as e:
        fmt.main([str(tmp_path), "--group-by", "sentence-id", "--dry-run"])
    assert e.value.code == 2
    assert not (tmp_path / "pretraining" / "train.jsonl").exists()


def test_known_group_by_field_keeps_documents_whole(tmp_path):
    """The valid path is untouched: each document lands on exactly one side."""
    out = tmp_path / "out"
    _corpus(tmp_path / "corpus")
    assert (
        fmt.main(
            [
                str(tmp_path / "corpus"),
                "--group-by",
                "sentence_id",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    train_docs = _read_docs(out / "train.jsonl")
    val_docs = _read_docs(out / "val.jsonl")
    assert train_docs == {"doc-0", "doc-1"} - val_docs
    assert val_docs, "one whole document must be held out"
    assert train_docs, "one whole document must be trained on"


# --- what the slice file name says about the records in it ------------------

@pytest.mark.parametrize("name,year,kind", [
    # The shape it always handled.
    ("italgiure_snciv_2023.dedup.jsonl", 2023, "snciv"),
    # The range names docs/corpus-acquisition.md tells you to copy. The year
    # used to be thrown away by int("2017-2024"), so the whole Consiglio di
    # Stato slice reached train.jsonl with no year at all.
    ("italgiure_cds_2017-2024.dedup.jsonl", 2024, "cds"),
    # A range that reaches into the held-out window takes its end: a year
    # too high refuses a corpus, a year too low lets one through.
    ("italgiure_snciv_2021-2026.dedup.jsonl", 2026, "snciv"),
    # A multi-word collection used to be cut to its first word, so the civil
    # and the criminal Cassazione slice both arrived as "cassazione".
    ("italgiure_cassazione_civile_2023.dedup.jsonl", 2023, "cassazione_civile"),
    ("italgiure_cassazione_penale_2023.dedup.jsonl", 2023, "cassazione_penale"),
    # Every legislation code used to be stamped "codice".
    ("legislazione_codice_civile.chunks.jsonl", None, "codice_civile"),
    ("legislazione_ricorsi_amministrativi.chunks.jsonl", None, "ricorsi_amministrativi"),
])
def test_a_slice_file_name_yields_its_year_and_collection(name, year, kind):
    assert fmt._infer_year_kind(Path(name)) == (year, kind)


def test_a_range_slice_stamps_a_year_the_held_out_gate_can_use(tmp_path):
    """End to end, because the year only matters once it is in train.jsonl:
    without it the held-out gate has nothing but the fetcher's id to read."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "italgiure_cds_2017-2024.dedup.jsonl").write_text(
        json.dumps({"text": "dispositivo", "sentence_id": "90135/2017"}) + "\n"
        + json.dumps({"text": "dispositivo", "sentence_id": "90136/2019"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert fmt.main([str(corpus), "--output", str(out), "--group-by", "none"]) == 0
    rows = [json.loads(ln) for f in sorted(out.glob("*.jsonl"))
            for ln in f.read_text(encoding="utf-8").splitlines()]
    assert rows, "the split must not have written nothing"
    assert {r["year"] for r in rows} == {2024}
    assert {r["kind"] for r in rows} == {"cds"}
