"""Tests for the held-out gate.

The gate exists because its failure mode is silent: a corpus containing the
evaluation years trains fine and reports a flattering number. So these tests
pin both directions — that a contaminated corpus fails, and that a clean one
passes — and, importantly, that the clean case would *not* pass if the check
were vacuous.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_corpus_holdout import main, record_source, record_year  # noqa: E402


def write(dirpath: Path, name: str, records: list[dict]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / name).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


CASS = {"text": "x", "source_id": "snciv/2023/1", "year": 2023, "kind": "snciv"}
CDS_OK = {"text": "x", "source_id": "cds/2019/7", "year": 2019, "kind": "cds"}
CDS_BAD = {"text": "x", "source_id": "cds/2025/9", "year": 2025, "kind": "cds"}


def test_clean_corpus_passes(tmp_path):
    write(tmp_path, "train.jsonl", [CASS, CDS_OK])
    assert main([str(tmp_path), "--require-source", "cds"]) == 0


def test_heldout_year_fails(tmp_path):
    write(tmp_path, "train.jsonl", [CASS, CDS_OK, CDS_BAD])
    assert main([str(tmp_path)]) == 1


def test_2026_fails_too(tmp_path):
    """The partition is 2025 *and later*, not 2025 alone."""
    write(tmp_path, "train.jsonl",
          [dict(CDS_BAD, source_id="cds/2026/1", year=2026)])
    assert main([str(tmp_path)]) == 1


def test_cassazione_2025_is_allowed(tmp_path):
    """Cassazione 2021-2026 is all training data — only CdS is held out.

    A blanket year check would be simpler and wrong: it would reject the
    corpus we are already training on.
    """
    write(tmp_path, "train.jsonl",
          [dict(CASS, source_id="snciv/2025/4", year=2025)])
    assert main([str(tmp_path)]) == 0


def test_missing_corpus_material_fails(tmp_path):
    """The other direction: an arm pointed at the old data directory."""
    write(tmp_path, "train.jsonl", [CASS])
    assert main([str(tmp_path), "--require-source", "cds"]) == 1
    # ...and without --require-source the same corpus is fine, which is what
    # makes the flag meaningful rather than decorative.
    assert main([str(tmp_path)]) == 0


def test_undated_heldout_record_fails(tmp_path):
    """A CdS record with no year cannot be shown to be outside the range.

    Treating it as safe is how a contaminated corpus passes: `year` is
    backfilled by the formatter, and a corpus assembled by hand may not carry
    it at all.
    """
    write(tmp_path, "train.jsonl", [{"text": "x", "kind": "cds"}])
    assert main([str(tmp_path)]) == 1


def test_year_read_from_id_when_field_absent(tmp_path):
    write(tmp_path, "train.jsonl",
          [{"text": "x", "kind": "cds", "source_id": "cds/2025/3"}])
    assert main([str(tmp_path)]) == 1


def test_val_file_is_checked_when_asked(tmp_path):
    write(tmp_path, "train.jsonl", [CDS_OK])
    write(tmp_path, "val.jsonl", [CDS_BAD])
    assert main([str(tmp_path)]) == 0                      # train only
    assert main([str(tmp_path), "--files", "train.jsonl", "val.jsonl"]) == 1


def test_missing_file_fails(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    assert main([str(tmp_path)]) == 1


def test_missing_directory_fails(tmp_path):
    assert main([str(tmp_path / "nope")]) == 1


def test_malformed_line_does_not_hide_a_violation(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "train.jsonl").write_text(
        "{not json\n" + json.dumps(CDS_BAD) + "\n", encoding="utf-8")
    assert main([str(tmp_path)]) == 1


@pytest.mark.parametrize("rec,expected", [
    ({"source_id": "CDS/2019/1"}, "cds/2019/1"),
    ({"source": "cds"}, "cds"),
    ({"kind": "cds"}, "cds"),
    ({}, ""),
])
def test_record_source(rec, expected):
    assert record_source(rec) == expected


@pytest.mark.parametrize("rec,expected", [
    ({"year": 2019}, 2019),
    ({"year": "2019"}, 2019),
    ({"source_id": "cds/2019/1"}, 2019),
    ({"sentence_id": "2019-00042"}, 2019),
    ({"source_id": "cds/12/1"}, None),
    ({}, None),
])
def test_record_year(rec, expected):
    assert record_year(rec) == expected
