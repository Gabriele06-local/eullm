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
