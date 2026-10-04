"""publish_hf.py refuses what should not be published, before sending anything."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish_hf", ROOT / "scripts" / "publish_hf.py")
publish_hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish_hf)

CARDS = ROOT / "model_cards"


def _gguf(tmp_path: Path, magic: bytes = b"GGUF") -> Path:
    p = tmp_path / "m-q4_k_m.gguf"
    p.write_bytes(magic + b"\x03\x00\x00\x00" + b"\x00" * 64)
    return p


def test_the_shipped_cards_pass_for_their_own_repository(tmp_path):
    for model in ("legal-it-8b", "legal-it-4b"):
        assert publish_hf.problems(f"eullm/{model}", _gguf(tmp_path), f"{model}-Q4_K_M.gguf",
                                   CARDS / model / "README.md") == []


def test_a_card_of_the_other_model_is_caught(tmp_path):
    found = publish_hf.problems("eullm/legal-it-4b", _gguf(tmp_path), "legal-it-4b-Q4_K_M.gguf",
                                CARDS / "legal-it-8b" / "README.md")
    assert any("not the card of legal-it-4b" in p for p in found)


def test_a_file_that_is_not_gguf_or_a_name_without_quant_is_refused(tmp_path):
    found = publish_hf.problems("eullm/legal-it-8b", _gguf(tmp_path, b"PK\x03\x04"),
                                "legal-it-8b.gguf", CARDS / "legal-it-8b" / "README.md")
    assert any("not a GGUF file" in p for p in found)
    assert any("must end with the quantization" in p for p in found)


def test_dry_run_hashes_and_sends_nothing(tmp_path, capsys):
    g = _gguf(tmp_path)
    rc = publish_hf.main(["--repo", "eullm/legal-it-8b", "--gguf", str(g),
                          "--name", "legal-it-8b-Q4_K_M.gguf",
                          "--card", str(CARDS / "legal-it-8b" / "README.md"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and publish_hf.sha256(g) in out and "nothing sent" in out
