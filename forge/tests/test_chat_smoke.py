"""The chat smoke test runs end to end on a tiny model and records a verdict."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "chat_smoke.py"


def test_chat_smoke_reports_whether_answers_end_their_turn(tmp_path, monkeypatch, capsys):
    pytest.importorskip("transformers")
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from eullm_forge.identity import ensure_chat_template

    special = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "[UNK]"]
    raw = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    raw.train_from_iterator(["user assistant Come ti chiami ? ricorso Stato articolo"],
                            trainers.WordLevelTrainer(special_tokens=special))
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]",
                                  eos_token="<|endoftext|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=special[:2])
    ensure_chat_template(tok)
    model_dir = tmp_path / "merged"
    tok.save_pretrained(model_dir)
    Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    )).save_pretrained(model_dir)

    spec = importlib.util.spec_from_file_location("chat_smoke", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out_csv = tmp_path / "smoke.csv"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["chat_smoke.py", str(model_dir), "--max-new-tokens", "5",
                                      "--csv", str(out_csv), "--label", "tiny"])
    assert mod.main() == 0

    printed = capsys.readouterr().out
    assert printed.count("[smoke] Q:") == len(mod.QUESTIONS)
    assert "answers ended their turn" in printed
    rows = list(csv.DictReader(out_csv.open()))
    assert rows[0]["label"] == "tiny"
    assert int(rows[0]["asked"]) == len(mod.QUESTIONS)
    assert 0 <= int(rows[0]["ended"]) <= len(mod.QUESTIONS)
