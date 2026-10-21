"""legal_eval.split_reasoning: only the answer after a reasoning block is graded."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "legal_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("legal_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_both_reasoning_formats_are_taken_apart():
    mod = _load()
    out = mod.split_reasoning("<think>\nla premessa è falsa\n</think>\n\n"
                              "No: da sei mesi a tre anni.")
    assert out == ("No: da sei mesi a tre anni.", "la premessa è falsa", True)
    out = mod.split_reasoning("[THINK]art. 640[/THINK]Risposta.")
    assert out == ("Risposta.", "art. 640", True)


def test_an_answer_without_reasoning_is_left_alone_and_a_cut_one_has_no_answer():
    mod = _load()
    assert mod.split_reasoning("Risposta diretta.") == ("Risposta diretta.", "", True)
    assert mod.split_reasoning("[THINK]sto ancora ragionando") == ("", "sto ancora ragionando",
                                                                   False)


class _Tok:
    """Enough of a tokenizer: a template that records its variables, and ids
    that decode to fixed strings, [THINK] among the special tokens."""

    all_special_tokens = ["<s>", "</s>", "<pad>", "[THINK]", "[/THINK]", "[INST]"]
    pieces = {1: "<s>", 2: "</s>", 3: "[THINK]", 4: "ragiono", 5: "[/THINK]", 6: "Risposta."}

    def __init__(self):
        self.seen = {}

    def apply_chat_template(self, messages, **kw):
        self.seen, self.messages = kw, messages
        return messages[-1]["content"]

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [1]}

    def decode(self, ids, skip_special_tokens=False):
        out = [self.pieces[i] for i in ids]
        if skip_special_tokens:
            out = [p for p in out if p not in self.all_special_tokens]
        return "".join(out)


def test_reasoning_effort_is_asked_only_when_thinking():
    mod, tok = _load(), _Tok()
    mod.chat_prompt(tok, "domanda", think=True)
    assert tok.seen["reasoning_effort"] == "high" and tok.seen["enable_thinking"] is True
    mod.chat_prompt(tok, "domanda", think=False)
    assert "reasoning_effort" not in tok.seen and tok.seen["enable_thinking"] is False


def test_a_system_prompt_goes_first_and_none_leaves_the_template_default():
    mod, tok = _load(), _Tok()
    mod.chat_prompt(tok, "domanda")
    assert tok.messages == [{"role": "user", "content": "domanda"}]
    mod.chat_prompt(tok, "domanda", system="Sei legal-it.")
    assert tok.messages == [{"role": "system", "content": "Sei legal-it."},
                            {"role": "user", "content": "domanda"}]


def test_mistral_think_markers_survive_decoding_and_are_split_off():
    mod, tok = _load(), _Tok()
    ids = [3, 4, 5, 6, 2]
    assert mod.decode_answer(tok, ids) == "ragionoRisposta."
    kept = mod.decode_answer(tok, ids, keep_reasoning=True)
    assert kept == "[THINK]ragiono[/THINK]Risposta."
    assert mod.split_reasoning(kept) == ("Risposta.", "ragiono", True)


def test_the_gguf_path_keeps_the_markers_that_skip_special_tokens_deletes():
    """legal_eval --gguf --think: [THINK] is a special token, so decoding it
    away left the reasoning glued to the answer and the keyword coverage
    graded the thinking block."""
    from eullm_forge.eval.gguf import generate_answers_gguf

    class Server:
        def complete(self, ids, max_new_tokens, **kw):
            return {"tokens": [3, 4, 5, 6, 2], "stop_type": "eos"}

    tok = _Tok()
    plain = generate_answers_gguf(Server(), tok, ["q"], max_new_tokens=8, end_ids=[2])
    kept = generate_answers_gguf(Server(), tok, ["q"], max_new_tokens=8, end_ids=[2],
                                 keep_reasoning=True)
    assert plain == [("ragionoRisposta.", True)]
    assert kept == [("[THINK]ragiono[/THINK]Risposta.", True)]
