"""`contains_word` matches a word, not any substring.

Expected 'W' used to pass "I don't know" (the w in "know") and expected 'a'
passed "span", so a model that answered neither still scored. The mode is
called contains_word: a wrong answer that merely contains the letters fails,
and a right answer in any case or punctuation still passes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "turboquant_quality", Path(__file__).resolve().parent / "turboquant_quality.py")
_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_mod)

check_answer = _mod.check_answer
contains_word = _mod.contains_word
TESTS = {t["id"]: t for t in _mod.TESTS}


def test_a_wrong_answer_is_not_passed_by_its_letters():
    assert check_answer(TESTS["fact02"], "I don't know")[0] is False
    assert check_answer(TESTS["code15"], "span")[0] is False


def test_the_right_answer_passes_bare_and_in_punctuation():
    assert check_answer(TESTS["fact02"], "W")[0] is True
    assert check_answer(TESTS["code15"], "a")[0] is True
    assert check_answer(TESTS["code15"], "Use the <a> tag.")[0] is True
    assert check_answer(TESTS["fact05"], "4%")[0] is True


def test_every_expected_word_matches_itself_bare_and_in_a_sentence():
    for t in _mod.TESTS:
        if t["check"] != "contains_word":
            continue
        assert check_answer(t, t["expected"])[0] is True, t["id"]
        assert check_answer(t, "La risposta e " + t["expected"] + ".")[0] is True, t["id"]


def test_contains_word_needs_no_word_character_on_either_side():
    # case is the caller's job: check_answer lowercases both sides first.
    assert contains_word("know", "w") is False
    assert contains_word("span", "a") is False
    assert contains_word("110", "10") is False
    assert contains_word("w", "w") is True
    assert contains_word("(w)", "w") is True
    assert contains_word("4%", "4%") is True
    assert contains_word("il 40% circa", "4%") is False
