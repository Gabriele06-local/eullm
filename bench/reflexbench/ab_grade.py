"""How an answer is graded: the number or the letter it gives, exact match
and token F1 against a reference, and a judge's verdict.

A model is asked to end with "Answer: <number>" or "Answer: <letter>"; what
follows the last "Answer:" is what it answered. Without one, the last number
it wrote, or a letter standing alone on its last line, is taken instead —
small models forget the format more often than large ones, and grading the
format rather than the answer would favour the large model for the wrong
reason.
"""

import re
import string

NUMBER = re.compile(r"-?\$?\s?\d[\d,]*(?:\.\d+)?")
ANSWER = re.compile(r"answer\s*(?:is)?\s*[:：]", re.IGNORECASE)
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def visible(text):
    """The answer without a reasoning block the model wrote first.

    With U+2212 MINUS SIGN folded to a hyphen: models emit it, and without
    the fold "−14" extracted as 14.0 and graded a correct determinant wrong.
    """
    return THINK.sub("", text or "").replace("\u2212", "-")


def after_answer(text):
    """What follows the last "Answer:" in `text`, or None."""
    matches = list(ANSWER.finditer(text))
    return text[matches[-1].end() :] if matches else None


def to_number(token):
    try:
        return float(token.replace("$", "").replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def extract_number(text):
    """The number answered: the first after the last "Answer:", else the
    last in the text; None when there is none."""
    text = visible(text)
    tail = after_answer(text)
    if tail is not None:
        found = NUMBER.search(tail)
        if found:
            return to_number(found.group())
    found = NUMBER.findall(text)
    return to_number(found[-1]) if found else None


def extract_letter(text, letters="ABCDE"):
    """The option answered: the first option letter after the last "Answer:",
    alone or in brackets, else a letter standing alone on the last line that
    has text; None when there is none."""
    text = visible(text)
    pattern = re.compile(rf"(?<![A-Za-z])[\(\[]?([{letters}])[\)\]\.]?(?![A-Za-z])")
    tail = after_answer(text)
    if tail is not None:
        found = pattern.search(tail)
        if found:
            return found.group(1)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        found = re.fullmatch(rf"[\(\[]?([{letters}])[\)\]\.]?", lines[-1])
        if found:
            return found.group(1)
    return None


def normalize(text):
    """Lower case, no punctuation, no articles, one space between words: the
    SQuAD normalization."""
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact(answer, reference):
    return normalize(visible(answer)) == normalize(str(reference))


def f1(answer, reference):
    """Token F1 between an answer and a reference, normalized."""
    a, r = normalize(visible(answer)).split(), normalize(str(reference)).split()
    if not a or not r:
        return float(a == r)
    common = {}
    for token in set(a) & set(r):
        common[token] = min(a.count(token), r.count(token))
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(a), same / len(r)
    return 2 * precision * recall / (precision + recall)


# An F1 at or above this counts as a correct answer.
F1_CORRECT = 0.5


def correct(item, answer):
    """Whether `answer` is right for `item`, for the graders with a
    reference; a `judge` item is graded by comparing the two models."""
    if item.grader == "number":
        got, want = extract_number(answer), to_number(str(item.answer))
        return got is not None and want is not None and abs(got - want) < 1e-6
    if item.grader == "letter":
        return extract_letter(answer) == str(item.answer).strip().upper()
    if item.grader == "exact":
        return exact(answer, item.answer)
    if item.grader == "f1":
        return f1(answer, item.answer) >= F1_CORRECT
    raise ValueError(f"{item.id}: a {item.grader} item is graded by a judge")


def judge_verdict(text):
    """A judge's reply read as "A", "B" or "tie"; None when it says neither."""
    text = visible(text).strip()
    tail = after_answer(text)
    for candidate in ([tail] if tail is not None else []) + [text]:
        found = re.search(r"(?<![A-Za-z])(A|B|tie|TIE|Tie)(?![A-Za-z])", candidate)
        if found:
            word = found.group(1)
            return "tie" if word.lower() == "tie" else word
    return None


def both_orders(first, second):
    """The verdict of a judge asked twice, the small model's answer first as
    A and then as B: who is better only when both orders say so, a tie
    otherwise. `first` and `second` are the two replies' verdicts."""
    if first == "A" and second == "B":
        return "small"
    if first == "B" and second == "A":
        return "large"
    return "tie"
