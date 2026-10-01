"""The codes-readout prompt, exactly as the engine renders and reads it.

The engine reads a decision model of our own through its *codes readout*
(`engine/src/inference/decision.rs`). Each question about a state becomes one
chat prompt — a fixed system text, then the state, then the question with
its answer codes — rendered with the model's own chat template, reasoning
switched off. Nothing is generated: the next-token distribution at the end
of the prompt is read once and restricted to the codes, `Yes`/`No` for a
yes/no question, `A`…`Z` for a choice, `0`…`9` for a score level, each of
which must be a single token right after the prompt.

A model trained on a prompt that differs from that one by a space is trained
for a prompt it will never be shown, and nothing fails: the engine still
reads probabilities, they only mean less. So everything here mirrors
decision.rs line for line — the system text, the labels, the order, what is
trimmed and what is not, how the template is applied, how the request's text
is kept apart from the template's control tokens, which spellings of a code
count — and `tests/test_decisions.py` holds the two together, reading the
constants out of decision.rs itself when the engine's source is there.

Only the standard library is needed here; the tokenizer is passed in.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

#: decision.rs `SYSTEM_PROMPT`.
SYSTEM_PROMPT = (
    "You are a decision function inside a software system. "
    "Read the state, then answer the question about it. "
    "Reply with exactly one of the allowed codes and nothing else."
)
#: decision.rs `STATE_LABEL` and `QUESTION_LABEL`: everything up to the end
#: of the second is the same for every question about one state.
STATE_LABEL = "State:\n"
QUESTION_LABEL = "\n\nQuestion:"

KINDS = ("noul", "choice", "score")
#: Options a code-readout model can answer, one letter each (`LETTER_CODES`).
LETTER_CODES = 26
#: Levels a score question may have, one digit each (`MAX_SCORE_LEVELS`).
MAX_SCORE_LEVELS = 10
#: Fewest options, or levels, a question may have (`MIN_OPTIONS`).
MIN_OPTIONS = 2

#: What Rust's `str::trim` removes: Unicode White_Space. Python's
#: `str.strip()` also removes U+001C–U+001F, which Rust keeps.
_RUST_WHITESPACE = (
    "\t\n\x0b\x0c\r \x85\xa0        "
    "        　"
)

#: The tag a template's reasoning block opens with. A template that opens
#: one at the end of the prompt has it stripped (`strip_preopened_thinking`
#: in inference/mod.rs), so the model's first token is its own.
THINK_START = "<think>"

#: User turns that carry what a real one can — newlines, brackets and
#: quotes, markup, a `</think>`, non-ASCII text — used, as decision.rs uses
#: them, to find the text a template puts around the user turn.
PROBES = (
    "State:\nx {a} [b] <c> \"d\" 'e' </think>\n\nQuestion: Is it?\nA) f: g\nAnswer Yes or No.",
    "State:\n«Il cliente» chiede — €12,50\n\nQuestion: Qual è?\n0) basso\n1) alto\nReply.",
)


def rust_trim(text: str) -> str:
    """`text.trim()` as Rust has it."""
    return text.strip(_RUST_WHITESPACE)


@dataclass(frozen=True)
class Question:
    """One question as the engine evaluates it (decision.rs `Question`).

    The order of `options` and `levels` is the order they are shown to the
    model and the order of every per-class result: class `i` is option `i`
    (letter `A` + i), level `i` (digit `i`), or for `noul` class 0 = Yes,
    class 1 = No.
    """

    kind: str
    instructions: str
    #: `choice`: (name, description); the description may be empty.
    options: tuple[tuple[str, str], ...] = ()
    #: `score`: level descriptions, lowest first.
    levels: tuple[str, ...] = ()
    #: `noul`: what an answer of true, and of false, means; empty when the
    #: question does not say.
    true_means: str = ""
    false_means: str = ""

    def n_classes(self) -> int:
        if self.kind == "noul":
            return 2
        return len(self.options) if self.kind == "choice" else len(self.levels)

    def labels(self) -> list[str]:
        """The answers' names in class order, as the API keys them."""
        if self.kind == "noul":
            return ["yes", "no"]
        if self.kind == "choice":
            return [name for name, _ in self.options]
        return [str(i) for i in range(len(self.levels))]

    def codes(self) -> list[str]:
        """The code the model answers each class with."""
        return [code(self.kind, i) for i in range(self.n_classes())]

    def problem(self) -> str | None:
        """Why a code-readout model cannot be asked this question, or None.

        decision.rs `Question::validate`, with the code readout's own limit
        of 26 options (the API allows 255 for a verdict model).
        """
        if self.kind not in KINDS:
            return f"unknown question type {self.kind!r}"
        if not rust_trim(self.instructions):
            return "empty instructions"
        texts = [self.instructions, self.true_means, self.false_means]
        texts += [t for option in self.options for t in option] + list(self.levels)
        if any("\0" in t for t in texts):
            return "NUL character"
        if self.kind == "choice":
            if not MIN_OPTIONS <= len(self.options) <= LETTER_CODES:
                return (f"a choice needs {MIN_OPTIONS} to {LETTER_CODES} options "
                        "for the code readout")
            names = [name for name, _ in self.options]
            if any(not rust_trim(name) for name in names):
                return "empty option name"
            if len(set(names)) != len(names):
                return "duplicate option name"
        if self.kind == "score":
            if not MIN_OPTIONS <= len(self.levels) <= MAX_SCORE_LEVELS:
                return f"a score needs {MIN_OPTIONS} to {MAX_SCORE_LEVELS} levels"
            if any(not rust_trim(level) for level in self.levels):
                return "empty level"
        return None


def code(kind: str, i: int) -> str:
    """decision.rs `CodeSet::code`."""
    if kind == "noul":
        return ("Yes", "No")[i]
    if kind == "choice":
        return chr(ord("A") + i)
    return str(i)


def code_set_size(kind: str) -> int:
    return {"noul": 2, "choice": LETTER_CODES, "score": MAX_SCORE_LEVELS}[kind]


def forms(kind: str, i: int) -> list[str]:
    """decision.rs `CodeSet::forms`: the spellings that count as code `i`
    when they are the answer's first token — the code, the code after a
    space and, for Yes/No only, both in lower case."""
    c = code(kind, i)
    spellings = [c, f" {c}"]
    if kind == "noul":
        spellings += [f" {c.lower()}", c.lower()]
    return spellings


def question_text(question: Question) -> str:
    """decision.rs `question_text`: the question's own part of the user
    turn, from the space after `QUESTION_LABEL` to the last line."""
    msg = f" {rust_trim(question.instructions)}\n"
    if question.kind == "noul":
        if rust_trim(question.true_means):
            msg += f"Yes means: {rust_trim(question.true_means)}\n"
        if rust_trim(question.false_means):
            msg += f"No means: {rust_trim(question.false_means)}\n"
        msg += "Answer Yes or No."
    elif question.kind == "choice":
        msg += "Options:\n"
        for i, (name, description) in enumerate(question.options):
            if not rust_trim(description):
                msg += f"{code('choice', i)}) {name}\n"
            else:
                msg += f"{code('choice', i)}) {name}: {rust_trim(description)}\n"
        msg += "Answer with the letter of the best option."
    else:
        msg += "Levels, from lowest to highest:\n"
        for i, level in enumerate(question.levels):
            msg += f"{i}) {rust_trim(level)}\n"
        msg += "Answer with the number of the level that fits best."
    return msg


def user_message(state: str, question: Question) -> str:
    """decision.rs `user_message`: the state first, the question last."""
    return f"{STATE_LABEL}{state}{QUESTION_LABEL}{question_text(question)}"


# --- the API's question format ------------------------------------------------


def python_json(value) -> str:
    """One line of JSON as `json.dumps(value, ensure_ascii=False)` writes it:
    how the engine shows a structured description or score level."""
    return json.dumps(value, ensure_ascii=False)


def compact_json(value) -> str:
    """The same with `(",", ":")` separators: how the engine shows
    structured instructions."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def state_text(state) -> str:
    """The state as a code-readout model reads it: a string as it is, any
    other JSON indented by two spaces (`serde_json::to_string_pretty`)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def _description(value) -> str:
    """systemone.rs `as_description`: a string as it is, nothing for null,
    anything else as one line of JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return python_json(value)


def _instructions(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)) and value:
        return compact_json(value)
    raise ValueError("\"instructions\" must be a string, or a non-empty object or array")


def _score_level(i: int, level) -> str:
    """systemone.rs `score_level`: the text the model reads for a level."""
    if isinstance(level, str):
        return level
    if isinstance(level, dict) and "label" in level:
        label = level.get("label")
        if not isinstance(label, str) or not rust_trim(label):
            raise ValueError(f"score level {i}: \"label\" must be a non-empty string")
        description = level.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(f"score level {i}: \"description\" must be a string")
        if any(key not in ("label", "description") for key in level):
            return python_json(level)
        return f"{label}: {description}" if description else label
    if isinstance(level, (dict, list)) and level:
        return python_json(level)
    raise ValueError(f"score level {i} must be a non-empty string or object")


def question_from_api(spec: dict) -> Question:
    """A System One question (`type`, `instructions`, `criteria`) as the
    engine evaluates it — systemone.rs `parse_question`."""
    if not isinstance(spec, dict):
        raise ValueError("a question must be an object")
    kind = spec.get("type")
    if kind not in KINDS:
        raise ValueError(f"unknown question type {kind!r}")
    instructions = _instructions(spec.get("instructions"))
    criteria = spec.get("criteria")
    if kind == "noul":
        means = {"true": "", "false": ""}
        if criteria is not None:
            if not isinstance(criteria, dict) or set(criteria) - set(means):
                raise ValueError("a noul's criteria is an object {\"true\": …, \"false\": …}")
            means.update({k: _description(v) for k, v in criteria.items()})
        return Question("noul", instructions, true_means=means["true"], false_means=means["false"])
    if kind == "choice":
        if not isinstance(criteria, dict):
            raise ValueError("a choice needs criteria: an object of option name → description")
        options = tuple((str(name), _description(d)) for name, d in criteria.items())
        return Question("choice", instructions, options=options)
    if not isinstance(criteria, list):
        raise ValueError("a score needs criteria: an array of levels, lowest first")
    return Question("score", instructions, levels=tuple(
        _score_level(i, level) for i, level in enumerate(criteria)))


def question_from_record(spec: dict) -> Question:
    """A question as a trace records it: the API's own shape, or the shape
    the engine evaluated — `options` (a list of `{name, description}` or of
    pairs, or an object), `levels`, `true_means`/`false_means`."""
    if not isinstance(spec, dict):
        raise ValueError("a question must be an object")
    if "criteria" in spec:
        return question_from_api(spec)
    kind = spec.get("type", spec.get("kind"))
    if kind not in KINDS:
        raise ValueError(f"unknown question type {kind!r}")
    instructions = _instructions(spec.get("instructions"))
    if kind == "noul":
        return Question("noul", instructions,
                        true_means=_description(spec.get("true_means")),
                        false_means=_description(spec.get("false_means")))
    if kind == "choice":
        raw = spec.get("options")
        if isinstance(raw, dict):
            options = [(str(n), _description(d)) for n, d in raw.items()]
        elif isinstance(raw, list):
            options = []
            for option in raw:
                if isinstance(option, dict):
                    options.append((str(option.get("name", "")),
                                    _description(option.get("description"))))
                elif isinstance(option, (list, tuple)) and option:
                    options.append((str(option[0]),
                                    _description(option[1] if len(option) > 1 else None)))
                else:
                    options.append((str(option), ""))
        else:
            raise ValueError("a choice records its options")
        return Question("choice", instructions, options=tuple(options))
    raw = spec.get("levels")
    if not isinstance(raw, list):
        raise ValueError("a score records its levels")
    return Question("score", instructions, levels=tuple(
        _score_level(i, level) for i, level in enumerate(raw)))


def question_to_api(question: Question) -> dict:
    """The System One question that the engine evaluates as `question`."""
    spec: dict = {"type": question.kind, "instructions": question.instructions}
    if question.kind == "noul":
        criteria = {k: v for k, v in (("true", question.true_means),
                                      ("false", question.false_means)) if v}
        if criteria:
            spec["criteria"] = criteria
    elif question.kind == "choice":
        spec["criteria"] = {name: (d or None) for name, d in question.options}
    else:
        spec["criteria"] = list(question.levels)
    return spec


def answer_index(question: Question, answer) -> int:
    """The class a right answer names: true/false (or yes/no) for a `noul`,
    the option's name for a `choice`, the level's number — or its text —
    for a `score`. Raises ValueError for an answer the question cannot have.
    """
    if question.kind == "noul":
        if isinstance(answer, bool):
            return 0 if answer else 1
        if isinstance(answer, (int, float)) and answer in (0, 1):
            return 0 if answer == 1 else 1
        text = str(answer).strip().lower()
        if text in ("yes", "true", "1"):
            return 0
        if text in ("no", "false", "0"):
            return 1
        raise ValueError(f"a noul answer is true or false, not {answer!r}")
    if question.kind == "choice":
        names = [name for name, _ in question.options]
        if isinstance(answer, str) and answer in names:
            return names.index(answer)
        raise ValueError(f"{answer!r} is not one of the options {names}")
    if isinstance(answer, bool):
        raise ValueError(f"a score answer is a level number, not {answer!r}")
    if isinstance(answer, float) and answer.is_integer():
        answer = int(answer)
    if isinstance(answer, str):
        text = answer.strip()
        if text.isdigit():
            answer = int(text)
        elif text in question.levels:
            return question.levels.index(text)
    if isinstance(answer, int) and 0 <= answer < len(question.levels):
        return answer
    raise ValueError(f"{answer!r} is not a level of 0..{len(question.levels) - 1}")


def answer_value(question: Question, index: int):
    """The right answer of class `index`, as feedback writes it."""
    if question.kind == "noul":
        return index == 0
    if question.kind == "choice":
        return question.options[index][0]
    return index


# --- the prompt, through a model's tokenizer ----------------------------------


def template_wrapper(render) -> tuple[str, str] | None:
    """decision.rs `template_wrapper`: the text a rendered prompt puts before
    and after the user turn, when every probe comes back verbatim between
    the same two strings."""
    found = None
    for probe in PROBES:
        rendered = render(probe)
        if rendered is None:
            return None
        at = rendered.find(probe)
        if at < 0:
            return None
        pair = (rendered[:at], rendered[at + len(probe):])
        if found is None:
            found = pair
        elif found != pair:
            return None
    return found


def template_cut(head: str, tail: str, controls: list[str]) -> tuple[str, str, str, str]:
    """decision.rs `TemplateCut::new`: the head cut after its last control
    token and the tail before its first, as `(head_controls, head_text,
    tail_text, tail_controls)`. What lies between is one run of text, which
    is tokenized with control tokens read as text."""
    controls = [c for c in controls if c]
    head_at = max((head.rfind(c) + len(c) for c in controls if c in head), default=0)
    tail_at = min((tail.find(c) for c in controls if c in tail), default=len(tail))
    return head[:head_at], head[head_at:], tail[:tail_at], tail[tail_at:]


@dataclass
class CodeReadout:
    """How the engine prompts and reads one model, worked out from its
    tokenizer as decision.rs works it out from the GGUF at load.

    The GGUF Forge exports carries this tokenizer's chat template and
    vocabulary, so what is resolved here is what the engine resolves: the
    template, rendered with reasoning off; the BOS token, added where the
    tokenizer adds one; the request's text tokenized with control tokens
    read as text, so a state holding `<|im_end|>` stays text; and, per
    code, the token ids that count as it.
    """

    tokenizer: object
    uses_template: bool = field(init=False)
    adds_bos: bool = field(init=False)
    wrapper: tuple[str, str] | None = field(init=False)
    cut: tuple[str, str, str, str] | None = field(init=False)
    #: Per question kind, per class, the token ids that count as its code.
    codes: dict[str, list[list[int]]] = field(init=False)

    def __post_init__(self):
        tok = self.tokenizer
        bos = getattr(tok, "bos_token_id", None)
        with_special = tok("x")["input_ids"]
        without = tok("x", add_special_tokens=False)["input_ids"]
        self.adds_bos = (
            bos is not None and len(with_special) > len(without) and with_special[0] == bos
        )
        self.uses_template = self._templated("x") is not None
        self.wrapper = template_wrapper(self.render)
        self.cut = None
        if self.wrapper:
            head, tail = self.wrapper
            cut = template_cut(head, tail, self._control_texts(head + tail))
            if self._cut_holds(cut):
                self.cut = cut
        probe = self.render("x")
        base = self.encode(probe, parse=True, bos=True)
        self.codes = {}
        for kind in KINDS:
            classes = []
            for i in range(code_set_size(kind)):
                tokens: list[int] = []
                for spelling in forms(kind, i):
                    extended = self.encode(probe + spelling, parse=True, bos=True)
                    if len(extended) == len(base) + 1 and extended[: len(base)] == base:
                        if extended[-1] not in tokens:
                            tokens.append(extended[-1])
                classes.append(tokens)
            # A token claimed by two codes would count its probability twice.
            owners = Counter(t for tokens in classes for t in tokens)
            self.codes[kind] = [[t for t in tokens if owners[t] == 1] for tokens in classes]

    # The rendering (inference/mod.rs `render_jinja_chat_template`, and
    # decision.rs `render_prompt`).

    def _templated(self, user: str) -> str | None:
        tok = self.tokenizer
        if not getattr(tok, "chat_template", None):
            return None
        try:
            text = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except Exception:  # the engine falls back to the plain prompt too
            return None
        if not isinstance(text, str):
            return None
        # llama.cpp drops the BOS text a template starts with when the
        # tokenizer adds BOS itself, and the tokenizer then adds it.
        bos = getattr(tok, "bos_token", None)
        if self.adds_bos and bos and text.startswith(bos):
            text = text[len(bos):]
        # A template that opens a reasoning block whatever it is told has the
        # tag stripped, so the answer's first token is the model's own.
        if THINK_START in str(tok.chat_template):
            content = text.rstrip()
            if content.endswith(THINK_START):
                text = content[: -len(THINK_START)]
        return text

    def render(self, user: str) -> str:
        """The whole prompt for one user turn, as text."""
        if self.uses_template:
            text = self._templated(user)
            if text is not None:
                return text
        return f"{SYSTEM_PROMPT}\n\n{user}\n\nAnswer:"

    # The tokenization (decision.rs `Wrapped`).

    def encode(self, text: str, parse: bool, bos: bool = False) -> list[int]:
        """`text` tokenized as llama.cpp does: control tokens read as such
        only with `parse`, BOS first only with `bos` and where the tokenizer
        adds one (`AddBos::Always`)."""
        ids = [self.tokenizer.bos_token_id] if bos and self.adds_bos else []
        if text:
            ids += self.tokenizer(
                text, add_special_tokens=False, split_special_tokens=not parse
            )["input_ids"]
        return ids

    def _control_texts(self, text: str) -> list[str]:
        """The text of every control token in `text`: the tokenizer's special
        tokens, which the GGUF marks as control tokens."""
        tok = self.tokenizer
        specials = {t.content for t in getattr(tok, "added_tokens_decoder", {}).values()
                    if getattr(t, "special", False)}
        if getattr(tok, "unk_token", None):
            specials.add(tok.unk_token)
        return sorted(s for s in specials if s and s in text)

    def _cut_tokens(self, user: str, cut) -> list[int]:
        head_controls, head_text, tail_text, tail_controls = cut
        return (self.encode(head_controls, parse=True, bos=True)
                + self.encode(head_text + user + tail_text, parse=False)
                + self.encode(tail_controls, parse=True))

    def _cut_holds(self, cut) -> bool:
        """decision.rs `cut_tokenization_holds`: tokenizing in the cut's
        pieces gives the whole prompt's tokens for every probe that holds no
        control token's text."""
        head, tail = self.wrapper
        checked = 0
        for probe in PROBES:
            if self.encode(probe, parse=True) != self.encode(probe, parse=False):
                continue
            whole = self.encode(head + probe + tail, parse=True, bos=True)
            if self._cut_tokens(probe, cut) != whole:
                return False
            checked += 1
        return checked > 0

    def refused(self, state: str, question: Question) -> bool:
        """decision.rs `refuse_control_text`: without a cut, a request whose
        text holds a control token's text is refused, not prompted."""
        if self.cut is not None:
            return False
        texts = (state, question_text(question))
        return any(self.encode(t, parse=True) != self.encode(t, parse=False) for t in texts)

    def tokens(self, state: str, question: Question) -> list[int] | None:
        """The prompt's tokens for one question about `state`, the answer
        read after the last of them; None for a request the engine refuses."""
        if self.refused(state, question):
            return None
        user = user_message(state, question)
        if self.cut is not None:
            return self._cut_tokens(user, self.cut)
        if self.wrapper is not None:
            head, tail = self.wrapper
            return self.encode(head + user + tail, parse=True, bos=True)
        return self.encode(self.render(user), parse=True, bos=True)

    # The answer codes (decision.rs `CodeTable`).

    def class_tokens(self, question: Question) -> list[list[int]]:
        """Per class, the token ids that count as its code. Raises ValueError
        for a question this model cannot answer, as the engine refuses it."""
        n = question.n_classes()
        table = self.codes[question.kind]
        if n > len(table):
            raise ValueError(f"at most {len(table)} classes for a {question.kind}, got {n}")
        classes = table[:n]
        for i, tokens in enumerate(classes):
            if not tokens:
                raise ValueError(
                    f"the {question.kind} code {code(question.kind, i)!r} is not a single "
                    "token after this model's prompt"
                )
        return classes

    def target(self, question: Question, index: int) -> int:
        """The token training puts the loss on for class `index`: the first
        spelling of its code the engine counts — the code itself wherever it
        is one token, as `Yes` after Qwen3's prompt."""
        return self.class_tokens(question)[index][0]

    def describe(self) -> str:
        """One line for the log, like the engine's load line."""
        if not self.uses_template:
            prompt = "plain-text prompt (no chat template)"
        elif self.cut is not None:
            prompt = "chat template, the request's text kept apart from its control tokens"
        else:
            prompt = "chat template, rendered per question"
        sets = ", ".join(
            f"{name} {sum(1 for t in self.codes[kind] if t)}/{len(self.codes[kind])}"
            for kind, name in (("noul", "yes/no"), ("choice", "letter"), ("score", "digit"))
        )
        return f"{prompt}; codes: {sets}"
