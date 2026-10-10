"""Who labels a decision nobody gave feedback on.

Feedback — what a person, a rule or a teacher recorded as the right answer
after the fact — is the label whenever there is one. For the rest, a
teacher answers the same question about the same state:

* **rules**: a Python function of yours, `rule(state, question_id,
  question, record)`, which returns the right answer — an option's name, a
  level's number, true or false — or None where it has nothing to say.
  Cheap and exact where it applies; the rest falls through.
* **a large model**: any OpenAI-compatible chat endpoint — EuLLM serving a
  large chat model, for instance — asked exactly the prompt the decision
  model will be asked, the engine's system text and user turn, and its
  reply parsed for a code. The state goes to that endpoint: point it at a
  server you would send the state to anyway.

The logged decision itself is a label only when explicitly allowed: a model
trained on its own answers learns to repeat them, mistakes included.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import threading
from pathlib import Path

from .prompt import (
    SYSTEM_PROMPT,
    Question,
    answer_index,
    question_to_api,
    user_message,
)

_THINKING = re.compile(r"<think>.*?</think>", re.S)
#: Markdown and quotes a chat model wraps a one-word answer in.
_WRAPPING = "*_`'\"([ \t\r\n"
#: What may follow a code for the reply still to be only that code:
#: "B", "B)", "B.", "B!", "Yes, it does", a new line. An emphatic code is
#: the same disobedience as a dotted one: without "!", "?" and ";", "Yes!"
#: counted as teacher_unparsed and its question went unlabelled -- silent
#: loss, no error, just a smaller set.
_AFTER_CODE = ").:,*_`'\"]\n!?;"


def parse_reply(text: str, question: Question) -> int | None:
    """The class a teacher's reply names, or None when it names none.

    The teacher was asked to reply with one code and nothing else, and a
    large model does. A reasoning block is skipped; one that never ended
    holds no answer. A letter that starts a sentence ("A good fit is B")
    is not an answer, nor is anything after the first word: guessing at a
    sentence would label the data with the guess.
    """
    text = _THINKING.sub("", text or "")
    if "<think>" in text:
        return None
    head = text.strip().lstrip(_WRAPPING)
    match = re.match(r"[A-Za-z]+|\d+", head)
    if not match:
        return None
    word, rest = match.group(0), head[match.end():]
    alone = not rest.strip() or rest[0] in _AFTER_CODE
    n = question.n_classes()
    if question.kind == "noul":
        if not alone:
            return None
        return {"yes": 0, "true": 0, "no": 1, "false": 1}.get(word.lower())
    if question.kind == "score":
        return int(word) if alone and word.isdigit() and int(word) < n else None
    if alone and len(word) == 1 and "A" <= word <= "Z" and ord(word) - ord("A") < n:
        return ord(word) - ord("A")
    # The option's name instead of its letter, alone on the line.
    reply = head.splitlines()[0].strip().rstrip(".").strip(_WRAPPING + ")").lower()
    names = [name.lower() for name, _ in question.options]
    return names.index(reply) if names.count(reply) == 1 else None


# --- rules -----------------------------------------------------------------------


def load_rules(spec: str):
    """The function `spec` names: `path/to/rules.py:function` or
    `package.module:function`."""
    # The drive letter comes off first: it is a colon, and on Windows
    # rpartition(":") would split there and leave "C" as the module. On POSIX
    # splitdrive is a no-op, so nothing about a POSIX spec changes.
    drive, spec = os.path.splitdrive(spec)
    target, sep, name = spec.rpartition(":")
    if not sep or not target or not name:
        raise ValueError(f"--rules takes FILE.py:FUNCTION or MODULE:FUNCTION, not {spec!r}")
    target = drive + target
    if target.endswith(".py") or "/" in target or "\\" in target:
        path = Path(target)
        if not path.is_file():
            raise FileNotFoundError(f"rules file not found: {path}")
        module_spec = importlib.util.spec_from_file_location(f"eullm_rules_{path.stem}", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    function = getattr(module, name, None)
    if not callable(function):
        raise ValueError(f"{spec}: no function {name!r} there")
    return function


class RulesTeacher:
    """Your function as a teacher. An answer it gives that the question
    cannot have — an option it does not offer — is an error in the rule,
    and raises rather than being dropped in silence."""

    name = "rules"

    def __init__(self, function):
        self.function = function

    def label(self, record: dict, question_id: str, question: Question) -> int | None:
        answer = self.function(record["state"], question_id, question_to_api(question), record)
        if answer is None:
            return None
        try:
            return answer_index(question, answer)
        except ValueError as e:
            raise ValueError(f"the rule answered {question_id!r} with {answer!r}: {e}") from e


# --- a large model -----------------------------------------------------------------


def post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    """POST JSON, return JSON; raises on an HTTP error."""
    import requests  # only a model teacher needs it

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


class ReplyCache:
    """The teacher's replies by prompt, one JSON line each, so a build run
    again — another split, a fixed rule — asks nothing it has asked before.
    A teacher that takes seconds a question is the slow part of a build."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.replies: dict[str, str] = {}
        self.lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self.replies[row["key"]] = row["reply"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    def get(self, key: str) -> str | None:
        return self.replies.get(key)

    def put(self, key: str, model: str, reply: str) -> None:
        with self.lock:
            self.replies[key] = reply
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "model": model, "reply": reply},
                                   ensure_ascii=False) + "\n")


class ChatTeacher:
    """A large model behind an OpenAI-compatible `/v1/chat/completions`,
    asked the decision model's own prompt at temperature 0.

    `max_tokens` leaves room for a model that reasons before it answers:
    the reply is read after its reasoning, and a reply cut off inside it is
    no answer rather than a wrong one.
    """

    def __init__(self, url: str, model: str, api_key: str | None = None,
                 timeout: float = 600.0, max_tokens: int = 1024,
                 cache: ReplyCache | None = None):
        base = url.rstrip("/")
        self.url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        self.model, self.api_key = model, api_key
        self.timeout, self.max_tokens, self.cache = timeout, max_tokens, cache
        self.name = f"teacher:{model}"

    def key(self, user: str) -> str:
        return hashlib.sha256(
            json.dumps([self.model, SYSTEM_PROMPT, user, self.max_tokens]).encode()
        ).hexdigest()

    def reply(self, state: str, question: Question) -> str:
        user = user_message(state, question)
        key = self.key(user)
        if self.cache is not None and (cached := self.cache.get(key)) is not None:
            return cached
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = post(self.url, {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }, headers, self.timeout)
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if self.cache is not None:
            self.cache.put(key, self.model, text)
        return text

    def label(self, state: str, question: Question) -> int | None:
        return parse_reply(self.reply(state, question), question)
