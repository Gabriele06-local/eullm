"""Decision traces, as the engine writes them with `EULLM_DECISION_TRACES`.

The engine keeps a state only as a SHA-256 in its audit trail, on purpose.
A server started with `EULLM_DECISION_TRACES=<dir>` also writes, locally and
with personal data redacted, what a decision model can be trained on:

* `<dir>/decisions.jsonl` — one line per computed decision: `schema` (1),
  `id` (the audit id), `timestamp`, `model`, `readout` (`codes` or
  `verdict`), `mode`, `state`, `questions` (as evaluated), `answers` (as
  `/v1/systemone` returned them), `policy_removed`, `client_disconnected`;
* `<dir>/feedback.jsonl` — what the right answers were: `schema` (1),
  `kind` (`feedback`), `timestamp`, `id` (a decision's audit id), `answers`
  ({question: a choice's option name | true/false for a noul | a level's
  number for a score}), `outcome` (optional text), `source` (`user`,
  `rule` or `teacher`).

Both are read tolerantly: unknown fields are ignored, a line that is not a
JSON object is skipped and counted, a question that cannot be read is
skipped with its reason, and files rotated next to the two
(`decisions-*.jsonl`) are read too. The two files are written by different
people at different times — the decisions by the server, the feedback by
whoever learns what the right answer was — so a feedback line may name a
decision that is not there; it is counted, not fatal.

Only the standard library is needed.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .prompt import Question, answer_index, question_from_record, state_text

SCHEMA = 1


def read_jsonl(path: str | Path) -> tuple[list[dict], int]:
    """The JSON objects in a JSONL file, and how many lines were not one.

    A line cut short by a crash, or written by something else, costs that
    line and nothing more.
    """
    rows: list[dict] = []
    bad = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                bad += 1
    return rows, bad


@dataclass
class Trace:
    """One computed decision: the state, its questions and what was logged."""

    id: str
    state: str
    #: Question id → question, in the order the request asked them.
    questions: dict[str, Question]
    #: Question id → the answer as `/v1/systemone` returned it.
    logged: dict[str, dict]
    readout: str | None = None
    model: str | None = None
    timestamp: str | None = None
    #: What a server-side policy took out before the model saw the
    #: questions, as recorded; the questions above are what was left.
    policy_removed: object = None
    client_disconnected: bool = False


@dataclass
class Feedback:
    """The right answers known for one decision, merged over every
    feedback line that names it: a later line corrects an earlier one."""

    #: Question id → the right answer, as feedback writes it.
    answers: dict
    #: Question id → `user`, `rule` or `teacher`.
    sources: dict
    outcome: str | None = None


@dataclass
class TraceSet:
    traces: list[Trace]
    feedback: dict[str, Feedback]
    stats: dict


def serving_state(record: dict) -> str:
    """The state a code-readout model will read for this decision.

    A structured state is shown to a code-readout model as indented JSON,
    to a verdict model as one line of it. A trace from a verdict model that
    holds a structured state is therefore re-indented: the model trained on
    it will be served through the code readout.
    """
    state = record.get("state")
    if not isinstance(state, str):
        return state_text(state)
    if record.get("readout") == "verdict" and state[:1] in "{[":
        try:
            value = json.loads(state)
        except json.JSONDecodeError:
            return state
        if isinstance(value, (dict, list)):
            return state_text(value)
    return state


def _questions(raw) -> tuple[dict[str, Question], Counter]:
    """Question id → question; `raw` is an object keyed by id, or a list of
    questions that carry their own `id`."""
    skipped: Counter = Counter()
    items = []
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        for n, spec in enumerate(raw):
            qid = spec.get("id", spec.get("name", str(n))) if isinstance(spec, dict) else str(n)
            items.append((str(qid), spec))
    out: dict[str, Question] = {}
    for qid, spec in items:
        try:
            out[str(qid)] = question_from_record(spec)
        except ValueError as e:
            skipped[f"unreadable question: {e}"] += 1
    return out, skipped


def _logged(raw) -> dict[str, dict]:
    """Question id → answer; `raw` is the response's `answers` object, or a
    list of audit-style records with `id`, `labels` and `probabilities`."""
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    if isinstance(raw, list):
        return {str(a["id"]): a for a in raw if isinstance(a, dict) and "id" in a}
    return {}


def logged_index(question: Question, answer: dict) -> int | None:
    """The class the logged decision chose: the most probable one, or for a
    noul P(yes) of one half or more."""
    labels = question.labels()
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, list) and isinstance(answer.get("labels"), list):
        probabilities = dict(zip(answer["labels"], probabilities))
    if question.kind == "noul":
        p = answer.get("noul")
        if p is None and isinstance(probabilities, dict):
            p = probabilities.get("yes")
        return None if not isinstance(p, (int, float)) else (0 if p >= 0.5 else 1)
    if question.kind == "choice" and answer.get("choice") in labels:
        return labels.index(answer["choice"])
    if isinstance(probabilities, dict):
        scored = [(probabilities.get(label), i) for i, label in enumerate(labels)]
        scored = [(p, i) for p, i in scored if isinstance(p, (int, float))]
        if len(scored) == len(labels):
            return max(scored)[1]
    return None


def _files(directory: Path, stem: str) -> list[Path]:
    main = directory / f"{stem}.jsonl"
    rotated = sorted(p for p in directory.glob(f"{stem}*.jsonl") if p != main)
    return rotated + ([main] if main.exists() else [])


def load_traces(directory: str | Path) -> TraceSet:
    """Every decision and the feedback on it in a traces directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a traces directory: {directory}")
    decision_files = _files(directory, "decisions")
    if not decision_files:
        raise FileNotFoundError(f"no decisions.jsonl in {directory}")

    stats: dict = {
        "decision_files": [str(p) for p in decision_files],
        "decisions": 0,
        "malformed_lines": 0,
        "newer_schema": 0,
        "unusable_decisions": Counter(),
        "skipped_questions": Counter(),
    }
    rows: list[dict] = []
    feedback_rows: list[dict] = []
    for path in decision_files:
        found, bad = read_jsonl(path)
        stats["malformed_lines"] += bad
        for row in found:
            (feedback_rows if row.get("kind") == "feedback" else rows).append(row)

    traces: list[Trace] = []
    for row in rows:
        if isinstance(row.get("schema"), int) and row["schema"] > SCHEMA:
            stats["newer_schema"] += 1
        if row.get("id") is None or row.get("state") is None:
            stats["unusable_decisions"]["no id or state"] += 1
            continue
        questions, skipped = _questions(row.get("questions"))
        stats["skipped_questions"].update(skipped)
        if not questions:
            stats["unusable_decisions"]["no readable question"] += 1
            continue
        traces.append(Trace(
            id=str(row["id"]),
            state=serving_state(row),
            questions=questions,
            logged=_logged(row.get("answers")),
            readout=row.get("readout"),
            model=row.get("model"),
            timestamp=row.get("timestamp"),
            policy_removed=row.get("policy_removed"),
            client_disconnected=bool(row.get("client_disconnected")),
        ))
    stats["decisions"] = len(traces)

    feedback_files = _files(directory, "feedback")
    stats["feedback_files"] = [str(p) for p in feedback_files]
    for path in feedback_files:
        found, bad = read_jsonl(path)
        stats["malformed_lines"] += bad
        feedback_rows += found
    feedback, fstats = merge_feedback(feedback_rows, {t.id for t in traces})
    stats["feedback"] = fstats
    return TraceSet(traces, feedback, stats)


def merge_feedback(rows: list[dict], known_ids: set[str]) -> tuple[dict[str, Feedback], dict]:
    """Feedback per decision id. Lines are taken in timestamp order (file
    order among equals), so a later correction overrides an earlier answer
    to the same question."""
    stats = {"lines": 0, "answers": 0, "corrections": 0, "orphans": 0, "unusable_lines": 0}
    indexed = []
    for n, row in enumerate(rows):
        answers = row.get("answers")
        if row.get("id") is None or not isinstance(answers, dict) or not answers:
            stats["unusable_lines"] += 1
            continue
        indexed.append((str(row.get("timestamp") or ""), n, row))
    indexed.sort(key=lambda x: (x[0], x[1]))
    merged: dict[str, Feedback] = {}
    for _, _, row in indexed:
        stats["lines"] += 1
        fid = str(row["id"])
        entry = merged.setdefault(fid, Feedback({}, {}))
        for qid, answer in row["answers"].items():
            if str(qid) in entry.answers and entry.answers[str(qid)] != answer:
                stats["corrections"] += 1
            entry.answers[str(qid)] = answer
            entry.sources[str(qid)] = str(row.get("source") or "user")
        if row.get("outcome") is not None:
            entry.outcome = str(row["outcome"])
    stats["answers"] = sum(len(f.answers) for f in merged.values())
    stats["orphans"] = sum(1 for fid in merged if fid not in known_ids)
    stats["decisions_with_feedback"] = sum(1 for fid in merged if fid in known_ids)
    return merged, stats


def feedback_index(question: Question, feedback: Feedback | None, qid: str):
    """`(class, source)` from the feedback on question `qid`, `None` when
    there is none, or `(None, reason)` when the answer does not fit it —
    an option the question did not offer, a level out of range."""
    if feedback is None or qid not in feedback.answers:
        return None
    try:
        return answer_index(question, feedback.answers[qid]), feedback.sources[qid]
    except ValueError as e:
        return None, str(e)
