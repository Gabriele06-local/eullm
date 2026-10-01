"""Decision traces → training examples for a decision model of your own.

One example per question of a traced decision: the state, the question as
the engine evaluated it, and the right answer — from feedback when there is
some, otherwise from a teacher (see `teachers`), otherwise, and only when
allowed, the logged decision itself. Nothing is rendered here: training
renders each example through the base model's own tokenizer with
`prompt.CodeReadout`, the same prompt the engine will show the trained
model.

Three things the builder does so that the numbers a run reports mean what
they say:

* **The same question about the same state is one example.** A client
  that asks again, or a decision traced twice, would otherwise count twice
  in training. Feedback on any of the copies wins over every teacher —
  the feedback on the latest decision when they disagree — and only a
  question no feedback covers goes to the rules, then the large model,
  then, if allowed, the log.
* **Dev and test are split by state, and stably.** Every question about a
  state lands on the same side, so a model is never tested on a state it
  was trained on. The side is a hash of the state, not a shuffle: traces
  keep arriving, and a state held out today stays held out when the set is
  built again next month. Traces made from a labelled set carry a split of
  their own (`splits.jsonl`, see `traces`), which is followed instead: the
  states of one RAG-gate question differ only in their passages, and a
  split by state would train on a question and then test on it.
* **Everything left out is counted**, with its reason, in `stats.json`:
  unreadable lines, questions a code-readout model cannot be asked (more
  than 26 options), feedback that names an option the question did not
  offer, questions nobody labelled.

Writes to the output directory:

    train.jsonl, dev.jsonl, test.jsonl   one example per line
    dev.labelled.jsonl, test.labelled.jsonl
                                         the same as labelled requests, the
                                         set `bench/reflexbench/qualify.py`
                                         reads (`--data`)
    stats.json                           what was kept, from where, and why
                                         the rest was not
    teacher-cache.jsonl                  the large model's replies, reused
                                         on the next build
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .prompt import (
    Question,
    answer_value,
    code,
    question_from_api,
    question_text,
    question_to_api,
)
from .traces import SPLITS, Trace, feedback_index, load_traces, logged_index


@dataclass
class Example:
    """One question about one state, and its right answer."""

    trace: str
    question_id: str
    state: str
    question: Question
    label: int
    source: str
    split: str = "train"

    def to_json(self) -> dict:
        return {
            "id": f"{self.trace}:{self.question_id}",
            "trace": self.trace,
            "question_id": self.question_id,
            "state": self.state,
            "question": question_to_api(self.question),
            "answer": answer_value(self.question, self.label),
            "label": self.label,
            "code": code(self.question.kind, self.label),
            "source": self.source,
            "split": self.split,
        }

    @classmethod
    def from_json(cls, row: dict) -> Example:
        question = question_from_api(row["question"])
        label = int(row["label"])
        if not 0 <= label < question.n_classes():
            raise ValueError(f"{row.get('id')}: label {label} out of range")
        return cls(row["trace"], row["question_id"], row["state"], question, label,
                   row.get("source", ""), row.get("split", "train"))


def read_examples(path: str | Path) -> list[Example]:
    """The examples of one split file."""
    with open(path, encoding="utf-8") as f:
        return [Example.from_json(json.loads(line)) for line in f if line.strip()]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_of(state: str, dev_share: float, test_share: float, seed: str) -> str:
    """The side a state — or anything else that must stay on one side, a
    question, a document — falls on: a hash of it, so that it never moves
    while the set grows."""
    bucket = int(_sha(f"{seed}\0{state}")[:12], 16) / 16**12
    if bucket < test_share:
        return "test"
    return "dev" if bucket < test_share + dev_share else "train"


def _record(trace: Trace) -> dict:
    """The decision as a rule sees it."""
    return {"id": trace.id, "state": trace.state, "model": trace.model,
            "timestamp": trace.timestamp, "readout": trace.readout, "answers": trace.logged}


@dataclass
class _Candidate:
    trace: Trace
    question_id: str
    question: Question
    label: int | None = None
    source: str | None = None


def build_dataset(
    trace_dirs: list[str],
    output_dir: str,
    rules=None,
    teacher=None,
    allow_logged: bool = False,
    dev_share: float = 0.1,
    test_share: float = 0.1,
    split_seed: str = "eullm-decisions",
    teacher_workers: int = 1,
    progress=None,
) -> dict:
    """Build the dataset; return its stats (also written to `stats.json`).

    Args:
        trace_dirs: directories the engine wrote decision traces to.
        output_dir: where the splits and stats go.
        rules: a `teachers.RulesTeacher`, or None.
        teacher: a `teachers.ChatTeacher`, or None.
        allow_logged: label what nobody else labelled with the logged
            decision itself.
        dev_share, test_share: the share of states held out for each, for
            the decisions `splits.jsonl` does not place.
        split_seed: changes which states are held out.
        teacher_workers: questions put to the large model at once.
        progress: `callable(str)` for progress lines, or None.
    """
    if not 0 <= dev_share < 1 or not 0 <= test_share < 1 or dev_share + test_share >= 1:
        raise ValueError("dev and test shares must leave something to train on")
    say = progress or (lambda _msg: None)
    stats: dict = {"traces": [], "questions": 0, "skipped_questions": Counter(),
                   "unusable_feedback": Counter(), "duplicates": 0, "conflicts": 0,
                   "unlabelled": 0, "teacher_unparsed": 0, "sources": Counter(),
                   "split_by": Counter()}

    # Every question of every decision, keyed by what the model will read.
    groups: dict[tuple[str, str], list[_Candidate]] = defaultdict(list)
    for directory in trace_dirs:
        traces = load_traces(directory)
        stats["traces"].append(_jsonable(dict(traces.stats, directory=str(directory))))
        stats["skipped_questions"].update(traces.stats["skipped_questions"])
        for trace in traces.traces:
            for qid, question in trace.questions.items():
                stats["questions"] += 1
                problem = question.problem()
                if problem:
                    stats["skipped_questions"][problem] += 1
                    continue
                candidate = _Candidate(trace, qid, question)
                found = feedback_index(question, traces.feedback.get(trace.id), qid)
                if found is not None:
                    label, detail = found
                    if label is None:
                        # The right answer is not among the options the model
                        # was shown: nothing to learn from this question as it
                        # was asked, and no teacher should overrule the person.
                        stats["unusable_feedback"][detail] += 1
                        continue
                    candidate.label, candidate.source = label, f"feedback:{detail}"
                groups[(_sha(trace.state), _sha(question_text(question)))].append(candidate)

    # One example per group: the latest feedback, else a teacher's label.
    chosen: dict[tuple[str, str], _Candidate] = {}
    for key, members in groups.items():
        stats["duplicates"] += len(members) - 1
        members.sort(key=lambda c: str(c.trace.timestamp or ""))
        labelled = [c for c in members if c.label is not None]
        if len({c.label for c in labelled}) > 1:
            stats["conflicts"] += 1
        chosen[key] = labelled[-1] if labelled else members[-1]

    pending = [c for c in chosen.values() if c.label is None]
    if rules is not None and pending:
        say(f"rules: {len(pending)} questions")
        for c in pending:
            label = rules.label(_record(c.trace), c.question_id, c.question)
            if label is not None:
                c.label, c.source = label, rules.name
        pending = [c for c in pending if c.label is None]
    if teacher is not None and pending:
        say(f"{teacher.name}: {len(pending)} questions")

        def ask(c: _Candidate):
            return c, teacher.label(c.trace.state, c.question)

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, teacher_workers)) as pool:
            for c, label in pool.map(ask, pending):
                done += 1
                if label is None:
                    stats["teacher_unparsed"] += 1
                else:
                    c.label, c.source = label, teacher.name
                if done % 100 == 0:
                    say(f"  {done}/{len(pending)}")
        pending = [c for c in pending if c.label is None]
    if allow_logged:
        for c in pending:
            answer = c.trace.logged.get(c.question_id)
            label = logged_index(c.question, answer) if answer else None
            if label is not None:
                c.label, c.source = label, "logged"
        pending = [c for c in pending if c.label is None]
    stats["unlabelled"] = len(pending)

    examples = []
    for c in chosen.values():
        if c.label is None:
            continue
        stats["sources"][c.source] += 1
        stats["split_by"]["splits.jsonl" if c.trace.split else "state"] += 1
        split = c.trace.split or split_of(c.trace.state, dev_share, test_share, split_seed)
        examples.append(Example(c.trace.id, c.question_id, c.trace.state, c.question, c.label,
                                c.source, split))
    # Stable: a decision's questions stay in the order its request asked
    # them, which `batched` evaluation depends on.
    examples.sort(key=lambda e: (e.split, e.trace))
    if not examples:
        raise ValueError(
            "no labelled question: give feedback, a teacher (--rules, --teacher-url) "
            "or --allow-logged"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = [e for e in examples if e.split == split]
        _write_jsonl(out / f"{split}.jsonl", (e.to_json() for e in rows))
        if split != "train":
            _write_jsonl(out / f"{split}.labelled.jsonl", labelled_items(rows))

    stats.update(describe(examples))
    stats["settings"] = {"dev_share": dev_share, "test_share": test_share,
                         "split_seed": split_seed, "allow_logged": allow_logged,
                         "rules": rules is not None,
                         "teacher": teacher.name if teacher is not None else None}
    stats = _jsonable(stats)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    return stats


def labelled_items(examples: list[Example]) -> list[dict]:
    """Examples as labelled requests, every question about a state in one:
    `{"id", "state", "questions": {id: question}, "answers": {id: answer},
    "sources": {id: source}}`. The same question id asked twice about a
    state, with different options, goes in a request of its own."""
    items: list[dict] = []
    open_items: dict[str, list[dict]] = defaultdict(list)
    for e in examples:
        slot = next((i for i in open_items[e.state] if e.question_id not in i["questions"]), None)
        if slot is None:
            slot = {"id": e.trace, "state": e.state, "questions": {}, "answers": {},
                    "sources": {}}
            open_items[e.state].append(slot)
            items.append(slot)
        slot["questions"][e.question_id] = question_to_api(e.question)
        slot["answers"][e.question_id] = answer_value(e.question, e.label)
        slot["sources"][e.question_id] = e.source
    return items


def describe(examples: list[Example]) -> dict:
    """Per split and per question id: how many examples, of which types,
    how the right answers are spread, and the share of the commonest — the
    accuracy of always giving it, which a trained model has to beat."""
    splits: dict = {}
    for split in SPLITS:
        rows = [e for e in examples if e.split == split]
        splits[split] = {
            "examples": len(rows),
            "states": len({e.state for e in rows}),
            "by_type": dict(Counter(e.question.kind for e in rows)),
        }
    by_question: dict = {}
    for e in examples:
        q = by_question.setdefault(e.question_id, {"type": e.question.kind, "examples": 0,
                                                   "answers": Counter(), "by_split": Counter()})
        q["examples"] += 1
        q["answers"][str(answer_value(e.question, e.label)).lower()] += 1
        q["by_split"][e.split] += 1
    for q in by_question.values():
        q["majority_share"] = max(q["answers"].values()) / q["examples"]
    return {"splits": splits, "by_question": by_question}


def _write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _jsonable(value):
    if isinstance(value, Counter):
        return dict(value.most_common())
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value
