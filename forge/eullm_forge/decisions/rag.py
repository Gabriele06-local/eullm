"""The RAG gate's labelled cases as decision traces, for a RAG-gate model.

ReflexBench's RAG gate (bench/reflexbench/ragbench.py) asks a decision
model, through `/v1/systemone`, whether the passages retrieved for a
question are enough to answer it: one choice among `answer`,
`retrieve_more` and `abstain` (`reflex-gate`), or one yes/no
(`reflex-yesno`). Its sets are labelled — MuSiQue's questions with every
passage they need, all but one, or none (rg_data.py), and the Italian
open-book set rg_openbook.py writes, each question with the article it was
written from or without it. This module writes such cases as the traces a
server writes with `EULLM_DECISION_TRACES`, schema 1, each case's label as
a person's feedback on it, so that `eullm-forge decisions build`, `train`
and `export` make a RAG-gate model of them and need nothing else.

Two things it does, so that the model is trained on what it will be asked
and measured on what it was not trained on:

* **The prompt is the gate's own.** The state and the questions are built
  by the gate's code — `rg_methods.request` and each method's `question`,
  what the Reflex methods send — not by a copy of it here. A model trained
  on a prompt off by a space is trained for one it is never shown, and
  nothing fails; tests/test_decisions_rag.py holds the two together. The
  states are written as they are, not redacted as a server's are: they are
  public text, and the prompt must be the one the gate sends.
* **The split is by question, and by document.** The contexts of one
  question differ only in their passages; the open-book pairs ask up to
  four questions of one article, and MuSiQue builds many questions on one
  single-hop question, and so on its paragraph. Split by state, as
  `decisions build` splits a server's traces, a model would be tested on
  questions — and articles, and paragraphs — it was trained on. Every
  question, with every other question about its document, falls on one
  side, chosen by a hash with a seed (`dataset.split_of`), so that a set
  converted again with more questions keeps each held-out one held out.
  The sides go in `splits.jsonl`, which `decisions build` follows.

The traces directory it writes:

    decisions.jsonl        one decision per case: its state and questions,
                           no answers, since no model decided it
    feedback.jsonl         the right answers, from the case's label
                           (source "user")
    splits.jsonl           each decision's side, and the case, question
                           and document it came from
    rag-test/<set>.jsonl   the test side's cases in ragbench.py's own
                           format, for `ragbench.py --data`
    import.json            what was read, with which settings, and how
                           many cases went where

The gate's code is bench/reflexbench of the EuLLM repository, and is
imported from there, as ragbench.py imports it: Forge is run from a
checkout of the repository.
"""

from __future__ import annotations

import json
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .dataset import split_of
from .prompt import question_from_api, question_to_record
from .traces import SCHEMA, SPLITS, SPLITS_FILE

#: The questions asked of every case: ragbench.py's two Reflex methods.
QUESTIONS = ("reflex-gate", "reflex-yesno")
DEFAULT_SPLIT_SEED = "eullm-rag-gate"
#: Written beside the traces. A directory that holds traces without it was
#: written by something else — a server — and is never overwritten.
IMPORT_FILE = "import.json"
#: The test side's cases, one file per set, as ragbench.py reads a set.
HELD_OUT_DIR = "rag-test"
#: Trace ids are UUIDs, as a server's are, derived from the set and the
#: case: a set converted again gets the same ones.
_NAMESPACE = uuid.UUID("0cf8024e-d85e-4839-b86b-95a01e9afd88")


def bench_dir() -> Path:
    """bench/reflexbench of the repository this Forge is in."""
    path = Path(__file__).resolve().parents[3] / "bench" / "reflexbench"
    if not (path / "rg_methods.py").is_file():
        raise FileNotFoundError(
            f"the RAG gate's code is not at {path}: run Forge from a checkout of the "
            "EuLLM repository"
        )
    return path


def gate_modules():
    """The RAG gate's own modules — `rg_data` (its sets), `rg_methods` (what
    it asks, and the right answers) and `rg_openbook` (which article an
    open-book question was written from) — imported from bench/reflexbench."""
    path = str(bench_dir())
    if path not in sys.path:
        sys.path.insert(0, path)
    import rg_data
    import rg_methods
    import rg_openbook

    return rg_data, rg_methods, rg_openbook


def case_document(case, rg_openbook) -> str | None:
    """The document `case`'s question was written from: the set's own word
    for it (an article, a group of MuSiQue questions resting on one
    paragraph), or, for an open-book set written before cases carried it,
    the article its key names. None for a set that names none."""
    return case.document or rg_openbook.document(case.group)


def import_rag(
    output_dir: str,
    sets: tuple[str, ...] = (),
    data: tuple[str, ...] = (),
    questions: tuple[str, ...] = QUESTIONS,
    limit: int = 0,
    seed: int = 1,
    passages: int = 5,
    dev_share: float = 0.1,
    test_share: float = 0.1,
    split_seed: str = DEFAULT_SPLIT_SEED,
    progress=None,
) -> dict:
    """Write the RAG gate's labelled cases as a traces directory; return what
    was written (also in `import.json`).

    Args:
        output_dir: the traces directory to write; one a server wrote is
            refused.
        sets: ragbench's public sets by name (`musique`), downloaded on
            first use as ragbench.py downloads them.
        data: sets of your own, as `ragbench.py --data` reads them
            (rg_openbook.py writes one).
        questions: which of the gate's questions every case is asked, by
            the name of the ragbench.py method that asks it.
        limit, seed, passages: for the public sets, as ragbench.py's
            `--limit` (questions; 0 for all), `--seed` and `--passages`.
        dev_share, test_share: the share of documents — of questions, for
            a set that names none — held out for each.
        split_seed: changes which are held out.
        progress: `callable(str)` for progress lines, or None.
    """
    if not 0 <= dev_share < 1 or not 0 <= test_share < 1 or dev_share + test_share >= 1:
        raise ValueError("dev and test shares must leave something to train on")
    rg_data, rg_methods, rg_openbook = gate_modules()
    unknown = [q for q in questions if q not in rg_methods.REFLEX]
    if unknown or not questions or len(set(questions)) != len(questions):
        raise ValueError(f"questions are some of {', '.join(rg_methods.REFLEX)}, each once; "
                         f"got {', '.join(questions) or 'none'}")
    if not sets and not data:
        raise ValueError("no set to read: name a public set (musique) or give a file")
    say = progress or (lambda _msg: None)
    out = Path(output_dir)
    _check_output(out)

    sources = [("ragbench", name) for name in sets] + [("file", str(path)) for path in data]
    datasets = []
    for kind, source in sources:
        say(f"reading {source}")
        datasets.append(rg_data.from_jsonl(source) if kind == "file"
                        else rg_data.load(source, limit, seed, passages))
    names = Counter(d.name for d in datasets)
    if any(n > 1 for n in names.values()):
        twice = ", ".join(name for name, n in names.items() if n > 1)
        raise ValueError(f"two sets are named {twice}: a set is named after its file, so "
                         "rename one")

    methods = [rg_methods.REFLEX[name] for name in questions]
    for m in methods:
        # As the engine parses it: a question it would refuse is refused
        # here, before anything is written.
        question_from_api(m.question)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    decisions, feedback, splits = [], [], []
    held_out: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    sides: dict = {side: {"cases": 0, "questions": set(), "documents": set(),
                          "labels": Counter()} for side in SPLITS}
    described = []
    for (kind, source), dataset in zip(sources, datasets):
        for case in dataset.cases:
            trace_id = str(uuid.uuid5(_NAMESPACE, f"{dataset.name}\0{case.id}"))
            if trace_id in seen:
                raise ValueError(f"{source}: case {case.id!r} is there twice")
            seen.add(trace_id)
            document = case_document(case, rg_openbook)
            side = split_of(document or case.group, dev_share, test_share, split_seed)
            # What each method sends for the case, as it sends it.
            bodies = [rg_methods.request(case, m.question) for m in methods]
            state = bodies[0]["state"]
            if any(body["state"] != state for body in bodies):
                raise ValueError(f"{source}: case {case.id!r} has a state per question")
            decisions.append({
                "schema": SCHEMA,
                "id": trace_id,
                "timestamp": stamp,
                # No model decided the case: what a server records of the
                # model that did is left empty.
                "model": None,
                "readout": None,
                "mode": None,
                "state": state,
                "questions": {m.name: question_to_record(question_from_api(body["questions"]["q"]))
                              for m, body in zip(methods, bodies)},
                "answers": {},
                "policy_removed": {},
                "client_disconnected": False,
            })
            feedback.append({
                "schema": SCHEMA,
                "kind": "feedback",
                "timestamp": stamp,
                "id": trace_id,
                "answers": {m.name: m.right(case) for m in methods},
                "outcome": f"labelled {case.label}: {dataset.name} case {case.id}",
                "source": "user",
            })
            splits.append({"id": trace_id, "split": side, "set": dataset.name, "case": case.id,
                           "question": case.group, "document": document,
                           "label": case.label})
            if side == "test":
                held_out[dataset.name].append({
                    "id": case.id, "group": case.group, "document": document,
                    "question": case.question, "passages": case.passages,
                    "label": case.label})
            counts = sides[side]
            counts["cases"] += 1
            counts["questions"].add((dataset.name, case.group))
            if document:
                counts["documents"].add((dataset.name, document))
            counts["labels"][case.label] += 1
        described.append({
            "name": dataset.name, "source": source if kind == "file" else f"ragbench {source}",
            "cases": len(dataset.cases),
            "questions": len({c.group for c in dataset.cases}),
            "documents": len({d for d in (case_document(c, rg_openbook)
                                          for c in dataset.cases) if d}),
        })
    if not decisions:
        raise ValueError("the sets hold no case")

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "decisions.jsonl", decisions)
    _write_jsonl(out / "feedback.jsonl", feedback)
    _write_jsonl(out / SPLITS_FILE, splits)
    held_out_dir = out / HELD_OUT_DIR
    for stale in held_out_dir.glob("*.jsonl"):
        stale.unlink()
    for name, rows in held_out.items():
        held_out_dir.mkdir(exist_ok=True)
        _write_jsonl(held_out_dir / f"{name}.jsonl", rows)

    report = {
        "written": stamp,
        "sets": described,
        "questions": {m.name: m.question for m in methods},
        "settings": {"limit": limit, "seed": seed, "passages": passages,
                     "dev_share": dev_share, "test_share": test_share,
                     "split_seed": split_seed},
        "splits": {side: {"cases": c["cases"], "questions": len(c["questions"]),
                          "documents": len(c["documents"]),
                          "labels": dict(sorted(c["labels"].items()))}
                   for side, c in sides.items()},
        "held_out": {name: f"{HELD_OUT_DIR}/{name}.jsonl" for name in held_out},
    }
    (out / IMPORT_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    return report


def _check_output(out: Path) -> None:
    """Refuse a directory holding traces this module did not write: a
    server's are the only copy of what it decided."""
    if not out.exists():
        return
    if not out.is_dir():
        raise FileExistsError(f"{out} is not a directory")
    traces = [p for stem in ("decisions", "feedback") for p in out.glob(f"{stem}*.jsonl")]
    if traces and not (out / IMPORT_FILE).is_file():
        raise FileExistsError(
            f"{out} holds traces import-rag did not write ({traces[0].name}): give another "
            "--output"
        )


def _write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
