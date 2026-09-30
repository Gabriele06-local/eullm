"""RAG sufficiency sets for ReflexBench: a question, the passages retrieved
for it, and what a RAG system should do with them.

  * `answer`: together the passages hold every fact the answer needs;
  * `retrieve_more`: they hold some of those facts, and at least one is
    missing;
  * `abstain`: none of them holds anything the answer needs.

MuSiQue (CC BY 4.0, https://github.com/StonyBrookNLP/musique) gives each
question 20 Wikipedia paragraphs: the 2 to 4 it needs, marked as
supporting, and others retrieved for being close to it. From each question
three contexts of `k` passages are drawn — every supporting paragraph
filled up with others (`answer`), all but one of them (`retrieve_more`),
none of them (`abstain`) — so the three differ in what they hold, not in
how the question reads. MuSiQue was built so that every hop needs its own
paragraph; the unanswerable half of its full release is made the same way,
by removing one. The set is read from a copy of the v1.0 release on
Hugging Face, at a fixed revision, downloaded on first use.
"""

import json
import pathlib
import random

from rb_data import fetch

LABELS = ("answer", "retrieve_more", "abstain")

MUSIQUE = (
    "https://huggingface.co/datasets/bdsaglam/musique/resolve/"
    "22873a405dd809893b22ada0b499299fb612d2df/musique_ans_v1.0_dev.jsonl"
)

SETS = {"musique": "MuSiQue dev: 2 to 4 hop questions over Wikipedia"}


class Case:
    """A question, the passages retrieved for it, and what to do with them.
    Cases of one `group` come from the same question."""

    def __init__(self, id, group, question, passages, label):
        self.id, self.group, self.question = id, group, question
        self.passages, self.label = passages, label

    @property
    def sufficient(self):
        return self.label == "answer"


class Dataset:
    def __init__(self, name, cases):
        self.name, self.cases = name, cases


def musique(limit=0, seed=1, k=5):
    """Three cases for each of `limit` questions (0: all 2,417)."""
    rows = [json.loads(line) for line in fetch(MUSIQUE).decode("utf-8").splitlines() if line]
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    cases = []
    for n in order[:limit] if limit else order:
        row = rows[n]
        support = [p for p in row["paragraphs"] if p["is_supporting"]]
        others = [p for p in row["paragraphs"] if not p["is_supporting"]]
        if not support or len(support) >= k or len(others) < k:
            continue
        rng = random.Random(f"{seed}:{row['id']}")
        rng.shuffle(others)
        dropped = rng.randrange(len(support))
        partial = support[:dropped] + support[dropped + 1 :]
        for label, kept in (("answer", support), ("retrieve_more", partial), ("abstain", [])):
            passages = kept + others[: k - len(kept)]
            rng.shuffle(passages)
            cases.append(
                Case(
                    f"{row['id']}:{label}",
                    row["id"],
                    row["question"],
                    [f"{p['title']}: {p['paragraph_text']}" for p in passages],
                    label,
                )
            )
    return Dataset("musique", cases)


def load(name, limit=0, seed=1, k=5):
    if name == "musique":
        return musique(limit, seed, k)
    raise ValueError(f"unknown set {name!r}: one of {', '.join(SETS)}")


def from_jsonl(path):
    """A set of your own, one JSON object per line:
    {"id": ..., "question": "...", "passages": ["...", ...],
     "label": "answer" | "retrieve_more" | "abstain", "group": optional}.
    A passage may also be {"title": ..., "text": ...}. Cases of one group
    stay on the same side of the split."""
    cases = []
    for n, line in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["label"] not in LABELS:
            raise ValueError(f"{path}: line {n + 1}: label must be one of {', '.join(LABELS)}")
        passages = [
            p if isinstance(p, str) else f"{p['title']}: {p['text']}" for p in row["passages"]
        ]
        case_id = str(row.get("id", n))
        group = str(row.get("group", case_id))
        cases.append(Case(case_id, group, row["question"], passages, row["label"]))
    return Dataset(pathlib.Path(path).stem, cases)


def split(dataset, seed=1):
    """Dev and test halves, by group: thresholds are fitted on dev and every
    method is scored on test, the same cases for all."""
    groups = sorted({c.group for c in dataset.cases})
    random.Random(f"split:{seed}").shuffle(groups)
    dev = set(groups[: len(groups) // 2])
    return (
        [c for c in dataset.cases if c.group in dev],
        [c for c in dataset.cases if c.group not in dev],
    )
