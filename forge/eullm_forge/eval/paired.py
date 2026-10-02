"""Is model A better than model B on the same questions, or is it noise?

Every candidate so far has been compared by its total on the 207-question
exam, and the totals sit between 160 and 170. A total throws away what the
exam knows: which questions each model got right. Two models that both score
169 can disagree on thirty questions, or on two; only the second pair is the
same model. The paired test asks the question that decides: of the questions
where exactly one of the two is right, how many go each way? With b of them
for A and c for B, under "no difference" each goes either way with
probability 1/2, so the two-sided exact binomial (McNemar) p-value is the
chance of a split at least as uneven as b:c.

On 207 questions with the 10-20% disagreement our models show, a real
difference needs to be some 11-16 questions to show up; on 900 it is
roughly half as many in proportion. That is why development decisions are
taken on the large development set, and the held-out exam is kept for the
end.

A second thing a total hides is how long the answers are. An LLM judge can
favour long answers or short ones (Dubois et al. 2024; Soumik 2026), and our
SFT models answer in 440 characters where their base answers in 1,160. So
every comparison also reports, among the questions where the two disagree,
how often the answer judged right is the longer one: far from one half, the
judge may be grading length as much as law.

Nothing here prints a question: ids name the article asked.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("termine_argomento", "termine", "contenuto", "inesistente")
_KIND = re.compile(r"^norm-(" + "|".join(KINDS) + r")-")

# What a person writes in the review sheet, read as the judge's labels.
HUMAN_LABELS = {"corretto": "correct", "corretta": "correct", "giusto": "correct",
                "parziale": "partial", "sbagliato": "wrong", "sbagliata": "wrong",
                "errato": "wrong", "errata": "wrong",
                "correct": "correct", "partial": "partial", "wrong": "wrong"}


def kind_of(item_id: str) -> str:
    """The question kind an exam id names (norm-<kind>-<code>-<article>)."""
    m = _KIND.match(item_id)
    return m.group(1) if m else "?"


@dataclass
class Graded:
    """One model's graded answers, by item id."""

    label: str
    grades: dict[str, str] = field(default_factory=dict)
    lengths: dict[str, int] = field(default_factory=dict)

    def right(self, item_id: str, lenient: bool = False) -> bool:
        g = self.grades.get(item_id)
        return g == "correct" or (lenient and g == "partial")

    def counts(self) -> Counter:
        return Counter(self.grades.values())

    def mean_length(self) -> float:
        return sum(self.lengths.values()) / len(self.lengths) if self.lengths else 0.0


def label_of(path: Path) -> str:
    """answers-v0.3-open.graded.jsonl -> v0.3-open"""
    name = path.name
    for suffix in (".graded.jsonl", ".jsonl"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name[len("answers-"):] if name.startswith("answers-") else name


def load_graded(path: Path) -> Graded:
    out = Graded(label_of(path))
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out.grades[r["id"]] = r.get("grade", "unparsed")
            out.lengths[r["id"]] = len(r.get("answer") or "")
    return out


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b discordant pairs one way, c the other."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@dataclass
class Comparison:
    """Model ``a`` against ``base`` on the items both answered."""

    a: str
    base: str
    n: int
    a_only: int          # a right, base not
    base_only: int       # base right, a not
    p: float
    a_only_lenient: int
    base_only_lenient: int
    p_lenient: float
    by_kind: dict[str, tuple[int, int]]
    longer_wins: float | None   # among discordant items, share where the right answer is longer

    @property
    def diff(self) -> int:
        return self.a_only - self.base_only


def compare(a: Graded, base: Graded) -> Comparison:
    ids = sorted(set(a.grades) & set(base.grades))
    ids = [i for i in ids if "unparsed" not in (a.grades[i], base.grades[i])]

    def split(lenient: bool) -> tuple[int, int]:
        x = sum(a.right(i, lenient) and not base.right(i, lenient) for i in ids)
        y = sum(base.right(i, lenient) and not a.right(i, lenient) for i in ids)
        return x, y

    b, c = split(False)
    bl, cl = split(True)
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    longer, discordant = 0, 0
    for i in ids:
        ar, br = a.right(i), base.right(i)
        if ar == br:
            continue
        by_kind[kind_of(i)][0 if ar else 1] += 1
        la, lb = a.lengths.get(i, 0), base.lengths.get(i, 0)
        if la != lb:
            discordant += 1
            longer += (la > lb) == ar
    return Comparison(a.label, base.label, len(ids), b, c, mcnemar_p(b, c), bl, cl,
                      mcnemar_p(bl, cl), {k: (v[0], v[1]) for k, v in sorted(by_kind.items())},
                      longer / discordant if discordant else None)


def human_agreement(rows: list[dict], models: dict[str, Graded]) -> dict:
    """How the judge's grades compare with a person's on the same answers.

    ``rows`` are the review sheet's lines (export_grade_review.py): ``chiave``
    is "<label>|<item id>", ``giudizio`` what the person wrote. Rows left
    blank or naming a model not given are not counted.
    """
    pairs = []
    for r in rows:
        human = HUMAN_LABELS.get(str(r.get("giudizio", "")).strip().lower())
        label, _, item = str(r.get("chiave", "")).partition("|")
        judge = models[label].grades.get(item) if label in models else None
        if human and judge and judge != "unparsed":
            pairs.append((judge, human))
    confusion = Counter(pairs)
    n = len(pairs)
    same = sum(v for (j, h), v in confusion.items() if j == h)
    binary = sum(v for (j, h), v in confusion.items() if (j == "correct") == (h == "correct"))
    return {
        "n": n,
        "same_label": same / n if n else None,
        "same_right_or_not": binary / n if n else None,
        # The two errors that move a comparison: the judge passing what a
        # person fails, and failing what a person passes.
        "judge_too_kind": confusion[("correct", "wrong")] + confusion[("correct", "partial")],
        "judge_too_harsh": confusion[("wrong", "correct")] + confusion[("partial", "correct")],
        "confusion": {f"{j}->{h}": v for (j, h), v in sorted(confusion.items())},
    }
