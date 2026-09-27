"""Evaluation harness — rank one model, or two models blind (A/B).

Model access is injected as ``generate_fn: (prompt) -> text`` so the harness is
decoupled from any specific runtime (EULLM Engine, HF, an OpenAI-compatible
endpoint). Nothing here imports ``torch``.
"""

from __future__ import annotations

from typing import Callable

from .dataset import EvalItem
from .judge import ABOutcome, Judge, blind_pairwise
from .metrics import QAResult, aggregate, score_item

GenerateFn = Callable[[str], str]


def collect_answers(items: list[EvalItem], generate_fn: GenerateFn) -> dict[str, str]:
    """Run ``generate_fn`` over every item's question, keyed by item id."""
    return {item.id: generate_fn(item.question) for item in items}


def evaluate_qa(items: list[EvalItem], answers: dict[str, str]) -> dict:
    """Score answers against items with the QA metrics.

    Items with no answer are scored on the empty string (a miss), so the
    denominator is always the full item count. Returns a dict with per-item
    results and an aggregate summary.

    An item with no keywords is *not measured* by keyword coverage and is
    reported as ``None`` rather than 1.0. ``keyword_coverage`` answers 1.0 for
    an empty keyword list as a per-item sentinel — nothing to miss — but in
    an aggregate that sentinel is a free point, and the held-out exam's
    ``contenuto`` family is exactly that: an item whose reference is the whole
    article, which no model can reproduce verbatim and which only the judge
    can assess. Left in, they put a floor under the number the gate reports
    and no model, however wrong, can go below it. Such items are excluded
    from the mean and counted in ``keyword_items``.
    """
    results: list[QAResult] = [score_item(answers.get(it.id, ""), it) for it in items]
    measured = [r.keyword_coverage for r, it in zip(results, items) if it.keywords]
    summary = aggregate(results)
    summary["keyword_coverage"] = (sum(measured) / len(measured)) if measured else float("nan")
    summary["keyword_items"] = len(measured)
    return {
        "summary": summary,
        "per_item": [
            {"id": r.id, "exact": r.exact,
             "keyword_coverage": (r.keyword_coverage if it.keywords else None)}
            for r, it in zip(results, items)
        ],
    }


def compare_models(
    items: list[EvalItem],
    answers_a: dict[str, str],
    answers_b: dict[str, str],
    judge: Judge,
    *,
    seed: int = 0,
) -> ABOutcome:
    """Blind A/B comparison of two models' answers using ``judge``."""
    return blind_pairwise(items, answers_a, answers_b, judge, seed=seed)
