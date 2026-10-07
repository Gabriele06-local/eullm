"""Render eval results (JSON-serializable dict + Markdown) and export a
lawyer/expert spot-check sheet.
"""

from __future__ import annotations

from .dataset import EvalItem
from .harness import evaluate_qa


def build_report(
    items: list[EvalItem],
    answers: dict[str, str],
    *,
    model_name: str = "",
    perplexity: float | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble a JSON-serializable evaluation report."""
    qa = evaluate_qa(items, answers)
    report = {
        "model": model_name,
        "n_items": len(items),
        "domains": sorted({it.domain for it in items}),
        "languages": sorted({it.lang for it in items}),
        "qa": qa["summary"],
        "perplexity": perplexity,
        "per_item": qa["per_item"],
    }
    if extra:
        report.update(extra)
    return report


def to_markdown(report: dict) -> str:
    """Render a report dict as a compact Markdown summary."""
    qa = report.get("qa", {})
    # Coverage is measured over the items that have keywords, not over all of
    # them -- say so, in the same words legal_eval.py prints ("over K of N
    # items"). Without the denominator "Items: 2" next to "Coverage: 100%"
    # reads as the whole set scoring full marks, when one of the two was not
    # measured at all.
    n_items = report.get("n_items", 0)
    keyword_items = qa.get("keyword_items", n_items)
    coverage = f"- Keyword coverage: **{_pct(qa.get('keyword_coverage'))}**"
    if keyword_items != n_items:
        coverage += f" (over {keyword_items} of {n_items} items)"
    lines = [
        f"# Eval report — {report.get('model') or '(model)'}",
        "",
        f"- Items: **{n_items}**",
        f"- Domains: {', '.join(report.get('domains', [])) or '—'}",
        f"- Languages: {', '.join(report.get('languages', [])) or '—'}",
        "",
        "## QA metrics",
        f"- Exact match: **{_pct(qa.get('exact_match'))}**",
        coverage,
    ]
    if report.get("perplexity") is not None:
        lines.append(f"- Perplexity (held-out): **{report['perplexity']:.2f}**")
    ab = report.get("ab")
    if ab:
        lines += [
            "",
            "## A/B (blind)",
            f"- A wins: **{ab['a_wins']}** · B wins: **{ab['b_wins']}** · ties: {ab['ties']}",
            f"- Win-rate A (decided): **{_pct(ab.get('win_rate_a'))}**",
        ]
    return "\n".join(lines) + "\n"


def spotcheck_markdown(items: list[EvalItem], answers: dict[str, str]) -> str:
    """A Markdown table for a human expert to score answers (F0-B spot-check)."""
    lines = [
        "# Human spot-check sheet",
        "",
        "Score each answer 0–2 (0 wrong · 1 partial · 2 correct) and add notes.",
        "",
        "| id | category | question | model answer | score (0-2) | notes |",
        "|----|----------|----------|--------------|:-----------:|-------|",
    ]
    for it in items:
        ans = answers.get(it.id, "").replace("\n", " ").replace("|", "\\|")
        q = it.question.replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {it.id} | {it.category} | {q} | {ans} |  |  |")
    return "\n".join(lines) + "\n"


def _pct(value: float | None) -> str:
    if value is None or value != value:  # None or NaN
        return "—"
    return f"{value * 100:.1f}%"
