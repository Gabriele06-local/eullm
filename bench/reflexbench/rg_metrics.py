"""What the RAG gate report says about a method.

The decision that matters most is two-way: let the large model answer from
these passages, or not. Of the cases where the passages do not suffice, the
share a method stops is how often it saves an answer made up from missing
facts (`caught`); of the cases where they do, the share it stops anyway is
answers lost for nothing (`blocked`). Both are given

  * at the method's own decision, where it makes one without labelled
    data (Reflex: its choice, or a yes above one half);
  * at a threshold on its score fitted on the dev half, the best balanced
    accuracy there, scored on the test half — what calibrating on a
    domain's own cases would buy, and the only way to use the embeddings.

AUROC says how well the score separates the two before any threshold, and
within a question, where the contexts differ only in what they hold, whether
it follows the passages; ECE, how far Reflex's probability of `answer` is
from how often it is right.
"""

import math
import statistics

LABELS = ("answer", "retrieve_more", "abstain")


def auroc(scores, positives):
    """The chance a sufficient case scores above an insufficient one, ties
    counting half (the Mann-Whitney statistic)."""
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return None
    ranked = sorted(scores)
    rank, i = {}, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j] == ranked[i]:
            j += 1
        rank[ranked[i]] = (i + j + 1) / 2  # 1-based, averaged over ties
        i = j
    total = sum(rank[s] for s in pos)
    return (total - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def within_question(cases, scores):
    """AUROC within each question, averaged: the three contexts of a
    question differ only in what they hold, so this says whether the score
    follows the passages, whatever the question's difficulty does to its
    level."""
    groups = {}
    for case, score in zip(cases, scores):
        groups.setdefault(case.group, []).append((score, case.sufficient))
    values = [
        auroc([s for s, _ in pairs], [y for _, y in pairs])
        for pairs in groups.values()
        if any(y for _, y in pairs) and not all(y for _, y in pairs)
    ]
    return statistics.mean(values) if values else None


def ece(probabilities, positives, bins=10):
    """Expected calibration error of P(sufficient), in equal-width bins."""
    if not probabilities:
        return None
    error = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [
            (p, y)
            for p, y in zip(probabilities, positives)
            if lo <= p < hi or (b == bins - 1 and p == 1.0)
        ]
        if inside:
            confidence = statistics.mean(p for p, _ in inside)
            accuracy = statistics.mean(1.0 if y else 0.0 for _, y in inside)
            error += len(inside) / len(probabilities) * abs(confidence - accuracy)
    return error


def two_way(predicted, positives):
    """`predicted`: True where the method lets the model answer."""
    pos = [p for p, y in zip(predicted, positives) if y]
    neg = [p for p, y in zip(predicted, positives) if not y]
    if not pos or not neg:
        return None
    caught = sum(not p for p in neg) / len(neg)
    blocked = sum(not p for p in pos) / len(pos)
    return {
        "accuracy": sum(p == y for p, y in zip(predicted, positives)) / len(positives),
        "balanced_accuracy": (caught + 1 - blocked) / 2,
        "caught": caught,
        "blocked": blocked,
    }


def fit_threshold(scores, positives):
    """The threshold with the best balanced accuracy: answer at or above it."""
    candidates = sorted(set(scores))
    if not candidates or all(positives) or not any(positives):
        return None
    cuts = [candidates[0]] + [(a + b) / 2 for a, b in zip(candidates, candidates[1:])]
    cuts.append(candidates[-1] + 1)
    best = max(
        cuts,
        key=lambda t: two_way([s >= t for s in scores], positives)["balanced_accuracy"],
    )
    return best


def three_way(predicted, labels):
    """Accuracy, macro-F1 and the confusion counts over the three labels."""
    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    for p, t in zip(predicted, labels):
        confusion[t][p] += 1
    f1 = []
    for label in LABELS:
        tp = confusion[label][label]
        predicted_n = sum(confusion[t][label] for t in LABELS)
        actual_n = sum(confusion[label].values())
        if predicted_n + actual_n:
            f1.append(2 * tp / (predicted_n + actual_n))
    return {
        "accuracy": sum(p == t for p, t in zip(predicted, labels)) / len(labels),
        "macro_f1": statistics.mean(f1) if f1 else None,
        "confusion": confusion,
    }


def fit_two_thresholds(scores, labels):
    """For a score that only says how likely the passages suffice: answer at
    or above the high threshold, abstain below the low one, retrieve more in
    between — the pair with the best macro-F1."""
    candidates = sorted(set(scores))
    cuts = [candidates[0]] + [(a + b) / 2 for a, b in zip(candidates, candidates[1:])]
    cuts.append(candidates[-1] + 1)
    step = max(1, len(cuts) // 60)  # a coarse grid: the pairs grow as its square
    grid = cuts[::step] + [cuts[-1]]
    best, best_f1 = None, -1.0
    for i, low in enumerate(grid):
        for high in grid[i:]:
            predicted = [label_for(s, low, high) for s in scores]
            f1 = three_way(predicted, labels)["macro_f1"]
            if f1 > best_f1:
                best, best_f1 = (low, high), f1
    return best


def label_for(score, low, high):
    if score >= high:
        return "answer"
    return "abstain" if score < low else "retrieve_more"


def summarize(dev, test):
    """`dev` and `test`: lists of (case, decision), one method on one set."""
    report = {"dev_cases": len(dev), "test_cases": len(test)}
    positives = [c.sufficient for c, _ in test]
    scores = [d.score for _, d in test]
    report["sufficient_share"] = sum(positives) / len(positives) if positives else None
    report["auroc"] = auroc(scores, positives)
    report["auroc_within_question"] = within_question([c for c, _ in test], scores)
    own = [d.choice for _, d in test]
    if all(choice is not None for choice in own):
        report["own"] = two_way([choice == "answer" for choice in own], positives)
        if all(choice in LABELS for choice in own):
            report["own_three_way"] = three_way(own, [c.label for c, _ in test])
    threshold = fit_threshold([d.score for _, d in dev], [c.sufficient for c, _ in dev])
    if threshold is not None:
        report["fitted_threshold"] = threshold
        report["fitted"] = two_way([s >= threshold for s in scores], positives)
    if all(d.choice is None for _, d in test) and dev:
        pair = fit_two_thresholds([d.score for _, d in dev], [c.label for c, _ in dev])
        report["fitted_thresholds"] = pair
        predicted = [label_for(s, *pair) for s in scores]
        report["fitted_three_way"] = three_way(predicted, [c.label for c, _ in test])
    if all(d.probabilities for _, d in test):
        report["ece"] = ece(scores, positives)
    ms = sorted(d.ms for _, d in dev + test)
    report["latency_ms"] = {
        "p50": ms[len(ms) // 2] if ms else None,
        "p95": ms[min(len(ms) - 1, math.ceil(0.95 * len(ms)) - 1)] if ms else None,
    }
    evaluated = [
        d.server["evaluated_tokens"] for _, d in dev + test if d.server.get("evaluated_tokens")
    ]
    if evaluated:
        report["evaluated_tokens_mean"] = statistics.mean(evaluated)
    return report
