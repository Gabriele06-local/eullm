"""What a decision model's answers on labelled questions say about it.

The same numbers `bench/reflexbench/qualify.py` gates on, computed here on
the dev split before and after training, from the same readout the engine
makes: each class's full-vocabulary log-probability (the log of the summed
probabilities of its code's spellings), the probabilities renormalized over
the classes, the most probable class as the answer.

* accuracy — the share of answers that are right;
* ECE — expected calibration error of the top answer, 15 equal-width bins,
  as `bench/decision_calibration.py` computes it: how far "90% sure" is
  from being right nine times in ten;
* NLL — mean −log p(right answer), which punishes confident mistakes;
* coverage — the share of the model's probability on a valid code at all.
"""

from __future__ import annotations

import math

ECE_BINS = 15


def class_result(logprobs: list[float], label: int, kind: str, temperature: float = 1.0) -> dict:
    """One answer from its classes' log-probabilities: decision.rs
    `calibrated_probabilities` with no prior — renormalized over the
    classes after dividing by `temperature`."""
    scaled = [lp / temperature for lp in logprobs]
    top_lp = max(scaled)
    exps = [math.exp(lp - top_lp) for lp in scaled]
    total = sum(exps)
    probabilities = [e / total for e in exps]
    top = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "kind": kind,
        "label": label,
        "logprobs": list(logprobs),
        "answer": top,
        "correct": top == label,
        "confidence": probabilities[top],
        "p_label": probabilities[label],
        "probabilities": probabilities,
        "coverage": min(1.0, sum(math.exp(lp) for lp in logprobs)),
    }


def at_temperature(results: list[dict], temperature: float) -> list[dict]:
    """The same answers read at another temperature."""
    return [class_result(r["logprobs"], r["label"], r["kind"], temperature) for r in results]


def fit_temperature(results: list[dict]) -> float:
    """The temperature with the lowest NLL on `results`, by golden-section
    search on log T in [0.05, 20] — bench/decision_calibration.py's fit. A
    fine-tuned model is usually too sure of itself (T > 1); the engine
    applies a temperature per request (`eullm.temperature`)."""

    def nll(log_t: float) -> float:
        t = math.exp(log_t)
        return sum(-math.log(max(r["p_label"], 1e-12)) for r in at_temperature(results, t))

    lo, hi = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = nll(a), nll(b)
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = nll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = nll(b)
    return math.exp((lo + hi) / 2)


def ece(results: list[dict], bins: int = ECE_BINS) -> float | None:
    if not results:
        return None
    counts = [[0, 0.0, 0.0] for _ in range(bins)]
    for r in results:
        b = min(int(r["confidence"] * bins), bins - 1)
        counts[b][0] += 1
        counts[b][1] += r["confidence"]
        counts[b][2] += 1.0 if r["correct"] else 0.0
    return sum(abs(hits - conf) for _, conf, hits in counts) / len(results)


def summarize(results: list[dict]) -> dict:
    """Per question type and over all of them."""
    out = {}
    groups = {"all": results}
    for r in results:
        groups.setdefault(r["kind"], []).append(r)
    for name, rows in groups.items():
        if not rows:
            continue
        out[name] = {
            "n": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "ece": ece(rows),
            "nll": sum(-math.log(max(r["p_label"], 1e-12)) for r in rows) / len(rows),
            "coverage": sum(r["coverage"] for r in rows) / len(rows),
        }
    return out
