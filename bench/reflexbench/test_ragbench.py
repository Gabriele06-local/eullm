"""Offline tests for the RAG gate of ReflexBench: the metrics, the loaders
on made-up files, and the methods against a stand-in server.

    python3 -m unittest discover -s bench/reflexbench
"""

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ragbench  # noqa: E402
import rg_data  # noqa: E402
import rg_methods  # noqa: E402
import rg_metrics  # noqa: E402


def case(group, label, question="q", passages=("p",)):
    return rg_data.Case(f"{group}:{label}", group, question, list(passages), label)


class MetricsTest(unittest.TestCase):
    def test_auroc(self):
        self.assertEqual(rg_metrics.auroc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]), 1.0)
        self.assertEqual(rg_metrics.auroc([0.1, 0.9], [True, False]), 0.0)
        # A tie counts half.
        self.assertEqual(rg_metrics.auroc([0.5, 0.5], [True, False]), 0.5)
        self.assertIsNone(rg_metrics.auroc([0.5], [True]))

    def test_within_question(self):
        cases = [
            case("a", "answer"),
            case("a", "abstain"),
            case("b", "answer"),
            case("b", "abstain"),
        ]
        # Question b sits lower, yet within each question the order is right.
        self.assertEqual(rg_metrics.within_question(cases, [0.9, 0.8, 0.3, 0.1]), 1.0)
        self.assertEqual(rg_metrics.auroc([0.9, 0.8, 0.3, 0.1], [True, False, True, False]), 0.75)

    def test_ece(self):
        self.assertEqual(rg_metrics.ece([1.0, 0.0], [True, False]), 0.0)
        self.assertAlmostEqual(rg_metrics.ece([0.9, 0.9], [False, False]), 0.9)

    def test_two_way(self):
        m = rg_metrics.two_way([True, False, False, True], [True, True, False, False])
        self.assertEqual((m["caught"], m["blocked"]), (0.5, 0.5))
        self.assertEqual(m["accuracy"], 0.5)
        self.assertIsNone(rg_metrics.two_way([True], [True]))

    def test_fit_threshold(self):
        t = rg_metrics.fit_threshold([0.1, 0.2, 0.7, 0.9], [False, False, True, True])
        self.assertTrue(0.2 < t <= 0.7)
        self.assertIsNone(rg_metrics.fit_threshold([0.1, 0.2], [True, True]))

    def test_three_way(self):
        m = rg_metrics.three_way(
            ["answer", "abstain", "abstain"], ["answer", "retrieve_more", "abstain"]
        )
        self.assertAlmostEqual(m["accuracy"], 2 / 3)
        self.assertEqual(m["confusion"]["retrieve_more"]["abstain"], 1)
        # F1: answer 1, retrieve_more 0, abstain 2/3.
        self.assertAlmostEqual(m["macro_f1"], (1 + 0 + 2 / 3) / 3)

    def test_fit_two_thresholds(self):
        scores = [0.9, 0.8, 0.5, 0.45, 0.1, 0.05]
        labels = ["answer", "answer", "retrieve_more", "retrieve_more", "abstain", "abstain"]
        low, high = rg_metrics.fit_two_thresholds(scores, labels)
        predicted = [rg_metrics.label_for(s, low, high) for s in scores]
        self.assertEqual(predicted, labels)

    def test_summarize(self):
        dev = [case("a", "answer"), case("a", "retrieve_more"), case("a", "abstain")]
        test = [case("b", "answer"), case("b", "retrieve_more"), case("b", "abstain")]

        def gate(score, choice):
            return rg_methods.Decision(
                score, choice, {"answer": score}, 10.0, {"evaluated_tokens": 7}
            )

        own = ["answer", "abstain", "abstain"]
        report = rg_metrics.summarize(
            [(c, gate(s, o)) for c, s, o in zip(dev, [0.8, 0.3, 0.1], own)],
            [(c, gate(s, o)) for c, s, o in zip(test, [0.7, 0.4, 0.2], own)],
        )
        self.assertEqual((report["dev_cases"], report["test_cases"]), (3, 3))
        self.assertEqual(report["auroc"], 1.0)
        self.assertEqual(report["own"]["caught"], 1.0)
        self.assertEqual(report["own"]["blocked"], 0.0)
        self.assertAlmostEqual(report["own_three_way"]["accuracy"], 2 / 3)
        self.assertEqual(report["fitted"]["balanced_accuracy"], 1.0)
        self.assertNotIn("fitted_three_way", report)
        self.assertIn("ece", report)
        self.assertEqual(report["evaluated_tokens_mean"], 7)

    def test_a_score_without_a_decision_gets_two_thresholds(self):
        dev = [case("a", label) for label in rg_metrics.LABELS]
        test = [case("b", label) for label in rg_metrics.LABELS]
        scores = [0.9, 0.5, 0.1]
        report = rg_metrics.summarize(
            [(c, rg_methods.Decision(s)) for c, s in zip(dev, scores)],
            [(c, rg_methods.Decision(s)) for c, s in zip(test, scores)],
        )
        self.assertNotIn("own", report)
        self.assertNotIn("ece", report)
        self.assertEqual(report["fitted_three_way"]["accuracy"], 1.0)


def musique_row(n, hops=2):
    paragraphs = [
        {"idx": i, "title": f"T{i}", "paragraph_text": f"text {i}", "is_supporting": i < hops}
        for i in range(20)
    ]
    return {"id": f"q{n}", "question": f"question {n}?", "paragraphs": paragraphs}


class DataTest(unittest.TestCase):
    def test_musique_draws_three_contexts_a_question(self):
        rows = [musique_row(0, hops=2), musique_row(1, hops=4)]
        data = "\n".join(json.dumps(r) for r in rows).encode()
        with mock.patch.object(rg_data, "fetch", lambda url: data):
            dataset = rg_data.load("musique", limit=0, seed=1, k=5)
        self.assertEqual(len(dataset.cases), 6)
        hops = {"q0": 2, "q1": 4}
        for c in dataset.cases:
            self.assertEqual(len(c.passages), 5)
            supporting = {f"T{i}" for i in range(hops[c.group])}
            held = sum(p.split(":")[0] in supporting for p in c.passages)
            wanted = {"answer": hops[c.group], "retrieve_more": hops[c.group] - 1, "abstain": 0}
            self.assertEqual(held, wanted[c.label], c.id)
        self.assertEqual({c.group for c in dataset.cases}, {"q0", "q1"})
        self.assertTrue(dataset.cases[0].passages[0].startswith("T"))

    def test_own_set(self):
        rows = [
            {
                "id": "x",
                "question": "q",
                "passages": ["a", {"title": "B", "text": "b"}],
                "label": "answer",
                "group": "g",
            },
            {"id": "y", "question": "q", "passages": ["a"], "label": "abstain", "group": "g"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "mine.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            dataset = rg_data.from_jsonl(path)
            self.assertEqual(dataset.cases[0].passages, ["a", "B: b"])
            self.assertTrue(dataset.cases[0].sufficient)
            path.write_text(json.dumps(dict(rows[1], label="maybe")), encoding="utf-8")
            with self.assertRaises(ValueError):
                rg_data.from_jsonl(path)

    def test_split_keeps_a_question_on_one_side(self):
        cases = [case(g, label) for g in "abcdef" for label in rg_data.LABELS]
        dev, test = rg_data.split(rg_data.Dataset("t", cases), seed=1)
        self.assertEqual((len(dev), len(test)), (9, 9))
        self.assertFalse({c.group for c in dev} & {c.group for c in test})


class MethodsTest(unittest.TestCase):
    c = case("g", "answer", "Who wrote it?", ["Book: written by Ann", "Ann: a writer"])

    def test_the_gate_asks_one_choice_about_the_passages(self):
        sent = []

        def post(url, payload, api_key, timeout):
            sent.append((url, payload))
            probabilities = {"answer": 0.7, "retrieve_more": 0.2, "abstain": 0.1}
            answer = {"type": "choice", "choice": "answer", "probabilities": probabilities}
            return {"model": "m", "answers": {"q": answer}, "eullm": {"evaluated_tokens": 42}}

        with mock.patch.object(rg_methods, "post", post):
            decision = rg_methods.ReflexGate("http://x/", None, None, 10).decide(self.c)
        url, payload = sent[0]
        self.assertEqual(url, "http://x/v1/systemone")
        self.assertEqual(
            payload["state"],
            "Question: Who wrote it?\n\nPassages:\n[1] Book: written by Ann\n\n[2] Ann: a writer",
        )
        question = payload["questions"]["q"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(list(question["criteria"]), list(rg_data.LABELS))
        self.assertEqual((decision.score, decision.choice), (0.7, "answer"))
        self.assertEqual(decision.server["model"], "m")

    def test_the_yes_no_decides_two_ways(self):
        def post(url, payload, api_key, timeout):
            self.assertEqual(payload["questions"]["q"]["type"], "noul")
            return {"answers": {"q": {"type": "noul", "noul": 0.3}}}

        with mock.patch.object(rg_methods, "post", post):
            decision = rg_methods.ReflexYesNo("http://x", "m", None, 10).decide(self.c)
        self.assertEqual((decision.score, decision.choice), (0.3, "not answer"))

    def test_embed_max_scores_the_closest_passage(self):
        vectors = {"Book: written by Ann": [1.0, 0.0], "Ann: a writer": [0.0, 1.0]}
        asked = []

        def post(url, payload, api_key, timeout):
            asked.extend(payload["input"])
            rows = [vectors.get(t, [0.6, 0.8]) for t in payload["input"]]
            return {"data": [{"index": i, "embedding": v} for i, v in enumerate(rows)]}

        with mock.patch("rb_methods.post", post):
            method = rg_methods.EmbedMax("http://x", "e", None, 10, "Q: ")
            self.assertAlmostEqual(method.decide(self.c).score, 0.8)
            method.decide(self.c)
        # The passages once, the question each time.
        self.assertEqual(sorted(asked), sorted(list(vectors) + ["Q: Who wrote it?"] * 2))


class TableTest(unittest.TestCase):
    def test_every_row_has_every_column(self):
        dev = [case("a", label) for label in rg_metrics.LABELS]
        test = [case("b", label) for label in rg_metrics.LABELS]
        metrics = rg_metrics.summarize(
            [(c, rg_methods.Decision(s)) for c, s in zip(dev, [0.9, 0.5, 0.1])],
            [(c, rg_methods.Decision(s)) for c, s in zip(test, [0.9, 0.5, 0.1])],
        )
        rows = ragbench.table(
            [{"set": "s", "method": "embed-max", "metrics": metrics}]
        ).splitlines()
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row.count("|") for row in rows}), 1)


if __name__ == "__main__":
    unittest.main()
