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
        # The two questions rest on the same paragraphs here: one document.
        self.assertEqual({c.document for c in dataset.cases}, {"musique/q0"})

    def test_questions_resting_on_one_paragraph_are_one_document(self):
        def para(title, supporting=True):
            return {"title": title, "paragraph_text": f"about {title}", "is_supporting": supporting}

        rows = [
            {"id": "2hop__1_2", "paragraphs": [para("A"), para("B")]},
            {"id": "2hop__2_3", "paragraphs": [para("B"), para("C")]},
            # A paragraph only retrieved, not needed, ties nothing.
            {"id": "2hop__4_5", "paragraphs": [para("D"), para("E"), para("A", False)]},
            # Tied to the first through the second.
            {"id": "2hop__3_6", "paragraphs": [para("C"), para("F")]},
        ]
        self.assertEqual(
            rg_data.shared_paragraphs(rows),
            {
                "2hop__1_2": "musique/2hop__1_2",
                "2hop__2_3": "musique/2hop__1_2",
                "2hop__4_5": "musique/2hop__4_5",
                "2hop__3_6": "musique/2hop__1_2",
            },
        )

    def test_own_set(self):
        rows = [
            {
                "id": "x",
                "question": "q",
                "passages": ["a", {"title": "B", "text": "b"}],
                "label": "answer",
                "group": "g",
            },
            {
                "id": "y",
                "question": "q",
                "passages": ["a"],
                "label": "abstain",
                "group": "g",
                "document": "doc-1",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "mine.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            dataset = rg_data.from_jsonl(path)
            self.assertEqual(dataset.cases[0].passages, ["a", "B: b"])
            self.assertTrue(dataset.cases[0].sufficient)
            self.assertEqual([c.document for c in dataset.cases], [None, "doc-1"])
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

    def test_what_is_sent_is_the_request_and_each_method_knows_its_right_answer(self):
        """Forge's RAG-gate traces are built from `request` and the methods'
        questions: what a method sends must be exactly that."""
        for name, method in rg_methods.REFLEX.items():
            sent = []

            def post(url, payload, api_key, timeout):
                sent.append(payload)
                answer = {"noul": 0.9, "choice": "answer", "probabilities": {"answer": 0.9}}
                return {"answers": {"q": answer}}

            with mock.patch.object(rg_methods, "post", post):
                method("http://x", "m", None, 10).decide(self.c)
            self.assertEqual(sent, [rg_methods.request(self.c, method.question, "m")], name)
        self.assertEqual(
            [rg_methods.ReflexGate.right(case("g", label)) for label in rg_data.LABELS],
            list(rg_data.LABELS),
        )
        self.assertEqual(
            [rg_methods.ReflexYesNo.right(case("g", label)) for label in rg_data.LABELS],
            [True, False, False],
        )

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


class OpenBookTest(unittest.TestCase):
    records = [
        {
            "code": "codice_civile",
            "article_num": "2043",
            "text": "Art. 2043. (Risarcimento per fatto illecito). Qualunque fatto doloso o "
            "colposo che cagiona ad altri un danno ingiusto obbliga a risarcire il danno.",
        },
        {
            "code": "codice_civile",
            "article_num": "2044",
            "text": "Art. 2044. (Legittima difesa). Non risponde del danno chi lo cagiona "
            "per legittima difesa.",
        },
        {
            "code": "codice_civile",
            "article_num": "2045",
            "text": "Art. 2045. (Stato di necessità). Al danneggiato è dovuta un'indennità "
            "quando il danno è cagionato per necessità.",
        },
        {
            "code": "codice_penale",
            "article_num": "52",
            "text": "Art. 52. (Difesa legittima). Non è punibile chi difende un diritto "
            "dal pericolo di un'offesa ingiusta, con un danno proporzionato.",
        },
    ]
    pair = {
        "instruction": "Testi normativi di riferimento:\n\n[1] ...\n\n"
        "Domanda: Chi deve risarcire un danno ingiusto causato con dolo o colpa?",
        "output": "Chi lo ha cagionato (art. 2043).",
        "task": "openbook_grounded",
        "named": False,
        "key": "ob-g-codice_civile-2043-v1",
    }

    def test_the_article_with_and_without(self):
        import rg_openbook

        index = rg_openbook.NormIndex(self.records)
        answer, abstain = rg_openbook.cases(self.pair, index, k=3)
        self.assertEqual(answer["question"], self.pair["instruction"].split("Domanda: ")[1])
        self.assertEqual((answer["label"], abstain["label"]), ("answer", "abstain"))
        self.assertEqual(answer["group"], abstain["group"])
        self.assertEqual(len(answer["passages"]), 3)
        own = "codice civile, art. 2043\n"
        self.assertTrue(any(p.startswith(own) for p in answer["passages"]))
        self.assertTrue(abstain["passages"])
        self.assertFalse(any(p.startswith(own) for p in abstain["passages"]))
        # Both name the article, which the pair's other questions share.
        self.assertEqual({answer["document"], abstain["document"]}, {"codice_civile/2043"})

        # What it writes is a set the gate reads.
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "legal.jsonl"
            path.write_text("\n".join(json.dumps(c) for c in (answer, abstain)), "utf-8")
            loaded = rg_data.from_jsonl(path)
        self.assertEqual([c.sufficient for c in loaded.cases], [True, False])
        self.assertEqual([c.document for c in loaded.cases], ["codice_civile/2043"] * 2)

    def test_a_set_written_before_cases_named_their_article_gets_it_from_the_key(self):
        import rg_openbook

        self.assertEqual(rg_openbook.document("ob-g-codice_civile-2043"), "codice_civile/2043")
        # Another question about the same article, and an article with a suffix.
        self.assertEqual(rg_openbook.document("ob-g-codice_civile-2043-v3"), "codice_civile/2043")
        self.assertEqual(
            rg_openbook.document("ob-g-codice_civile-2043-bis-v1"), "codice_civile/2043-bis"
        )
        self.assertEqual(rg_openbook.document("h-codice_penale-52"), "codice_penale/52")
        for key in ("2hop__1_2", "ob-g-codice_civile", "ob-m-codice_civile-9000-3", ""):
            self.assertIsNone(rg_openbook.document(key), key)

    def test_questions_by_rubrica_without_the_pairs(self):
        import rg_openbook

        records = self.records + [
            {"code": "codice_civile", "article_num": "1", "text": "Art. 1. (Definizioni). ..."},
            {
                "code": "codice_civile",
                "article_num": "2047",
                "text": "Art. 2047. (Danno cagionato dall'incapace). Il risarcimento è dovuto "
                "da chi è tenuto alla sorveglianza, salvo che provi di non aver potuto "
                "impedire il fatto.",
            },
        ]
        index = rg_openbook.NormIndex(records)
        drawn = rg_openbook.heading_cases(index, k=3)
        questions = sorted({c["question"] for c in drawn})
        self.assertIn(
            "Che cosa prevede la legge in materia di risarcimento per fatto illecito?", questions
        )
        # A rubrica that names no topic makes no question.
        self.assertFalse(any("definizioni" in q for q in questions))
        self.assertEqual(len(drawn), 2 * len(questions))
        # "Stato di necessità": retrieval finds nothing else, so there is no
        # context without the article, and no question.
        self.assertFalse(any("necessità" in q for q in questions))
        one = [c for c in drawn if c["group"] == "h-codice_civile-2043"]
        self.assertEqual([c["label"] for c in one], ["answer", "abstain"])
        self.assertEqual({c["document"] for c in one}, {"codice_civile/2043"})
        self.assertEqual(len(rg_openbook.heading_cases(index, k=3, limit=1)), 2)

    def test_only_questions_asked_by_topic(self):
        import rg_openbook

        index = rg_openbook.NormIndex(self.records)
        self.assertEqual(rg_openbook.cases(dict(self.pair, named=True), index), [])
        self.assertEqual(rg_openbook.cases(dict(self.pair, task="openbook_absent"), index), [])
        unknown = dict(self.pair, key="ob-g-codice_civile-9999")
        self.assertEqual(rg_openbook.cases(unknown, index), [])


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
