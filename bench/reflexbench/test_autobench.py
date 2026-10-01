"""Offline tests for AutoBench: the graders, the metrics, the loaders on
made-up files, and the routers against a stand-in server.

    python3 -m unittest discover -s bench/reflexbench
"""

import hashlib
import io
import json
import os
import pathlib
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_data  # noqa: E402
import ab_grade  # noqa: E402
import ab_methods  # noqa: E402
import ab_metrics  # noqa: E402
import autobench  # noqa: E402


def item(id, grader="letter", answer="B", text="q"):
    return ab_data.Item(id, "t", [{"role": "user", "content": text}], answer, grader)


class GradeTest(unittest.TestCase):
    def test_the_number_after_the_last_answer(self):
        self.assertEqual(ab_grade.extract_number("2 + 2 = 4\nAnswer: $1,234.50"), 1234.5)
        self.assertEqual(ab_grade.extract_number("Answer: 3. No wait.\nAnswer: -7"), -7)
        # No "Answer:": the last number written.
        self.assertEqual(ab_grade.extract_number("so she makes 9 * 2 = 18 dollars"), 18)
        self.assertIsNone(ab_grade.extract_number("I do not know"))
        # A reasoning block does not count.
        self.assertEqual(ab_grade.extract_number("<think>Answer: 5</think>Answer: 6"), 6)

    def test_the_letter_after_the_last_answer(self):
        self.assertEqual(ab_grade.extract_letter("The sun.\nAnswer: (B)"), "B")
        self.assertEqual(ab_grade.extract_letter("Answer: C. Photosynthesis"), "C")
        self.assertEqual(ab_grade.extract_letter("Answer: the answer is D"), "D")
        # No "Answer:": a letter alone on the last line.
        self.assertEqual(ab_grade.extract_letter("Thinking it over.\nA"), "A")
        self.assertIsNone(ab_grade.extract_letter("Probably the second one"))

    def test_correct_by_grader(self):
        self.assertTrue(ab_grade.correct(item("1", "number", "18"), "Answer: 18"))
        self.assertFalse(ab_grade.correct(item("1", "number", "18"), "Answer: 19"))
        self.assertTrue(ab_grade.correct(item("2", "letter", "b"), "Answer: B"))
        self.assertTrue(ab_grade.correct(item("3", "exact", "The Rome"), "rome!"))
        self.assertTrue(ab_grade.correct(item("4", "f1", "the city of Rome"), "Rome city"))
        with self.assertRaises(ValueError):
            ab_grade.correct(item("5", "judge", None), "x")

    def test_f1(self):
        self.assertEqual(ab_grade.f1("a b c", "a b c"), 1.0)
        self.assertEqual(ab_grade.f1("x", "y"), 0.0)
        # Precision 1, recall 2/4: "of" and "italy" are not in the answer.
        self.assertAlmostEqual(ab_grade.f1("rome city", "city of rome italy"), 2 / 3)

    def test_a_judge_is_believed_only_when_both_orders_agree(self):
        self.assertEqual(ab_grade.judge_verdict("Answer: A"), "A")
        self.assertEqual(ab_grade.judge_verdict("It is a TIE"), "tie")
        self.assertIsNone(ab_grade.judge_verdict("Both are fine"))
        self.assertEqual(ab_grade.both_orders("A", "B"), "small")
        self.assertEqual(ab_grade.both_orders("B", "A"), "large")
        # Position bias: A twice is no verdict.
        self.assertEqual(ab_grade.both_orders("A", "A"), "tie")
        self.assertEqual(ab_grade.both_orders(None, "B"), "tie")


class MetricsTest(unittest.TestCase):
    def test_the_bootstrap_is_seeded_and_brackets_the_difference(self):
        a = [1.0] * 60 + [0.0] * 40
        b = [1.0] * 50 + [0.0] * 50
        ci = ab_metrics.paired_bootstrap(a, b, seed=1)
        self.assertEqual(ci, ab_metrics.paired_bootstrap(a, b, seed=1))
        self.assertTrue(ci[0] <= 0.1 <= ci[1], ci)
        self.assertEqual(ab_metrics.paired_bootstrap(b, b), (0.0, 0.0))
        self.assertIsNone(ab_metrics.paired_bootstrap([], []))

    def test_the_fitted_threshold_avoids_the_most_calls_within_a_point(self):
        # Five items the small model gets right, five it gets wrong where
        # the large one is right; the score ranks the first five higher.
        scores = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.1]
        small = [True] * 5 + [False] * 5
        large = [True] * 10
        self.assertEqual(ab_metrics.fit_threshold(scores, small, large), 0.55)
        # With ten points to give, the next item goes small too.
        self.assertEqual(ab_metrics.fit_threshold(scores, small, large, max_loss=10), 0.5)
        # Nothing to lose: every item small.
        self.assertEqual(ab_metrics.fit_threshold(scores, [True] * 10, large), 0.1)
        self.assertIsNone(ab_metrics.fit_threshold([], [], []))

    def test_cohen_kappa(self):
        self.assertEqual(ab_metrics.cohen_kappa(["a", "b"], ["a", "b"]), 1.0)
        self.assertAlmostEqual(
            ab_metrics.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]), 0
        )
        self.assertIsNone(ab_metrics.cohen_kappa([], []))

    def test_summarize(self):
        items = [item(str(n)) for n in range(4)]
        report = ab_metrics.summarize(
            items,
            routed=[True, True, False, False],
            small_correct=[True, False, False, True],
            large_correct=[True, True, True, False],
            large_ms=[1000.0, 2000.0, 3000.0, 4000.0],
            scores=[0.9, 0.2, 0.3, 0.8],
            latency=[10.0, 20.0, 30.0, 40.0],
        )
        self.assertEqual(report["calls_avoided"], 0.5)
        self.assertEqual(report["large_gpu_s_avoided"], 3.0)
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["accuracy_large"], 0.75)
        self.assertEqual((report["lost"], report["gained"]), (1, 0))
        # small_ok: right, or both wrong — items 0 and 3; scored 0.9 and 0.8.
        self.assertEqual(report["auroc"], 1.0)
        self.assertEqual(report["latency_ms"], {"p50": 20.0, "p95": 40.0})

    def test_the_kill_criterion(self):
        def r(router, avoided, ci):
            return {
                "set": "s",
                "router": router,
                "metrics": {"calls_avoided": avoided, "delta_ci95": ci},
            }

        rows = [
            r("reflex (fitted)", 0.4, (-0.02, 0.01)),
            r("length (fitted)", 0.3, (-0.02, 0.01)),
            r("knn (fitted)", 0.5, (-0.05, -0.01)),
        ]
        self.assertEqual(ab_metrics.kill_criterion(rows), {"s": []})
        rows.append(r("knn (own)", 0.45, (-0.03, 0.0)))
        self.assertEqual(ab_metrics.kill_criterion(rows), {"s": ["knn (own)"]})


class DataTest(unittest.TestCase):
    def test_gsm8k(self):
        rows = [
            {"question": "Ann has 3 apples and buys 1,000 more.", "answer": "3 + 1000\n#### 1,003"}
        ]
        items = ab_data.gsm8k_items("\n".join(json.dumps(r) for r in rows).encode())
        self.assertEqual(items[0].answer, "1003")
        self.assertEqual(items[0].grader, "number")
        self.assertIn(ab_data.NUMBER_INSTRUCTION, items[0].messages[0]["content"])

    def test_arc_relabels_numbered_options(self):
        choices = [{"label": str(n), "text": f"option {n}"} for n in range(1, 5)]
        row = {"id": "x1", "question": {"stem": "Which?", "choices": choices}, "answerKey": "3"}
        items = ab_data.arc_items(json.dumps(row).encode(), "arc-easy")
        self.assertEqual(items[0].answer, "C")
        self.assertIn("C. option 3", items[0].messages[0]["content"])
        self.assertEqual(items[0].id, "arc-easy-x1")

    def test_mmlu(self):
        items = ab_data.mmlu_items([("global_facts", ["Q?", "a", "b", "c", "d", "D"])])
        content = items[0].messages[0]["content"]
        self.assertTrue(content.startswith("(global facts) Q?\n\nA. a"), content)
        self.assertEqual(items[0].answer, "D")

    def test_mmlu_is_read_out_of_its_archive_and_the_archive_dropped(self):
        rows = [["What is 1+1?", "1", "2", "3", "4", "B"], ["Bad row"]]
        text = "\n".join(",".join(r) for r in rows).encode()
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            info = tarfile.TarInfo("data/test/elementary_mathematics_test.csv")
            info.size = len(text)
            archive.addfile(info, io.BytesIO(text))
        blob = buffer.getvalue()

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.dict(os.environ, {"REFLEXBENCH_CACHE": tmp}),
                mock.patch.object(ab_data, "MMLU_SHA256", hashlib.sha256(blob).hexdigest()),
                mock.patch("urllib.request.urlopen", lambda url, timeout: Response(blob)),
            ):
                dataset = ab_data.load("mmlu")
            self.assertEqual([i.answer for i in dataset.items], ["B"], "the bad row skipped")
            self.assertIn("(elementary mathematics) What is 1+1?", dataset.items[0].text)
            self.assertEqual(
                sorted(p.name for p in pathlib.Path(tmp).iterdir()), ["mmlu-test.jsonl"]
            )

    def test_a_file_that_is_not_the_pinned_one_is_refused(self):
        with self.assertRaises(ValueError):
            ab_data.checked(b"changed", "0" * 64, "test.jsonl")

    def test_a_zip_member_is_read_with_range_requests(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("big.bin", os.urandom(300_000))
            archive.writestr("wanted.jsonl", b'{"id": 1}\n')
        blob = buffer.getvalue()
        ranges = []

        class Response:
            def __init__(self, request):
                self.headers = {"Content-Length": str(len(blob))}
                self.status = 206
                spec = request.headers.get("Range")
                if spec:
                    start, end = map(int, spec.split("=")[1].split("-"))
                    ranges.append((start, end))
                    self.data = blob[start : end + 1]

            def read(self):
                return self.data

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with mock.patch.object(ab_data.RemoteFile, "BLOCK", 4096):
            with mock.patch("urllib.request.urlopen", lambda request, timeout: Response(request)):
                with zipfile.ZipFile(ab_data.RemoteFile("http://x/a.zip")) as archive:
                    self.assertEqual(archive.read("wanted.jsonl"), b'{"id": 1}\n')
        fetched = sum(end - start + 1 for start, end in ranges)
        self.assertLess(fetched, len(blob) // 4, "only the member and the directory")

    def test_own_set(self):
        rows = [
            {"id": "a", "prompt": "2+2?", "answer": "4", "grader": "number"},
            {
                "id": "b",
                "messages": [{"role": "user", "content": "Tell a joke"}],
                "grader": "judge",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "mine.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            dataset = ab_data.from_jsonl(path)
            by_id = {i.id: i for i in dataset.items}
            self.assertEqual(by_id["a"].text, "2+2?")
            self.assertEqual(by_id["b"].grader, "judge")
            for bad in (
                {"id": "c", "prompt": "x", "grader": "number"},
                {"id": "d", "prompt": "x", "messages": [], "answer": "1"},
                {"id": "e", "prompt": "x", "answer": "1", "grader": "vibes"},
            ):
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ValueError, msg=bad):
                    ab_data.from_jsonl(path)

    def test_split_is_fixed_by_the_seed(self):
        dataset = ab_data.Dataset("t", [item(str(n)) for n in range(10)])
        dev, test = ab_data.split(dataset, seed=1)
        self.assertEqual((len(dev), len(test)), (5, 5))
        self.assertEqual([i.id for i in dev], [i.id for i in ab_data.split(dataset, seed=1)[0]])
        self.assertFalse({i.id for i in dev} & {i.id for i in test})


ROUTE = {
    "model": "small-m",
    "reason": "decided",
    "fallback": "large-m",
    "candidates": [
        {"model": "small-m", "description": "short", "probability": 0.7, "resident": True},
        {"model": "large-m", "description": "long", "probability": 0.3, "resident": True},
    ],
    "excluded": [],
    "confidence": 0.4,
    "decision_model": "jev",
    "decision_ms": 12.5,
    "state": "Request to answer, with its context.",
    "question": {
        "type": "choice",
        "instructions": "Which model?",
        "criteria": {"small-m": "short", "large-m": "long"},
    },
    "route_id": "r1",
}


class MethodsTest(unittest.TestCase):
    def test_reflex_asks_api_route_with_the_request_a_client_sends(self):
        sent = []

        def post(url, payload, api_key, timeout):
            sent.append((url, payload))
            return ROUTE

        with mock.patch.object(ab_methods, "post", post):
            router = ab_methods.Reflex("http://x/", "small-m", "large-m", None, 10)
            decision = router.decide(item("1", text="Hi"))
        url, payload = sent[0]
        self.assertEqual(url, "http://x/api/route")
        self.assertEqual(payload["model"], "auto")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Hi"}])
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertIs(payload["cache_prompt"], False)
        self.assertEqual((decision.score, decision.small), (0.7, True))
        self.assertEqual(decision.server["decision_ms"], 12.5)

    def test_reflex_refuses_a_server_routing_other_models(self):
        with mock.patch.object(ab_methods, "post", lambda *a: ROUTE):
            router = ab_methods.Reflex("http://x", "qwen3-4b", "qwen3-8b", None, 10)
            with self.assertRaises(SystemExit):
                router.decide(item("1"))

    def test_a_replay_reverses_the_options_about_the_same_state(self):
        sent = []

        def post(url, payload, api_key, timeout):
            sent.append((url, payload))
            answer = {"choice": "large-m", "probabilities": {"small-m": 0.2, "large-m": 0.8}}
            return {"answers": {"route": answer}, "eullm": {"request_ms": 9.0}}

        reflex_decision = ab_methods.Decision(0.7, True, 1.0, ROUTE)
        spec = {"name": "reversed", "reverse": True, "instructions": "Pick one."}
        with mock.patch.object(ab_methods, "post", post):
            replay = ab_methods.Replay(spec, "http://x", "small-m", None, 10)
            decision = replay.decide(reflex_decision)
        url, payload = sent[0]
        self.assertEqual(url, "http://x/v1/systemone")
        self.assertEqual(payload["state"], ROUTE["state"])
        question = payload["questions"]["route"]
        self.assertEqual(list(question["criteria"]), ["large-m", "small-m"])
        self.assertEqual(question["instructions"], "Pick one.")
        self.assertEqual((decision.score, decision.small), (0.2, False))
        self.assertEqual(replay.name, "reflex:reversed")

    def test_knn_scores_by_the_neighbours_the_small_model_handled(self):
        vectors = {"near": [1.0, 0.0], "far": [0.0, 1.0], "query": [0.9, 0.1]}

        def post(url, payload, api_key, timeout):
            rows = [vectors[t] for t in payload["input"]]
            return {"data": [{"index": i, "embedding": v} for i, v in enumerate(rows)]}

        with mock.patch.object(ab_methods, "post", post):
            knn = ab_methods.Knn("http://x", "e", None, 10, k=1)
            knn.fit([item("1", text="near"), item("2", text="far")], [True, False])
            decision = knn.decide(item("3", text="query"))
        self.assertEqual((decision.score, decision.small), (1.0, True))

    def test_the_knn_embedder_is_unloaded_before_stage_3_unless_reserved(self):
        def ps(reserved):
            return {"models": [{"name": "e", "eullm": {"slot": "embedding", "reserved_companion": reserved}}]}

        sent = []
        for reserved, expected in ((False, "unloaded"), (True, "reserved, kept")):
            state = {"version": {}, "ps": ps(reserved)}
            with mock.patch.object(ab_methods, "server_state", lambda *a, s=state: s), \
                    mock.patch.object(ab_methods, "post", lambda url, body, *a: sent.append((url, body))):
                self.assertEqual(ab_methods.release_embedder("http://x/", "e", None, 10), expected)
        self.assertEqual(sent, [("http://x/api/embed", {"model": "e", "input": "-", "keep_alive": 0})])
        with mock.patch.object(ab_methods, "server_state", lambda *a: {"ps": {"models": []}}):
            self.assertEqual(ab_methods.release_embedder("http://x", "e", None, 10), "not loaded")

    def test_a_streamed_answer_is_timed_to_its_first_piece(self):
        lines = [
            {"model": "m", "message": {"content": "Hel"}, "done": False},
            {"model": "m", "message": {"content": "lo"}, "done": False},
            {
                "model": "m",
                "message": {"content": ""},
                "done": True,
                "eval_count": 2,
                "load_duration": 5_000_000,
                "eullm": {"route": {"reason": "decided"}},
            },
        ]

        class Response:
            headers = {"X-EuLLM-Model": "m"}

            def __iter__(self):
                return iter([(json.dumps(line) + "\n").encode() for line in lines])

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with mock.patch("urllib.request.urlopen", lambda request, timeout: Response()):
            answer = ab_methods.generate("http://x", item("1"), "auto", None, 10)
        self.assertEqual(answer.text, "Hello")
        self.assertEqual((answer.model, answer.eval_count, answer.load_ms), ("m", 2, 5.0))
        self.assertEqual(answer.route, {"reason": "decided"})
        self.assertEqual(answer.headers["x-eullm-model"], "m")
        self.assertLessEqual(answer.ttft_ms, answer.total_ms)


class ReportTest(unittest.TestCase):
    def test_every_router_is_scored_and_the_table_has_every_column(self):
        items = [item(f"i{n}", text="x" * (n + 1)) for n in range(8)]
        dataset = ab_data.Dataset("t", items)
        answers = {}
        for n, it in enumerate(items):
            small = ab_methods.Answer("Answer: B", "small-m", 10.0, 100.0)
            large = ab_methods.Answer("Answer: B", "large-m", 20.0, 400.0)
            small.correct, large.correct = n % 2 == 0, True
            answers[autobench.answer_key(it, "small-m")] = small
            answers[autobench.answer_key(it, "large-m")] = large

        def post(url, payload, api_key, timeout):
            if url.endswith("/api/route"):
                return ROUTE
            rows = [[1.0, float(len(t))] for t in payload["input"]]
            return {"data": [{"index": i, "embedding": v} for i, v in enumerate(rows)]}

        args = autobench.parse_args(
            ["--small", "small-m", "--large", "large-m", "--embed-model", "e", "--k", "2"]
        )
        details = io.StringIO()
        with mock.patch.object(ab_methods, "post", post):
            rows = autobench.rows_for_set(args, dataset, answers, details)
        routers = [r["router"] for r in rows]
        for expected in (
            "always-large",
            "always-small",
            "oracle",
            "reflex (own)",
            "random (Reflex's share)",
            "knn (own)",
        ):
            self.assertIn(expected, routers)
        own = next(r for r in rows if r["router"] == "reflex (own)")["metrics"]
        self.assertEqual(own["calls_avoided"], 1.0)
        self.assertEqual(own["reasons"], {"decided": 4})
        self.assertEqual(own["decision_model"], "jev")
        text = autobench.table(rows)
        header = text.splitlines()[0]
        for column in autobench.COLUMNS:
            self.assertIn(column, header)
        for line in text.splitlines()[2:]:
            self.assertEqual(line.count("|"), len(autobench.COLUMNS) + 1, line)
        self.assertEqual(len(details.getvalue().splitlines()), 8)

    def test_end_to_end_compares_auto_with_the_model_alone(self):
        items = [item(f"i{n}") for n in range(8)]
        dataset = ab_data.Dataset("t", items)
        answers = {}
        for it in items:
            for model, ttft in (("small-m", 10.0), ("large-m", 20.0)):
                answer = ab_methods.Answer("Answer: B", model, ttft, 100.0)
                answer.correct = True
                answers[autobench.answer_key(it, model)] = answer
        _, test = ab_data.split(dataset, 1)
        asked = []

        def generate(url, it, model, api_key, timeout, think=False, max_tokens=768):
            asked.append(model)
            if it is test[0]:
                raise ab_methods.ServerError(500, "boom")
            chosen = "small-m" if it is test[1] else "large-m"
            # One answer that is not what its model gave alone.
            text = "Answer: C" if it is test[2] else "Answer: B"
            answer = ab_methods.Answer(text, chosen, 30.0, 120.0)
            answer.route = {"reason": "decided", "decision_ms": 12.0, "model": chosen}
            return answer

        args = autobench.parse_args(
            ["--small", "small-m", "--large", "large-m", "--concurrency", "1,2", "--e2e-limit", "3"]
        )
        with mock.patch.object(ab_methods, "generate", generate):
            rows = autobench.e2e_rows(args, dataset, answers)
        self.assertEqual(set(asked), {"auto"})
        self.assertEqual([r["concurrency"] for r in rows], [1, 2])
        m = rows[0]["metrics"]
        self.assertEqual((m["items"], m["answered"], m["error_count"]), (3, 2, 1))
        self.assertEqual(m["same_answer"], 0.5)
        self.assertEqual(m["routed_small"], 1)
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["accuracy_large"], 1.0)
        self.assertEqual(m["reasons"], {"decided": 2})
        self.assertEqual(m["ttft_large_alone_ms"]["p50"], 20.0)
        self.assertEqual(m["routing_ms"]["p50"], 12.0)
        text = autobench.e2e_table(rows)
        for column in autobench.E2E_COLUMNS:
            self.assertIn(column, text.splitlines()[0])
        for line in text.splitlines()[2:]:
            self.assertEqual(line.count("|"), len(autobench.E2E_COLUMNS) + 1, line)

    def test_kept_answers_are_reused_only_when_asked_the_same_way(self):
        args = autobench.parse_args(["--url", "http://x", "--small", "s", "--large", "l"])
        how = autobench.answers_how(args)
        self.assertIs(how["cache_prompt"], False)
        answer = ab_methods.Answer("4", "s", 1.0, 2.0).to_json()
        rows = [
            {"key": "a", "how": how, "answer": answer},
            # Kept before `cache_prompt`, or with another limit: asked again.
            {"key": "b", "answer": answer},
            {"key": "c", "how": dict(how, max_tokens=16), "answer": answer},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "answers.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            self.assertEqual(list(autobench.load_answers(str(path), how)), ["a"])

    def test_concurrency_levels_are_numbers(self):
        args = autobench.parse_args(["--small", "s", "--large", "l"])
        self.assertEqual(args.concurrency, [1, 4, 16])
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            autobench.parse_args(["--small", "s", "--large", "l", "--concurrency", "1,x"])
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            autobench.parse_args(["--small", "s", "--large", "l", "--concurrency", "0"])


if __name__ == "__main__":
    unittest.main()
