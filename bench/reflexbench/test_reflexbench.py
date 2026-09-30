"""Offline tests for ReflexBench: the metrics, BM25, the loaders on made-up
files, and the Reflex client against a stand-in server. No network, no
model:

    python3 -m unittest discover -s bench/reflexbench
"""

import io
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_data  # noqa: E402
import rb_methods  # noqa: E402
import rb_metrics  # noqa: E402
import reflexbench  # noqa: E402


def tools(*names):
    return [rb_data.Tool(n, f"{n} tool", f"spec of {n}") for n in names]


class MetricsTest(unittest.TestCase):
    def test_worst_rank_is_the_needed_tool_ranked_lowest(self):
        self.assertEqual(rb_metrics.worst_rank(["a", "b", "c"], ["c", "a"]), 3)

    def test_k_for_keeps_the_coverage_asked(self):
        ranks = list(range(1, 101))
        self.assertEqual(rb_metrics.k_for(ranks, 0.95), 95)
        self.assertEqual(rb_metrics.k_for(ranks, 0.99), 99)
        self.assertEqual(rb_metrics.k_for([4], 0.95), 4)

    def test_summarize(self):
        catalog = tools("a", "b", "c", "d")
        items = [
            rb_data.Item("1", "r1", catalog, ["a"]),
            rb_data.Item("2", "r2", catalog, ["b", "c"]),
            rb_data.Item("3", "r3", catalog, []),
        ]
        rankings = [
            rb_methods.Ranking(["a", "b", "c", "d"], 10.0, none_score=-2.0, best_score=1.0),
            rb_methods.Ranking(["b", "d", "c", "a"], 20.0, none_score=3.0, best_score=1.0),
            rb_methods.Ranking(["d", "c", "b", "a"], 30.0, none_score=2.0, best_score=0.5),
        ]
        m = rb_metrics.summarize(items, rankings)
        self.assertEqual(m["items"], 3)
        self.assertEqual(m["items_needing_tools"], 2)
        # Item 1 is whole at k=1, item 2 only at k=3, where "c" is.
        self.assertEqual(m["recall_at"]["1"], 0.5)
        self.assertEqual(m["recall_at"]["3"], 1.0)
        self.assertEqual(m["k95"], 3)
        self.assertEqual(m["mrr"], 1.0)
        # Three specs of four, all the same length, kept on both items.
        self.assertAlmostEqual(m["spec_kept_at_k95"], 0.75)
        # Item 3 needs nothing and "no tool" won; item 2 needs two and it won too.
        self.assertEqual(m["abstain_accuracy"], 1.0)
        self.assertEqual(m["false_abstain_rate"], 0.5)
        self.assertEqual(m["latency_ms"], {"p50": 20.0, "p95": 30.0})
        self.assertNotIn("evaluated_tokens_mean", m)

    def test_no_abstain_metrics_when_not_asked(self):
        items = [rb_data.Item("1", "r", tools("a", "b"), [])]
        m = rb_metrics.summarize(items, [rb_methods.Ranking(["b", "a"], 1.0)])
        self.assertNotIn("recall_at", m)
        self.assertNotIn("abstain_accuracy", m)


class BM25Test(unittest.TestCase):
    def test_words_split_names(self):
        self.assertEqual(
            rb_methods.words("getWeather_forecast2Day"), ["get", "weather", "forecast2", "day"]
        )

    def test_the_matching_tool_comes_first(self):
        catalog = [
            rb_data.Tool("get_weather", "Current weather and forecast for a city", ""),
            rb_data.Tool("send_email", "Send an email message to a recipient", ""),
            rb_data.Tool("stock_price", "Latest price of a stock ticker", ""),
        ]
        item = rb_data.Item("1", "Rain in Rome tomorrow? Check the forecast", catalog, [])
        ranking = rb_methods.BM25(rb_data.Dataset("t", [item], True)).rank(item)
        self.assertEqual(ranking.order[0], "get_weather")
        self.assertEqual(sorted(ranking.order), ["get_weather", "send_email", "stock_price"])
        self.assertIsNone(ranking.none_score)


class EmbeddingsTest(unittest.TestCase):
    def test_ranks_by_cosine_and_embeds_each_tool_once(self):
        vectors = {"a: a tool": [1.0, 0.0], "b: b tool": [0.0, 1.0], "c: c tool": [0.6, 0.8]}
        asked = []

        def post(url, payload, api_key, timeout):
            asked.extend(payload["input"])
            rows = [vectors.get(text, [0.1, 1.0]) for text in payload["input"]]
            return {"data": [{"index": i, "embedding": v} for i, v in enumerate(rows)]}

        item = rb_data.Item("1", "r", tools("a", "b", "c"), ["b"])
        with (
            mock.patch.object(rb_methods, "post", post),
            mock.patch.dict(rb_methods.Embeddings.seen, clear=True),
        ):
            embed = rb_methods.Embeddings("http://x", "m", None, 10, query_prefix="Q: ")
            self.assertEqual(embed.rank(item).order, ["b", "c", "a"])
            rb_methods.Embeddings("http://x", "m", None, 10).rank(item)
        self.assertEqual(sorted(asked), ["Q: r", "a: a tool", "b: b tool", "c: c tool", "r"])


def lines(rows):
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


class LoadersTest(unittest.TestCase):
    def test_metatool(self):
        files = {
            "dataset/plugin_des.json": json.dumps({"WeatherTool": "old", "MailTool": "Send mail"}),
            "dataset/big_tool_des.json": json.dumps({"WeatherTool": "Weather forecasts"}),
            "dataset/data/all_clean_data.csv": "Query,Tool\nWill it rain?,WeatherTool\n"
            "Mail Bob,MailTool\n",
            "dataset/data/multi_tool_query_golden.json": json.dumps(
                [{"query": "Rain? Then mail Bob", "tool": ["WeatherTool", "MailTool"]}]
            ),
        }

        def fetch(url):
            return files[url[len(rb_data.METATOOL) :]].encode()

        with mock.patch.object(rb_data, "fetch", fetch):
            single = rb_data.load("metatool-single")
            multi = rb_data.load("metatool-multi")
        self.assertTrue(single.fixed_catalog)
        self.assertEqual(sorted(i.request for i in single.items), ["Mail Bob", "Will it rain?"])
        catalog = single.items[0].candidates
        self.assertIs(catalog, single.items[1].candidates)
        self.assertEqual([t.name for t in catalog], ["MailTool", "WeatherTool"])
        self.assertEqual(catalog[1].description, "Weather forecasts")
        self.assertEqual(multi.items[0].needed, ["WeatherTool", "MailTool"])

    def test_bfcl(self):
        functions = [
            {"name": "get_weather", "description": "Weather for a city", "parameters": {}},
            {"name": "send_email", "description": "Send an email", "parameters": {}},
        ]
        user = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Rome?"}]
        rows = [
            {"id": "m_0", "question": [user], "function": functions},
            # A name offered twice cannot be scored apart; nothing offered,
            # nothing to select: both left out.
            {"id": "m_1", "question": [user], "function": functions + functions[:1]},
            {"id": "m_2", "question": [user], "function": []},
        ]
        answers = [
            {"id": "m_0", "ground_truth": [{"get_weather": {"city": ["Rome"]}}]},
            {"id": "m_1", "ground_truth": [{"send_email": {}}]},
        ]
        files = {
            "BFCL_v3_live_multiple.json": lines(rows),
            "possible_answer/BFCL_v3_live_multiple.json": lines(answers),
        }

        def fetch(url):
            return files[url[len(rb_data.BFCL) :]]

        with mock.patch.object(rb_data, "fetch", fetch):
            data = rb_data.load("bfcl-live-multiple")
        self.assertEqual(data.name, "bfcl-live-multiple")
        self.assertFalse(data.fixed_catalog)
        self.assertEqual([i.id for i in data.items], ["m_0"])
        item = data.items[0]
        self.assertEqual(item.request, "Rome?")
        self.assertEqual(item.needed, ["get_weather"])
        self.assertEqual(json.loads(item.candidates[0].spec), functions[0])

    def test_bfcl_irrelevance_needs_no_tool(self):
        row = {
            "id": "irrelevance_0",
            "question": [[{"role": "user", "content": "Tell me a joke"}]],
            "function": [{"name": "area", "description": "Area of a shape"}],
        }
        with mock.patch.object(rb_data, "fetch", lambda url: lines([row])):
            data = rb_data.load("bfcl-irrelevance")
        self.assertEqual(data.items[0].needed, [])

    def test_own_set_needs_only_tools_it_offers(self):
        row = {"id": "x", "request": "r", "needed": ["z"], "tools": [{"name": "a"}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bad.jsonl"
            path.write_bytes(lines([row]))
            with self.assertRaises(ValueError):
                rb_data.from_jsonl(path)

    def test_unknown_set(self):
        with self.assertRaises(ValueError):
            rb_data.load("nope")

    def test_own_set_and_resized(self):
        catalog = [{"name": n, "description": f"{n} tool"} for n in "abcde"]
        rows = [
            {"id": "x", "request": "r1", "needed": ["c"], "tools": catalog},
            {"id": "y", "request": "r2", "needed": [], "tools": catalog},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "mine.jsonl"
            path.write_bytes(lines(rows))
            data = rb_data.from_jsonl(path)
        self.assertEqual(data.name, "mine")
        self.assertTrue(data.fixed_catalog)
        self.assertIs(data.items[0].candidates, data.items[1].candidates)
        self.assertEqual(data.items[1].needed, [])

        small = rb_data.resized(data, 3, seed=1)
        self.assertEqual(small.name, "mine@3")
        self.assertFalse(small.fixed_catalog)
        names = [t.name for t in small.items[0].candidates]
        self.assertEqual(len(names), 3)
        self.assertIn("c", names)
        self.assertEqual(names, sorted(names))
        again = rb_data.resized(data, 3, seed=1)
        self.assertEqual(names, [t.name for t in again.items[0].candidates])


SCORES = {"a": 0.1, "b": -1.0, "c": 2.0, "d": 0.5, rb_methods.NONE: -3.0}


def server(max_options):
    """A stand-in `/v1/systemone` that refuses a question with more than
    `max_options` options, as EuLLM does one too long for the model."""
    sent = []

    def post(url, payload, api_key, timeout):
        sent.append(payload)
        if any(len(q["criteria"]) > max_options for q in payload["questions"].values()):
            raise rb_methods.ServerError(400, "question too long: nothing was truncated")
        return {
            "answers": {
                name: {"eullm": {"scores": {o: SCORES[o] for o in q["criteria"]}}}
                for name, q in payload["questions"].items()
            },
            "eullm": {"evaluated_tokens": 42, "prefix_reused": True},
        }

    return post, sent


class ReflexTest(unittest.TestCase):
    item = rb_data.Item("1", "Will it rain?", tools("a", "b", "c", "d"), ["c"])

    def test_layout_a_splits_a_question_too_long(self):
        post, sent = server(max_options=3)
        reflex = rb_methods.Reflex("http://x/", "A", None, None, 10, abstain=True)
        with mock.patch.object(rb_methods, "post", post):
            ranking = reflex.rank(self.item)
        self.assertEqual(ranking.order, ["c", "d", "a", "b"])
        self.assertEqual((ranking.best_score, ranking.none_score), (2.0, -3.0))
        self.assertFalse(ranking.abstains())
        # Four tools and "none" were refused; two questions of two fit.
        self.assertEqual(len(sent), 2)
        self.assertEqual(reflex.chunk, 2)
        payload = sent[-1]
        self.assertEqual(payload["state"], "Will it rain?")
        self.assertNotIn("model", payload)
        first = payload["questions"]["tools_0"]
        self.assertEqual(first["type"], "choice")
        self.assertEqual(
            first["criteria"], {"a": "a tool", "b": "b tool", "none": rb_methods.NONE_TEXT}
        )
        self.assertEqual(list(payload["questions"]["tools_1"]["criteria"]), ["c", "d", "none"])
        self.assertEqual(ranking.server["questions"], 2)
        self.assertEqual(ranking.server["evaluated_tokens"], 42)

    def test_layout_b_puts_the_catalog_in_the_state(self):
        post, sent = server(max_options=10)
        reflex = rb_methods.Reflex("http://x", "B", "m", None, 10, abstain=False)
        with mock.patch.object(rb_methods, "post", post):
            ranking = reflex.rank(self.item)
        self.assertEqual(ranking.order, ["c", "d", "a", "b"])
        self.assertIsNone(ranking.none_score)
        payload = sent[0]
        self.assertEqual(payload["model"], "m")
        self.assertTrue(payload["state"].startswith("The tools available:\n- a: a tool\n"))
        question = payload["questions"]["tools_0"]
        self.assertIn('"Will it rain?"', question["instructions"])
        self.assertEqual(question["criteria"], dict.fromkeys("abcd"))

    def test_without_none_every_question_keeps_two_options(self):
        post, sent = server(max_options=2)
        reflex = rb_methods.Reflex("http://x", "A", None, None, 10, abstain=False)
        with mock.patch.object(rb_methods, "post", post):
            ranking = reflex.rank(self.item)
        self.assertEqual(ranking.order, ["c", "d", "a", "b"])
        self.assertEqual(reflex.chunk, 3)
        sizes = [len(q["criteria"]) for q in sent[-1]["questions"].values()]
        self.assertEqual(sizes, [2, 2])

    def test_one_tool_without_none_is_no_decision(self):
        def post(url, payload, api_key, timeout):
            raise AssertionError("nothing to ask")

        item = rb_data.Item("1", "r", tools("a"), ["a"])
        reflex = rb_methods.Reflex("http://x", "A", None, None, 10, abstain=False)
        with mock.patch.object(rb_methods, "post", post):
            ranking = reflex.rank(item)
        self.assertEqual((ranking.order, ranking.server["questions"]), (["a"], 0))

    def test_split_evenly(self):
        def sizes(n, size):
            return [len(part) for part in rb_methods.split(list(range(n)), size)]

        self.assertEqual(sizes(199, 99), [67, 66, 66])
        self.assertEqual(sizes(7, 3), [3, 2, 2])
        self.assertEqual(sizes(4, 255), [4])
        for n in range(2, 60):
            self.assertGreaterEqual(min(sizes(n, 3)), 2, n)

    def test_abstains_when_no_tool_scores_higher(self):
        self.assertTrue(rb_methods.Ranking(["a"], 1.0, none_score=1.0, best_score=0.5).abstains())
        self.assertFalse(rb_methods.Ranking(["a"], 1.0, best_score=0.5).abstains())

    def test_other_errors_are_not_retried(self):
        def post(url, payload, api_key, timeout):
            raise rb_methods.ServerError(503, "no decision model loaded")

        reflex = rb_methods.Reflex("http://x", "A", None, None, 10, abstain=False)
        with mock.patch.object(rb_methods, "post", post):
            with self.assertRaises(rb_methods.ServerError):
                reflex.rank(self.item)
        self.assertEqual(reflex.chunk, 255)

    def test_a_model_without_verdict_scores_is_refused(self):
        def post(url, payload, api_key, timeout):
            return {"answers": {"tools_0": {"answer": "a"}}}

        reflex = rb_methods.Reflex("http://x", "A", None, None, 10, abstain=False)
        with mock.patch.object(rb_methods, "post", post):
            with self.assertRaises(ValueError):
                reflex.rank(self.item)


class TwoStageTest(unittest.TestCase):
    def test_reflex_ranks_the_shortlist_and_the_rest_follow(self):
        item = rb_data.Item("1", "r", tools("a", "b", "c", "d", "e"), ["d"])
        shown = []

        class Embeddings:
            def rank(self, item):
                return rb_methods.Ranking(["e", "d", "a", "b", "c"], 5.0)

        class Reflex:
            abstain = True

            def rank(self, item):
                shown.append([t.name for t in item.candidates])
                return rb_methods.Ranking(["d", "a", "e"], 20.0, -1.0, 2.0, {"questions": 1})

        ranking = rb_methods.TwoStage(Embeddings(), Reflex(), shortlist=3).rank(item)
        # The shortlist in the catalog's order, not the embeddings'.
        self.assertEqual(shown, [["a", "d", "e"]])
        self.assertEqual(ranking.order, ["d", "a", "e", "b", "c"])
        self.assertEqual(ranking.ms, 25.0)
        self.assertEqual((ranking.none_score, ranking.best_score), (-1.0, 2.0))


class BuildMethodsTest(unittest.TestCase):
    dataset = rb_data.Dataset("t", [rb_data.Item("1", "r", tools("a", "b"), ["a"])], True)

    def args(self, **changes):
        args = dict(
            methods="bm25,two-stage",
            embed_model=None,
            embed_query_prefix="Q:\\n",
            url="http://x",
            api_key=None,
            timeout=1,
            model=None,
            abstain="on",
            shortlist=20,
        )
        args.update(changes)
        return types.SimpleNamespace(**args)

    def test_embeddings_are_needed_for_two_stages(self):
        with mock.patch("sys.stderr", io.StringIO()):
            methods = reflexbench.build_methods(self.args(), self.dataset)
        self.assertEqual([m.name for m in methods], ["bm25"])

    def test_two_stage(self):
        methods = reflexbench.build_methods(self.args(embed_model="e"), self.dataset)
        self.assertEqual([m.name for m in methods], ["bm25", "two-stage"])
        two = methods[1]
        self.assertTrue(two.abstain)
        self.assertEqual((two.reflex.layout, two.shortlist), ("A", 20))
        self.assertEqual(two.embeddings.query_prefix, "Q:\n")

    def test_unknown_method(self):
        with self.assertRaises(SystemExit):
            reflexbench.build_methods(self.args(methods="nope"), self.dataset)


class TableTest(unittest.TestCase):
    def test_every_row_has_every_column(self):
        full = rb_metrics.summarize(
            [ReflexTest.item], [rb_methods.Ranking(["c", "a", "b", "d"], 5.0)]
        )
        empty = {"items": 0, "latency_ms": {"p50": None, "p95": None}}
        out = reflexbench.table(
            [
                {"set": "s", "method": "bm25", "metrics": full},
                {"set": "s", "method": "reflex-a", "metrics": empty},
            ]
        )
        rows = out.splitlines()
        self.assertEqual(len(rows), 4)
        self.assertEqual({row.count("|") for row in rows}, {16})
        self.assertIn("| s | bm25 | 1 | 100.0% |", rows[2])


if __name__ == "__main__":
    unittest.main()
