"""End to end against a real EuLLM: the command started on stdio, as an MCP
client starts it, every tool once, and one of EuLLM's own refusals. Skipped
unless REFLEX_MCP_E2E_URL names a running server with a decision model;
REFLEX_MCP_E2E_EMBED_MODEL names the embedding model for the two stages, and
EULLM_API_KEY is passed on when set:

    EULLM_AUDIT_DIR=/tmp/reflex-mcp-audit eullm serve --port 11602 \\
        --decision-model jev-style-0.8b-decision-v3-gguf-q4_k_m \\
        --embedding-model qwen3-embedding-0.6b-gguf-q8_0
    REFLEX_MCP_E2E_URL=http://127.0.0.1:11602 \\
    REFLEX_MCP_E2E_EMBED_MODEL=qwen3-embedding-0.6b-gguf-q8_0 \\
        python3 -m unittest discover -s adapters/reflex-mcp/tests -p test_e2e.py -v

It prints what each tool answered and how long it took. The assertions on
answers are the obvious cases only: the weather tool for a weather request,
a passage that states the answer against passages about something else.
"""

import json
import os
import pathlib
import sys
import time
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "src"

from mcp import Client, StdioServerParameters  # noqa: E402

URL = os.environ.get("REFLEX_MCP_E2E_URL")
EMBED = os.environ.get("REFLEX_MCP_E2E_EMBED_MODEL")
# Qwen3-Embedding's instruction for the request, as ReflexBench measured it.
PREFIX = "Instruct: Given a user request, retrieve the tools needed to handle it\\nQuery:"

CATALOG = [
    {"name": "get_weather", "description": "Current weather and the forecast for a city"},
    {"name": "send_email", "description": "Send an email message to a recipient"},
    {"name": "create_invoice", "description": "Create an invoice for a customer"},
    {"name": "book_meeting", "description": "Book a meeting room and invite people"},
    {"name": "stock_price", "description": "Latest price of a stock ticker"},
    {"name": "translate_text", "description": "Translate text between two languages"},
    {"name": "search_flights", "description": "Search flights between two airports"},
    {"name": "set_reminder", "description": "Set a reminder at a date and time"},
]
QUESTION = "Who wrote the novel The Name of the Rose?"
SUFFICIENT = [
    "The Name of the Rose: The Name of the Rose is the 1980 debut novel by Italian author "
    "Umberto Eco. It is a historical murder mystery set in an Italian monastery in 1327.",
    "Bologna: Bologna is the capital of the Emilia-Romagna region in northern Italy.",
]
IRRELEVANT = [
    "Bologna: Bologna is the capital of the Emilia-Romagna region in northern Italy.",
    "Rome: Rome has a Mediterranean climate, with hot, dry summers and mild, wet winters.",
]


@unittest.skipUnless(URL, "set REFLEX_MCP_E2E_URL to a running EuLLM with a decision model")
class EndToEndTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        env = {"EULLM_URL": URL, "PYTHONPATH": str(SRC)}
        if EMBED:
            env.update(EULLM_EMBED_MODEL=EMBED, EULLM_EMBED_QUERY_PREFIX=PREFIX)
        if os.environ.get("EULLM_API_KEY"):
            env["EULLM_API_KEY"] = os.environ["EULLM_API_KEY"]
        self.command = StdioServerParameters(
            command=sys.executable, args=["-m", "eullm_reflex_mcp"], env=env
        )

    async def test_every_tool(self):
        results = {}
        async with Client(self.command, read_timeout_seconds=600) as client:

            async def call(label, tool, **arguments):
                started = time.perf_counter()
                result = await client.call_tool(tool, arguments)
                seconds = time.perf_counter() - started
                shown = result.content[0].text if result.is_error else result.structured_content
                print(f"\n{label} ({seconds:.2f} s): {json.dumps(shown, ensure_ascii=False)}")
                results[label] = result
                return result

            await call("model_info", "model_info")
            await call(
                "select_tools",
                "select_tools",
                request="Will it rain in Rome tomorrow? Do I need an umbrella?",
                tools=CATALOG,
                shortlist=4,
            )
            await call(
                "select_tools, no tool needed",
                "select_tools",
                request="Thanks, that is all for today!",
                tools=CATALOG,
            )
            await call("rag_gate, sufficient", "rag_gate", question=QUESTION, passages=SUFFICIENT)
            await call("rag_gate, irrelevant", "rag_gate", question=QUESTION, passages=IRRELEVANT)
            await call(
                "decide",
                "decide",
                state="Help! My payouts have been failing for 3 days.",
                questions={
                    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
                    "team": {
                        "type": "choice",
                        "instructions": "Which team should handle it?",
                        "criteria": {"billing": "Payments and payouts", "tech": "Bugs"},
                    },
                },
            )
            await call(
                "decide, one option",
                "decide",
                state="x",
                questions={"q": {"type": "choice", "instructions": "?", "criteria": {"a": None}}},
            )

        for label, result in results.items():
            if label != "decide, one option":
                self.assertFalse(result.is_error, label)
        info = results["model_info"].structured_content
        self.assertIsNotNone(info["decision_model"], "EuLLM has no decision model loaded")

        selection = results["select_tools"].structured_content
        self.assertEqual(selection["method"], "two-stage" if EMBED else "reflex")
        self.assertEqual(selection["tools"][0]["name"], "get_weather")
        total = sum(t["probability"] for t in selection["tools"]) + selection["none_probability"]
        self.assertAlmostEqual(total, 1.0, places=2)

        sufficient = results["rag_gate, sufficient"].structured_content
        irrelevant = results["rag_gate, irrelevant"].structured_content
        self.assertGreater(
            sufficient["probabilities"]["answer"], irrelevant["probabilities"]["answer"]
        )

        answers = results["decide"].structured_content["answers"]
        self.assertEqual(set(answers), {"is_urgent", "team"})
        refused = results["decide, one option"]
        self.assertTrue(refused.is_error)
        self.assertIn("422 invalid_question", refused.content[0].text)


if __name__ == "__main__":
    unittest.main()
