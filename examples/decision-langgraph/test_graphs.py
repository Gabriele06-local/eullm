"""Both graphs against a stand-in EuLLM: an HTTP server from the standard
library that answers `/v1/systemone`, `/v1/embeddings` and
`/v1/chat/completions` in EuLLM's shapes, with the decisions each test sets.
No network, no model.

    pip install -r examples/decision-langgraph/requirements.txt
    python -m unittest discover -s examples/decision-langgraph
"""

import contextlib
import hashlib
import http.server
import io
import json
import math
import os
import pathlib
import re
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The stand-in listens on 127.0.0.1: a proxy set for this machine must not
# get those requests, and LangSmith tracing, when switched on, must not send
# the tests anywhere.
for _name in ("NO_PROXY", "no_proxy"):
    os.environ[_name] = ",".join(filter(None, [os.environ.get(_name), "127.0.0.1", "localhost"]))
for _name in (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
):
    os.environ.pop(_name, None)

from langgraph.types import Command  # noqa: E402

import eullm_client  # noqa: E402
import rag_graph  # noqa: E402
import triage_graph  # noqa: E402

DIMENSIONS = 64


def vector(text):
    """A bag of words hashed into a few dimensions: texts that share words
    are close, the same text always gets the same vector."""
    v = [0.0] * DIMENSIONS
    for word in re.findall(r"[a-z]+", text.lower()):
        v[int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIMENSIONS] += 1.0
    return v


def choice(probabilities):
    best = max(probabilities, key=probabilities.get)
    return {"type": "choice", "choice": best, "probabilities": probabilities, "confidence": 0.0}


def noul(p):
    return {"type": "noul", "noul": p}


class StandIn(http.server.ThreadingHTTPServer):
    """EuLLM, as far as the graphs can tell."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests = []  # (path, headers with lower-case names, payload)
        self.decide = lambda payload: {}  # the answers to a /v1/systemone payload
        self.reply = "A draft."
        self.fail = None  # (status, body) to answer every request with
        self.misbehave = None  # a broken answer, by name: Handler.MISBEHAVIOUR
        # Polled often, so that shutting it down at the end of a test is quick.
        threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        ).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def sent(self, path):
        return [payload for p, _, payload in self.requests if p == path]

    def close(self):
        self.shutdown()
        self.server_close()


class Handler(http.server.BaseHTTPRequestHandler):
    #: Answers a request the way something other than a well-behaved server
    #: would, named so a test can ask for one. All three leave the client's
    #: request past the status line, where urllib's own error handling ends.
    STALL, HTML, TRUNCATED = "stall", "html", "truncated"

    def _raw(self, data, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        server = self.server
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        headers = {k.lower(): v for k, v in self.headers.items()}
        server.requests.append((self.path, headers, payload))
        if server.misbehave == self.STALL:
            # The status line, then not a byte more: the socket stays open and
            # the client's own timeout is what ends it. Writing a partial body
            # instead would close the read with IncompleteRead, not a timeout.
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "40")
            # Without this the handler's HTTP/1.0 closes the connection when it
            # returns and the client reads IncompleteRead rather than waiting.
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.flush()
            time.sleep(6)
            return
        if server.misbehave == self.HTML:
            self._raw(b"<html><body>502 Bad Gateway</body></html>", "text/html")
            return
        if server.misbehave == self.TRUNCATED:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "400")
            self.end_headers()
            self.wfile.write(b'{"answers": {"q": {"type": "cho')
            self.close_connection = True
            return
        status = 200
        if server.fail:
            status, body = server.fail
        elif self.path == "/v1/systemone":
            body = {"model": "stand-in-decision", "answers": server.decide(payload)}
        elif self.path == "/v1/embeddings":
            rows = [vector(text) for text in payload["input"]]
            body = {"data": [{"index": i, "embedding": v} for i, v in enumerate(rows)]}
        elif self.path == "/v1/chat/completions":
            body = {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": server.reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        else:
            status, body = 404, {"error": f"no route for {self.path}"}
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class WithStandIn(unittest.TestCase):
    def setUp(self):
        self.server = StandIn()
        self.addCleanup(self.server.close)
        self.eullm = eullm_client.EuLLM(self.server.url, api_key="secret", timeout=10)
        self.chat = self.eullm.chat_model("chat-model")


# --- triage ----------------------------------------------------------------

CLEAR = {"billing": 0.9, "technical": 0.05, "account": 0.03, "sales": 0.02}
UNSURE = {"billing": 0.5, "account": 0.45, "technical": 0.03, "sales": 0.02}


class TriageTest(WithStandIn):
    def setUp(self):
        super().setUp()
        self.app = triage_graph.build_graph(self.eullm, self.chat, decision_model="jev")

    def answers(self, team, blocked, person):
        self.server.decide = lambda payload: {
            "team": choice(team),
            "blocked": noul(blocked),
            "person": noul(person),
        }

    def run_ticket(self, ticket, thread):
        return self.app.invoke({"ticket": ticket}, {"configurable": {"thread_id": thread}})

    def resume(self, value, thread):
        return self.app.invoke(Command(resume=value), {"configurable": {"thread_id": thread}})

    def test_a_clear_ticket_goes_to_its_team_which_drafts_the_reply(self):
        self.answers(CLEAR, blocked=0.8, person=0.1)
        ticket = triage_graph.SAMPLES[0]
        state = self.run_ticket(ticket, "clear")
        self.assertEqual(state["team"], "billing")
        self.assertEqual(state["priority"], "high")
        self.assertEqual(state["routed_by"], "model")
        self.assertEqual(state["reply"], "A draft.")
        self.assertEqual(state["decision"]["team"], CLEAR)

        # One decision: the ticket as the state, three questions about it.
        (decision,) = self.server.sent("/v1/systemone")
        self.assertEqual(decision["model"], "jev")
        self.assertEqual(decision["state"]["subject"], ticket["subject"])
        self.assertEqual(decision["state"]["message"], ticket["message"])
        kinds = {name: q["type"] for name, q in decision["questions"].items()}
        self.assertEqual(kinds, {"team": "choice", "blocked": "noul", "person": "noul"})
        self.assertEqual(decision["questions"]["team"]["criteria"], triage_graph.TEAMS)

        # The billing team's node drafted it, told the ticket has priority.
        (chat,) = self.server.sent("/v1/chat/completions")
        self.assertEqual(chat["model"], "chat-model")
        system, human = chat["messages"]
        self.assertIn("billing team", system["content"])
        self.assertIn("priority", system["content"])
        self.assertIn(ticket["message"], human["content"])
        # EuLLM's fields, in the body itself.
        self.assertIs(chat["think"], False)
        self.assertEqual(chat["max_tokens"], 400)

        # The key goes with every request, to both endpoints.
        self.assertEqual(len(self.server.requests), 2)
        for _, headers, _ in self.server.requests:
            self.assertEqual(headers["authorization"], "Bearer secret")

    def test_a_ticket_a_person_must_handle_waits_for_one(self):
        self.answers(CLEAR, blocked=0.2, person=0.8)
        state = self.run_ticket(triage_graph.SAMPLES[4], "must")
        waiting = state["__interrupt__"][0].value
        self.assertEqual(waiting["reason"], "a person must handle it (0.80)")
        self.assertEqual(waiting["team"], CLEAR)
        self.assertEqual(state["priority"], "normal")
        self.assertFalse(self.server.sent("/v1/chat/completions"))

        # The person keeps it: the graph ends, and nothing is drafted.
        state = self.resume("", "must")
        self.assertEqual(state["team"], "person")
        self.assertNotIn("reply", state)
        self.assertFalse(self.server.sent("/v1/chat/completions"))

    def test_an_unsure_team_goes_to_a_person_who_picks_one(self):
        self.answers(UNSURE, blocked=0.1, person=0.1)
        state = self.run_ticket(triage_graph.SAMPLES[7], "unsure")
        reason = state["__interrupt__"][0].value["reason"]
        self.assertEqual(reason, "unsure of the team: billing 0.50, account 0.45, technical 0.03")

        state = self.resume("account", "unsure")
        self.assertEqual((state["team"], state["routed_by"]), ("account", "person"))
        self.assertEqual(state["reply"], "A draft.")
        (chat,) = self.server.sent("/v1/chat/completions")
        self.assertIn("account team", chat["messages"][0]["content"])
        # Resuming does not ask the decision model again.
        self.assertEqual(len(self.server.sent("/v1/systemone")), 1)

    def test_the_thresholds_are_the_policy_s_and_nothing_else(self):
        decision = {"team": {"a": 0.7, "b": 0.3}, "person": 0.49, "blocked": 0.0}
        policy = triage_graph.policy
        self.assertEqual(policy(decision, 0.5, 0.6), ("a", ""))
        self.assertEqual(policy(dict(decision, person=0.5), 0.5, 0.6)[0], "person")
        self.assertEqual(policy(decision, 0.5, 0.71)[0], "person")

    def test_a_team_cannot_take_a_node_s_name(self):
        with self.assertRaises(ValueError):
            triage_graph.build_graph(self.eullm, self.chat, {"person": "x", "billing": "y"})

    def test_the_command_line(self):
        def decide(payload):
            unsure = payload["state"]["subject"] == "Change of email"
            return {
                "team": choice(UNSURE if unsure else CLEAR),
                "blocked": noul(0.1),
                "person": noul(0.1),
            }

        self.server.decide = decide
        tickets = [triage_graph.SAMPLES[0], triage_graph.SAMPLES[7]]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "tickets.json"
            path.write_text(json.dumps(tickets), encoding="utf-8")
            out = io.StringIO()
            argv = ["--url", self.server.url, "--chat-model", "m", "--tickets", str(path), "--ask"]
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                # A mistyped team is asked again.
                with mock.patch("builtins.input", side_effect=["acount", "account"]) as asked:
                    triage_graph.main(argv)
        text = out.getvalue()
        self.assertEqual(asked.call_count, 2)  # only the unsure ticket went to a person
        self.assertIn("no team is called 'acount'", text)
        self.assertRegex(text, r"1  Charged twice for my yearly plan\s+→ billing")
        self.assertIn("a person chose account", text)
        self.assertEqual(text.count("│ A draft."), 2)


# --- RAG -------------------------------------------------------------------

DOCS = {
    "refunds.md": "# Refunds\n\nYearly plans are refunded within thirty days.\n\n"
    "Monthly plans are never refunded.\n",
    "plans.md": "# Plans\n\nThe team plan is billed yearly.\n\nThe personal plan is billed "
    "monthly.\n",
    "offices.md": "# Offices\n\nThe only office is in Turin.\n",
}
QUESTION = "Are yearly plans refunded?"


def gate(*choices):
    """A /v1/systemone stand-in that answers the gate with `choices`, one
    per request, the last one again once they run out."""
    left = list(choices)

    def decide(payload):
        name = left.pop(0) if len(left) > 1 else left[0]
        probabilities = {"answer": 0.1, "retrieve_more": 0.1, "abstain": 0.1}
        probabilities[name] = 0.8
        return {"gate": choice(probabilities)}

    return decide


class RagTest(WithStandIn):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.docs = pathlib.Path(tmp.name)
        for name, text in DOCS.items():
            (self.docs / name).write_text(text, encoding="utf-8")
        self.passages = rag_graph.load_documents(self.docs)
        self.index = rag_graph.Index(self.eullm, "embedder", self.passages)

    def graph(self, **kwargs):
        kwargs.setdefault("k", 2)
        return rag_graph.build_graph(self.eullm, self.chat, self.index, "jev", **kwargs)

    def test_the_documents_are_cut_into_titled_passages(self):
        texts = [p["text"] for p in self.passages]
        self.assertEqual(len(texts), 5)
        self.assertIn("Refunds: Yearly plans are refunded within thirty days.", texts)
        bundled = rag_graph.load_documents(rag_graph.DOCUMENTS)
        self.assertGreaterEqual(len(bundled), 15)
        self.assertTrue(all(re.match(r"[A-Z][^:]+: \S", p["text"]) for p in bundled))

    def test_the_passages_are_embedded_once_and_the_question_with_its_instruction(self):
        (passages,) = self.server.sent("/v1/embeddings")
        self.assertEqual(passages["model"], "embedder")
        self.assertEqual(passages["input"], [p["text"] for p in self.passages])
        self.server.decide = gate("answer")
        self.graph().invoke({"question": QUESTION})
        question = self.server.sent("/v1/embeddings")[1]
        self.assertEqual(question["input"], [rag_graph.QUERY_INSTRUCTION + QUESTION])

    def test_when_the_passages_suffice_the_chat_model_answers_from_them(self):
        self.server.decide = gate("answer")
        state = self.graph().invoke({"question": QUESTION})
        self.assertEqual(state["answer"], "A draft.")
        self.assertEqual(state["rounds"], 1)
        # The closest passage first, k at a time.
        self.assertEqual(len(state["passages"]), 2)
        self.assertEqual(
            state["passages"][0]["text"], "Refunds: Yearly plans are refunded within thirty days."
        )

        (decision,) = self.server.sent("/v1/systemone")
        self.assertEqual(decision["model"], "jev")
        self.assertEqual(
            decision["state"],
            rag_graph.gate_state(QUESTION, [p["text"] for p in state["passages"]]),
        )
        (chat,) = self.server.sent("/v1/chat/completions")
        system, human = chat["messages"]
        self.assertEqual(system["content"], rag_graph.ANSWER)
        first = "[1] Refunds: Yearly plans are refunded within thirty days."
        self.assertIn(first, human["content"])
        self.assertTrue(human["content"].endswith(f"Question: {QUESTION}"))

    def test_a_missing_fact_sends_the_graph_back_for_more(self):
        self.server.decide = gate("retrieve_more", "retrieve_more", "answer")
        state = self.graph(max_more=2).invoke({"question": QUESTION})
        self.assertEqual(state["rounds"], 3)
        texts = [p["text"] for p in state["passages"]]
        self.assertEqual(len(texts), 5)  # 2 + 2 + the last one: every passage, once
        self.assertEqual(sorted(texts), sorted(p["text"] for p in self.passages))
        similarities = [p["similarity"] for p in state["passages"]]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
        # Each gate saw every passage retrieved so far.
        held = [d["state"].count("\n[") for d in self.server.sent("/v1/systemone")]
        self.assertEqual(held, [2, 4, 5])
        self.assertEqual(state["answer"], "A draft.")

    def test_it_abstains_when_its_retrievals_run_out(self):
        self.server.decide = gate("retrieve_more")
        state = self.graph(max_more=1).invoke({"question": QUESTION})
        self.assertEqual(state["rounds"], 2)
        self.assertEqual(state["abstained"], "a fact is still missing after 2 retrievals")
        self.assertNotIn("answer", state)
        self.assertFalse(self.server.sent("/v1/chat/completions"))

    def test_it_abstains_when_every_passage_was_retrieved(self):
        self.server.decide = gate("retrieve_more")
        state = self.graph(k=3, max_more=5).invoke({"question": QUESTION})
        self.assertEqual(state["rounds"], 2)
        self.assertEqual(
            state["abstained"], "a fact is still missing, and every passage was retrieved"
        )

    def test_nothing_relevant_abstains_at_once_without_generating(self):
        self.server.decide = gate("abstain")
        state = self.graph().invoke({"question": "Where is the office in Lisbon?"})
        self.assertEqual(state["rounds"], 1)
        self.assertEqual(state["abstained"], "the passages hold nothing the answer needs")
        self.assertFalse(self.server.sent("/v1/chat/completions"))

    def test_a_zero_vector_is_not_divided_by_zero(self):
        self.assertTrue(all(math.isfinite(x) for x in rag_graph.unit([0.0, 0.0])))

    def test_a_calibrated_threshold_decides_instead_of_the_model(self):
        def state(answer, retrieve_more, abstain):
            p = {"answer": answer, "retrieve_more": retrieve_more, "abstain": abstain}
            said = {"probabilities": p, "choice": max(p, key=p.get)}
            return {"gate": said, "rounds": 1, "passages": [{}], "ranking": [0, 1]}

        decide = rag_graph.gate_decision
        # The model would abstain; P(answer) reaches the threshold: answer.
        self.assertEqual(decide(state(0.3, 0.2, 0.5), 2, 0.25)[0], "answer")
        self.assertEqual(decide(state(0.3, 0.2, 0.5), 2, None)[0], "abstain")
        # Below it, the likelier of the other two, though the model chose answer.
        self.assertEqual(decide(state(0.5, 0.3, 0.2), 2, 0.6)[0], "retrieve_more")

        self.server.decide = lambda payload: {
            "gate": choice({"answer": 0.3, "retrieve_more": 0.2, "abstain": 0.5})
        }
        answered = self.graph(answer_threshold=0.25).invoke({"question": QUESTION})
        self.assertEqual(answered["answer"], "A draft.")

    def test_the_gate_asks_what_ragbench_measured(self):
        bench = HERE.parent.parent / "bench" / "reflexbench"
        if not (bench / "rg_methods.py").exists():
            self.skipTest("not inside the EuLLM repository")
        sys.path.insert(0, str(bench))
        try:
            import rg_data
            import rg_methods
        finally:
            sys.path.remove(str(bench))
        self.assertEqual(rag_graph.GATE, rg_methods.GATE)
        self.assertEqual(rag_graph.OPTIONS, rg_methods.OPTIONS)
        passages = ["Book: written by Ann", "Ann: a writer"]
        case = rg_data.Case("g:answer", "g", "Who wrote it?", passages, "answer")
        self.assertEqual(rag_graph.gate_state("Who wrote it?", passages), rg_methods.state(case))

    def test_the_command_line(self):
        self.server.decide = gate("retrieve_more", "answer")
        out = io.StringIO()
        argv = [QUESTION, "--url", self.server.url, "--chat-model", "m", "--docs", str(self.docs)]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rag_graph.main(argv + ["--k", "2"])
        lines = [line.split()[0] for line in out.getvalue().splitlines()[2:]]
        self.assertEqual(lines, ["retrieve", "gate", "retrieve", "gate", "answer"])


# --- the client --------------------------------------------------------------


class ClientTest(WithStandIn):
    def test_an_error_says_what_the_server_said(self):
        self.server.fail = (422, {"error": "question too long"})
        with self.assertRaisesRegex(eullm_client.EuLLMError, "HTTP 422: .*question too long"):
            self.eullm.decide("state", {})

    def test_an_unreachable_server_says_to_start_it(self):
        url = self.server.url
        self.server.close()
        with self.assertRaisesRegex(eullm_client.EuLLMError, "eullm serve"):
            eullm_client.EuLLM(url, timeout=2).embed(["x"], "m")

    def test_a_server_that_stops_mid_answer_is_an_eullm_error(self):
        """The body is read after the status line, where URLError stops.

        Both graphs catch EuLLMError and nothing else, so a TimeoutError or a
        JSONDecodeError from a stalled or proxied server reaches the user as a
        traceback instead of the message the graphs write.
        """
        client = eullm_client.EuLLM(self.server.url, timeout=2)
        self.server.misbehave = Handler.STALL
        with self.assertRaisesRegex(eullm_client.EuLLMError, "no answer within 2s"):
            client.decide("state", {})
        self.server.misbehave = Handler.HTML
        with self.assertRaisesRegex(eullm_client.EuLLMError, "not JSON"):
            client.decide("state", {})
        self.server.misbehave = Handler.TRUNCATED
        with self.assertRaisesRegex(eullm_client.EuLLMError, "stopped halfway"):
            client.decide("state", {})

    def test_vectors_come_back_in_input_order(self):
        texts = ["one", "two words", "three words here"]
        self.assertEqual(self.eullm.embed(texts, "m"), [vector(t) for t in texts])


if __name__ == "__main__":
    unittest.main()
