"""A stand-in EuLLM for the tests: `POST /v1/systemone`, `POST
/v1/embeddings`, `GET /v1/models` and `GET /api/version` on 127.0.0.1, with
the shapes docs/engine.md gives them, errors included. Standard library
only; it records every request it is sent."""

import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "Jev-Style-0.8B-Decision-v3-Q4_K_M"

# EuLLM's refusal of a question too long for a Jev-Style model's budget.
TOO_LONG = (
    "question, options and readout need 2100 tokens; Jev-Style-0.8B-Decision-v3 allows 2048 "
    "— nothing was truncated: shorten the question or the options, or split the options over "
    "several questions"
)


class StandIn:
    """`scores` gives each option its verdict score, by option name or by
    (question id, option name), 0 when not given; a `noul` reads its P(yes)
    from `scores` by its instructions. A question with more than
    `max_options` options is refused as too long. `vectors` gives the
    embedding of a text, `DEFAULT_VECTOR` when not given."""

    DEFAULT_VECTOR = [0.0, 0.0, 1.0]

    def __init__(
        self,
        scores=None,
        vectors=None,
        max_options=None,
        temperature=0.88,
        api_key=None,
        decision_model=MODEL,
        embed_models=("qwen3-embedding",),
        readout="verdict",
        delay=0.0,
    ):
        self.scores, self.vectors = scores or {}, vectors or {}
        self.max_options, self.temperature = max_options, temperature
        self.api_key, self.decision_model = api_key, decision_model
        self.embed_models, self.readout, self.delay = embed_models, readout, delay
        self.requests = []  # (method, path, headers, body), as received

    def __enter__(self):
        standin = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.answer("GET")

            def do_POST(self):
                self.answer("POST")

            def answer(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                standin.requests.append((method, self.path, dict(self.headers), body))
                status, reply = standin.route(method, self.path, dict(self.headers), body)
                if standin.delay:
                    time.sleep(standin.delay)
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        class Server(ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                # A client that gave up waiting (the timeout test) is not an
                # error of the stand-in's; anything else is printed.
                if not isinstance(sys.exc_info()[1], ConnectionError):
                    super().handle_error(request, client_address)

        self.server = Server(("127.0.0.1", 0), Handler)
        # shutdown() waits for the loop to look again: 0.5 s by default, a
        # stall in every test, where it is called from the event loop.
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def sent(self, path):
        """The bodies of the requests made to `path`, in order."""
        return [body for _, p, _, body in self.requests if p == path]

    # --- What EuLLM answers ---------------------------------------------------

    def route(self, method, path, headers, body):
        if self.api_key and headers.get("Authorization") != f"Bearer {self.api_key}":
            return 401, error("unauthorized", "missing or invalid API key")
        if (method, path) == ("POST", "/v1/systemone"):
            return self.systemone(body)
        if (method, path) == ("POST", "/v1/embeddings"):
            return self.embeddings(body)
        if (method, path) == ("GET", "/v1/models"):
            return 200, self.models()
        if (method, path) == ("GET", "/api/version"):
            return 200, {"version": "0.7.22", "api_port": 11434, "model_swaps": 0}
        return 404, {"error": "not found"}

    def score(self, question, option):
        return self.scores.get((question, option), self.scores.get(option, 0.0))

    def systemone(self, body):
        if not self.decision_model:
            return 400, error(
                "model_not_loaded",
                "no decision model is loaded: start the server with --decision-model",
            )
        answers = {}
        for qid, question in body["questions"].items():
            kind = question.get("type")
            if kind == "noul":
                p = self.scores.get(question["instructions"], 0.5)
                answers[qid] = {"type": "noul", "noul": p, "eullm": {"scores": {"yes": 0.0}}}
            elif kind == "choice":
                options = list(question["criteria"])
                if len(options) < 2:
                    return 422, error(
                        "invalid_question", "a choice question needs 2 to 255 options", qid
                    )
                if self.max_options and len(options) > self.max_options:
                    return 422, error("input_budget_exceeded", TOO_LONG, qid)
                scores = {o: self.score(qid, o) for o in options}
                probabilities = softmax(scores, self.temperature)
                best = max(options, key=probabilities.get)
                k = len(options)
                extra = {"scores": scores, "raw_probabilities": softmax(scores, 1.0)}
                if self.readout == "codes":
                    extra = {"logprobs": {o: math.log(p) for o, p in probabilities.items()}}
                answers[qid] = {
                    "type": "choice",
                    "choice": best,
                    "probabilities": probabilities,
                    "confidence": max(0.0, (k * probabilities[best] - 1) / (k - 1)),
                    "eullm": extra,
                }
            else:
                return 422, error("invalid_question", f"unknown question type {kind!r}", qid)
        return 200, {
            "model": self.decision_model,
            "answers": answers,
            "usage": {"input_tokens": 42, "output_tokens": 0},
            "timing": {"total_ms": 12.5},
            "eullm": {
                "readout": self.readout,
                "mode": "shared_prefix",
                "temperature": self.temperature,
                "evaluated_tokens": 42,
                "request_ms": 12.5,
            },
        }

    def embeddings(self, body):
        model = body.get("model")
        if model not in self.embed_models:
            return 404, {"error": f"model {model!r} not found; pull it first"}
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        data = [
            {
                "object": "embedding",
                "embedding": self.vectors.get(t, self.DEFAULT_VECTOR),
                "index": i,
            }
            for i, t in enumerate(inputs)
        ]
        # Out of order on purpose: the index says which input a vector is.
        data.reverse()
        return 200, {"object": "list", "data": data, "model": model, "usage": {}}

    def models(self):
        data = [{"id": "qwen3-embedding", "object": "model", "owned_by": "eullm"}]
        models = []
        if self.decision_model:
            data.append(
                {
                    "id": self.decision_model,
                    "object": "model",
                    "owned_by": "eullm",
                    "context_tokens": 8192,
                    "head_max_tokens": 2048,
                    "eullm": {"slot": "decision", "readout": self.readout},
                }
            )
            models.append({"name": self.decision_model, "description": "", "release_date": ""})
        return {"object": "list", "data": data, "models": models}


def error(code, message, question=None):
    detail = {"code": code, "message": message}
    if question:
        detail["question"] = question
    return {"error": detail}


def softmax(scores, temperature):
    top = max(scores.values())
    weights = {k: math.exp((v - top) / temperature) for k, v in scores.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}
