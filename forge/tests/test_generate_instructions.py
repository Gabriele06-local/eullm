"""Tests for the instruction-generation driver.

Generating from an evaluation set puts the exam's answers into the training
data with nothing downstream noticing, so the refusal must not depend on the
filename's capitalisation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_instructions.py"


def _load():
    spec = importlib.util.spec_from_file_location("generate_instructions", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gen_mod = _load()


def test_eval_corpus_refused_regardless_of_case(tmp_path, monkeypatch, capsys):
    for name in ("val.jsonl", "VAL.jsonl", "Val.jsonl", "cds_eval.jsonl", "CDS_eval.jsonl"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "generate_instructions.py",
                "--corpus",
                str(tmp_path / name),
                "--out",
                str(tmp_path / "out.jsonl"),
                "--limit",
                "1",
                "--dry-run",
            ],
        )
        assert gen_mod.main() == 2
        assert "evaluation set" in capsys.readouterr().err


# ── the EULLM Engine backend ─────────────────────────────────────────────
# Requests to real HTTP servers on localhost, standing in for `eullm serve`:
# what is sent (the sampling must match the transformers backend's) and
# that the servers share the work.

import json  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402


def _server(name: str, seen: list):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((name, self.path, body))
            if body["messages"][-1]["content"] == "fail":
                self.send_response(500)
                self.end_headers()
                return
            out = json.dumps({"message": {"role": "assistant", "content": f"{name}:ok"},
                              "eval_count": 7, "done": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_the_engine_backend_spreads_requests_and_samples_like_transformers():
    seen: list = []
    a, url_a = _server("a", seen)
    b, url_b = _server("b", seen)
    try:
        gen = gen_mod.EullmGenerator([url_a, url_b + "/"], "qwen3-30b", 512, concurrency=4)
        convs = [[{"role": "user", "content": f"q{i}"}] for i in range(4)]
        texts, tokens = gen.generate(convs)
    finally:
        a.shutdown()
        b.shutdown()
    assert texts == ["a:ok", "b:ok", "a:ok", "b:ok"]
    assert tokens == 28
    assert {path for _, path, _ in seen} == {"/api/chat"}
    body = seen[0][2]
    assert body["model"] == "qwen3-30b" and body["stream"] is False
    assert body["options"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                               "repeat_penalty": 1.0, "num_predict": 512}


def test_a_failed_request_is_an_empty_answer_not_a_crash():
    seen: list = []
    srv, url = _server("a", seen)
    try:
        gen = gen_mod.EullmGenerator([url], "m", 16, concurrency=2)
        texts, tokens = gen.generate([[{"role": "user", "content": "fail"}],
                                      [{"role": "user", "content": "q"}]])
    finally:
        srv.shutdown()
    assert texts == ["", "a:ok"] and tokens == 7
