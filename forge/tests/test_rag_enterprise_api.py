"""rag_enterprise_eval.Api: an expired token logs in again instead of ending the run."""

from __future__ import annotations

import http.server
import importlib.util
import json
import threading
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rag_enterprise_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("rag_enterprise_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _serve(valid: set[str], logins: list[str]):
    """A RAG Enterprise that accepts only the tokens in `valid`, issuing t1, t2, ..."""
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/api/auth/login":
                logins.append(f"t{len(logins) + 1}")
                return self._send(200, {"access_token": logins[-1]})
            if self.headers.get("Authorization", "").removeprefix("Bearer ") in valid:
                return self._send(200, {"answer": "ok", "sources": []})
            self._send(401, {"error": "invalid or expired token"})

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_an_expired_token_logs_in_again_and_the_query_is_answered():
    mod = _load()
    logins: list[str] = []
    valid = {"t1"}
    srv = _serve(valid, logins)
    try:
        api = mod.Api(f"http://127.0.0.1:{srv.server_port}")
        api.login("admin", "pw")
        assert api.query("q", None)["answer"] == "ok"
        valid.clear()
        valid.add("t2")            # t1 expires; the next login issues t2
        assert api.query("q", None)["answer"] == "ok"
        assert logins == ["t1", "t2"] and api.token == "t2"
    finally:
        srv.shutdown()


def test_a_401_that_a_new_login_does_not_cure_still_stops_the_run():
    mod = _load()
    logins: list[str] = []
    srv = _serve(set(), logins)
    try:
        api = mod.Api(f"http://127.0.0.1:{srv.server_port}")
        api.login("admin", "pw")
        with pytest.raises(SystemExit, match="HTTP 401"):
            api.query("q", None)
        assert logins == ["t1", "t2"]   # one new login, not a loop
    finally:
        srv.shutdown()
