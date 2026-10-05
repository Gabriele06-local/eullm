#!/usr/bin/env python3
"""A stand-in for the eullm binary, for the tests: `--version`, `list`, and
`serve --port N`, whose server answers /api/version, /api/generate and
/api/chat the way the engine does (streamed NDJSON, Ollama's fields).

Models it knows: every id except those starting with "missing", which get a
404 like an id the store does not have, and those starting with "huge",
refused the way --fit-strict refuses a model too large for the free VRAM.
Every answer ends with "Answer: 4", so a GSM8K-style item whose answer is 4
grades correct. FAKE_EULLM_DELAY_S sets the time per token.
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DELAY = float(os.environ.get("FAKE_EULLM_DELAY_S", "0.001"))
WORDS = ["Il", " mare", " era", " calmo", "."]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/version":
            self._json(200, {"version": "0.0.0-fake"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        model = body.get("model", "")
        if model.startswith("missing"):
            self._json(404, {"error": f"model '{model}' not found"})
            return
        if model.startswith("huge"):
            self._json(500, {"error": f"Failed to load model '{model}': --fit-strict: model "
                                      f"'{model}' does not fully fit in the currently free VRAM; "
                                      "not loading."})
            return
        n = int((body.get("options") or {}).get("num_predict") or body.get("num_predict") or 16)
        n = min(n, 32)
        chat = self.path == "/api/chat"
        prompt = body.get("prompt") or json.dumps(body.get("messages"))
        pieces = [WORDS[i % len(WORDS)] for i in range(max(n - 4, 1))] + ["\nAnswer: 4"]
        if not body.get("stream", True):
            time.sleep(DELAY * len(pieces))
            self._json(200, {"model": model, "response": "".join(pieces), "done": True,
                             "eval_count": len(pieces), "prompt_eval_count": len(prompt) // 4,
                             "load_duration": 1000000})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            data = (json.dumps(obj) + "\n").encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        started = time.time()
        for piece in pieces:
            time.sleep(DELAY)
            send({"model": model, "message": {"content": piece}} if chat
                 else {"model": model, "response": piece, "done": False})
        send({"model": model, "done": True, "eval_count": len(pieces),
              "prompt_eval_count": len(prompt) // 4,
              "eval_duration": int((time.time() - started) * 1e9),
              "prompt_eval_duration": 0, "load_duration": 2000000})
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main(argv):
    if "--version" in argv:
        print("eullm 0.0.0-fake (test)")
        return 0
    if argv and argv[0] == "list":
        print("qwen3-8b\nqwen3-4b")
        return 0
    if argv and argv[0] == "serve":
        port = int(argv[argv.index("--port") + 1])
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
