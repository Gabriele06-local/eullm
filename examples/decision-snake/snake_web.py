#!/usr/bin/env python3
"""Snake decided by EuLLM's `/v1/systemone`, shown in a browser.

The same game as `snake.py` — the same facts, the same model question, the
same rules about what code decides — drawn on a page: the board, the options
the model was shown at every move with the probability it gave each, how
long the decision took, and what the game has scored so far. The page is
served from your own machine and the game talks to your own EuLLM server;
nothing else is involved.

    eullm serve --decision-model jev-style-0.8b-decision-v3-gguf-q4_k_m
    python examples/decision-snake/snake_web.py
    # then open http://127.0.0.1:8765

Buttons on the page start and pause the game, start a new one, switch
between the model and the plain rule, and set the speed. On a CPU, where a
decision takes a second or two, tick "wait for every answer". Only the
Python standard library is needed.

On a server without a screen, forward the port over SSH and open the page
on your own computer, or serve it on every address with `--host 0.0.0.0`:

    ssh -L 8765:127.0.0.1:8765 you@server   # then, on the server:
    python examples/decision-snake/snake_web.py --no-browser
"""

import argparse
import collections
import http.server
import json
import os
import pathlib
import queue
import random
import statistics
import sys
import threading
import time
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snake  # noqa: E402  (the game next to this file)

PAGE = pathlib.Path(__file__).with_name("snake.html")


class Show:
    """The game running in its own thread, and the pages watching it."""

    def __init__(self, args):
        self.args = args
        self.client = snake.Client(
            args.url, args.model, args.api_key, snake.INSTRUCTIONS, args.timeout
        )
        self.lock = threading.Lock()
        self.watchers = []
        self.rng = random.Random(args.seed)
        self.running = True
        self.player = "model"
        self.tick = args.tick
        self.wait = args.wait
        self.decider = snake.Decider(
            self.client, self.player, 0.0 if self.wait else self.tick
        )
        self.scores = []
        self.error = None
        self.new_game()
        self.frame = self.snapshot(None)

    def new_game(self):
        self.game = snake.Game(self.args.width, self.args.height, self.rng)
        self.moves = 0
        self.idle = 0
        self.end = None
        self.counts = collections.Counter()
        self.latencies = []
        self.model_name = None

    def control(self, request):
        with self.lock:
            action = request.get("action")
            if action == "start":
                self.running, self.error = True, None
            elif action == "pause":
                self.running = False
            elif action == "new":
                self.new_game()
                self.running, self.error = True, None
            if request.get("player") in ("model", "rule"):
                self.player = request["player"]
            if isinstance(request.get("tick_ms"), (int, float)):
                self.tick = min(2.0, max(0.02, request["tick_ms"] / 1000.0))
            if isinstance(request.get("wait"), bool):
                self.wait = request["wait"]
            self.decider.player = self.player
            self.decider.tick = 0.0 if self.wait else self.tick
            self.publish(self.snapshot(None))

    def snapshot(self, decision):
        game = self.game
        frame = {
            "width": game.width,
            "height": game.height,
            "snake": game.snake,
            "food": game.food,
            "heading": game.heading,
            "score": game.score,
            "length": len(game.snake),
            "moves": self.moves,
            "games": len(self.scores) + 1,
            "scores": self.scores[-30:],
            "running": self.running,
            "player": self.player,
            "tick_ms": round(self.tick * 1000),
            "wait": self.wait,
            "end": self.end,
            "error": self.error,
            "model": self.model_name,
            "question": self.client.instructions,
            "stats": {
                "model": self.counts["model"],
                "best": self.counts["best"],
                "late": self.counts["late"],
                "forced": self.counts["forced"],
                "one_way": self.counts["one way"],
                "tail": self.counts["tail"],
                "no_safe": self.counts["no safe move"],
                "median_ms": round(statistics.median(self.latencies), 1)
                if self.latencies
                else None,
            },
            "decision": None,
        }
        if decision is not None:
            answer = decision.answer or {}
            frame["decision"] = {
                "move": decision.move,
                "how": decision.how,
                "state": decision.state,
                "options": snake.describe(decision.facts),
                "best": sorted(snake.best(decision.facts)),
                "probabilities": answer.get("probabilities"),
                "confidence": answer.get("confidence"),
                "ms": round(answer["ms"], 1) if "ms" in answer else None,
            }
        return frame

    def publish(self, frame):
        self.frame = frame
        for watcher in list(self.watchers):
            # Never block the game on a page: a full queue loses its oldest.
            try:
                watcher.put_nowait(frame)
            except queue.Full:
                try:
                    watcher.get_nowait()
                    watcher.put_nowait(frame)
                except (queue.Empty, queue.Full):
                    pass

    def run(self):
        while True:
            with self.lock:
                running, over = self.running and not self.error, self.end is not None
            if not running:
                time.sleep(0.05)
                continue
            if over:
                time.sleep(2.0)  # the final board stays on screen a moment
                with self.lock:
                    if self.end is not None and self.running:
                        self.new_game()
                        self.publish(self.snapshot(None))
                continue
            started = time.perf_counter()
            game = self.game
            try:
                decision = self.decider.decide(game)
            except (SystemExit, Exception) as e:
                # The server could not be reached, or answered something
                # else: the page says so, and Start tries again.
                with self.lock:
                    self.error = str(e) or type(e).__name__
                    self.publish(self.snapshot(None))
                continue
            with self.lock:
                # A new game started while the model was thinking: this
                # decision was for the old board.
                if self.game is game:
                    self.step(decision)
            pause = self.tick - (time.perf_counter() - started)
            if pause > 0:
                time.sleep(pause)

    def step(self, decision):
        game = self.game
        if decision is None:
            self.finish("no move left")
            self.publish(self.snapshot(None))
            return
        self.counts[decision.how] += 1
        if decision.how == "model":
            self.latencies.append(decision.answer["ms"])
            self.model_name = decision.answer.get("model") or self.model_name
            if decision.move in snake.best(decision.facts):
                self.counts["best"] += 1
        before = game.score
        crashed = not game.play(decision.move)
        self.moves += 1
        self.idle = 0 if game.score > before else self.idle + 1
        if crashed:
            self.finish(f"crashed moving {decision.move}")
        elif game.food is None:
            self.finish("the board is full: won")
        elif self.idle > game.width * game.height * 2:
            self.finish(f"stopped: {self.idle} moves without eating")
        self.publish(self.snapshot(decision))

    def finish(self, why):
        self.end = why
        self.scores.append(self.game.score)


def handler(show):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # quiet: the game prints nothing per request

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/events":
                self.stream()
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/control":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self.send_error(400)
                return
            show.control(request if isinstance(request, dict) else {})
            self.send_response(204)
            self.end_headers()

        def stream(self):
            """Server-sent events: one frame per move."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            watcher = queue.Queue(maxsize=64)
            with show.lock:
                show.watchers.append(watcher)
                watcher.put(show.frame)
            try:
                while True:
                    try:
                        frame = watcher.get(timeout=15)
                        # A slow page skips frames rather than falling behind.
                        while not watcher.empty():
                            frame = watcher.get_nowait()
                        data = "data: " + json.dumps(frame) + "\n\n"
                    except queue.Empty:
                        data = ": still here\n\n"
                    self.wfile.write(data.encode())
                    self.wfile.flush()
            except OSError:  # the page was closed
                pass
            finally:
                with show.lock:
                    show.watchers.remove(watcher)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--url", default="http://localhost:11434", help="EuLLM server URL"
    )
    parser.add_argument(
        "--model", default=None, help="decision model (default: the one loaded)"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EULLM_API_KEY"),
        help="API key, when the server requires one (default: $EULLM_API_KEY)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="address the page is served on; 0.0.0.0 to open it from another machine",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument(
        "--tick", type=float, default=0.15, help="seconds per move (default 0.15)"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="wait for every answer, for a CPU (also a checkbox on the page)",
    )
    parser.add_argument("--seed", type=int, default=None, help="food placement seed")
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="per-request timeout, seconds"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser window"
    )
    args = parser.parse_args()
    if args.width < 5 or args.height < 5:
        parser.error("the board must be at least 5x5")

    show = Show(args)
    threading.Thread(target=show.run, daemon=True).start()
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler(show))
    server.daemon_threads = True
    local = args.host in ("0.0.0.0", "::", "")
    address = f"http://{'127.0.0.1' if local else args.host}:{args.port}"
    print(f"Snake on {address} — decisions from {args.url}. Ctrl+C to stop.")
    if local:
        print(
            f"Served on every address: from another machine, open this one's on port {args.port}."
        )
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        show.decider.close()


if __name__ == "__main__":
    main()
