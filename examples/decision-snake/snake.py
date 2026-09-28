#!/usr/bin/env python3
"""Snake played by a decision model through EuLLM's `POST /v1/systemone`.

Code does what code is good at: it lists the moves that do not crash
straight away and computes exact facts about each one — how far the food is
afterwards (shortest path), how much room is left (flood fill), whether the
snake could still reach its own tail, which is what keeps it alive. The
model does what it is good at: it reads those facts and picks one move,
with a single `choice` question per step. Nothing is generated. And what
code knows for certain, code decides: the model chooses among the safe
moves only; with a single safe move, or none, there is nothing to judge,
and code plays the one there is, or the one with the most room.

The game never waits for the model. Every tick starts a request and plays
whatever answer has arrived when the tick ends; a late answer is dropped
and a plain rule plays that tick instead. `--tick 0` waits for every
answer — a turn-based game, for a CPU, where one decision takes about a
second — and is the default with `--headless`.

`snake_web.py` next to this file shows the same game in a browser, with
every option the model was shown and the probability it gave each.

Start the server with a decision model — a Jev-Style model is the one
trained for this — then run the game:

    eullm pull hf.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF:Q4_K_M
    eullm serve --decision-model jev-style-0.8b-decision-v3-gguf-q4_k_m
    python examples/decision-snake/snake.py

    # no display, ten games, then the plain rule alone on the same boards
    python examples/decision-snake/snake.py --headless --games 10 --seed 1
    python examples/decision-snake/snake.py --headless --games 10 --seed 1 --player rule

Only the Python standard library is needed.
"""

import argparse
import collections
import concurrent.futures
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

INSTRUCTIONS = (
    "Choose the snake's next move. Never choose a dead end or a risky move "
    "while a safe one exists. Among the safe moves, choose the one that "
    "reaches the food soonest."
)


class Game:
    """The board, the snake (head first) and the food."""

    def __init__(self, width, height, rng):
        self.width, self.height, self.rng = width, height, rng
        y = height // 2
        self.snake = [(3, y), (2, y), (1, y)]
        self.heading = "right"
        self.score = 0
        self.food = None
        self.place_food()

    def inside(self, cell):
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def place_food(self):
        body = set(self.snake)
        free = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in body
        ]
        # No free cell left: the snake fills the board and has won.
        self.food = self.rng.choice(free) if free else None

    def after(self, move):
        """The snake after `move`, or None when the move crashes."""
        dx, dy = MOVES[move]
        hx, hy = self.snake[0]
        head = (hx + dx, hy + dy)
        # The tail moves out of the way unless the snake grows.
        body = self.snake if head == self.food else self.snake[:-1]
        if not self.inside(head) or head in body:
            return None
        return [head] + body

    def play(self, move):
        """Make `move`; False when it crashes."""
        snake = self.after(move)
        if snake is None:
            return False
        eats = snake[0] == self.food
        self.snake, self.heading = snake, move
        if eats:
            self.score += 1
            self.place_food()
        return True


def flood(game, snake):
    """Every cell the head of `snake` can reach, with its distance, when its
    body blocks the way. The tail does not: it moves on as the snake does."""
    blocked = set(snake[1:-1])
    start = snake[0]
    seen = {start: 0}
    queue = collections.deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in MOVES.values():
            cell = (x + dx, y + dy)
            if cell in seen or cell in blocked or not game.inside(cell):
                continue
            seen[cell] = seen[(x, y)] + 1
            queue.append(cell)
    return seen


def options(game):
    """Every move that does not crash at once, with the facts about it."""
    found = {}
    for move in MOVES:
        snake = game.after(move)
        if snake is None:
            continue
        reach = flood(game, snake)
        eats = snake[0] == game.food
        found[move] = {
            "eats": eats,
            "food_steps": 0 if eats else reach.get(game.food),
            "room": len(reach),  # the head's own cell included
            "tail": snake[-1] in reach,
            "length": len(snake),
        }
    return found


def danger(fact):
    """Why a move could kill the snake later, or None when it is safe."""
    if fact["room"] < fact["length"]:
        cells = "cell" if fact["room"] == 1 else "cells"
        return f"dead end: only {fact['room']} {cells} of room for a snake of {fact['length']}"
    if not fact["tail"]:
        return f"risky: the tail would be out of reach, {fact['room']} cells of room"
    return None


def describe(facts):
    """Every option's facts as the model reads them: a danger when there is
    one, else the steps to the food and the room left."""
    texts = {}
    for move, fact in facts.items():
        warning = danger(fact)
        if warning:
            texts[move] = warning
        elif fact["eats"]:
            texts[move] = "eats the food now"
        elif fact["food_steps"] is None:
            texts[move] = (
                f"the food cannot be reached from there; {fact['room']} cells of room"
            )
        else:
            # "1 steps", not "1 step": with the same word after every count
            # the options differ in the number alone, and on the same boards
            # the Jev-Style 0.8B put 0.74 on the best move against 0.66.
            texts[move] = (
                f"food {fact['food_steps']} steps away; {fact['room']} cells of room; "
                "the tail stays within reach"
            )
    return texts


def value(fact):
    """How good a move is, as far as the facts can tell. A safe move: the
    food reachable first, then the fewest steps to it, or with the food out
    of reach the most room. A dangerous one: room enough for the body first,
    then the most room, where the tail has the longest to free the way."""
    if danger(fact) is None:
        steps = fact["food_steps"]
        reachable = steps is not None
        return (True, reachable, -steps if reachable else fact["room"])
    return (False, fact["room"] >= fact["length"], fact["room"])


def best(facts):
    """The moves the facts cannot tell apart from the best one."""
    top = max(value(f) for f in facts.values())
    return {m for m, f in facts.items() if value(f) == top}


def rule(game, facts):
    """The baseline, and what plays when an answer is late: the best move by
    the facts, straight on among equals, then the one with the most room."""
    return max(
        facts,
        key=lambda m: (value(facts[m]), m == game.heading, facts[m]["room"]),
    )


def state_text(game):
    """Where the food is, and nothing about the way the snake is heading:
    named in the state, the heading pulled the Jev-Style 0.8B towards
    carrying straight on even when the facts said otherwise. Without it, it
    picked the best move on 40 boards out of 40."""
    hx, hy = game.snake[0]
    fx, fy = game.food
    parts = []
    if fy != hy:
        parts.append(f"{abs(fy - hy)} {'up' if fy < hy else 'down'}")
    if fx != hx:
        parts.append(f"{abs(fx - hx)} {'left' if fx < hx else 'right'}")
    return (
        f"A game of Snake on a {game.width}x{game.height} board. The snake is "
        f"{len(game.snake)} cells long. The food is {' and '.join(parts)} of the head."
    )


class Client:
    """`POST /v1/systemone`, one choice question per call."""

    def __init__(self, url, model, api_key, instructions, timeout):
        self.url = url.rstrip("/") + "/v1/systemone"
        self.model, self.api_key = model, api_key
        self.instructions, self.timeout = instructions, timeout

    def decide(self, state, criteria):
        """The model's pick among `criteria` (move -> its facts)."""
        payload = {
            "state": state,
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": self.instructions,
                    "criteria": criteria,
                }
            },
        }
        if self.model:
            payload["model"] = self.model
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers=headers
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise SystemExit(f"{self.url}: HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise SystemExit(
                f"{self.url}: {e.reason} — is `eullm serve` running?"
            ) from None
        answer = body["answers"]["move"]
        return {
            "move": answer["choice"],
            "probabilities": answer["probabilities"],
            "confidence": answer["confidence"],
            "ms": (time.perf_counter() - started) * 1000.0,
            "model": body.get("model"),
        }


class Decision:
    """One move: which, how it was chosen (`model`, `late`, `rule`,
    `forced`, `no safe move`), the options it was chosen from and, when the
    model chose, its answer and the state it read."""

    def __init__(self, move, how, facts, answer=None, state=None):
        self.move, self.how, self.facts = move, how, facts
        self.answer, self.state = answer, state

    def status(self):
        if self.how == "model":
            a = self.answer
            shares = ", ".join(f"{m} {p:.2f}" for m, p in a["probabilities"].items())
            return (
                f"model: {self.move}  ({shares}; confidence {a['confidence']:.2f}; "
                f"{a['ms']:.0f} ms)"
            )
        if self.how == "late":
            return f"answer late: the rule played {self.move}"
        return f"{self.how}: {self.move}"


class Decider:
    """Picks every move. The model chooses among the safe moves. Code plays
    what needs no judgement — the only safe move, or with none the move with
    the most room — and the rule plays a tick whose answer is late, or every
    move with `player="rule"`.

    One request at a time: while an answer is still on its way, the rule
    plays and the server is not asked again."""

    def __init__(self, client, player="model", tick=0.0):
        self.client, self.player, self.tick = client, player, tick
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.pending = None

    def decide(self, game):
        """The next move; None when every move crashes."""
        facts = options(game)
        if not facts:
            return None
        safe_moves = {m: f for m, f in facts.items() if danger(f) is None}
        started = time.perf_counter()
        # What is left when no move is safe is a comparison of room, which
        # code does exactly: offered a dead end next to a risky move with
        # ten times the room, the Jev-Style 0.8B took the dead end, and
        # those moves ended its games.
        if not safe_moves:
            return Decision(rule(game, facts), "no safe move", facts)
        if len(safe_moves) == 1:
            return Decision(next(iter(safe_moves)), "forced", safe_moves)
        if self.player == "rule":
            return Decision(rule(game, safe_moves), "rule", safe_moves)
        state = state_text(game)
        asked = None
        if self.pending is None or self.pending.done():
            self.pending = asked = self.pool.submit(
                self.client.decide, state, describe(safe_moves)
            )
        answer = None
        if asked is not None:
            wait = None
            if self.tick > 0:
                wait = max(0.0, self.tick - (time.perf_counter() - started))
            try:
                answer = asked.result(timeout=wait)
            except concurrent.futures.TimeoutError:
                pass
        if answer is not None and answer["move"] in safe_moves:
            return Decision(answer["move"], "model", safe_moves, answer, state)
        return Decision(rule(game, safe_moves), "late", safe_moves)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


class Screen:
    """The board, redrawn in place with ANSI escape codes."""

    def __init__(self, color):
        self.color = color
        if os.name == "nt":
            os.system("")  # turns on escape codes in the Windows console
        sys.stdout.write("\x1b[2J\x1b[?25l")  # clear once, hide the cursor

    def paint(self, text, code):
        return f"\x1b[{code}m{text}\x1b[0m" if self.color else text

    def draw(self, game, lines):
        cells = {cell: self.paint("o", "32") for cell in game.snake[1:]}
        cells[game.snake[0]] = self.paint("@", "1;32")
        if game.food:
            cells[game.food] = self.paint("*", "1;31")
        border = "+" + "-" * (game.width * 2) + "+"
        rows = [border]
        for y in range(game.height):
            row = "".join(cells.get((x, y), " ") + " " for x in range(game.width))
            rows.append("|" + row + "|")
        rows.append(border)
        out = "\x1b[H" + "".join(row + "\x1b[K\n" for row in rows + lines) + "\x1b[J"
        sys.stdout.write(out)
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\x1b[?25h")  # the cursor back
        sys.stdout.flush()


def play_game(args, decider, rng, screen, log):
    """One game; returns its record. With `log`, every move the model made
    goes to it as one JSON line."""
    game = Game(args.width, args.height, rng)
    counts = collections.Counter()
    latencies = []
    idle = 0
    end = "move limit reached"
    if screen:
        screen.draw(game, [f"score 0   length {len(game.snake)}", "starting"])
    for tick in range(args.max_ticks):
        if game.food is None:
            end = "the board is full: won"
            break
        started = time.perf_counter()
        decision = decider.decide(game)
        if decision is None:
            end = "no move left"
            break
        counts[decision.how] += 1
        if decision.how == "model":
            latencies.append(decision.answer["ms"])
            ideal = best(decision.facts)
            if decision.move in ideal:
                counts["best"] += 1
            if log:
                entry = {
                    "state": decision.state,
                    "options": describe(decision.facts),
                    "model": decision.move,
                    "best": sorted(ideal),
                    "probabilities": decision.answer["probabilities"],
                    "confidence": decision.answer["confidence"],
                    "ms": round(decision.answer["ms"], 1),
                }
                log.write(json.dumps(entry) + "\n")
        before = game.score
        if not game.play(decision.move):
            end = f"crashed moving {decision.move}"
            break
        idle = 0 if game.score > before else idle + 1
        if screen:
            lines = [
                f"score {game.score}   length {len(game.snake)}   move {tick + 1}",
                decision.status(),
            ]
            screen.draw(game, lines)
        if args.tick > 0:
            time.sleep(max(0.0, args.tick - (time.perf_counter() - started)))
        if idle > args.width * args.height * 2:
            end = "going round in circles: stopped"
            break
    return {
        "score": game.score,
        "length": len(game.snake),
        "end": end,
        "counts": counts,
        "latencies": latencies,
    }


def summary(n, result, player):
    c = result["counts"]
    line = (
        f"game {n}: score {result['score']}, length {result['length']}, {result['end']}"
    )
    if player == "model" and c["model"] + c["late"]:
        lat = result["latencies"]
        line += f"; {c['model']} moves by the model"
        if lat:
            line += f" (median {statistics.median(lat):.0f} ms)"
        line += (
            f", {c['best']} of them among the best by the facts; "
            f"{c['late']} late, {c['forced']} forced, "
            f"{c['no safe move']} with no safe move"
        )
    return line


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
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument(
        "--tick",
        type=float,
        default=None,
        help="seconds per move; 0 waits for every answer (default: 0.15, 0 with --headless)",
    )
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument(
        "--max-ticks", type=int, default=2000, help="moves per game at most"
    )
    parser.add_argument("--seed", type=int, default=None, help="food placement seed")
    parser.add_argument(
        "--player",
        choices=["model", "rule"],
        default="model",
        help="who picks the moves: the model, or the plain rule alone (the baseline)",
    )
    parser.add_argument(
        "--instructions", default=INSTRUCTIONS, help="the question asked each move"
    )
    parser.add_argument(
        "--headless", action="store_true", help="no display; print a summary"
    )
    parser.add_argument("--no-color", dest="color", action="store_false")
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="per-request timeout, seconds"
    )
    parser.add_argument(
        "--log", default=None, help="write every move the model made to this JSONL file"
    )
    args = parser.parse_args()
    if args.width < 5 or args.height < 5:
        parser.error("the board must be at least 5x5")
    if args.tick is None:
        args.tick = 0.0 if args.headless else 0.15

    client = Client(args.url, args.model, args.api_key, args.instructions, args.timeout)
    rng = random.Random(args.seed)
    screen = None if args.headless else Screen(args.color)
    log = open(args.log, "w", encoding="utf-8") if args.log else None
    decider = Decider(client, args.player, args.tick)
    results = []
    try:
        for n in range(1, args.games + 1):
            results.append(play_game(args, decider, rng, screen, log))
            if screen:
                screen.close()
            print(summary(n, results[-1], args.player))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        decider.close()
        if screen:
            screen.close()
        if log:
            log.close()
    if len(results) > 1:
        scores = [r["score"] for r in results]
        print(
            f"{len(results)} games: mean score {statistics.mean(scores):.1f}, "
            f"best {max(scores)}, worst {min(scores)}"
        )


if __name__ == "__main__":
    main()
