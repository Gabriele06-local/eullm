#!/usr/bin/env python3
"""A conversation of several turns with one server, and how long each turn
takes to read its prompt.

    multiturn_probe.py --url http://127.0.0.1:18600 --api eullm \
        --model qwen3.6-27b --turns 5 --label eullm

Prints one JSON line per turn: the prompt's tokens, the time the server
spent reading it (`prompt_ms`) and the wall time of the turn.

WHY. A hybrid model such as Qwen3.6 keeps a recurrent state per sequence
that cannot be cut back to an earlier token, so a server that kept the
previous turn in the slot can reuse it only if the new prompt extends it
exactly; otherwise it reads the whole conversation again, every turn. If
`prompt_ms` grows with the whole conversation, every turn is read from the
start; if it stays near the new message's share, the turn before was
reused. llama-server keeps checkpoints of the state for this
(`--ctx-checkpoints`, on by default); EuLLM has the same flag, off by
default.

The conversation opens with a long document (--doc-tokens, nominal, at 4
characters a token) and then asks one short question per turn; answers are
greedy, without thinking, and short (--answer), so the turn's time is
mostly its prompt. With --conversations 2 a second conversation, over a
document that differs from its first token, follows the first: its first
turn is read on a server already warm, where the first conversation's
also pays for the server's first request.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

PARAGRAPH = (
    "The harbour office kept a ledger of every ship that entered the bay, with the hour, "
    "the cargo, the name of the pilot and the weather at the mouth of the river. "
)
QUESTIONS = [
    "In one sentence, what does the ledger record?",
    "Who would read such a ledger, and why? Answer briefly.",
    "Name one thing the ledger leaves out.",
    "Suggest a title for the document, in five words or fewer.",
    "Summarize our conversation so far in one sentence.",
    "What would you ask the harbour master? One question.",
    "Rewrite the first question more formally.",
    "Thank you. Reply with a single word.",
]


def post(url: str, path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def turn(args, messages: list) -> tuple:
    """(answer text, prompt tokens, prompt ms) of one turn."""
    if args.api == "eullm":
        r = post(args.url, "/api/chat", {
            "model": args.model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0.0, "top_k": 1, "num_predict": args.answer},
        }, args.timeout)
        return (r["message"]["content"], r.get("prompt_eval_count"),
                round(r.get("prompt_eval_duration", 0) / 1e6, 1))
    r = post(args.url, "/v1/chat/completions", {
        "model": args.model, "messages": messages, "stream": False,
        "temperature": 0.0, "top_k": 1, "max_tokens": args.answer,
        "chat_template_kwargs": {"enable_thinking": False},
    }, args.timeout)
    t = r.get("timings") or {}
    tokens = (t.get("prompt_n") or 0) + (t.get("cache_n") or 0) or \
        (r.get("usage") or {}).get("prompt_tokens")
    return r["choices"][0]["message"]["content"], tokens, round(t.get("prompt_ms", 0), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--api", choices=("eullm", "llama"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--doc-tokens", type=int, default=6000)
    ap.add_argument("--answer", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--conversations", type=int, default=1)
    args = ap.parse_args()

    body = (PARAGRAPH * (args.doc_tokens * 4 // len(PARAGRAPH) + 1))[: args.doc_tokens * 4]
    for c in range(args.conversations):
        doc = f"Document {c + 1}. {body}"
        messages = [{"role": "user", "content": f"{doc}\n\n{QUESTIONS[0]}"}]
        for n in range(args.turns):
            t0 = time.time()
            try:
                answer, tokens, prompt_ms = turn(args, messages)
            except Exception as e:  # one failed turn ends the run, and says why
                print(json.dumps({"label": args.label, "conversation": c + 1, "turn": n + 1,
                                  "error": str(e)[:300]}))
                return 1
            wall = time.time() - t0
            print(json.dumps({"label": args.label, "conversation": c + 1, "turn": n + 1,
                              "prompt_tokens": tokens, "prompt_ms": prompt_ms,
                              "wall_s": round(wall, 2)}), flush=True)
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": QUESTIONS[(n + 1) % len(QUESTIONS)]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
