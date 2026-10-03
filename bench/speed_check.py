#!/usr/bin/env python3
"""How fast an OpenAI-compatible server writes an answer and reads a prompt.

    python3 bench/speed_check.py --url http://127.0.0.1:11434/v1
    python3 bench/speed_check.py --url http://127.0.0.1:8080/v1 --prompt-tokens 32000

The two numbers an engine's README quotes, measured the same way on any server
that speaks /v1/chat/completions — EuLLM, llama-server, vLLM, or an engine
written for one model — so that two of them can be compared on one machine:

* writes answers: tokens per second of a 256-token answer to a short request,
  thinking off: a story unless `--write-prompt` asks for something else (a
  piece of code, say, which a drafting engine predicts better than prose).
  Taken from the second of two runs, so that the first one's warm-up (graphs
  captured, caches filled) does not count;
* reads a prompt: tokens per second over a long document the server has never
  seen, with a one-token answer, so that the time is the reading. A fresh
  number at the start of the document defeats any prompt cache, which would
  otherwise turn a second run into a measurement of nothing.

The token counts are the server's own, from the response's `usage`; the times
are this client's, so they include the HTTP round trip — negligible against
seconds of work, and the same for every server. Thinking is turned off both
ways a server may expect it: `"think": false` (EuLLM) and
`"reasoning_effort": "none"` (OpenAI style); a server ignores the one it does
not know. Standard library only.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

STORY = "Write a story of about 400 words about a lighthouse keeper and a storm."


def post(url, body, api_key, timeout):
    """POST `body` to `url`: the decoded response and the seconds it took."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answer = json.load(response)
    except urllib.error.HTTPError as e:
        sys.exit(f"{url}: HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
    except urllib.error.URLError as e:
        sys.exit(f"{url}: {e.reason}: is the server running?")
    return answer, time.perf_counter() - started


def first_model(base, api_key, timeout):
    """The first model `/v1/models` lists: the one a server with a single
    model loaded answers with."""
    request = urllib.request.Request(base + "/models")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            models = json.load(response).get("data") or []
    except urllib.error.URLError as e:
        sys.exit(f"{base}/models: {e.reason}: is the server running?")
    if not models:
        sys.exit(f"{base}/models lists no model: pass --model")
    return models[0]["id"]


def document(tokens, seed):
    """About `tokens` tokens of plain English prose, different for every
    `seed` from its first word on, so no server has any of it cached."""
    rng = random.Random(seed)
    subjects = ["The committee", "The engineer", "The auditor", "The council", "The team"]
    verbs = ["reviewed", "approved", "questioned", "postponed", "documented"]
    objects = ["the budget", "the schedule", "the contract", "the incident", "the report"]
    lines = [f"Reference {seed}."]
    # About 24 tokens a sentence in the Qwen tokenizer (2,665 for 111 sentences).
    for i in range(max(1, tokens // 24)):
        lines.append(
            f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} "
            f"for item {i} on day {rng.randint(1, 365)}, and noted it in the minutes."
        )
    return " ".join(lines)


def chat(base, model, content, max_tokens, args):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "think": False,
        "reasoning_effort": "none",
    }
    answer, seconds = post(base + "/chat/completions", body, args.api_key, args.timeout)
    usage = answer.get("usage") or {}
    return usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0, seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:11434/v1", help="the server's /v1")
    parser.add_argument("--model", help="model name; default: the first /v1/models lists")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=8000,
        help="length of the document read; the server's context must hold it",
    )
    parser.add_argument(
        "--write-prompt",
        default=STORY,
        help="the request whose answer is timed (default: a short story)",
    )
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    base = args.url.rstrip("/")
    model = args.model or first_model(base, args.api_key, args.timeout)
    print(f"{base}  model {model}")

    runs = [chat(base, model, args.write_prompt, 256, args) for _ in range(2)]
    _, written, seconds = runs[-1]
    first = runs[0][1] / runs[0][2] if runs[0][2] else 0
    print(
        f"writes answers: {written / seconds:6.1f} tokens/s  "
        f"({written} tokens in {seconds:.1f} s; the first run, warming up: {first:.1f})"
    )

    text = document(args.prompt_tokens, random.randrange(10**9))
    read, _, seconds = chat(base, model, text + "\n\nReply with one word: done.", 1, args)
    print(f"reads a prompt: {read / seconds:6.1f} tokens/s  ({read} tokens in {seconds:.1f} s)")


if __name__ == "__main__":
    main()
