#!/usr/bin/env python3
"""Does an answer keep coming while the server reads another request's long
prompt? Roadmap item 0.7-D, checked on any machine.

    eullm serve --batch-size 2 --default-model qwen3-8b
    python3 bench/interleave_check.py --url http://127.0.0.1:11434 --model qwen3-8b

Streams one answer from /api/generate and, once its first tokens have
arrived, sends a second request with a long prompt of its own (a document of
`--prompt-tokens` tokens no server has seen) and a one-token answer. Prints
how long that second request took, which is its prompt's reading, the
longest pause between two tokens of the streamed answer while it ran, and
the answer's speed before and during it. A server that reads a prompt whole
before anything else moves stops the answer for the whole reading; one that
reads it in chunks between the answer's tokens (EuLLM with `--batch-size` 2
or more) keeps it coming, a little slower. With `--batch-size 1` the second
request waits for the first instead, which this reports too.

Standard library only.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speed_check import document  # noqa: E402

COUNT = "Count from 1 to 3000 in words, one number per line, without stopping."


def post(url, body, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def stream(url, model, tokens, timeout, arrivals, failure):
    """The streamed answer: the arrival time of each of its lines."""
    body = {
        "model": model,
        "prompt": COUNT,
        "stream": True,
        "think": False,
        "options": {"num_predict": tokens, "temperature": 0},
    }
    try:
        with post(url + "/api/generate", body, timeout) as response:
            for line in response:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    failure.append(chunk["error"])
                    return
                if chunk.get("response"):
                    arrivals.append(time.perf_counter())
                if chunk.get("done"):
                    return
    except (urllib.error.URLError, OSError, ValueError) as e:
        failure.append(str(e))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--url", default="http://127.0.0.1:11434", help="the server, Ollama API"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt-tokens", type=int, default=8000, help="the long prompt"
    )
    parser.add_argument(
        "--answer-tokens", type=int, default=1024, help="the streamed answer"
    )
    parser.add_argument(
        "--after", type=int, default=16, help="tokens streamed before the prompt"
    )
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    base = args.url.rstrip("/")

    arrivals, failure = [], []
    reader = threading.Thread(
        target=stream,
        args=(base, args.model, args.answer_tokens, args.timeout, arrivals, failure),
    )
    reader.start()
    while len(arrivals) < args.after and reader.is_alive():
        time.sleep(0.01)
    if failure or len(arrivals) < args.after:
        sys.exit(
            f"the streamed answer failed: {failure[0] if failure else 'it ended too soon'}"
        )

    text = document(args.prompt_tokens, random.randrange(10**9))
    body = {
        "model": args.model,
        "prompt": text + "\n\nReply with one word: done.",
        "stream": False,
        "think": False,
        "cache_prompt": False,
        "options": {"num_predict": 1, "temperature": 0},
    }
    started = time.perf_counter()
    try:
        with post(base + "/api/generate", body, args.timeout) as response:
            read = json.load(response).get("prompt_eval_count") or 0
    except urllib.error.HTTPError as e:
        sys.exit(
            f"the long prompt: HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
        )
    ended = time.perf_counter()
    reader.join()
    if failure:
        sys.exit(f"the streamed answer failed: {failure[0]}")

    before = [t for t in arrivals if t < started]
    during = [t for t in arrivals if started <= t <= ended]
    gaps = [
        b - a for a, b in zip(arrivals, arrivals[1:]) if b >= started and a <= ended
    ]
    rate = (len(before) - 1) / (before[-1] - before[0]) if len(before) > 1 else 0
    print(f"{base}  model {args.model}")
    print(f"long prompt:    {read} tokens read in {ended - started:.2f} s")
    print(f"answer before:  {rate:6.1f} tokens/s")
    if arrivals[-1] < ended:
        print(
            f"The answer ended before the long prompt's request did ({len(during)} tokens "
            "after it was sent): either the server took that request only once the answer "
            "was done (one slot: --batch-size 1), or the answer is too short for this "
            "reading (raise --answer-tokens)."
        )
        return 1
    print(
        f"answer during:  {len(during) / (ended - started):6.1f} tokens/s  ({len(during)} tokens)"
    )
    print(f"longest pause:  {1000 * max(gaps):6.0f} ms")
    if not during:
        print("The answer stopped while the prompt was read.")
    else:
        print("The answer kept coming while the prompt was read.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
