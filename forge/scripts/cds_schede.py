#!/usr/bin/env python3
"""One card per Consiglio di Stato ruling, written by a teacher model.

    python forge/scripts/cds_schede.py \\
        --chunks $WORK/datasets/legal_it_amm/train.jsonl \\
        --openga $WORK/eval/openga/cds-sentenze-20*.csv \\
        --ids $WORK/eval/cds/cds-train-ids.txt \\
        --serve $WORK/gguf/qwen3-30b-a3b-instruct-2507/qwen3-30b-a3b-instruct-2507-q8_0.gguf \\
        --gpus 0,1,2 --out $WORK/eullm_runs/cds/schede.jsonl

Step 1 of the case-law plan (research report of 2026-10-05); see
`eullm_forge.caselaw.schede` for what a card holds and why. With
``--serve`` one llama-server per GPU is started on the teacher GGUF and
the rulings are spread over them; with ``--url`` the servers are already
running.

Resumable and safe to chain: a ruling already carded, or refused, is not
asked again, and ``--stop-after`` stops taking new rulings in time for a
2-hour link to end on its own. Refusals go to ``<out>.rejects.jsonl`` with
their reason and no text.

Every 100 cards it prints throughput (cards/min, prompt and generated
tokens per second): the first link of a chain is the throughput measure
step 0 asks for. It never prints a ruling, a card or a question.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.caselaw import attach_meta, load_openga, load_rulings, ruling_view  # noqa: E402
from eullm_forge.caselaw.schede import CardRejected, messages, parse_card  # noqa: E402


def ask(url: str, msgs: list[dict], max_tokens: int, temperature: float,
        timeout: float = 900.0) -> dict:
    body = {"messages": msgs, "temperature": temperature, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def done_ids(*paths: Path) -> set[str]:
    out: set[str] = set()
    for p in paths:
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.add(json.loads(line)["id"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", nargs="+", type=Path, required=True)
    ap.add_argument("--openga", nargs="*", type=Path, default=[])
    ap.add_argument("--ids", type=Path, required=True,
                    help="the rulings to card, one id per line (cds_split.py output)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--url", action="append", default=[],
                    help="a running OpenAI-compatible server (repeatable)")
    ap.add_argument("--serve", help="teacher GGUF: start one llama-server per --gpus entry")
    ap.add_argument("--gpus", default="0", help="with --serve: GPUs, one server each")
    ap.add_argument("--llama-server", help="llama-server binary (default as in legal_eval.py)")
    ap.add_argument("--parallel", type=int, default=8, help="requests in flight per server")
    ap.add_argument("--ctx", type=int, default=16384, help="context per request (tokens)")
    ap.add_argument("--max-chars", type=int, default=24000, help="ruling text shown (chars)")
    ap.add_argument("--max-tokens", type=int, default=1500, help="card length limit (tokens)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--teacher", default="", help="name recorded in every card")
    ap.add_argument("--limit", type=int, default=0, help="card at most this many (0: all)")
    ap.add_argument("--stop-after", type=float, default=0,
                    help="seconds after which no new ruling is started (0: never)")
    args = ap.parse_args(argv)
    if bool(args.url) == bool(args.serve):
        ap.error("give either --url or --serve")

    wanted = [line.strip() for line in args.ids.open(encoding="utf-8") if line.strip()]
    rejects_path = args.out.with_name(args.out.name.replace(".jsonl", "") + ".rejects.jsonl")
    skip = done_ids(args.out, rejects_path)
    todo_ids = [i for i in wanted if i not in skip]
    if args.limit:
        todo_ids = todo_ids[:args.limit]
    print(f"[schede] {len(wanted):,} rulings asked, {len(skip & set(wanted)):,} done already, "
          f"{len(todo_ids):,} to do", flush=True)
    if not todo_ids:
        print("[schede] nothing left to do", flush=True)
        return 0
    rulings = load_rulings(args.chunks)
    if args.openga:
        attach_meta(rulings, load_openga(args.openga))
    missing = [i for i in todo_ids if i not in rulings]
    if missing:
        print(f"[schede] {len(missing):,} ids not in the chunk files, skipped", file=sys.stderr)
    todo = [rulings[i] for i in todo_ids if i in rulings]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    stats = Counter()
    lock = threading.Lock()

    with contextlib.ExitStack() as stack:
        urls = list(args.url)
        if args.serve:
            from eullm_forge.eval.gguf import LlamaServer
            for gpu in args.gpus.split(","):
                srv = LlamaServer(args.serve, binary=args.llama_server, parallel=args.parallel,
                                  ctx_per_slot=args.ctx, devices=gpu,
                                  log_path=args.out.with_name(f"server-gpu{gpu}.log"))
                urls.append(stack.enter_context(srv).url)
            print(f"[schede] {len(urls)} teacher server(s) up after "
                  f"{time.monotonic() - t0:.0f} s", flush=True)
        teacher = args.teacher or Path(args.serve or "").stem or "teacher"
        out = stack.enter_context(args.out.open("a", encoding="utf-8"))
        rej = stack.enter_context(rejects_path.open("a", encoding="utf-8"))
        t_start = time.monotonic()

        def work(n: int, r):
            url = urls[n % len(urls)]
            msgs = messages(ruling_view(r.text, args.max_chars), section=r.section, year=r.year)
            try:
                reply = ask(url, msgs, args.max_tokens, args.temperature)
            except (urllib.error.URLError, OSError, ValueError) as e:
                return r, None, CardRejected("request", type(e).__name__), {}
            usage = reply.get("usage") or {}
            try:
                card = parse_card(reply["choices"][0]["message"]["content"])
            except CardRejected as e:
                return r, None, e, usage
            return r, card, None, usage

        def record(r, card, err, usage):
            with lock:
                stats["prompt"] += usage.get("prompt_tokens", 0)
                stats["completion"] += usage.get("completion_tokens", 0)
                if err is not None:
                    stats["rejected"] += 1
                    stats["why:" + err.reason] += 1
                    if err.reason != "request":     # a failed request is asked again next run
                        rej.write(json.dumps({"id": r.id, "reason": err.reason}) + "\n")
                        rej.flush()
                    return
                stats["cards"] += 1
                row = {"id": r.id, "year": r.year, "numero": r.number, "nrg": r.nrg,
                       "sezione": r.section, "data": r.meta.get("DATA_PUBBLICAZIONE"),
                       "esito_openga": r.meta.get("ESITO_PROVVEDIMENTO"),
                       # what the appeal was about: how step 4 keeps the art. 9 and 10
                       # GDPR categories (antimafia, residence permits, health) out of
                       # anything trained on
                       "oggetto": r.meta.get("OGGETTO_RICORSO"), "teacher": teacher,
                       **card}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                if stats["cards"] % 100 == 0:
                    report(stats, t_start)

        in_flight = max(1, args.parallel * len(urls))
        with ThreadPoolExecutor(max_workers=in_flight) as pool:
            pending = set()
            it = iter(enumerate(todo))
            stopped = False
            while True:
                while not stopped and len(pending) < in_flight:
                    if args.stop_after and time.monotonic() - t0 > args.stop_after:
                        stopped = True
                        print("[schede] time is up: finishing what is in flight", flush=True)
                        break
                    nxt = next(it, None)
                    if nxt is None:
                        stopped = True
                        break
                    pending.add(pool.submit(work, *nxt))
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    record(*fut.result())
        report(stats, t_start)
    reasons = ", ".join(f"{k[4:]} {v}" for k, v in sorted(stats.items()) if k.startswith("why:"))
    print(f"[schede] {stats['cards']:,} cards, {stats['rejected']:,} refused"
          + (f" ({reasons})" if reasons else "") + f" -> {args.out}", flush=True)
    return 0


def report(stats: Counter, t_start: float) -> None:
    dt = max(1e-6, time.monotonic() - t_start)
    n = stats["cards"] + stats["rejected"]
    print(f"[schede] {stats['cards']:,} cards in {dt / 60:.1f} min: {n / dt * 60:.1f} rulings/min, "
          f"prompt {stats['prompt'] / dt:,.0f} tok/s, generated {stats['completion'] / dt:,.0f} "
          f"tok/s, refused {stats['rejected']:,}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
