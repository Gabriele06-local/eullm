#!/usr/bin/env python3
"""The practical-questions exam (roadmap 1.1), written by a teacher from the law.

    python forge/scripts/make_practical_exam.py $WORK/norms/legislazione_*.chunks.jsonl \\
        --serve $WORK/gguf/<teacher>.gguf --gpus 0,1,2 \\
        --exclude-pairs $WORK/eullm_runs/grpo/prompts*.jsonl \\
            $WORK/eullm_runs/stage3/openbook*.jsonl \\
        --out-dev $WORK/eval/practical-dev.jsonl --out-heldout $WORK/eval/practical-heldout.jsonl

See `eullm_forge.eval.practical` for what an item is and why. For each code,
``--per-code`` groups of related articles are drawn (an article and those its
text refers to) and the teacher writes one practical question per group; the
refused replies are counted by reason. ``--heldout-share`` of the groups,
chosen by a hash of their first article, go to the held-out file, the rest to
the development file.

Articles a model was trained on (``--exclude-pairs``: open-book pairs and
GRPO prompts, read by `norm_exam.trained_articles`) are never in a group: an
exam that asks what training answered measures recall of the training set.

Quiet: it prints counts, never a question, since half of what it writes is
held out. With ``--serve`` one llama-server per GPU is started on the teacher
GGUF (as cds_schede.py does); with ``--url`` the servers are already running.
An existing output is not overwritten without ``--force``: the held-out
exam is drawn once.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import save_eval_set  # noqa: E402
from eullm_forge.eval.norm_exam import articles_from_records, trained_articles  # noqa: E402
from eullm_forge.eval.practical import generate, groups, heldout, to_item  # noqa: E402


def ask(url: str, msgs: list[dict], max_tokens: int, temperature: float,
        timeout: float = 900.0) -> str:
    body = {"messages": msgs, "temperature": temperature, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        reply = json.loads(r.read())
    return ((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def read_jsonl(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            out.extend(json.loads(ln) for ln in f if ln.strip())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("norms", nargs="+", type=Path, help="legislazione_*.chunks.jsonl")
    ap.add_argument("--out-dev", type=Path, required=True)
    ap.add_argument("--out-heldout", type=Path, required=True)
    ap.add_argument("--per-code", type=int, default=60, help="groups drawn per code")
    ap.add_argument("--max-refs", type=int, default=2, help="referred articles per group")
    ap.add_argument("--heldout-share", type=float, default=0.34)
    ap.add_argument("--exclude-pairs", nargs="*", type=Path, default=[],
                    help="training files whose articles are never in a group")
    ap.add_argument("--url", action="append", default=[],
                    help="a running OpenAI-compatible server (repeatable)")
    ap.add_argument("--serve", help="teacher GGUF: start one llama-server per --gpus entry")
    ap.add_argument("--gpus", default="0", help="with --serve: GPUs, one server each")
    ap.add_argument("--llama-server", help="llama-server binary (default as in legal_eval.py)")
    ap.add_argument("--parallel", type=int, default=8, help="requests in flight per server")
    ap.add_argument("--ctx", type=int, default=16384, help="context per request (tokens)")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--teacher", default="", help="name recorded in every item")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args(argv)
    if bool(args.url) == bool(args.serve):
        ap.error("give either --url or --serve")
    for p in (args.out_dev, args.out_heldout):
        if p.exists() and not args.force:
            ap.error(f"{p} exists: the exam is drawn once (--force to draw it again)")

    articles = articles_from_records(read_jsonl(args.norms))
    exclude = frozenset(trained_articles(read_jsonl(args.exclude_pairs)))
    todo = groups(articles, args.per_code, random.Random(args.seed), args.max_refs, exclude)
    print(f"[practical] {len(articles):,} articles, {len(exclude):,} excluded as trained on, "
          f"{len(todo):,} groups over {len({g[0].code for g in todo})} codes", flush=True)
    if not todo:
        return 1

    stats: Counter = Counter()
    dev, held = [], []
    t0 = time.monotonic()
    with contextlib.ExitStack() as stack:
        urls = list(args.url)
        if args.serve:
            from eullm_forge.eval.gguf import LlamaServer
            for gpu in args.gpus.split(","):
                srv = LlamaServer(args.serve, binary=args.llama_server, parallel=args.parallel,
                                  ctx_per_slot=args.ctx, devices=gpu,
                                  log_path=args.out_dev.with_name(f"practical-server-gpu{gpu}.log"))
                urls.append(stack.enter_context(srv).url)
            print(f"[practical] {len(urls)} teacher server(s) up after "
                  f"{time.monotonic() - t0:.0f} s", flush=True)
        teacher = args.teacher or Path(args.serve or "").stem or "teacher"

        def ask_one(n: int, msgs: list[dict]) -> str:
            return ask(urls[n % len(urls)], msgs, args.max_tokens, args.temperature)

        for group, parsed, why in generate(todo, ask_one, args.parallel * len(urls)):
            if why:
                stats["refused:" + why] += 1
                continue
            split = "heldout" if heldout(group[0], args.heldout_share) else "dev"
            (held if split == "heldout" else dev).append(to_item(group, parsed, split, teacher))
            stats["multi" if len(parsed["articoli"]) > 1 else "single"] += 1

    # the items in a stable order, whatever order the servers answered in
    dev.sort(key=lambda it: it.id)
    held.sort(key=lambda it: it.id)
    save_eval_set(dev, args.out_dev)
    save_eval_set(held, args.out_heldout)
    refused = sum(v for k, v in stats.items() if k.startswith("refused:"))
    print(f"[practical] {len(dev):,} development and {len(held):,} held-out items "
          f"({stats['multi']:,} needing more than one article), {refused:,} refused, "
          f"{time.monotonic() - t0:.0f} s", flush=True)
    for k, v in sorted(stats.items()):
        if k.startswith("refused:"):
            print(f"[practical]   {k[8:]}: {v:,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
