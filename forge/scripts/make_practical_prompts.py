#!/usr/bin/env python3
"""GRPO prompts of practical questions (roadmap 1.2), written by a teacher from the law.

    python forge/scripts/make_practical_prompts.py \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --exclude-exam $WORK/eval/practical-*.jsonl $WORK/eval/norm-exam*.jsonl \\
        --serve $WORK/gguf/qwen3.6-27b/qwen3.6-27b-q8_0.gguf --gpus 0,1,2 \\
        --embedder Qwen/Qwen3-Embedding-0.6B --reranker Qwen/Qwen3-Reranker-0.6B \\
        --mix $WORK/eullm_runs/grpo/prompts-judge.jsonl \\
        --out $WORK/eullm_runs/grpo/prompts-practical.jsonl

The same questions the practical exam asks (`eullm_forge.eval.practical`:
groups of articles the law links, a teacher's question, the articles a right
answer cites, a reference and a rubric), as training prompts: the question
with the texts retrieval finds for it, as the exam and RAG Enterprise ask it,
graded during training by the judge (kind "pratica" is one of
`rl.JUDGED_TYPES`), against the reference and the rubric the exam grades with.

No article of an ``--exclude-exam`` file is in a group -- for a practical
exam that is every article of an item's group, not only the one its id names.
Articles a model was trained on are not left out here, unlike for the exam:
these prompts are training.

``--mix`` puts the rows of another GRPO prompts file in front, unchanged: the
deadline, absent and content questions keep what the models do on those
while the practical ones are learnt. Counts only, never a question.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import NormIndex, load_eval_set, open_book_prompt  # noqa: E402
from eullm_forge.eval.norm_exam import articles_from_records  # noqa: E402
from eullm_forge.eval.practical import (  # noqa: E402
    KIND,
    exam_articles,
    generate,
    groups,
    to_item,
)


def ask(url: str, msgs: list[dict], max_tokens: int, temperature: float,
        timeout: float = 900.0) -> str:
    import urllib.request
    body = {"messages": msgs, "temperature": temperature, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        reply = json.loads(r.read())
    return ((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def prompt_row(item, index, k: int) -> dict:
    """One GRPO row, the shape make_grpo_prompts.py writes for a judged question."""
    found = index.search(item.question, k)
    return {"id": item.id.replace("norm-", "pr-", 1),
            "prompt": [{"role": "user", "content": open_book_prompt(item.question, found)}],
            "tipo": KIND, "keywords": [],
            "code": item.metadata["code"], "articolo": item.metadata["articolo"],
            "question": item.question, "reference": item.reference, "rubric": item.rubric}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--exclude-exam", nargs="*", type=Path, default=[],
                    help="exams whose articles are in no group")
    ap.add_argument("--per-code", type=int, default=150, help="groups drawn per code")
    ap.add_argument("--max-refs", type=int, default=2)
    ap.add_argument("--mix", nargs="*", type=Path, default=[],
                    help="GRPO prompts files whose rows come first, unchanged")
    ap.add_argument("-k", type=int, default=3, help="retrieved texts per question, as the exam")
    ap.add_argument("--embedder", help="retrieve with BM25 fused with this embedding model")
    ap.add_argument("--reranker", help="with --embedder: reorder with this model")
    ap.add_argument("--retrieval-cache", type=Path)
    ap.add_argument("--url", action="append", default=[])
    ap.add_argument("--serve", help="teacher GGUF: start one llama-server per --gpus entry")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--llama-server")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if bool(args.url) == bool(args.serve):
        ap.error("give either --url or --serve")

    exclude = exam_articles([it for p in args.exclude_exam for it in load_eval_set(p)])
    if args.embedder:
        from eullm_forge.eval.dense import build_hybrid
        index = build_hybrid(args.norms, args.embedder, reranker_id=args.reranker,
                             cache_dir=args.retrieval_cache)
    else:
        index = NormIndex.from_files(args.norms)
    articles = articles_from_records(index.records)
    todo = groups(articles, args.per_code, random.Random(args.seed), args.max_refs, exclude)
    print(f"[practical-prompts] {len(articles):,} articles, {len(exclude):,} exam articles "
          f"left out, {len(todo):,} groups", flush=True)

    stats: Counter = Counter()
    rows = []
    t0 = time.monotonic()
    with contextlib.ExitStack() as stack:
        urls = list(args.url)
        if args.serve:
            from eullm_forge.eval.gguf import LlamaServer
            for gpu in args.gpus.split(","):
                srv = LlamaServer(args.serve, binary=args.llama_server, parallel=args.parallel,
                                  ctx_per_slot=args.ctx, devices=gpu,
                                  log_path=args.out.with_name(f"practical-server-gpu{gpu}.log"))
                urls.append(stack.enter_context(srv).url)
        teacher = Path(args.serve or "").stem or "teacher"

        def ask_one(n: int, msgs: list[dict]) -> str:
            return ask(urls[n % len(urls)], msgs, args.max_tokens, args.temperature)

        for group, parsed, why in generate(todo, ask_one, args.parallel * len(urls)):
            if why:
                stats["refused:" + why] += 1
                continue
            rows.append(prompt_row(to_item(group, parsed, "train", teacher), index, args.k))

    rows.sort(key=lambda r: r["id"])
    mixed = []
    for p in args.mix:
        with p.open(encoding="utf-8") as f:
            mixed.extend(json.loads(ln) for ln in f if ln.strip())
    out = mixed + rows
    random.Random(args.seed).shuffle(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    refused = sum(v for k, v in stats.items() if k.startswith("refused:"))
    kinds = Counter(r.get("tipo") for r in out)
    print(f"[practical-prompts] {len(rows):,} practical prompts ({refused:,} refused), "
          f"{len(mixed):,} mixed in; {', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))}; "
          f"{time.monotonic() - t0:.0f} s -> {args.out}", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
