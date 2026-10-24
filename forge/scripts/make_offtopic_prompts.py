#!/usr/bin/env python3
"""GRPO prompts on matters the collection does not cover: the right answer abstains.

    python forge/scripts/make_offtopic_prompts.py --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --offtopic $WORK/eullm_runs/gate/offtopic.jsonl \\
        --embedder Qwen/Qwen3-Embedding-0.6B --reranker Qwen/Qwen3-Reranker-0.6B \\
        --out $WORK/eullm_runs/grpo/prompts-offtopic.jsonl

The context gate learnt to tell these questions apart and then stopped
almost every trap too (2026-10-24): the skill belongs in the model. Each
teacher-written question on a matter outside the collection
(make_offtopic_questions.py) is asked with the texts retrieval finds for it,
as the product asks, without the missing-article note; the reward is the
abstention one (``assente`` in `eullm_forge.rl.rewards`): say the texts do
not hold it, and name no deadline. In the mix the practical and
false-premise prompts, whose texts do answer, keep the model from learning
to abstain everywhere. Counts only, never a question.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import NormIndex, open_book_prompt  # noqa: E402


def prompt_row(item: dict, index, k: int) -> dict:
    """One GRPO row, the shape make_grpo_prompts.py writes, rewarded as an abstention."""
    found = index.search(item["question"], k)
    return {"id": f"{item['id']}-grpo",
            "prompt": [{"role": "user", "content": open_book_prompt(item["question"], found)}],
            "tipo": "assente", "keywords": [], "topic": item.get("topic", "")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--offtopic", type=Path, required=True,
                    help="make_offtopic_questions.py output")
    ap.add_argument("--limit", type=int, default=0, help="at most this many prompts (0: all)")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--embedder")
    ap.add_argument("--reranker")
    ap.add_argument("--retrieval-cache", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    items = [json.loads(line) for line in args.offtopic.open(encoding="utf-8") if line.strip()]
    random.Random(args.seed).shuffle(items)
    if args.limit:
        items = items[:args.limit]
    if args.embedder:
        from eullm_forge.eval.dense import build_hybrid
        index = build_hybrid(args.norms, args.embedder, reranker_id=args.reranker,
                             cache_dir=args.retrieval_cache)
    else:
        index = NormIndex.from_files(args.norms)
    rows = [prompt_row(it, index, args.k) for it in items]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print(f"[offtopic] {len(rows)} prompts on {len({r['topic'] for r in rows})} matters "
          f"-> {args.out}", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
