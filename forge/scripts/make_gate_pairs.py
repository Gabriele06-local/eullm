#!/usr/bin/env python3
"""Labelled examples for the context gate (`eullm_forge.datasets.context_gate`).

    python forge/scripts/make_gate_pairs.py --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --openbook $WORK/eullm_runs/stage3/openbook-v04.jsonl \\
        --premise $WORK/eullm_runs/premise5/grpo.jsonl \\
        --offtopic $WORK/eullm_runs/gate/offtopic.jsonl \\
        --exclude-exam $WORK/eval/norm-exam-*.jsonl $WORK/eval/practical-*.jsonl \\
                       $WORK/eval/lexia/lexia-trappole.jsonl \\
        --embedder Qwen/Qwen3-Embedding-0.6B --reranker Qwen/Qwen3-Reranker-0.6B \\
        --out-train $WORK/eullm_runs/gate/train.jsonl --out-dev $WORK/eullm_runs/gate/dev.jsonl

Each question written from an article -- grounded open-book pairs, the
practical exam's GRPO prompts (``--practical``), the false-premise prompts
-- gives two examples, its article in the texts and taken out; the
false-premise questions on nonexistent articles and the off-topic questions
give one each, both insufficient. The texts are
retrieved the way the released models retrieve them (``--embedder``,
``--reranker``), since that is what the gate will read.

The development set holds articles and off-topic matters the training set
never sees, so its score says how the gate does on what it was not shown.
No article of an ``--exclude-exam`` set is used, including the ones an
outside set's questions only retrieve. Counts only, never a question.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.context_gate import (  # noqa: E402
    from_article,
    from_question,
    openbook_article,
    question_of,
)
from eullm_forge.datasets.premise_gen import excluded_articles, retrieved_articles  # noqa: E402
from eullm_forge.eval import NormIndex, load_eval_set  # noqa: E402


def read(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--openbook", nargs="+", type=Path, required=True,
                    help="open-book SFT pairs (openbook_grounded rows are used)")
    ap.add_argument("--premise", nargs="*", type=Path, default=[],
                    help="false-premise GRPO prompts: with their article and without it, "
                         "and the nonexistent-article ones as insufficient")
    ap.add_argument("--practical", nargs="*", type=Path, default=[],
                    help="practical-exam GRPO prompts (make_practical_prompts.py)")
    ap.add_argument("--max-per-source", type=int, default=3000,
                    help="premise and practical questions used at most, each")
    ap.add_argument("--offtopic", nargs="*", type=Path, default=[],
                    help="make_offtopic_questions.py output")
    ap.add_argument("--exclude-exam", nargs="*", type=Path, default=[])
    ap.add_argument("--max-openbook", type=int, default=6000,
                    help="grounded pairs used at most (two examples each)")
    ap.add_argument("--max-nonexistent", type=int, default=1500)
    ap.add_argument("--dev-share", type=float, default=0.1)
    ap.add_argument("-k", type=int, default=3, help="texts per question, as the exam")
    ap.add_argument("--embedder")
    ap.add_argument("--reranker")
    ap.add_argument("--retrieval-cache", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-train", type=Path, required=True)
    ap.add_argument("--out-dev", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.embedder:
        from eullm_forge.eval.dense import build_hybrid
        index = build_hybrid(args.norms, args.embedder, reranker_id=args.reranker,
                             cache_dir=args.retrieval_cache)
    else:
        index = NormIndex.from_files(args.norms)
    held_out = [it for p in args.exclude_exam for it in load_eval_set(p)]
    exclude = excluded_articles(held_out) | retrieved_articles(index, held_out, args.k)
    rng = random.Random(args.seed)
    skipped: Counter = Counter()

    # (question, code, article, key, source) of every question written from an article
    asked = []
    grounded = [p for p in read(args.openbook) if openbook_article(p)]
    rng.shuffle(grounded)
    for p in grounded[:args.max_openbook]:
        asked.append((question_of(p["instruction"]), *openbook_article(p), str(p["key"]),
                      "openbook"))
    premise = read(args.premise)
    for source, rows in (("premise", [r for r in premise if r.get("tipo") not in
                                      (None, "premessa_inesistente")]),
                         ("practical", read(args.practical))):
        rows = [r for r in rows if r.get("code") and r.get("articolo")]
        rng.shuffle(rows)
        for r in rows[:args.max_per_source]:
            asked.append((question_of(r["prompt"][-1]["content"]), r["code"],
                          str(r["articolo"]), str(r["id"]), source))
    asked = [a for a in asked if (a[1], a[2]) not in exclude]
    articles = sorted({(a[1], a[2]) for a in asked})
    rng.shuffle(articles)
    dev_articles = set(articles[:int(round(len(articles) * args.dev_share))])

    train, dev = [], []
    for question, code, number, key, source in asked:
        made = from_article(question, code, number, index, rng, key, k=args.k)
        if not made:
            skipped[f"{source}:article_not_in_index"] += 1
        for ex in made:
            ex["source"] = source
        (dev if (code, number) in dev_articles else train).extend(made)

    fake = [r for r in premise if r.get("tipo") == "premessa_inesistente"]
    rng.shuffle(fake)
    for r in fake[:args.max_nonexistent]:
        ex = from_question(question_of(r["prompt"][-1]["content"]), index,
                           "articolo_inesistente", f"{r['id']}-gate", k=args.k)
        (dev if rng.random() < args.dev_share else train).append(ex)

    off = read(args.offtopic)
    topics = sorted({r["topic"] for r in off})
    rng.shuffle(topics)
    dev_topics = set(topics[:max(1, int(round(len(topics) * args.dev_share)))] if topics else [])
    for r in off:
        ex = from_question(r["question"], index, "fuori_ambito", f"{r['id']}-gate", k=args.k)
        (dev if r["topic"] in dev_topics else train).append(ex)

    rng.shuffle(train)
    rng.shuffle(dev)
    write(args.out_train, train)
    write(args.out_dev, dev)
    for name, rows in (("train", train), ("dev", dev)):
        c = Counter(r["reason"] for r in rows)
        src = Counter(r.get("source", "-") for r in rows if r["label"])
        print(f"[gate] {name}: {len(rows)} examples, "
              + ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
              + "; sufficient ones from " + ", ".join(f"{k} {v}" for k, v in sorted(src.items())),
              flush=True)
    print(f"[gate] {len(exclude)} exam articles left out; dev holds {len(dev_articles)} "
          f"articles and {len(dev_topics)} off-topic matters never trained on"
          + "".join(f"; {k} {v}" for k, v in sorted(skipped.items())), flush=True)
    return 0 if train and dev else 1


if __name__ == "__main__":
    sys.exit(main())
