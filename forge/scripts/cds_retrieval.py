#!/usr/bin/env python3
"""How often retrieval finds the ruling a question is about, per index setting.

    python forge/scripts/cds_retrieval.py \\
        --chunks $WORK/datasets/legal_it_amm/train.jsonl $WORK/datasets/legal_it_amm/val.jsonl \\
        --openga $WORK/eval/openga/cds-sentenze-2017-2024.csv \\
        --cards $WORK/eullm_runs/cds/schede*.jsonl \\
        --questions $WORK/eval/cds/dev-cards.jsonl --dev-ids $WORK/eval/cds/cds-dev-ids.txt \\
        --setting chunks prefix prefix+cards --embedder Qwen/Qwen3-Embedding-0.6B \\
        --cache-dir $WORK/eval/cds/retrieval-cache

Step 1's acceptance test (research report of 2026-10-05): the share of
development questions whose ruling is among the first 1, 3 and 10 rulings
retrieved, for the index as it would be built today (``chunks``), with
each chunk carrying its ruling's card prefix (``prefix``), and with the
cards as units of their own besides (``prefix+cards``). Step 1 passes when
recall@3 gains 5 points over ``chunks``.

The questions are those of ``--questions`` (cards of the DEVELOPMENT
rulings written by a different model than the cards in the index -- the
same model would write questions in the very words of its own cards), the
research questions and the exam questions apart. The index holds every
ruling, the development ones included, as it will in use.

``--limit`` asks a fixed random sample (``--seed``), the same for every
setting, so the settings are compared on the same questions. The 8,000-odd
questions of all 1,300 development rulings, each reranked, took more than
two hours per setting, and the first run lost three links that way
(2026-10-06): each setting's rows now go to ``--csv`` as soon as it is
measured, and a setting already there is skipped, so the next link carries
on where the last stopped.

Counts only, never a question or a ruling.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.caselaw import attach_meta, load_openga, load_rulings  # noqa: E402
from eullm_forge.caselaw.index import (  # noqa: E402
    RulingIndex,
    SparseBM25,
    build_units,
    shard_vectors,
    units_key,
)

SETTINGS = {"chunks": (False, False), "prefix": (True, False), "prefix+cards": (True, True)}


def read_jsonl(paths) -> list[dict]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def questions_of(rows: list[dict], dev: set[str] | None) -> list[tuple[str, str, str]]:
    """(kind, question, ruling id) for every question of the given cards."""
    out = []
    for r in rows:
        if dev is not None and r["id"] not in dev:
            continue
        out += [("ricerca", q, r["id"]) for q in r.get("domande_ricerca", [])]
        out += [("esame", q["domanda"], r["id"]) for q in r.get("domande_esame", [])]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", nargs="+", type=Path, required=True)
    ap.add_argument("--openga", nargs="*", type=Path, default=[])
    ap.add_argument("--cards", nargs="*", type=Path, default=[],
                    help="cards of the indexed rulings (cds_schede.py), for prefix and card units")
    ap.add_argument("--questions", nargs="+", type=Path, required=True,
                    help="cards whose questions are asked (the development rulings')")
    ap.add_argument("--dev-ids", type=Path, help="ask only the questions of these rulings")
    ap.add_argument("--setting", nargs="+", choices=list(SETTINGS), default=["chunks"])
    ap.add_argument("--embedder", help="also embeddings, fused with BM25")
    ap.add_argument("--reranker", help="with --embedder: reorder the fused list")
    ap.add_argument("--cache-dir", type=Path, help="where unit embeddings are cached")
    ap.add_argument("--limit", type=int, default=0,
                    help="ask a random sample of this many questions (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=Path, help="append one row per setting and kind here")
    args = ap.parse_args(argv)

    dev = None
    if args.dev_ids:
        dev = {ln.strip() for ln in args.dev_ids.open(encoding="utf-8") if ln.strip()}
    asked = questions_of(read_jsonl(args.questions), dev)
    if args.limit and len(asked) > args.limit:
        asked = random.Random(args.seed).sample(asked, args.limit)
    if not asked:
        print("[ret] no questions", file=sys.stderr)
        return 1
    chunks = [c for c in read_jsonl(args.chunks)
              if (c.get("kind") or str(c.get("sentence_id", "")).split("/")[0]) == "cds"]
    rulings = load_rulings(args.chunks)
    if args.openga:
        attach_meta(rulings, load_openga(args.openga))
    cards = {c["id"]: c for c in read_jsonl(args.cards)} if args.cards else {}
    print(f"[ret] {len(asked):,} questions on {len({q[2] for q in asked}):,} rulings; index of "
          f"{len(rulings):,} rulings, {len(chunks):,} chunks, {len(cards):,} cards", flush=True)

    embed = rerank = None
    if args.embedder:
        from eullm_forge.eval.dense import Embedder, Reranker
        embedder = Embedder(args.embedder, batch_size=32, max_length=768)
        embed = embedder
        if args.reranker:
            rerank = Reranker(args.reranker)

    done = set()
    if args.csv and args.csv.exists():
        with args.csv.open(encoding="utf-8") as f:
            done = {(r["setting"], r["retrieval"]) for r in csv.DictReader(f)}
    how_of = ("bm25" if embed is None else "bm25+dense" + (", rerank" if rerank else ""))
    for name in args.setting:
        if (name, how_of) in done:
            print(f"[ret] {name}: already in {args.csv}, skipped", flush=True)
            continue
        prefix_chunks, card_units = SETTINGS[name]
        if (prefix_chunks or card_units) and not cards:
            print(f"[ret] {name}: needs --cards, skipped", file=sys.stderr)
            continue
        t0 = time.monotonic()
        units = build_units(rulings, chunks, cards=cards, prefix_chunks=prefix_chunks,
                            card_units=card_units)
        index = RulingIndex(units, bm25=SparseBM25([u.text for u in units]))
        how = "bm25"
        if embed is not None:
            key = units_key(units, args.embedder)
            index.vectors = shard_vectors([u.text[:3000] for u in units], embed.encode,
                                          args.cache_dir, key)
            index.query_fn = lambda q: embed.encode([q], query=True)[0]
            index.reranker = rerank
            how = "bm25+dense" + (", rerank" if rerank else "")
        print(f"[ret] {name}: {len(units):,} units, built in {time.monotonic() - t0:.0f} s "
              f"({how})", flush=True)
        rows = []
        hits = defaultdict(lambda: [0, 0, 0, 0])            # @1, @3, @10, n
        for kind, q, target in asked:
            found = index.search(q, k=10)
            h = hits[kind]
            h[0] += target in found[:1]
            h[1] += target in found[:3]
            h[2] += target in found[:10]
            h[3] += 1
        for kind, (a1, a3, a10, n) in sorted(hits.items()):
            print(f"[ret] {name:<13} {kind:<8} n={n:<5} recall@1 {a1 / n:.3f}  "
                  f"recall@3 {a3 / n:.3f}  recall@10 {a10 / n:.3f}", flush=True)
            rows.append([name, how, kind, n, f"{a1 / n:.4f}", f"{a3 / n:.4f}", f"{a10 / n:.4f}"])
        if args.csv and rows:
            new = not args.csv.exists()
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["setting", "retrieval", "kind", "n", "recall1", "recall3",
                                "recall10"])
                w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
