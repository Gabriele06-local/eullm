#!/usr/bin/env python3
"""How often retrieval finds each question's own article, at several settings.

    python forge/scripts/check_retrieval.py $WORK/eval/norm-exam-dev.jsonl \\
        $WORK/eval/norm-exam-dev2.jsonl \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl --boost 0 3 6

Prints, per question type and per ``heading_boost``, the share of questions
whose article is first (top1) and among the first three (top3). Counts only,
never a question — but choose settings on DEVELOPMENT sets: tuning retrieval
against the held-out exam turns it into one more development set.

``--pairs`` measures on the open-book training pairs instead: the teacher's
questions that ask by topic in its own words (``named`` false), each with the
article it was written from. The exam's topic questions quote the rubrica
verbatim, so a heavy rubrica weight wins there by construction; the teacher
paraphrases, the way people do, and is the fairer test of a weight.

``--embedder`` adds retrieval by meaning (`eullm_forge.eval.dense`): for each
model given, the embedding ranking alone ("dense") and fused with BM25
("hybrid"), and, with ``--reranker``, the fused list reordered by the
reranker ("hybrid+rerank"). These load models, so they run in a GPU job
(sbatch_retrieval.slurm); the document embeddings are cached in
``--cache-dir`` and computed once.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import EvalItem, NormIndex, load_eval_set  # noqa: E402
from eullm_forge.eval.norm_exam import retrieval_hits  # noqa: E402


def topic_questions(path: Path) -> list[EvalItem]:
    """The teacher's by-topic questions of an open-book pairs file, as items
    that name the article they were written from."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            key = str(p.get("key", ""))
            if p.get("task") != "openbook_grounded" or p.get("named", True) \
                    or not key.startswith("ob-g-"):
                continue
            code, _, number = key[len("ob-g-"):].partition("-")
            number = re.sub(r"-v\d+$", "", number)
            question = p["instruction"].rsplit("Domanda: ", 1)[-1].strip()
            items.append(EvalItem(id=key, domain="legal", lang="it", question=question,
                                  metadata={"tipo": "argomento_insegnante", "code": code,
                                            "articolo": number}))
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exams", nargs="*", type=Path, help="exam JSONL files, pooled")
    ap.add_argument("--pairs", nargs="+", type=Path, default=[],
                    help="open-book pairs: measure on their topic questions")
    ap.add_argument("--pairs-limit", type=int, default=0,
                    help="use only the first N topic questions of --pairs (0 = all)")
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--boost", nargs="+", type=int, default=[0, 3])
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--embedder", nargs="+", default=[],
                    help="embedding models (hub id or snapshot dir), e.g. Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--reranker", help="reranker model, e.g. Qwen/Qwen3-Reranker-4B")
    ap.add_argument("--cache-dir", type=Path, help="where document embeddings are kept")
    ap.add_argument("--csv", type=Path, help="append one row per setting and kind here")
    args = ap.parse_args(argv)

    items = [it for path in args.exams for it in load_eval_set(path)]
    topic = [q for path in args.pairs for q in topic_questions(path)]
    items.extend(topic[:args.pairs_limit] if args.pairs_limit else topic)
    if not items:
        ap.error("give exam files or --pairs")
    records = NormIndex.from_files(args.norms).records
    print(f"[retrieval] {len(items)} items from {len(args.exams) + len(args.pairs)} file(s)")
    rows = []

    def report(setting: str, index) -> None:
        for kind, r in retrieval_hits(items, index, k=args.k).items():
            print(f"[retrieval] {setting:<40} {kind:<20} n={r['n']:<4} "
                  f"top1={r['top1']:.2f} top{args.k}={r[f'top{args.k}']:.2f}", flush=True)
            rows.append([setting, kind, r["n"], f"{r['top1']:.3f}", f"{r[f'top{args.k}']:.3f}"])

    for boost in args.boost:
        report(f"bm25 boost={boost}", NormIndex(records, heading_boost=boost))

    if args.embedder:
        from eullm_forge.eval.dense import (
            Embedder,
            HybridIndex,
            Reranker,
            cached_doc_vectors,
            files_fingerprint,
        )

        base = NormIndex(records)
        reranker = Reranker(args.reranker) if args.reranker else None
        for model_id in args.embedder:
            emb = Embedder(model_id)
            vecs = cached_doc_vectors(base, emb, args.cache_dir, files_fingerprint(args.norms))
            # every question once, up front: one batched pass, not one per item
            qs = sorted({it.question for it in items})
            qvec = dict(zip(qs, emb.encode(qs, query=True)))
            name = model_id.rstrip("/").split("/")[-1]
            report(f"dense {name}", HybridIndex(base, vecs, qvec.__getitem__, use_bm25=False))
            report(f"hybrid {name}", HybridIndex(base, vecs, qvec.__getitem__))
            if reranker:
                rname = args.reranker.rstrip("/").split("/")[-1]
                report(f"hybrid {name} + {rname}",
                       HybridIndex(base, vecs, qvec.__getitem__, reranker))
            del emb

    if args.csv:
        import csv

        new = not (args.csv.exists() and args.csv.stat().st_size > 0)
        with args.csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["setting", "kind", "n", "top1", f"top{args.k}"])
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
