#!/usr/bin/env python3
"""How many questions of the held-out exam ask about an article the
instruction pairs were generated from. Prints counts, never questions.

The exam of 16-10-2026 was drawn from every Normattiva article
(make_norm_exam.py over norms-src), and the pairs from passages of the
training split (generate_instructions.py over norms/train.jsonl), with no
article left out of either: an exam article can have pairs about it. The
pairs do not record their article, but their jobs are deterministic for a
corpus and a seed, and the key hashes the passage, so the jobs are derived
again here and each accepted key traced to its record (``Job.record``):

    python forge/scripts/jupiter/pairs_exam_overlap.py \\
        --corpus "$FORGE_DATA_DIR/norms/train.jsonl" \\
        --pairs "$FORGE_DATA_DIR/norms-instruct/pairs.jsonl" \\
        --exam "$FORGE_DATA_DIR/eval/norm-exam.jsonl" --seeds 0 1 2 3

With ``--write-clean OUT`` it also writes the pairs that touch no exam
article, for instruction tuning without the overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eullm_forge.datasets.instruct_gen import GenConfig, make_jobs  # noqa: E402
from eullm_forge.eval import load_eval_set  # noqa: E402
from eullm_forge.eval.retrieval import record_articles  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def articles_per_record(records: list[dict]) -> list[set[tuple[str, str]]]:
    """(code, article) of each record. A chunk with no article header of its
    own continues the article before it in the same code, as the exam's
    builder reads them."""
    out, last = [], {}
    for r in records:
        code = str(r.get("code", ""))
        arts = record_articles(r)
        if arts:
            last[code] = arts[-1]
        elif code in last:
            arts = [last[code]]
        out.append({(code, a) for a in arts})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, required=True,
                    help="the train.jsonl the pairs came from")
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--exam", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--limit", type=int, default=20000, help="--limit the generator ran with")
    ap.add_argument("--write-clean", type=Path, help="the pairs touching no exam article")
    args = ap.parse_args(argv)

    records = read_jsonl(args.corpus)
    arts = articles_per_record(records)
    key_arts: dict[str, set[tuple[str, str]]] = {}
    for seed in args.seeds:
        for j in make_jobs(records, args.limit, seed=seed, cfg=GenConfig()):
            key_arts.setdefault(j.key, set()).update(arts[j.record])
    pairs = read_jsonl(args.pairs)
    exam = {(it.metadata.get("code"), str(it.metadata.get("articolo")))
            for it in load_eval_set(args.exam) if it.metadata.get("tipo") != "inesistente"}
    traced = [p for p in pairs if p.get("key") in key_arts]
    touching = [p for p in traced if key_arts[p["key"]] & exam]
    hit = set().union(*(key_arts[p["key"]] & exam for p in touching)) if touching else set()
    print(f"[overlap] pairs {len(pairs):,}, traced to their article {len(traced):,}")
    print(f"[overlap] pairs about an exam article: {len(touching):,}")
    print(f"[overlap] exam articles with pairs about them: {len(hit)} of {len(exam)}")
    print(f"[overlap] by task: {dict(Counter(p.get('task') for p in touching))}")
    if args.write_clean:
        bad = {p["key"] for p in touching}
        with open(args.write_clean, "w", encoding="utf-8") as f:
            for p in pairs:
                if p.get("key") not in bad:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"[overlap] {len(pairs) - len(bad):,} pairs written to {args.write_clean}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
