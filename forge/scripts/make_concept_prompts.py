#!/usr/bin/env python3
"""GRPO prompts asking for a concept by name, where the texts hold the answer.

    python forge/scripts/make_concept_prompts.py --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --exclude-exam $WORK/eval/norm-exam*.jsonl $WORK/eval/practical-*.jsonl \\
                       $WORK/eval/lexia/lexia-trappole.jsonl \\
        --embedder Qwen/Qwen3-Embedding-0.6B --reranker Qwen/Qwen3-Reranker-0.6B \\
        --out $WORK/eullm_runs/grpo/prompts-concept.jsonl

Every 8B round since 2026-10-21 answers a plain question or two fewer than
the published model, and on 10-26 r2q declined to define "recidiva
reiterata" with art. 99 c.p. among its texts: the premise and off-topic
prompts pay for declining, and nothing paid for answering a question that
names no article. Here each article with a rubrica gives a question on the
concept it names ("Che cosa si intende per «Recidiva»?"), asked with the
texts retrieval finds, without the note, as the product asks; only the
questions whose texts include the article are kept, so a right answer is
always there to give. Judged as ``contenuto``: the article's text is the
reference, graded with the exam's content rubric.

No article of an ``--exclude-exam`` set is used, nor the ones an outside
set's questions only retrieve. Counts only, never a question.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.premise_gen import excluded_articles, retrieved_articles  # noqa: E402
from eullm_forge.eval import NormIndex, load_eval_set, open_book_prompt  # noqa: E402
from eullm_forge.eval.norm_exam import (  # noqa: E402
    CODE_LABELS,
    CONTENT_RUBRIC,
    _flat,
    _usable,
    articles_from_records,
)

TEMPLATES = ("Che cosa si intende per «{h}»?",
             "Che cosa prevede la legge in materia di «{h}»?",
             "Spiegami in breve: {h}.")
# Headings that name no concept a user would ask about.
_GENERIC = re.compile(
    r"^(?:disposizion|norm[ae]\b|ambito|definizion|abrogat|oggetto|finalit|entrata in vigore|"
    r"modific|articol|casi\b|altr[ie]\b|rinvio|applicazion|clausola|regime transitorio)",
    re.IGNORECASE)


def concept(heading: str) -> str | None:
    """The concept a rubrica names, as a question can say it, or None."""
    h = heading.strip(" .;:()")
    if not 4 <= len(h) <= 70 or _GENERIC.match(h) or not h[:1].isalpha():
        return None
    return h


def prompt_row(article, question: str, index, k: int) -> dict | None:
    """A judged row, or None when the texts retrieval finds lack the article."""
    found = index.search(question, k)
    if not any(r.get("code") == article.code and article.number in index.articles_of(r)
               for r in found):
        return None
    return {"id": f"concept-{article.code}-{article.number}",
            "prompt": [{"role": "user", "content": open_book_prompt(question, found)}],
            "tipo": "contenuto", "keywords": [], "code": article.code,
            "articolo": article.number, "question": question,
            "reference": _flat(article.text), "rubric": CONTENT_RUBRIC}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--exclude-exam", nargs="*", type=Path, default=[])
    ap.add_argument("--per-code", type=int, default=80, help="questions drawn per code")
    ap.add_argument("-k", type=int, default=3, help="texts per question, as the exam")
    ap.add_argument("--embedder")
    ap.add_argument("--reranker")
    ap.add_argument("--retrieval-cache", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    records = []
    for p in args.norms:
        with p.open(encoding="utf-8") as f:
            records += [json.loads(line) for line in f if line.strip()]
    if args.embedder:
        from eullm_forge.eval.dense import build_hybrid
        index = build_hybrid(args.norms, args.embedder, reranker_id=args.reranker,
                             cache_dir=args.retrieval_cache)
    else:
        index = NormIndex(records)
    held_out = [it for p in args.exclude_exam for it in load_eval_set(p)]
    exclude = excluded_articles(held_out) | retrieved_articles(index, held_out, args.k)
    rng = random.Random(args.seed)

    by_code: dict[str, list] = {}
    for a in articles_from_records(records).values():
        if (a.code in CODE_LABELS and _usable(a) and (a.code, a.number) not in exclude
                and concept(a.heading)):
            by_code.setdefault(a.code, []).append(a)
    rows, skipped = [], Counter()
    for code in sorted(by_code):
        pool = sorted(by_code[code], key=lambda a: a.number)
        for a in rng.sample(pool, min(args.per_code, len(pool))):
            question = rng.choice(TEMPLATES).format(h=concept(a.heading))
            row = prompt_row(a, question, index, args.k)
            if row is None:
                skipped["texts_lack_the_article"] += 1
            else:
                rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print(f"[concept] {len(rows)} prompts from {len(by_code)} codes, {len(exclude)} exam "
          f"articles left out" + "".join(f"; {k} {v}" for k, v in skipped.items())
          + f" -> {args.out}", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
