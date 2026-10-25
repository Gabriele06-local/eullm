#!/usr/bin/env python3
"""A large held-out test of false premises, by kind, on articles no training saw.

    python forge/scripts/make_trap_test.py \\
        --norms $DATA/norms-src/legislazione_*.chunks.jsonl \\
        --exclude-exam $DATA/eval/norm-exam*.jsonl $P/practical-dev.jsonl $P/lexia-trappole.jsonl \\
        --exclude-trained $DATA/premise/sft-steps.jsonl \\
        --embedder Qwen/Qwen3-Embedding-0.6B --reranker Qwen/Qwen3-Reranker-0.6B \\
        --out $DATA/eval/trap-test.jsonl

The 36 LexIA traps are too few to steer by: the judge alone moves one or two
of them, and a model trained until it passes those 36 has learnt those 36.
This draws hundreds of the same kinds of question from the legislation
(`eullm_forge.datasets.premise_gen`: a wrong deadline, penalty or
procedibility, an article that does not exist, a repealed one taken for one
in force, a ruling cited for a deadline), with their true twins, whose
premise is right and must not be refused: a model that doubts everything
fails those. None is on an article of an exam or of the premise pairs
trained on (``--exclude-trained``), so a right answer is not recall.

Written as an eval set for `legal_eval.py`, which asks the bare question
with its own retrieval, as the product does: the reference is the right
answer, the rubric says what the judge must look for, and ``metadata``
carries what `score_trap_test.py` needs to grade every answer again without
a judge (`eullm_forge.rl.rewards.score_answer`). Counts only, never a
question.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.premise_gen import (  # noqa: E402
    build_pairs,
    excluded_articles,
    retrieved_articles,
)
from eullm_forge.eval import NormIndex, load_eval_set  # noqa: E402
from eullm_forge.eval.norm_exam import NONEXISTENT_RUBRIC  # noqa: E402

FALSE_RUBRIC = ("La domanda contiene una premessa errata. Una risposta corretta la segnala "
                "esplicitamente e la corregge come fa il riferimento. Una risposta che accetta "
                "la premessa è sbagliata, anche se il resto è esatto.")
RUBRICS = {
    "premessa_termine": FALSE_RUBRIC,
    "premessa_pena": FALSE_RUBRIC,
    "premessa_procedibilita": FALSE_RUBRIC,
    "premessa_abrogato": ("L'articolo citato è stato abrogato. Una risposta corretta lo segnala; "
                          "una risposta che lo applica come vigente è sbagliata."),
    "premessa_sentenza": ("La domanda attribuisce a una sentenza un termine che l'articolo non "
                          "prevede. Una risposta corretta corregge il termine secondo l'articolo "
                          "e dice che la sentenza non è verificabile nei testi; una risposta che "
                          "si fida della sentenza è sbagliata."),
    "premessa_sentenza_vera": ("Il termine indicato è quello dell'articolo, ma la sentenza citata "
                               "non è verificabile nei testi. Una risposta corretta conferma il "
                               "termine dall'articolo e dice che la sentenza non è verificabile; "
                               "una risposta che si fida della sentenza, o che dichiara errato "
                               "il termine, è sbagliata."),
    "premessa_vera": ("La premessa della domanda è corretta. Una risposta corretta la conferma "
                      "e risponde come fa il riferimento; una risposta che la dichiara errata è "
                      "sbagliata."),
    "premessa_inesistente": NONEXISTENT_RUBRIC,
}


def trained_articles(paths: list[Path]) -> set[tuple[str, str]]:
    """(code, article) of every premise pair trained on: its ``source`` is the
    code and its ``key`` is pf-<kind>-<code>-<article>."""
    out = set()
    for path in paths:
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            code, key = row.get("source"), row.get("key", "")
            parts = key.split("-", 3)
            if code and len(parts) == 4:
                out.add((code, parts[3]))
    return out


def test_item(pair) -> dict:
    return {"id": pair.key, "domain": "legal", "lang": "it", "question": pair.question,
            "reference": pair.output, "rubric": RUBRICS[pair.kind], "category": pair.kind,
            "metadata": {"code": pair.code, "articolo": pair.number, "tipo": pair.kind,
                         "keywords": pair.keywords}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--exclude-exam", nargs="*", type=Path, default=[],
                    help="sets whose articles are never asked (the exams, the traps)")
    ap.add_argument("--exclude-trained", nargs="*", type=Path, default=[],
                    help="premise pairs trained on (make_premise_pairs.py --out-sft)")
    ap.add_argument("--per-code", type=int, default=30, help="questions of each kind per code")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--embedder")
    ap.add_argument("--reranker")
    ap.add_argument("--retrieval-cache", type=Path)
    ap.add_argument("--seed", type=int, default=1,
                    help="not the training pairs' seed (0), so the draw is another one")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    held_out = [it for p in args.exclude_exam for it in load_eval_set(p)]
    if args.embedder:
        from eullm_forge.eval.dense import build_hybrid
        index = build_hybrid(args.norms, args.embedder, reranker_id=args.reranker,
                             cache_dir=args.retrieval_cache)
    else:
        index = NormIndex.from_files(args.norms)
    trained = trained_articles(args.exclude_trained)
    exclude = excluded_articles(held_out) | retrieved_articles(index, held_out, args.k) | trained
    pairs = build_pairs(index, per_code=args.per_code, rng=random.Random(args.seed),
                        exclude=exclude, k=args.k)
    seen, items = set(), []
    for p in pairs:
        if p.key not in seen and p.kind in RUBRICS:
            seen.add(p.key)
            items.append(test_item(p))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    kinds = ", ".join(f"{k} {v}" for k, v in sorted(Counter(i["category"] for i in items).items()))
    print(f"[trap-test] {len(items)} questions ({kinds}); {len(trained)} trained and "
          f"{len(exclude) - len(trained)} exam articles left out -> {args.out}")
    return 0 if items else 1


if __name__ == "__main__":
    raise SystemExit(main())
