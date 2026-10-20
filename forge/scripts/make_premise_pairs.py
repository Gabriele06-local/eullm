#!/usr/bin/env python3
"""Questions on a false premise, for SFT and for GRPO (`eullm_forge.datasets.premise_gen`).

    python forge/scripts/make_premise_pairs.py \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --exclude-exam $WORK/eval/norm-exam-*.jsonl $WORK/eval/practical-*.jsonl \\
                       $WORK/eval/lexia/lexia-trappole.jsonl \\
        --embedder Qwen/Qwen3-Embedding-0.6B --reranker Qwen/Qwen3-Reranker-0.6B \\
        --out-sft $WORK/eullm_runs/stage3/premise-steps.jsonl \\
        --out-grpo $WORK/eullm_runs/grpo/prompts-premise.jsonl

A wrong deadline, an article that does not exist, a repealed article taken
for one in force: the premise is false by construction, so the right answer
is known without a teacher, and `eullm_forge.rl.rewards` checks it.

``--grpo-share`` of the pairs become GRPO prompts and the rest SFT pairs, on
different articles, so RL is not asked what SFT already answered; the split
does not depend on which files are written. ``--format plain`` writes the
SFT answers in one paragraph instead of the three checked steps, for the
ablation between the two: the same seed gives the same pairs.

No article of an ``--exclude-exam`` file is used, whether the exam records
it in its metadata or only names it in a question (the LexIA traps, which
stay a test and are never trained on). Counts only, never a question.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.premise_gen import FORMATS, build_pairs, excluded_articles  # noqa: E402
from eullm_forge.eval import NormIndex, load_eval_set  # noqa: E402


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
    ap.add_argument("--exclude-exam", nargs="*", type=Path, default=[],
                    help="exams and test sets whose articles are never used")
    ap.add_argument("--per-code", type=int, default=80,
                    help="pairs of each kind per code; a code with fewer gives all it has")
    ap.add_argument("--format", choices=FORMATS, default="steps",
                    help="steps: verifica, norma, risposta; plain: one paragraph")
    ap.add_argument("--bis-share", type=float, default=0.5,
                    help="share of the nonexistent articles made as the -bis of a real one")
    ap.add_argument("--grpo-share", type=float, default=0.5,
                    help="share of the pairs written as GRPO prompts instead of SFT pairs")
    ap.add_argument("-k", type=int, default=3, help="retrieved texts per question, as the exam")
    ap.add_argument("--embedder", help="retrieve with BM25 fused with this embedding model "
                    "(eullm_forge.eval.dense), as the released models do")
    ap.add_argument("--reranker", help="with --embedder: reorder the fused list with this model")
    ap.add_argument("--retrieval-cache", type=Path, help="where the document embeddings are cached")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-sft", type=Path)
    ap.add_argument("--out-grpo", type=Path)
    args = ap.parse_args(argv)
    if not args.out_sft and not args.out_grpo:
        ap.error("give --out-sft, --out-grpo or both")
    if not 0 <= args.grpo_share <= 1:
        ap.error("--grpo-share must be in [0, 1]")

    exclude = excluded_articles([it for p in args.exclude_exam for it in load_eval_set(p)])
    if args.embedder:
        from eullm_forge.eval.dense import build_hybrid
        index = build_hybrid(args.norms, args.embedder, reranker_id=args.reranker,
                             cache_dir=args.retrieval_cache)
    else:
        index = NormIndex.from_files(args.norms)
    rng = random.Random(args.seed)
    skipped: dict[str, int] = {}
    pairs = build_pairs(index, per_code=args.per_code, rng=rng, fmt=args.format,
                        exclude=exclude, k=args.k, bis_share=args.bis_share, stats=skipped)

    # Split by article, so a GRPO prompt is never on an article SFT answered.
    articles = sorted({(p.code, p.number) for p in pairs})
    rng.shuffle(articles)
    # Split the same way whichever files are asked for: the plain-format run
    # writes only its SFT file, on the same articles as the steps run's.
    to_grpo = set(articles[:int(round(len(articles) * args.grpo_share))])
    grpo = [p.grpo_row() for p in pairs if (p.code, p.number) in to_grpo]
    sft = [p.sft_row() for p in pairs if (p.code, p.number) not in to_grpo]
    rng.shuffle(grpo)
    rng.shuffle(sft)
    if args.out_sft:
        write(args.out_sft, sft)
    if args.out_grpo:
        write(args.out_grpo, grpo)

    kinds = ", ".join(f"{k} {v}" for k, v in sorted(Counter(p.kind for p in pairs).items()))
    print(f"[premise] {len(pairs)} pairs ({kinds})"
          f", {len(exclude)} exam articles left out; {len(sft)} SFT ({args.format})"
          f"{f' -> {args.out_sft}' if args.out_sft else ''}, {len(grpo)} GRPO"
          f"{f' -> {args.out_grpo}' if args.out_grpo else ''}")
    if skipped:
        print("[premise] skipped: " + ", ".join(f"{k} {v}" for k, v in sorted(skipped.items())))
    return 0 if pairs else 1


if __name__ == "__main__":
    sys.exit(main())
