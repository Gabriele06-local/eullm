#!/usr/bin/env python3
"""Open-book prompts with checkable answers, for GRPO (grpo_train.py).

    python forge/scripts/make_grpo_prompts.py \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --exclude-exam $WORK/eval/norm-exam-v3.jsonl $WORK/eval/norm-exam-v4.jsonl \\
                       $WORK/eval/norm-exam-dev*.jsonl \\
        --out $WORK/eullm_runs/grpo/prompts.jsonl

Draws questions the way the exam does (`norm_exam.build_exam`) and keeps the
kinds a program can grade (see `eullm_forge.rl.rewards`): deadlines, asked by
article or by topic, and articles that do not exist. A share of the by-topic
deadline questions is copied with its own article taken out of the retrieved
texts (``assente``), so that saying "the texts do not hold it" is rewarded
where it is true and only there.

Every prompt is built by the code the exam and the engine use
(`NormIndex.search`, `missing_article_note`, `open_book_prompt`), so RL
trains on exactly the input the model will be graded and used on.

No article of an ``--exclude-exam`` file is drawn: training on the exam's
articles would turn the exam into a memory test. The file holds counts and
prompts, never an exam item.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import NormIndex, load_eval_set, open_book_prompt  # noqa: E402
from eullm_forge.eval.norm_exam import build_exam  # noqa: E402
from eullm_forge.eval.retrieval import record_articles  # noqa: E402
from eullm_forge.rl import DEADLINE_TYPES  # noqa: E402

KEPT = DEADLINE_TYPES | {"inesistente"}


def exam_articles(paths: list[Path]) -> set[tuple[str, str]]:
    """(code, article) of every item of the given exams."""
    out = set()
    for path in paths:
        for it in load_eval_set(path):
            code, art = it.metadata.get("code"), it.metadata.get("articolo")
            if code and art:
                out.add((code, str(art)))
    return out


def prompt_row(item, index: NormIndex, k: int, absent: bool = False) -> dict | None:
    """One GRPO row: the open-book prompt and what its answer is checked against.

    ``absent`` drops the item's own article from the retrieved texts; None
    when nothing else is left to show.
    """
    code, art = item.metadata["code"], str(item.metadata["articolo"])
    if absent:
        # index.articles_of, not record_articles: a chunk that continues the
        # article carries no header of its own, so record_articles reports
        # none, and the article's own text stayed in the prompt -- shown to
        # the model, and paid for denying it, since an assente row scores 1.0
        # only for abstaining.
        found = [r for r in index.search(item.question, k + 6)
                 if not (r.get("code") == code and art in index.articles_of(r))][:k]
        if not found:
            return None
        content, tipo = open_book_prompt(item.question, found), "assente"
    else:
        found = index.search(item.question, k)
        note = index.missing_article_note(item.question)
        content, tipo = open_book_prompt(item.question, found, note=note), item.metadata["tipo"]
    return {"id": item.id + ("-assente" if absent else ""),
            "prompt": [{"role": "user", "content": content}],
            "tipo": tipo, "keywords": list(item.keywords or []),
            "code": code, "articolo": art}


def cap_share(rows: list[dict], tipo: str, share: float, rng: random.Random) -> list[dict]:
    """``rows`` with at most ``share`` of the result of kind ``tipo``,
    dropping a random sample of that kind and nothing else."""
    kind = [r for r in rows if r["tipo"] == tipo]
    other = [r for r in rows if r["tipo"] != tipo]
    if not 0 <= share < 1:
        raise ValueError(f"share must be in [0, 1), got {share}")
    keep = min(len(kind), int(share * len(other) / (1 - share)))
    return other + rng.sample(kind, keep)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--exclude-exam", nargs="+", type=Path, default=[],
                    help="exams whose articles are never drawn (held-out and dev)")
    ap.add_argument("--per-code", type=int, default=400,
                    help="questions of each kind per code; a code with fewer gives all it has")
    ap.add_argument("--absent-share", type=float, default=0.3,
                    help="share of by-topic deadline questions also asked without their article")
    ap.add_argument("--inesistente-share", type=float, default=0.2,
                    help="largest share of the prompts that may ask about a nonexistent article")
    ap.add_argument("-k", type=int, default=3, help="retrieved texts per question, as the exam")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    exclude = exam_articles(args.exclude_exam)
    index = NormIndex.from_files(args.norms)
    items = [it for it in build_exam(index.records, per_code=args.per_code, seed=args.seed,
                                     exclude=exclude)
             if it.metadata.get("tipo") in KEPT]
    rng = random.Random(args.seed)
    rows = [r for it in items if (r := prompt_row(it, index, args.k))]
    topic = [it for it in items if it.metadata["tipo"] == "termine_argomento"]
    for it in rng.sample(topic, int(round(len(topic) * args.absent_share))):
        if row := prompt_row(it, index, args.k, absent=True):
            rows.append(row)
    # The exam builder makes per_code // 5 nonexistent articles in every code
    # but a deadline question only for an article with exactly one deadline,
    # so left alone the nonexistent ones were 61% of the first prompts file
    # (2026-10-01). Models already answer them right every time: a group of
    # eight right answers teaches nothing and costs a full generation, and a
    # file that is mostly "say it is not there" pushes towards refusing.
    rows = cap_share(rows, "inesistente", args.inesistente_share, rng)
    rng.shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = Counter(r["tipo"] for r in rows)
    kinds = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    print(f"[grpo] {len(rows)} prompts ({kinds}), {len(exclude)} exam articles left out "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
