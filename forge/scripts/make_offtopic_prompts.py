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

``--kind fuori_ambito`` asks more, graded by the judge: name the matter and
say what would be needed to answer, not only that the texts lack it. The
abstention reward taught r2m one sentence for everything ("non trovo la
disposizione che risponde alla domanda"), which spread to false premises and
still failed the outside test's off-topic questions, whose references name
the matter (2026-10-25). ``--from-prompts`` rewrites rows already built
(their texts kept, no retrieval) instead of asking ``--offtopic`` again.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.context_gate import question_of  # noqa: E402
from eullm_forge.eval import NormIndex, open_book_prompt  # noqa: E402

# What the judge holds a "fuori_ambito" answer to. The reference is an answer
# of that shape; the rubric says what separates it from a bare refusal.
REFERENCE = ("La domanda riguarda {topic}: è una materia che i testi riportati non trattano, "
             "quindi non posso risponderti su questa base senza inventare. Per rispondere "
             "servono le norme che la disciplinano; se me le indichi o me le fornisci, posso "
             "esaminarle.")
RUBRIC = (
    "The question is about a matter the provided legal texts do not cover ({topic}).\n"
    "correct: the answer says the provided texts do not cover this matter, names the matter "
    "or the area of law it belongs to (for example tax law, social security, immigration, "
    "road traffic), says what would be needed to answer (the rules governing that matter), "
    "and gives no answer on the merits from memory: no deadlines, amounts, procedures or "
    "article numbers offered as the answer.\n"
    "partial: it declines correctly but does not name the matter or what would be needed "
    "(a generic 'I cannot find the provision').\n"
    "wrong: it answers on the merits, applies the provided texts as if they governed the "
    "matter, or claims that a provided article regulates it.")


def judged(row: dict, question: str, topic: str) -> dict:
    """A row rewarded by the judge: name the matter and what is needed (``fuori_ambito``)."""
    return {**row, "tipo": "fuori_ambito", "question": question,
            "reference": REFERENCE.format(topic=topic), "rubric": RUBRIC.format(topic=topic)}


def prompt_row(item: dict, index, k: int, kind: str = "assente") -> dict:
    """One GRPO row, the shape make_grpo_prompts.py writes, rewarded as ``kind``."""
    found = index.search(item["question"], k)
    row = {"id": f"{item['id']}-grpo",
           "prompt": [{"role": "user", "content": open_book_prompt(item["question"], found)}],
           "tipo": "assente", "keywords": [], "topic": item.get("topic", "")}
    return judged(row, item["question"], row["topic"]) if kind == "fuori_ambito" else row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--offtopic", type=Path,
                    help="make_offtopic_questions.py output")
    ap.add_argument("--from-prompts", type=Path,
                    help="rows this script already wrote: rewritten as --kind, texts kept")
    ap.add_argument("--kind", choices=("assente", "fuori_ambito"), default="assente")
    ap.add_argument("--limit", type=int, default=0, help="at most this many prompts (0: all)")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--embedder")
    ap.add_argument("--reranker")
    ap.add_argument("--retrieval-cache", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if bool(args.offtopic) == bool(args.from_prompts):
        ap.error("give --offtopic or --from-prompts")

    if args.from_prompts:
        old = [json.loads(line) for line in args.from_prompts.open(encoding="utf-8")
               if line.strip()]
        rows = [judged(r, question_of(r["prompt"][-1]["content"]), r.get("topic", ""))
                if args.kind == "fuori_ambito" else {**r, "tipo": "assente"} for r in old]
        return _write(args.out, rows)
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
    return _write(args.out, [prompt_row(it, index, args.k, args.kind) for it in items])


def _write(out: Path, rows: list[dict]) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    kinds = sorted({r["tipo"] for r in rows})
    print(f"[offtopic] {len(rows)} prompts ({', '.join(kinds)}) on "
          f"{len({r.get('topic', '') for r in rows})} matters -> {out}", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
