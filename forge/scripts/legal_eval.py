#!/usr/bin/env python3
"""Does a stage-3 model answer legal questions CORRECTLY? The gate before a release.

`chat_smoke.py` answers whether a model ends its turn. legal-it-4b v0.1c
passed it, 3 of 3, and then, read by a person, got both legal questions
wrong: article 2043 of the civil code became contractual good faith (artt.
1176 and 1375), and the ricorso straordinario was placed in article 111 of
the Constitution. Fluent, confident, invented. A format check cannot see
that, and "the loss went down" cannot either.

This asks every question of the held-out seed set
(`eullm_forge/eval/data/legal_it_heldout.seed.jsonl`), greedy, and scores
each answer by keyword coverage: the share of the terms a correct answer has
to contain that it does contain. Crude, and meant to be: it is
deterministic, it needs no judge model, and it ranks candidates on the same
questions. The answers are written out in full because the score only says
which model to READ first, not whether it is right. The seed items are
marked "da validare" and stay so until a lawyer has checked them.

Run it on every candidate AND on a reference, so a number means something:

    python forge/scripts/legal_eval.py <merged-dir or HF id> --label NAME \
        --csv legal-eval.csv --answers answers-NAME.jsonl

Qwen/Qwen3-4B-Instruct-2507 is the reference: same size, Qwen's own
instruction tuning. A vertical that does not beat it on its own domain is not
worth shipping. Exit status is 0 whatever the scores; this reports.

OPEN BOOK. With ``--norms`` every question is asked with the text of the
norms `NormIndex` retrieves for it placed in front (the named article when
the question names one, BM25 otherwise). Same model, same questions, the
text in hand: the difference between the two runs is how much of the error
is memory rather than understanding. The retrieved records are written into
the answers file, so a wrong answer can be told apart from a wrong retrieval.

    python forge/scripts/legal_eval.py <model> --label NAME-open \
        --norms "$CORPUS_DIR"/legislazione_*.chunks.jsonl ...

The keyword score is a first sort. `judge_answers.py` grades the answers
files against the references with a large model, which is the number to
decide on.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import (  # noqa: E402
    NormIndex,
    evaluate_qa,
    load_eval_set,
    load_seed,
    open_book_prompt,
)
from eullm_forge.eval.retrieval import label as norm_label  # noqa: E402


def summary_row(label: str, model: str, report: dict, ended: int) -> list:
    """One CSV row per model: the numbers that rank candidates."""
    s = report["summary"]
    full = sum(1 for r in report["per_item"] if r["keyword_coverage"] == 1.0)
    return [time.strftime("%Y-%m-%dT%H:%M:%S"), label, model, s["n"],
            f"{s['keyword_coverage']:.3f}", full, ended]


CSV_HEADER = ["timestamp", "label", "model", "items", "keyword_coverage",
              "fully_covered", "ended_turn"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="merged HF directory or hub id")
    ap.add_argument("--label", default="")
    ap.add_argument("--items", help="eval set JSONL (default: the legal-it seed)")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--csv", help="append the summary row here")
    ap.add_argument("--answers", help="write every answer, with its score, here")
    ap.add_argument("--norms", nargs="+",
                    help="legislazione_*.chunks.jsonl files: ask OPEN BOOK, "
                         "with the retrieved norms in the prompt")
    ap.add_argument("--k", type=int, default=3, help="norms retrieved per question")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = load_eval_set(args.items) if args.items else load_seed()
    index = NormIndex.from_files(args.norms) if args.norms else None
    if index:
        print(f"[eval] open book: {len(index.records):,} legislation records, "
              f"{args.k} per question", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    end_ids = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
               if isinstance(i, int) and i >= 0]

    answers, contexts, ended = {}, {}, 0
    for it in items:
        content = it.question
        if index:
            found = index.search(it.question, args.k)
            contexts[it.id] = [norm_label(r) for r in found]
            content = open_book_prompt(it.question, found)
            print(f"\n[eval] {it.id} reads: {'; '.join(contexts[it.id]) or 'nothing found'}")
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, eos_token_id=end_ids)
        new = out[0, ids["input_ids"].shape[1]:]
        ended += len(new) > 0 and int(new[-1]) in end_ids
        answers[it.id] = tok.decode(new, skip_special_tokens=True).strip()
        print(f"\n[eval] {it.id}: {it.question}\n{answers[it.id]}", flush=True)

    report = evaluate_qa(items, answers)
    for r in report["per_item"]:
        print(f"[eval] {r['id']:<20} keywords {r['keyword_coverage']:.2f}")
    s = report["summary"]
    print(f"\n[eval] {args.label or args.model}: keyword coverage "
          f"{s['keyword_coverage']:.3f} over {s['n']} items, "
          f"{ended}/{s['n']} ended their turn", flush=True)

    if args.answers:
        with open(args.answers, "w", encoding="utf-8") as f:
            for it, r in zip(items, report["per_item"]):
                f.write(json.dumps({"id": it.id, "question": it.question,
                                    "answer": answers[it.id], "reference": it.reference,
                                    "rubric": it.rubric,
                                    "keyword_coverage": r["keyword_coverage"],
                                    "context": contexts.get(it.id)},
                                   ensure_ascii=False) + "\n")
    if args.csv:
        path = Path(args.csv)
        new_file = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(CSV_HEADER)
            w.writerow(summary_row(args.label, args.model, report, ended))
    return 0


if __name__ == "__main__":
    sys.exit(main())
