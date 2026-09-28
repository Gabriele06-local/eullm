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
    # None for an item the keyword metric does not measure, so it is not
    # counted here either.
    full = sum(1 for r in report["per_item"] if r["keyword_coverage"] == 1.0)
    return [time.strftime("%Y-%m-%dT%H:%M:%S"), label, model, s["n"],
            f"{s['keyword_coverage']:.3f}", full, ended,
            s.get("keyword_items", s["n"])]


# keyword_items is last so the columns before it keep their position: it says
# how many of `items` keyword_coverage and fully_covered were measured over,
# which is not all of them once an exam carries judge-graded items.
CSV_HEADER = ["timestamp", "label", "model", "items", "keyword_coverage",
              "fully_covered", "ended_turn", "keyword_items"]


def append_csv_row(path: Path, row: list[str]) -> None:
    """Append one summary row, writing the header if the file has none yet.

    Existence is not enough: `open("a")` creates the file and the writer's
    buffer is only flushed after the first row, so a run interrupted in that
    window leaves a 0-byte CSV. The next run sees a file that exists, skips
    the header, and the first model's row lands where the header belongs —
    `csv.DictReader` then reads no rows and the ranking is silently empty.
    The header is flushed before the row rather than after it.
    """
    needs_header = not (path.exists() and path.stat().st_size > 0)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if needs_header:
            w.writerow(CSV_HEADER)
            f.flush()
        w.writerow(row)
        f.flush()


def chat_prompt(tok, content: str) -> str:
    """One user turn in the model's chat format, with thinking switched off.

    Qwen3's hybrid models (Qwen3-8B and the like) think by default: they open
    with a <think> block that eats the answer budget and is graded as the
    answer. ``enable_thinking=False`` is the switch their template reads; a
    template that has no such switch (Qwen3-4B-Instruct-2507, base models)
    ignores the extra variable, so every model is asked the same way.
    """
    return tok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def generate_answers(model, tok, prompts: list[str], *, batch_size: int,
                     max_new_tokens: int, end_ids: list[int]) -> list[tuple[str, bool]]:
    """Greedy answers to already-templated prompts, ``batch_size`` at a time.

    Returns (answer, ended_its_turn) per prompt. In a batch a finished
    sequence is padded until the longest one ends, so "ended" is whether an
    end token was generated at all, and the answer is cut there.
    """
    import torch

    tok.padding_side = "left"          # decoder-only: pad before the prompt
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out_all = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=end_ids, pad_token_id=tok.pad_token_id)
        for row in out[:, enc["input_ids"].shape[1]:].tolist():
            cut = next((j for j, t in enumerate(row) if t in end_ids), None)
            text = tok.decode(row[:cut] if cut is not None else row,
                              skip_special_tokens=True).strip()
            out_all.append((text, cut is not None))
    return out_all


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
    ap.add_argument("--batch-size", type=int, default=0,
                    help="questions generated together (default: 16 on GPU, 1 on CPU)")
    ap.add_argument("--quiet", action="store_true",
                    help="print totals only, never a question or an answer: for the "
                         "held-out exam, whose questions nobody improving the models "
                         "should read (see make_norm_exam.py)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = load_eval_set(args.items) if args.items else load_seed()
    index = NormIndex.from_files(args.norms) if args.norms else None
    if index:
        print(f"[eval] open book: {len(index.records):,} legislation records, "
              f"{args.k} per question", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if cuda:
        model.to("cuda")
    model.eval()
    end_ids = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
               if isinstance(i, int) and i >= 0]
    batch = args.batch_size or (16 if cuda else 1)
    print(f"[eval] {len(items)} items on {'GPU' if cuda else 'CPU'}, batch {batch}", flush=True)

    contexts, prompts = {}, []
    for it in items:
        content = it.question
        if index:
            found = index.search(it.question, args.k)
            note = index.missing_article_note(it.question)
            contexts[it.id] = [norm_label(r) for r in found] + ([note] if note else [])
            content = open_book_prompt(it.question, found, note=note)
        prompts.append(chat_prompt(tok, content))
    results = generate_answers(model, tok, prompts, batch_size=batch,
                               max_new_tokens=args.max_new_tokens, end_ids=end_ids)
    answers = {it.id: text for it, (text, _) in zip(items, results)}
    ended = sum(e for _, e in results)
    if not args.quiet:
        for it in items:
            if index:
                print(f"\n[eval] {it.id} reads: {'; '.join(contexts[it.id]) or 'nothing found'}")
            print(f"\n[eval] {it.id}: {it.question}\n{answers[it.id]}", flush=True)

    report = evaluate_qa(items, answers)
    if not args.quiet:
        for r in report["per_item"]:
            cov = r["keyword_coverage"]
            shown = "not scored" if cov is None else f"{cov:.2f}"
            print(f"[eval] {r['id']:<20} keywords {shown}")
    s = report["summary"]
    print(f"\n[eval] {args.label or args.model}: keyword coverage "
          f"{s['keyword_coverage']:.3f} over {s.get('keyword_items', s['n'])} of "
          f"{s['n']} items, "
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
        append_csv_row(Path(args.csv), summary_row(args.label, args.model, report, ended))
    return 0


if __name__ == "__main__":
    sys.exit(main())
