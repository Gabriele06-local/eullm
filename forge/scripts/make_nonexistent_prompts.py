#!/usr/bin/env python3
"""OPD prompts about articles that do not exist, in the form the exam asks them.

    python forge/scripts/make_nonexistent_prompts.py \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --exclude $WORK/eval/norm-exam-devbig.jsonl \\
        --mix-with $WORK/eullm_runs/opd/prompts-q14.jsonl \\
        --out $WORK/eullm_runs/opd/prompts-q14-nx.jsonl

The Qwen3.5 students answered every "Che cosa prevede l'art. 3434 del codice
civile?" of the statute exam with "non è possibile rispondere": a refusal,
right in substance, that the judge grades wrong because it never says the
article may not exist -- 0 against 71 for the published 4B, which says "potrebbe
non esistere, oppure il riferimento potrebbe essere errato". A teacher note on
the absent-article rows did not move it (2026-10-14): those rows remove an
article that exists and carry no note, while the exam's question names a number
past the end of the code and its prompt opens with retrieval's own "Nota: nella
raccolta normativa non è presente art. N". The student had never met that
prompt, so it answered it its own way.

These rows are that prompt: an article number past the last one the collection
holds for its code, retrieved and worded exactly as legal_eval.py does
(NormIndex.search, missing_article_note, open_book_prompt), with a teacher note
saying how to answer it. Numbers the exams in --exclude ask about are never
drawn. With --mix-with, the rows of that prompts file come first, unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import NormIndex  # noqa: E402
from eullm_forge.eval.norm_exam import CODE_LABELS  # noqa: E402
from eullm_forge.eval.retrieval import open_book_prompt  # noqa: E402

QUESTIONS = (
    "Che cosa prevede l'art. {n} {of}?",
    "Cosa dice l'articolo {n} {of}?",
    "Quale termine prevede l'art. {n} {of}?",
    "Mi spieghi l'art. {n} {of}?",
)
NOTE = (
    "Nota riservata, che chi fa la domanda non vede: l'art. {n} non esiste {in_}: "
    "nella raccolta il testo arriva all'art. {last}. Rispondi in una o due frasi che "
    "l'art. {n} non compare nei testi a disposizione e che probabilmente non esiste "
    "{in_}, oppure che il numero potrebbe essere errato e va verificato. Non attribuire "
    "un contenuto all'articolo chiesto e non descrivere altri articoli.")


def last_articles(index: NormIndex) -> dict[str, int]:
    """The highest article number the collection holds, per code it labels."""
    last: dict[str, int] = {}
    for r in index.records:
        code = r.get("code")
        if code not in CODE_LABELS:
            continue
        for a in index.articles_of(r):
            m = re.match(r"\d+", str(a))
            if m:
                last[code] = max(last.get(code, 0), int(m.group()))
    return last


def asked(paths: list[Path]) -> set[tuple[str, int]]:
    """(code, article) of every item in the exams given: never drawn here."""
    out = set()
    for p in paths:
        for ln in p.open(encoding="utf-8"):
            if not ln.strip():
                continue
            meta = json.loads(ln).get("metadata") or {}
            m = re.match(r"\d+", str(meta.get("articolo", "")))
            if meta.get("code") and m:
                out.add((meta["code"], int(m.group())))
    return out


def rows_for(index: NormIndex, per_code: int, exclude: set[tuple[str, int]],
             rng: random.Random, k: int = 3) -> list[dict]:
    rows = []
    for code, last in sorted(last_articles(index).items()):
        of, in_ = CODE_LABELS[code]
        drawn: set[int] = set()
        tries = 0
        while len(drawn) < per_code and tries < per_code * 50:
            tries += 1
            n = last + rng.randint(1, 1000)
            if n in drawn or (code, n) in exclude:
                continue
            drawn.add(n)
            question = rng.choice(QUESTIONS).format(n=n, of=of)
            note = index.missing_article_note(question)
            if not note:   # the question did not read as naming that article of that code
                continue
            student = open_book_prompt(question, index.search(question, k), note=note)
            teacher = NOTE.format(n=n, in_=in_, last=last) + "\n\n---\n\n" + student
            rows.append({"id": f"nx-{code}-{n}", "kind": "nonexistent",
                         "student": [{"role": "user", "content": student}],
                         "teacher": [{"role": "user", "content": teacher}]})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--exclude", nargs="*", type=Path, default=[],
                    help="exam files whose articles are never drawn")
    ap.add_argument("--per-code", type=int, default=60)
    ap.add_argument("--mix-with", type=Path, help="an OPD prompts file whose rows come first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    index = NormIndex.from_files(args.norms)
    rows = rows_for(index, args.per_code, asked(args.exclude), random.Random(args.seed))
    mixed = ([json.loads(ln) for ln in args.mix_with.open(encoding="utf-8") if ln.strip()]
             if args.mix_with else [])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in mixed + rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print(f"[nonexistent] {len(rows):,} rows over {len({r['id'].split('-')[1] for r in rows})} "
          f"codes, after {len(mixed):,} mixed rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
