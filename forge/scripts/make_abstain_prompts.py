#!/usr/bin/env python3
"""OPD prompts that teach the model to say when the texts do not hold the answer.

    python forge/scripts/make_abstain_prompts.py $WORK/eullm_runs/stage3/openbook-v04.jsonl \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --absent 1500 --grounded 1500 --no-text 800 \\
        --mix-with $WORK/eullm_runs/opd/prompts-v3.jsonl --mix 2000 \\
        --out $WORK/eullm_runs/opd/prompts-abstain.jsonl

On 2026-10-08, asked with no texts, legal-it-8b cited the wrong articles with
confidence and confirmed article numbers the user had made up. The models
answer from the texts they are given and were never taught what to do when
those texts do not hold the answer. Three kinds of row, in opd_train.py's
format (student and teacher message lists):

* ``absent``: a by-topic question with its own article taken out of the
  retrieved texts (`openbook_gen.absent_context_pair`, continuation chunks
  included since #735). The teacher reads the same texts behind a private
  note saying the answer is not among them, so it says so; the student, who
  sees no note, learns to tell from the texts themselves;
* ``statute``: the same questions with their article present, so that
  abstaining is learnt as a difference and not as a habit. With
  opd_train.py's ``--anchor-statutes`` they are taught by the student before
  the run, which holds the answers it already gives;
* ``no_text``: the question alone, as asked in a chat with nothing
  retrieved. The teacher's private note forbids citing articles from memory;

plus, with ``--mix-with``, rows of an existing OPD prompts file (the case-law
ones), so that the skills already learnt keep being trained.

No model is loaded; it runs on a login node. It prints counts only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.openbook_gen import absent_context_pair  # noqa: E402
from eullm_forge.eval import NormIndex  # noqa: E402

ABSENT_NOTE = (
    "Nota riservata, che chi fa la domanda non vede: l'articolo che risponde a questa domanda "
    "NON è tra i testi qui sotto. Rispondi che nei testi disponibili non hai trovato la norma che "
    "risponde alla domanda. Non indicare articoli a memoria e non ricavare la risposta da testi "
    "che parlano d'altro.")
NO_TEXT_NOTE = (
    "Nota riservata, che chi fa la domanda non vede: non ti è stato fornito nessun testo "
    "normativo. Non citare numeri di articoli né il loro contenuto a memoria, perché potresti "
    "sbagliarli. Rispondi che senza il testo della norma non puoi dare una risposta sicura, e "
    "indica soltanto, in termini generali, quale materia o quale codice andrebbe consultato.")


def _row(rid: str, kind: str, student: str, teacher: str | None = None) -> dict:
    return {"id": rid, "kind": kind,
            "student": [{"role": "user", "content": student}],
            "teacher": [{"role": "user", "content": teacher if teacher is not None else student}]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", type=Path, help="open-book pairs (openbook_gen output)")
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--absent", type=int, default=1500)
    ap.add_argument("--grounded", type=int, default=1500)
    ap.add_argument("--no-text", type=int, default=800)
    ap.add_argument("--mix-with", type=Path, help="an OPD prompts file to draw rows from")
    ap.add_argument("--mix", type=int, default=0, help="rows drawn from --mix-with")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    pairs = [json.loads(ln) for ln in args.pairs.open(encoding="utf-8") if ln.strip()]
    topic = [p for p in pairs if p.get("task") == "openbook_grounded"
             and not p.get("named", True) and "Domanda: " in p.get("instruction", "")]
    rng.shuffle(topic)
    index = NormIndex.from_files(args.norms)

    rows: list[dict] = []
    absent_keys = []
    for p in topic:
        if len(absent_keys) >= args.absent:
            break
        a = absent_context_pair(p, index)
        if a is None:
            continue
        rows.append(_row(a["key"], "absent", a["instruction"],
                         f"{ABSENT_NOTE}\n\n---\n\n{a['instruction']}"))
        absent_keys.append(p)
    # the same questions with their article present first, then other ones
    grounded = absent_keys + [p for p in topic if p not in absent_keys]
    for p in grounded[:args.grounded]:
        rows.append(_row(str(p.get("key")), "statute", p["instruction"]))
    for p in rng.sample(topic, min(args.no_text, len(topic))):
        question = p["instruction"].rsplit("Domanda: ", 1)[-1].strip()
        rows.append(_row(f"{p.get('key')}-notext", "no_text", question,
                         f"{NO_TEXT_NOTE}\n\nDomanda: {question}"))
    n_mix = 0
    if args.mix_with and args.mix:
        other = [json.loads(ln) for ln in args.mix_with.open(encoding="utf-8") if ln.strip()]
        case = [r for r in other if r.get("kind") == "caselaw"]
        for r in rng.sample(case, min(args.mix, len(case))):
            rows.append(r)
            n_mix += 1
    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    counts = ", ".join(f"{k} {v:,}" for k, v in sorted(kinds.items()))
    print(f"[abstain] {len(rows):,} rows: {counts} ({len(topic):,} by-topic pairs available) "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
