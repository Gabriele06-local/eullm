#!/usr/bin/env python3
"""The trap test asked in other words: each question rewritten as people ask.

    python forge/scripts/paraphrase_trap_test.py $DATA/eval/trap-test.jsonl \\
        --servers http://127.0.0.1:18500,... --model <name in the engine> \\
        --out $DATA/eval/trap-test-para.jsonl

`make_trap_test.py` asks its questions in the templates the premise pairs are
trained on, and the models trained on them pass 98% of it (judge, 24-10-2026)
while the LexIA traps, written by people, stay at 11 of 36: the test measured
the templates. Here a teacher (Qwen3-30B-A3B-Instruct-2507 through EULLM
Engine, as for the pairs) rewrites every question in three styles -- a client
telling a case, a short colloquial question, a professional taking the fact
for granted -- keeping every fact of it, the false one included, and
answering nothing. A rewrite is kept only if it still carries every number
of the question (article, deadline, penalty, ruling) and corrects nothing;
the reference, the rubric and the grading metadata are the original's.
Counts only, never a question.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

STYLES = {
    "caso": "un caso concreto: una persona racconta in due o tre frasi la sua situazione "
            "e poi fa la domanda",
    "breve": "una domanda breve e colloquiale, con parole diverse da quelle originali",
    "tecnico": "un professionista (avvocato o praticante) che chiede conferma in tono "
               "tecnico, dando il fatto per scontato",
}
PROMPT = (
    "Riscrivi la domanda seguente come la scriverebbe una persona reale che si rivolge a un "
    "assistente legale, nello stile indicato.\n"
    "Regole:\n"
    "- mantieni esattamente tutti i fatti della domanda, anche se ti sembrano sbagliati: lo "
    "stesso articolo e lo stesso codice (o lo stesso nome del reato), gli stessi numeri, "
    "termini, pene e sentenze, scritti in cifre come nell'originale;\n"
    "- non correggere nulla, non aggiungere fatti nuovi, non rispondere;\n"
    "- scrivi soltanto la domanda riscritta, in italiano, senza virgolette e senza premesse.\n\n"
    "Stile: {style}\n\nDomanda: {question}"
)
# A rewrite that says the premise is wrong has answered the question.
CORRECTS = re.compile(r"\berrat[aoie]\b|in realtà|non è corrett|non esiste|abrogat", re.I)
NUMBER = re.compile(r"\d+(?:[./]\d+)*")


def faithful(original: str, rewrite: str) -> bool:
    """Every number of the question still there, nothing corrected, a sane length."""
    if not 15 <= len(rewrite) <= 900 or "\n\n" in rewrite.strip():
        return False
    if CORRECTS.search(rewrite) and not CORRECTS.search(original):
        return False
    have = set(NUMBER.findall(rewrite))
    return all(n in have for n in NUMBER.findall(original))


def clean(text: str) -> str:
    text = text.strip().strip('"«»“”').strip()
    return re.sub(r"^(domanda( riscritta)?|riscrittura)\s*:\s*", "", text, flags=re.I).strip()


def main(argv: list[str] | None = None) -> int:
    from generate_instructions import DEFAULT_MODEL, EullmGenerator

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", type=Path, help="the trap test (make_trap_test.py --out)")
    ap.add_argument("--servers", required=True, help="EULLM Engine URLs, comma-separated")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    items = [json.loads(ln) for ln in args.items.open(encoding="utf-8") if ln.strip()]
    styles = [s for s in args.styles.split(",") if s in STYLES]
    jobs = [(it, s) for it in items for s in styles]
    gen = EullmGenerator(args.servers.split(","), args.model, max_new_tokens=300,
                         concurrency=args.concurrency)
    texts, tokens = gen.generate([[{"role": "user", "content": PROMPT.format(
        style=STYLES[s], question=it["question"])}] for it, s in jobs])
    kept, dropped = [], Counter()
    for (it, style), text in zip(jobs, texts, strict=True):
        rewrite = clean(text)
        if not faithful(it["question"], rewrite):
            dropped[style] += 1
            continue
        kept.append({**it, "id": f"{it['id']}-{style}", "question": rewrite,
                     "metadata": {**it["metadata"], "paraphrase_of": it["id"], "style": style}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    by_style = Counter(it["metadata"]["style"] for it in kept)
    print(f"[paraphrase] {len(kept)} of {len(jobs)} rewrites kept "
          f"({', '.join(f'{s} {by_style[s]}' for s in styles)}; dropped "
          f"{', '.join(f'{s} {dropped[s]}' for s in styles)}), {tokens:,} tokens -> {args.out}")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
