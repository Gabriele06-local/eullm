#!/usr/bin/env python3
"""SFT pairs for answering from case law and citing it, from clean ruling cards.

    python forge/scripts/make_caselaw_sft.py $WORK/eullm_runs/cass/clean-civ.jsonl \\
        --kind snciv --mix $WORK/eullm_runs/stage3/openbook-v04.jsonl \\
        --out $WORK/eullm_runs/cass/sft-civ.jsonl

The skill RAG uses: a question, the case law retrieval puts in front of the
model, an answer that says what the right ruling holds and cites it, and
says so when none of them answers. Built only from the cards
`export_clean_cards.py` let out -- abstract principles under made-up
references -- so no ruling text, number or party reaches the weights.

For each exam question of a card (``domande_esame``: question, reference
answer) a pair is written in the case-law prompt the exam and the product
use (`caselaw.prompts.caselaw_prompt`):

* ``caselaw_grounded`` -- the card and ``k - 1`` others that BM25 finds for
  the question, in a shuffled order; the answer is the card's reference
  answer followed by its made-up citation. The student learns to pick the
  right ruling among related ones and to cite the one it used;
* ``caselaw_absent`` (``--absent-share`` of them) -- the same question with
  the card left out: only the others, and an answer saying the rulings
  shown do not decide the question. A model that cites a ruling it was not
  shown invents, the failure that costs lawyers sanctions;
* ``caselaw_wrong_cite`` (``--wrong-cite-share``) -- the question attributes
  the principle to a ruling that is not among those shown (a made-up number
  no card has); the answer says that ruling is not there, then answers from
  the one that is. Legal-it-8b let "Sezioni Unite n. 18474/2025" pass
  uncorrected, with another ruling in front of it (a tester's report of
  2026-10-18, on reddit.com).

Every citation in an answer ends with the passage's number, "[2]": the
same tester could not check a single answer, since none named the passage
it came from. ``[n]`` is what a RAG interface links to.

A passage is the card's principles and norms, the form a "massima" takes.
``--mix`` puts rows of other SFT files (the statute open-book pairs) beside
these, unchanged, so the statute skill the model has is trained too.

Counts only, never a card, a question or an answer.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.caselaw import court  # noqa: E402
from eullm_forge.caselaw.index import SparseBM25  # noqa: E402
from eullm_forge.caselaw.prompts import caselaw_prompt  # noqa: E402

ABSENT_ANSWER = (
    "Tra le sentenze riportate non ne trovo una che decida questa questione: "
    "affrontano temi diversi. Non posso quindi risponderti citando la giurisprudenza "
    "che ho davanti. Se riformuli la domanda, o indichi gli estremi di una pronuncia, "
    "posso cercare quella pertinente."
)


def label(card: dict) -> str:
    """A card's made-up citation: "Cass. civ., Sez. III, n. 51234/2023"."""
    e = card.get("estremi") or {}
    parts = [e.get("autorita") or "", e.get("sezione") or "",
             f"n. {e['numero']}/{e['anno']}" if e.get("numero") and e.get("anno") else ""]
    return ", ".join(p for p in parts if p)


def passage(card: dict) -> str:
    """What a card shows as a retrieved text: its principles, then its norms."""
    text = " ".join(card.get("principi", []))
    norms = "; ".join(card.get("norme", []))
    return text + (f"\nNorme: {norms}" if norms else "")


WRONG_CITE = ("Tra le sentenze riportate non c'è {fake}, quindi non posso confermare che "
              "abbia affermato questo principio. ")


def grounded_answer(reference: str, cite: str, pos: int) -> str:
    """The reference answer, then the citation and the passage's number: "... [2]."."""
    reference = reference.strip()
    if reference and reference[-1] not in ".!?":
        reference += "."
    return f"{reference} In questo senso {cite} [{pos}]."


def fake_label(card: dict, rng: random.Random, taken: set[str]) -> str:
    """A citation in the card's court and section whose number no card has."""
    e = card.get("estremi") or {}
    while True:
        fake = dict(e, numero=rng.randrange(50000, 60000))
        cite = label({"estremi": fake})
        if cite not in taken:
            return cite


def build(cards: list[dict], kind: str, k: int, absent_share: float,
          rng: random.Random, wrong_cite_share: float = 0.0) -> tuple[list[dict], Counter]:
    the_court = court(kind)
    stats: Counter = Counter()
    bm25 = SparseBM25([passage(c) for c in cards])
    taken = {label(c) for c in cards}
    rows = []
    for n, card in enumerate(cards):
        for i, q in enumerate(card.get("domande_esame", [])):
            question, reference = q.get("domanda", "").strip(), q.get("risposta", "").strip()
            if not question or not reference:
                stats["skipped:empty"] += 1
                continue
            others = [j for j in bm25.ranking(question, k + 4) if j != n][:k]
            draw = rng.random()
            absent = draw < absent_share and len(others) >= 1
            wrong = not absent and draw < absent_share + wrong_cite_share
            if absent:
                shown = others[:k]
                output = ABSENT_ANSWER
                task = "caselaw_absent"
            else:
                shown = [n] + others[:k - 1]
                rng.shuffle(shown)
                output = grounded_answer(reference, label(card), shown.index(n) + 1)
                task = "caselaw_grounded"
            if wrong:
                fake = fake_label(card, rng, taken)
                question = f"{question} Mi risulta che lo abbia stabilito {fake}: è così?"
                output = WRONG_CITE.format(fake=fake) + output
                task = "caselaw_wrong_cite"
            prompt = caselaw_prompt(question, [(label(cards[j]), passage(cards[j]))
                                               for j in shown], the_court)
            stats[task] += 1
            rows.append({"instruction": prompt, "output": output, "task": task,
                         "source": kind, "key": f"{card.get('id', n)}-{i}"})
    return rows, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cards", nargs="+", type=Path, help="export_clean_cards.py output")
    ap.add_argument("--kind", default="snciv", help="court of the cards (labels the prompt)")
    ap.add_argument("-k", type=int, default=3, help="rulings shown per question, as the exam")
    ap.add_argument("--absent-share", type=float, default=0.15)
    ap.add_argument("--wrong-cite-share", type=float, default=0.10,
                    help="questions citing a ruling that is not among those shown")
    ap.add_argument("--mix", nargs="*", type=Path, default=[],
                    help="SFT files whose rows are added unchanged (statute open-book pairs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        court(args.kind)
    except ValueError as e:
        ap.error(str(e))

    cards = []
    for p in args.cards:
        with p.open(encoding="utf-8") as f:
            cards += [json.loads(ln) for ln in f if ln.strip()]
    if any(not c.get("estremi") for c in cards):
        ap.error("cards without made-up references: give export_clean_cards.py output, "
                 "never the cards of the rulings themselves")
    rng = random.Random(args.seed)
    rows, stats = build(cards, args.kind, args.k, args.absent_share, rng,
                        args.wrong_cite_share)
    mixed = []
    for p in args.mix:
        with p.open(encoding="utf-8") as f:
            mixed += [json.loads(ln) for ln in f if ln.strip()]
    out = rows + mixed
    rng.shuffle(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print(f"[caselaw-sft] {len(cards):,} cards: {stats['caselaw_grounded']:,} grounded, "
          f"{stats['caselaw_absent']:,} absent and {stats['caselaw_wrong_cite']:,} wrong-citation "
          f"pairs, {len(mixed):,} mixed in, "
          f"{len(out):,} written -> {args.out}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
