#!/usr/bin/env python3
"""Prompts for the privileged-context distillation (opd_train.py), from the ruling cards.

    python forge/scripts/make_opd_prompts.py \\
        --chunks $WORK/datasets/legal_it_amm/train.jsonl $WORK/datasets/legal_it_amm/val.jsonl \\
        --openga $WORK/eval/openga/cds-sentenze-2017-2024.csv \\
        --cards $WORK/eullm_runs/cds/schede.jsonl $WORK/eullm_runs/cds/schede-b.jsonl \\
        --dev-ids $WORK/eval/cds/cds-dev-ids.txt \\
        --statutes $WORK/eullm_runs/grpo/prompts-hyb06.jsonl \\
        --out $WORK/eullm_runs/opd/prompts.jsonl

Step 4 of the case-law plan (research report of 2026-10-05). For each
carded TRAINING ruling, up to ``--per-ruling`` of its research questions:

* the student's prompt is the question with the ``-k`` passages the
  case-law index retrieves for it (`eullm_forge.caselaw.index`, chunks with
  their card prefix and the cards themselves, ``--index``), as it will be
  asked in use;
* the teacher's is the same with the ruling the question came from in
  front (`caselaw.prompts.privileged_prompt`).

Left out, and checked rather than assumed:

* every development ruling (``--dev-ids``): no prompt is written from one;
* rulings whose appeal is about the GDPR's special categories and criminal
  matters (art. 9 and 10), told by OpenGA's OGGETTO_RICORSO
  (``--exclude-oggetto``): antimafia measures, residence permits, health,
  minors and the like. The rulings stay in the index; they are never the
  teacher's privileged source.

``--mix`` of the rows are statute questions from an open-book prompts file
(make_grpo_prompts.py output) with the teacher shown exactly what the
student is: what the models already do is held in place while the new
skill is learnt.

Counts only, never a question or a ruling.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.caselaw import attach_meta, load_openga, load_rulings, ruling_view  # noqa: E402
from eullm_forge.caselaw.index import RulingIndex, SparseBM25, build_units  # noqa: E402
from eullm_forge.caselaw.prompts import (  # noqa: E402
    caselaw_prompt,
    privileged_prompt,
    ruling_label,
)

SENSITIVE = (r"antimafia|interditti|informativ[ae] prefettizi|soggiorno|cittadinanza|"
             r"protezione internazionale|asilo|espulsion|rimpatri|sanit|salute|vaccin|disabil|"
             r"invalidit|handicap|minor[ei]|adozion|stupefacent|penal|reato|condann|casellario|"
             r"religio|culto|sindacal|orientamento sessuale|genetic|biometric")


def read_jsonl(paths) -> list[dict]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", nargs="+", type=Path, required=True)
    ap.add_argument("--openga", nargs="*", type=Path, default=[])
    ap.add_argument("--cards", nargs="+", type=Path, required=True)
    ap.add_argument("--dev-ids", type=Path, required=True)
    ap.add_argument("--statutes", type=Path, help="open-book statute prompts to mix in")
    ap.add_argument("--mix", type=float, default=0.25, help="share of statute rows")
    ap.add_argument("--per-ruling", type=int, default=3)
    ap.add_argument("--max-rulings", type=int, default=0, help="0: all")
    ap.add_argument("-k", type=int, default=3, help="passages in the student's prompt")
    ap.add_argument("--exclude-oggetto", default=SENSITIVE,
                    help="regex on OGGETTO_RICORSO: rulings never used as a source")
    ap.add_argument("--index", choices=["prefix", "prefix+cards"], default="prefix+cards",
                    help="units of the case-law index: chunks with their card prefix, and the "
                         "cards as units of their own besides (the retrieval check of "
                         "2026-10-07: recall@3 +5.4 and +7.2 points over plain chunks)")
    ap.add_argument("--embedder", help="retrieve with embeddings too (as cds_retrieval.py)")
    ap.add_argument("--reranker")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    dev = {ln.strip() for ln in args.dev_ids.open(encoding="utf-8") if ln.strip()}
    rulings = load_rulings(args.chunks)
    if args.openga:
        attach_meta(rulings, load_openga(args.openga))
    cards = {c["id"]: c for c in read_jsonl(args.cards)}
    sensitive = re.compile(args.exclude_oggetto, re.I)
    rng = random.Random(args.seed)

    chosen, n_dev, n_sens = [], 0, 0
    for rid, card in sorted(cards.items()):
        if rid in dev:
            n_dev += 1
            continue
        oggetto = card.get("oggetto") or (rulings[rid].meta.get("OGGETTO_RICORSO")
                                          if rid in rulings else "") or ""
        if sensitive.search(oggetto):
            n_sens += 1
            continue
        if rid in rulings:
            chosen.append(rid)
    rng.shuffle(chosen)
    if args.max_rulings:
        chosen = chosen[:args.max_rulings]

    chunks = [c for c in read_jsonl(args.chunks)
              if (c.get("kind") or str(c.get("sentence_id", "")).split("/")[0]) == "cds"]
    # Development rulings never reach a prompt, as a source or as a retrieved
    # passage. Their units are skipped when ranked rather than left out of the
    # index, so the index stays the one the exam and the retrieval check use,
    # and its cached embeddings (keyed on the units) still apply.
    units = build_units(rulings, chunks, cards=cards, prefix_chunks=True,
                        card_units=args.index == "prefix+cards")
    index = RulingIndex(units, bm25=SparseBM25([u.text for u in units]))
    rank_all = index.ranked_units
    index.ranked_units = lambda q: [i for i in rank_all(q) if units[i].ruling not in dev]
    if args.embedder:
        from eullm_forge.caselaw.index import shard_vectors, units_key
        from eullm_forge.eval.dense import Embedder, Reranker
        emb = Embedder(args.embedder, batch_size=32, max_length=768)
        index.vectors = shard_vectors([u.text[:3000] for u in units], emb.encode,
                                      args.cache_dir, units_key(units, args.embedder))
        index.query_fn = lambda q: emb.encode([q], query=True)[0]
        index.reranker = Reranker(args.reranker) if args.reranker else None

    def meta_of(rid: str) -> dict:
        r = rulings.get(rid)
        return {"sezione": getattr(r, "section", ""), "numero": rid.split("/", 1)[-1]}

    rows, n_citable = [], 0
    for rid in chosen:
        qs = list(cards[rid].get("domande_ricerca", []))
        rng.shuffle(qs)
        for q in qs[:args.per_ruling]:
            passages, found = [], set()
            for i in index.ranked_units(q):
                u = units[i]
                passages.append((ruling_label(meta_of(u.ruling)), u.text))
                found.add(u.ruling)
                if len(passages) >= args.k:
                    break
            source = (ruling_label(meta_of(rid)), ruling_view(rulings[rid].text, 12000))
            n_citable += rid in found
            rows.append({"id": f"{rid}#{len(rows)}", "kind": "caselaw", "ruling": rid,
                         "student": [{"role": "user", "content": caselaw_prompt(q, passages)}],
                         "teacher": [{"role": "user",
                                      "content": privileged_prompt(
                                          q, passages, source, citable=rid in found)}]})
    n_case = len(rows)
    if args.statutes and args.mix > 0:
        stat = read_jsonl([args.statutes])
        want = min(len(stat), int(round(n_case * args.mix / (1 - args.mix))))
        for r in rng.sample(stat, want):
            rows.append({"id": f"statute#{r.get('id', len(rows))}", "kind": "statute",
                         "student": r["prompt"], "teacher": r["prompt"]})
    rng.shuffle(rows)
    assert not any(r.get("ruling") in dev for r in rows), "a development ruling got through"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print(f"[opd-prompts] {len(rows):,} rows ({n_case:,} case-law from {len(chosen):,} rulings, "
          f"{len(rows) - n_case:,} statute; source among the passages in {n_citable:,}); "
          f"left out {n_dev:,} development and {n_sens:,} sensitive rulings -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
