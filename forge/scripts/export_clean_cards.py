#!/usr/bin/env python3
"""The ruling cards with nothing of the case left in them: case law without personal data.

    python forge/scripts/export_clean_cards.py \\
        $WORK/eullm_runs/cds/schede.jsonl $WORK/eullm_runs/cds/schede-b.jsonl \\
        --dev-ids $WORK/eval/cds/cds-dev-ids.txt \\
        --out $WORK/eullm_runs/cds/schede-clean.jsonl

The Consiglio di Stato rulings are pseudonymised, not anonymous: personal
data. A card (cds_schede.py) is what a teacher distils
out of a ruling -- principles in the abstract, norms applied, outcome,
questions -- and it was written to hold the law and not the case. This keeps
only the cards that are, checked rather than assumed, and takes off what
still points back at a ruling:

* every card of a development ruling (``--dev-ids``) is left out;
* so is every card of an appeal about the GDPR's special categories or
  criminal matters, read from OpenGA's OGGETTO_RICORSO and from the card's
  own "materia" (`SENSITIVE_OGGETTO`), and every card whose appeal has no
  OGGETTO to read;
* and every card whose text still names a party, a place or a figure of the
  case: a pseudonym placeholder, a tax code or IBAN (the cards' own check,
  run again), a named municipality, province, prefecture, university, local
  health or hospital authority, regional administrative court or company, an
  amount in euro;
* what is written has no ruling id, number, appeal number, section, date,
  OGGETTO or teacher. In their place are made-up references
  (``estremi``): a section of the court, a year of the corpus, and a number
  from 50000 to 59999, above any count the Consiglio di Stato reaches in a
  year, so a made-up reference can never be a real ruling's;
* the cards are shuffled before they are numbered, and no table from the
  new numbers to the rulings is kept anywhere.

``--kind snciv`` exports Corte di cassazione civil cards the same way. They
have no OGGETTO, so the special categories are read from the card's own
subject and text (`schede.CASS_SENSITIVE`, wider than the administrative
list: family, health and personal status come with civil cases), and the
made-up references are "Cass. civ." of 2021-2024. Criminal rulings
(``snpen``) are refused outright.

A model trained on these learns to answer from, and cite, the reference in
front of it, which is the skill; it cannot learn a real ruling's number,
which is what it would otherwise hallucinate. Counts only, never a card.

Whether the result is anonymous data is a legal judgement this script does
not make: it removes what it can check for, and says how much it removed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.caselaw import court  # noqa: E402
from eullm_forge.caselaw.schede import CASS_SENSITIVE, SENSITIVE_OGGETTO, _leaks  # noqa: E402

CONTENT = ("principi", "norme", "esito", "materia", "domande_ricerca", "domande_esame")
SECTIONS = ("Sez. III", "Sez. IV", "Sez. V", "Sez. VI", "Sez. VII")
YEARS = range(2017, 2025)        # the corpus: OpenGA 2017-2024
NUMBERS = range(50000, 60000)
# The Corte di cassazione's civil sections, and the years of its corpus
# (ItalGiure 2021-2026, training years to 2024). Its civil numbering stays
# under 40,000 a year, so 50000-59999 is no real ruling's either.
CASS_SECTIONS = ("Sez. I", "Sez. II", "Sez. III", "Sez. lav.", "Sez. trib.")
CASS_YEARS = range(2021, 2025)

_SENSITIVE = re.compile(SENSITIVE_OGGETTO, re.IGNORECASE)
_CASS_SENSITIVE = re.compile(CASS_SENSITIVE, re.IGNORECASE)
# A named body of the case. Capitalised on purpose: "comune di residenza"
# and "azienda sanitaria" in lower case are the law speaking in general.
# Regions are not in it: "l.r. Regione Lombardia n. 12/2005" is a law, and
# regional laws are in a good share of the cards' norms.
_NAMED = re.compile(
    r"\b(?:Comune|Provincia|Citt[àa] metropolitana|Prefettura|Questura|Universit[àa])"
    r"\s+(?:di\s+|del\s+|della\s+|dell['’]\s*)?[A-Z][a-zà-ù]+"
    r"|\b(?:ASL|A\.S\.L\.|ASP|ATS|Azienda\s+(?:Sanitaria|Ospedaliera|USL))\b"
    r"|\bT\.?A\.?R\.?\s+(?:per\s+(?:il|la|le|l['’])\s*)?(?:[A-Z][a-zà-ù]+)"
    r"|\b[A-Z][\w&'.-]*\s+(?:S\.?p\.?A|S\.?r\.?l|S\.?n\.?c|S\.?a\.?s)\b\.?")
_AMOUNT = re.compile(r"(?:€|\beuro\b)\s*\d|\d[\d.]*(?:,\d+)?\s*(?:€|\beuro\b)", re.IGNORECASE)


def why_not(row: dict, kind: str = "cds") -> str | None:
    """Why this card may not leave, or None if it may."""
    flat = json.dumps({k: row.get(k) for k in CONTENT}, ensure_ascii=False)
    if kind == "cds":
        oggetto = str(row.get("oggetto") or "").strip()
        if not oggetto:
            return "no_oggetto"
        if _SENSITIVE.search(oggetto):
            return "sensitive_oggetto"
        if _SENSITIVE.search(str(row.get("materia") or "")):
            return "sensitive_materia"
    else:
        # no OGGETTO for the Cassazione: the card's subject, then all its text
        if _CASS_SENSITIVE.search(str(row.get("materia") or "")):
            return "sensitive_materia"
        if _CASS_SENSITIVE.search(flat):
            return "sensitive_content"
    leak = _leaks(flat)
    if leak:
        return leak
    if _NAMED.search(flat):
        return "named_body"
    if _AMOUNT.search(flat):
        return "amount"
    return None


def made_up_reference(rng: random.Random, kind: str = "cds") -> dict:
    """A reference no real ruling has (see the module docstring)."""
    cds = kind == "cds"
    year = rng.choice(YEARS if cds else CASS_YEARS)
    day = date(year, 1, 1) + timedelta(days=rng.randrange(365))
    return {"autorita": court(kind).cite, "sezione": rng.choice(SECTIONS if cds else CASS_SECTIONS),
            "numero": rng.choice(NUMBERS), "anno": year, "data": day.isoformat()}


def export(rows: list[dict], dev: set[str], rng: random.Random,
           kind: str = "cds") -> tuple[list[dict], Counter]:
    stats: Counter = Counter()
    seen: set[str] = set()
    kept = []
    for row in rows:
        stats["read"] += 1
        if "reason" in row:              # a refusal from cds_schede.py, not a card
            stats["refusal_rows"] += 1
            continue
        rid = str(row.get("id", ""))
        # cards written before the court was recorded are the Consiglio di Stato's
        if (row.get("court") or (rid.split("/", 1)[0] if "/" in rid else "cds")) != kind:
            stats["other_court"] += 1
            continue
        if rid in dev:
            stats["dev"] += 1
            continue
        if rid in seen:
            stats["duplicate"] += 1
            continue
        seen.add(rid)
        reason = why_not(row, kind)
        if reason:
            stats["dropped:" + reason] += 1
            continue
        kept.append({k: row[k] for k in CONTENT if k in row})
    rng.shuffle(kept)
    out = [{"id": f"card-{i:06d}", "estremi": made_up_reference(rng, kind), **card}
           for i, card in enumerate(kept, 1)]
    stats["written"] = len(out)
    return out, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cards", nargs="+", type=Path, help="cds_schede.py output files (JSONL)")
    ap.add_argument("--dev-ids", type=Path, required=True,
                    help="development rulings: their cards never leave")
    ap.add_argument("--kind", default="cds",
                    help="court of the cards: cds, or snciv for the Cassazione's civil sections")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.kind == "snpen":
        ap.error("criminal rulings are never exported: art. 10 GDPR (see the module docstring)")
    try:
        court(args.kind)
    except ValueError as e:
        ap.error(str(e))

    dev = {ln.strip() for ln in args.dev_ids.open(encoding="utf-8") if ln.strip()}
    if not dev:
        ap.error(f"{args.dev_ids} is empty: refusing to export without the dev split")
    rows = []
    for p in args.cards:
        with p.open(encoding="utf-8") as f:
            rows.extend(json.loads(ln) for ln in f if ln.strip())
    out, stats = export(rows, dev, random.Random(args.seed), args.kind)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    dropped = sum(v for k, v in stats.items() if k.startswith("dropped:"))
    print(f"[clean-cards] {stats['read']:,} rows read: {stats['refusal_rows']:,} refusals, "
          f"{stats['dev']:,} dev, {stats['duplicate']:,} duplicates, "
          f"{stats['other_court']:,} of another court, {dropped:,} dropped, "
          f"{stats['written']:,} written -> {args.out}")
    for k, v in sorted(stats.items()):
        if k.startswith("dropped:"):
            print(f"[clean-cards]   {k[8:]}: {v:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
