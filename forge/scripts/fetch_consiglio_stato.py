#!/usr/bin/env python3
"""Fetch Consiglio di Stato rulings as a held-out evaluation corpus.

**Why a different court at all.** The training corpus is Corte di Cassazione
(civile + penale, 2021-2026) plus codes and the Constitution. Any perplexity
measured on a slice of that corpus is measured on text the model's training
touched — and `format_pretraining.py` split it by *chunk*, so even the
nominal validation set shares documents with training: chunk 3 trained on,
chunk 4 held out, same parties and same citations on both sides. That
inflated the first quality result by an amount the measurement could not
bound, and demoted it from a result to an upper bound.

The Consiglio di Stato is not in the corpus at any granularity. It is Italian
legal prose of the same register, from a court the model has never read, so a
number measured here is clean by construction — and it answers the better
question: did the student learn Italian legal language, or memorise one court?

**Where the data comes from.** Two sources, because neither is enough alone:

* The OpenGA open-data portal publishes CdS ruling *metadata* — 17 columns,
  CC BY 4.0, monthly updates, no text. 16.7 MB covers 2017-2024, which works
  out to ~240 bytes per ruling: an index, not a corpus.
* The full text lives on the institutional portal, addressed by the two keys
  the metadata carries: `NUMERO_RICORSO` is the URL's `nrg` and
  `NUMERO_PROVVEDIMENTO` is its `nomeFile`.

So this reads the index, builds the addresses, and fetches the texts.

**It fetches few, slowly, and on purpose.** An evaluation corpus needs a few
hundred documents, not seventy thousand: 40 perplexity chunks is ~73 kB. The
default is 300 rulings with a second between requests, against a public
institution's server that owes us nothing. Do not raise the rate to make it
finish sooner.

Output is JSONL in the same shape as the training corpus, so
`make_ppl_corpus.py` consumes it with no changes.

Usage:
    # 1. the index (CC BY 4.0), once
    curl -o cds-2025.csv 'https://openga.giustizia-amministrativa.it/dataset/\
1112a570-e037-4611-82e2-b2a206149225/resource/\
edabfe63-db01-419c-a9d4-9f7d7a288062/download/cds-sentenze-2025.csv'

    # 2. the texts
    python forge/scripts/fetch_consiglio_stato.py \\
        --index cds-2025.csv --out $WORK/eval/cds-2025.jsonl --limit 300

    # 3. the perplexity corpus, unchanged
    python forge/scripts/make_ppl_corpus.py \\
        --val $WORK/eval/cds-2025.jsonl \\
        --out $WORK/eval/cds-40chunks.txt --target-chunks 40

Run it where the portal answers: it returns 503 to datacenter addresses.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PORTAL = "https://portali.giustizia-amministrativa.it/portale/pages/istituzionale/visualizza"

# The suffix on `nomeFile` is a document-kind code and is not always _11, so
# the ones seen in the wild are tried in order rather than assumed. Guessing a
# single value would silently yield an empty corpus.
NOME_FILE_SUFFIXES = ("_11", "_01", "_21")

# A ruling that parses to less than this is a stub, an error page rendered
# with HTTP 200, or a cookie banner. Discarding it here keeps the corpus from
# quietly filling with boilerplate that would flatter every model equally.
MIN_CHARS = 1500

# At least one of these must appear. Every Italian ruling has some section
# heading; a page with none of them is not a ruling, whatever its length.
MARKERS = re.compile(
    r"(?i)\b(fatto|diritto|p\.\s*q\.\s*m|per questi motivi|ha pronunciato|"
    r"il consiglio di stato|svolgimento del processo|motivi della decisione)\b"
)


def strip_html(raw: str) -> str:
    """HTML to plain text, without a parser dependency.

    Deliberately crude: script and style go first, then tags, then entities,
    then whitespace. Court pages are static documents, not applications, and
    a heavier dependency here would have to be installed on a login node.
    """
    raw = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = html.unescape(raw)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_a_ruling(text: str) -> bool:
    return len(text) >= MIN_CHARS and bool(MARKERS.search(text))


def build_url(nrg: str, provvedimento: str, suffix: str) -> str:
    q = urllib.parse.urlencode(
        {
            "nodeRef": "",
            "schema": "cds",
            "nrg": nrg,
            "nomeFile": f"{provvedimento}{suffix}.html",
            "subDir": "Provvedimenti",
        }
    )
    return f"{PORTAL}?{q}"


def fetch(url: str, timeout: int, user_agent: str) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            charset = r.headers.get_content_charset() or "utf-8"
            return r.read().decode(charset, errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def read_index(path: Path, only_sentenze: bool) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        head = f.read(8192)
        f.seek(0)
        delim = ";" if head.count(";") > head.count(",") else ","
        rows = list(csv.DictReader(f, delimiter=delim))
    if only_sentenze:
        rows = [
            r for r in rows
            if (r.get("TIPO_PROVVEDIMENTO") or "").strip().upper() == "SENTENZA"
        ]
    return rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--index", required=True, type=Path,
                   help="OpenGA CDS metadata CSV")
    p.add_argument("--out", required=True, type=Path,
                   help="JSONL to write (appended; already-fetched ids are skipped)")
    p.add_argument("--limit", type=int, default=300,
                   help="how many rulings to fetch (default: 300 — an eval "
                        "corpus needs hundreds, not thousands)")
    p.add_argument("--delay", type=float, default=1.0,
                   help="seconds between requests (default: 1.0). This is a "
                        "public institution's server; do not lower it.")
    p.add_argument("--seed", type=int, default=42,
                   help="sampling seed, so the same index gives the same corpus")
    p.add_argument("--timeout", type=int, default=45)
    p.add_argument("--user-agent",
                   default="eullm-eval/0.1 (research; contact info@i3k.eu)",
                   help="identify honestly; anonymous bulk scraping of a court "
                        "portal is both rude and a good way to get blocked")
    p.add_argument("--all-kinds", action="store_true",
                   help="keep ordinanze and decreti too (default: SENTENZA only)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.index.is_file():
        raise SystemExit(f"[err] no such index: {args.index}")

    rows = read_index(args.index, only_sentenze=not args.all_kinds)
    if not rows:
        raise SystemExit(
            f"[err] no usable rows in {args.index}. Expected the OpenGA CDS "
            f"columns (TIPO_PROVVEDIMENTO, NUMERO_RICORSO, "
            f"NUMERO_PROVVEDIMENTO)."
        )

    # Resume rather than refetch: this is slow by design, and a rerun after an
    # interruption should cost only what is missing.
    seen: set[str] = set()
    if args.out.is_file():
        for line in args.out.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line).get("source_id", ""))
            except ValueError:
                continue
        print(f"[cds] {len(seen):,} already in {args.out} — skipping those",
              file=sys.stderr)

    random.Random(args.seed).shuffle(rows)

    print(f"[cds] index     {args.index} ({len(rows):,} candidates)", file=sys.stderr)
    print(f"[cds] target    {args.limit} rulings, {args.delay}s apart", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = attempted = 0
    rejected_short = rejected_http = 0

    with args.out.open("a", encoding="utf-8") as sink:
        for row in rows:
            if written >= args.limit:
                break
            nrg = (row.get("NUMERO_RICORSO") or "").strip()
            prov = (row.get("NUMERO_PROVVEDIMENTO") or "").strip()
            if not nrg or not prov:
                continue
            source_id = f"cds/{prov}"
            if source_id in seen:
                continue

            attempted += 1
            text = None
            for suffix in NOME_FILE_SUFFIXES:
                raw = fetch(build_url(nrg, prov, suffix), args.timeout,
                            args.user_agent)
                time.sleep(args.delay)
                if raw is None:
                    rejected_http += 1
                    continue
                candidate = strip_html(raw)
                if looks_like_a_ruling(candidate):
                    text = candidate
                    break
                rejected_short += 1
            if text is None:
                continue

            rec = {
                "text": text,
                "source_id": source_id,
                "sentence_id": source_id,
                "year": int(row.get("ANNO_PUBBLICAZIONE") or 0) or None,
                "kind": "cds",
                "chunk_index": 0,
                "chunk_total": 1,
                "sezione": row.get("NOME_SEZIONE"),
                "data_pubblicazione": row.get("DATA_PUBBLICAZIONE"),
                "esito": row.get("ESITO_PROVVEDIMENTO"),
            }
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            written += 1
            seen.add(source_id)
            if written % 25 == 0:
                print(f"[cds] {written}/{args.limit} …", file=sys.stderr)

    print(f"\n[cds] written   {written} rulings → {args.out}", file=sys.stderr)
    print(f"[cds] attempted {attempted}", file=sys.stderr)
    print(f"[cds] discarded {rejected_short} too short / not a ruling, "
          f"{rejected_http} network or HTTP failures", file=sys.stderr)
    if written == 0:
        print(
            "[cds] Nothing was written. If every attempt failed at the network "
            "layer, the portal is refusing this host — it answers 503 to "
            "datacenter addresses. Run this from a normal connection.",
            file=sys.stderr,
        )
        return 1
    print("[cds] next: forge/scripts/make_ppl_corpus.py --val "
          f"{args.out} --out <corpus.txt> --target-chunks 40", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
