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
import http.cookiejar
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# The document host, and it is not the one the portal's own links advertise.
#
# A search result gives
# portali.giustizia-amministrativa.it/portale/pages/istituzionale/visualizza,
# and a browser following it ends up on mdp.giustizia-amministrativa.it with a
# shorter path. Opening one ruling by hand and reading the address bar is what
# established that — the first version used the advertised URL and fetched
# nothing. Both are kept, canonical first: if the redirect ever moves again,
# the fallback keeps this working rather than failing wholesale.
PORTALS = (
    "https://mdp.giustizia-amministrativa.it/visualizza/",
    "https://portali.giustizia-amministrativa.it/portale/pages/istituzionale/visualizza",
)

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


# Where the ruling actually starts. What precedes it is the court's internal
# document header, which the served HTML carries and the browser view hides:
# a urn, two generated filenames, a Windows path into their document store
# (`U:\DocumentiGA\Magistrati\…`), the drafting magistrate's name and a
# handful of timestamps. None of it is the judgment, all of it would be
# measured as if it were, and one line of it is somebody's internal file path.
START_ANCHORS = (
    re.compile(r"REPUBBLICA ITALIANA", re.I),
    re.compile(r"N\.\s*\d+/\d{4}\s*REG\.\s*PROV\.\s*COLL", re.I),
    re.compile(r"Il Consiglio di Stato\s+in sede giurisdizionale", re.I),
    re.compile(r"Il Consiglio di Stato", re.I),
)

# Belt and braces for the case where no anchor matches and the preamble would
# otherwise survive: these two patterns are unmistakably internal and never
# appear in the text of a judgment.
INTERNAL_JUNK = (
    re.compile(r"[A-Z]:\\[^\s]+"),
    re.compile(r"urn:nir:[^\s]+"),
)


def trim_preamble(text: str) -> tuple[str, bool]:
    """Cut the internal header. Returns (text, anchored)."""
    best = None
    for rx in START_ANCHORS:
        m = rx.search(text)
        if m and (best is None or m.start() < best):
            best = m.start()
    if best is not None and best > 0:
        return text[best:].lstrip(), True
    for rx in INTERNAL_JUNK:
        text = rx.sub(" ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip(), best is not None


def looks_like_a_ruling(text: str) -> bool:
    return len(text) >= MIN_CHARS and bool(MARKERS.search(text))


def build_url(nrg: str, provvedimento: str, suffix: str, portal: str) -> str:
    q = urllib.parse.urlencode(
        {
            "nodeRef": "",
            "schema": "cds",
            "nrg": nrg,
            "nomeFile": f"{provvedimento}{suffix}.html",
            "subDir": "Provvedimenti",
        }
    )
    return f"{portal}?{q}"


def make_opener() -> urllib.request.OpenerDirector:
    """An opener that keeps cookies, like any ordinary HTTP client.

    The document host answers 401 to a bare urllib request and serves the
    same page to a browser with no login at all. That is a bot filter, not
    authentication: it wants the headers every browser sends and the session
    cookie the portal hands out on first contact. Keeping a cookie jar and
    sending the usual headers is what a normal client does, not a way past an
    access control — these are public judgments, published to be read.
    """
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def browser_headers(user_agent: str, referer: str | None = None) -> dict[str, str]:
    h = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    if referer:
        h["Referer"] = referer
    return h


def warm_up(opener: urllib.request.OpenerDirector, timeout: int,
            user_agent: str) -> str:
    """Visit the portal once so the session cookie exists before the first
    document request. Returns a diagnosis, empty when it worked."""
    url = "https://www.giustizia-amministrativa.it/web/guest/dcsnprr"
    req = urllib.request.Request(url, headers=browser_headers(user_agent))
    try:
        with opener.open(req, timeout=timeout) as r:
            r.read(2048)
        return ""
    except Exception as e:  # noqa: BLE001 — any failure here is non-fatal
        return f"{type(e).__name__}: {e}"


def fetch(url: str, timeout: int, user_agent: str,
          opener: urllib.request.OpenerDirector | None = None,
          referer: str | None = None) -> tuple[str | None, str]:
    """Return (body, reason). `reason` is "ok" or a short diagnosis.

    The reason is returned rather than swallowed because the three ways this
    fails need three different responses and look identical from the outside:
    503 means the portal is refusing this host and no amount of retrying will
    help, 404 means the nomeFile suffix is wrong and another should be tried,
    and a timeout means the network. The first version returned None for all
    of them, so a run that failed 900 times could not say why.
    """
    req = urllib.request.Request(url, headers=browser_headers(user_agent, referer))
    open_it = opener.open if opener is not None else urllib.request.urlopen
    try:
        with open_it(req, timeout=timeout) as r:
            charset = r.headers.get_content_charset() or "utf-8"
            return r.read().decode(charset, errors="replace"), "ok"
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return None, f"network: {e.reason}"
    except TimeoutError:
        return None, "timeout"
    except OSError as e:
        return None, f"os: {e}"


def read_index(path: Path, only_sentenze: bool) -> list[dict]:
    """Parse the OpenGA CSV, refusing anything that is not one.

    A failed download leaves an HTML error page at the path the caller asked
    for, and `csv.DictReader` parses HTML happily — into one column of
    nonsense. Without this check that surfaces an hour later as "every fetch
    failed", which is the wrong diagnosis entirely.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        head = f.read(8192)
        if head.lstrip()[:1] == "<":
            raise SystemExit(
                f"[err] {path} is HTML, not CSV — the index download failed and "
                f"saved an error page. Re-download it and check the first line:\n"
                f"      head -1 {path}"
            )
        f.seek(0)
        delim = ";" if head.count(";") > head.count(",") else ","
        rows = list(csv.DictReader(f, delimiter=delim))

    required = {"TIPO_PROVVEDIMENTO", "NUMERO_RICORSO", "NUMERO_PROVVEDIMENTO"}
    present = set(rows[0].keys()) if rows else set()
    missing = required - present
    if missing:
        raise SystemExit(
            f"[err] {path} is missing the columns this needs: "
            f"{', '.join(sorted(missing))}.\n"
            f"      Found: {', '.join(sorted(present)) or '(no columns)'}\n"
            f"      Expected the OpenGA 'CDS - Sentenze' CSV."
        )

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
    p.add_argument("--timeout", type=int, default=15,
                   help="seconds per request (default: 15). A ruling costs up "
                        "to six requests, so a high value turns a failing run "
                        "into minutes of silence before anything is reported.")
    p.add_argument("--user-agent",
                   default=("Mozilla/5.0 (compatible; eullm-eval/0.1; "
                            "research; +mailto:info@i3k.eu)"),
                   help="kept honest and browser-shaped at once: the plain "
                        "identifier alone drew HTTP 401 from the document "
                        "host, and an anonymous browser string would hide "
                        "who is asking. This says both.")
    p.add_argument("--give-up-after", type=int, default=12,
                   help="stop after this many failed REQUESTS while nothing "
                        "has been fetched (default: 12). Counted in requests, "
                        "not rulings: a ruling costs up to six, so a limit in "
                        "rulings is half an hour of apparent hang.")
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

    opener = make_opener()
    problem = warm_up(opener, args.timeout, args.user_agent)
    if problem:
        print(f"[cds] warm-up failed ({problem}) — continuing, but the "
              f"document host may answer 401 without a session cookie",
              file=sys.stderr)
    else:
        print("[cds] session  established", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = attempted = 0
    rejected_short = 0
    reasons: dict[str, int] = {}
    requests_made = failed_requests = unanchored = 0

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
            for portal in PORTALS:
                for suffix in NOME_FILE_SUFFIXES:
                    raw, reason = fetch(
                        build_url(nrg, prov, suffix, portal),
                        args.timeout, args.user_agent, opener=opener,
                        referer="https://www.giustizia-amministrativa.it/",
                    )
                    requests_made += 1
                    time.sleep(args.delay)
                    if raw is None:
                        reasons[reason] = reasons.get(reason, 0) + 1
                        failed_requests += 1
                        # Say something on every early attempt. Six requests
                        # per ruling at the default timeout is minutes of
                        # silence, which reads as a hang — and did.
                        if written == 0:
                            print(f"[cds]   attempt {requests_made}: {reason}",
                                  file=sys.stderr)
                        continue
                    if raw.lstrip()[:5] == "%PDF-":
                        reasons["PDF, not HTML"] = (
                            reasons.get("PDF, not HTML", 0) + 1
                        )
                        continue
                    candidate, anchored = trim_preamble(strip_html(raw))
                    if looks_like_a_ruling(candidate):
                        text = candidate
                        if not anchored:
                            unanchored += 1
                        break
                    reasons["fetched but not a ruling"] = (
                        reasons.get("fetched but not a ruling", 0) + 1
                    )
                    rejected_short += 1
                if text is not None:
                    break

            # Stop early when nothing is working, counting REQUESTS.
            #
            # Counting rulings was the second version of this mistake. The
            # first had no limit at all and ran ninety minutes; then the limit
            # counted rulings, and since each ruling costs up to six requests
            # of `--timeout` seconds, eight of them is over half an hour of
            # apparent hang. The unit that costs time is the request, so that
            # is the unit the limit has to be in.
            if text is None:
                if failed_requests >= args.give_up_after and written == 0:
                    print(
                        f"\n[cds] {failed_requests} requests failed and nothing "
                        f"has been fetched — stopping rather than working "
                        f"through {len(rows):,} more rulings.",
                        file=sys.stderr,
                    )
                    break
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
    if unanchored:
        print(f"[cds] {unanchored} kept without a start anchor — their\n"
              f"[cds]   internal header was scrubbed by pattern instead of cut,\n"
              f"[cds]   so check a couple by hand", file=sys.stderr)
    if reasons:
        print("[cds] failures by reason:", file=sys.stderr)
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"[cds]   {count:5d}  {reason}", file=sys.stderr)

    if written == 0:
        # Name the likely cause from what was actually observed, instead of
        # offering the same three guesses whatever happened.
        top = max(reasons, key=reasons.get) if reasons else ""
        print("\n[cds] Nothing was written.", file=sys.stderr)
        if "503" in top:
            print(
                "[cds] Every request came back 503: the portal is refusing this\n"
                "[cds] host. It does that for datacenter addresses. Run this from\n"
                "[cds] an ordinary connection, or from the cluster's login node.",
                file=sys.stderr,
            )
        elif "404" in top:
            print(
                "[cds] Every request came back 404: the addresses are wrong, not\n"
                f"[cds] refused. The nomeFile suffixes tried were "
                f"{', '.join(NOME_FILE_SUFFIXES)}; open one ruling in a browser\n"
                "[cds] from the portal's own search and read the suffix out of the\n"
                "[cds] URL, then add it to NOME_FILE_SUFFIXES.",
                file=sys.stderr,
            )
        elif "not a ruling" in top:
            print(
                "[cds] Pages were fetched but none parsed as a ruling — the portal\n"
                "[cds] is probably returning a search form or a consent page at\n"
                "[cds] HTTP 200. Save one by hand and look at it.",
                file=sys.stderr,
            )
        elif top:
            print(f"[cds] Dominant failure: {top}. Check connectivity first.",
                  file=sys.stderr)
        return 1
    print("[cds] next: forge/scripts/make_ppl_corpus.py --val "
          f"{args.out} --out <corpus.txt> --target-chunks 40", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
