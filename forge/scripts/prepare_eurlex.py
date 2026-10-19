#!/usr/bin/env python3
"""EU acts in Italian, as saved from the Publications Office, into training chunks.

Reads a directory of ``<CELEX>.html.gz`` (or ``.html``) files, one act each,
in the two layouts the Office serves: the XHTML of the acts published since
2014 (``oj-ti-art`` article headings, ``oj-sti-art`` titles) and the plain
HTML before (an ``Articolo N`` paragraph). Fetching them is not done here.

Each act is cut where an article begins: the text before the first article
(title, citations, recitals) is the act's ``preambolo``, every article one
record, every annex (``ALLEGATO``) one more. Each record starts with the
act's title, so a chunk says which act it comes from. Footnotes are left
out. Then the chunker of prepare_legislation.py, into one file:

    python forge/scripts/prepare_eurlex.py ~/eurlex-it --output <corpus dir>
    # -> <corpus dir>/legislazione_eurlex.chunks.jsonl, code eurlex_<CELEX>

which make_norms_split.py takes beside the Normattiva files. The Italian
text of an EU act is official and public: no personal data.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_legislation import chunk_records  # noqa: E402

from eullm_forge.datasets.chunk import ChunkConfig  # noqa: E402

ARTICLE = re.compile(
    r"^Articolo\s+(unico|\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?"
    r"(?:\s*[a-z])?)\s*$", re.IGNORECASE)
ANNEX = re.compile(r"^ALLEGAT[OI](?:\s+([IVXLC]+|\d+|[A-Z]))?\s*$")
SKIP_CLASSES = ("note", "signatory")
# The pre-2014 pages open with EUR-Lex's own lines: a legal notice, the CELEX
# number, a row of asterisks. Not the act.
BOILERPLATE = re.compile(r"^(?:Avis juridique important|\*+|[0-9]{5}[A-Z][0-9]{4})$")
TITLE_CLASSES = ("oj-doc-ti", "doc-ti")
SUBTITLE_CLASSES = ("oj-sti-art", "sti-art")


class Paragraphs(HTMLParser):
    """The text of every <p> (and table cell), in order, with its class."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[tuple[str, str]] = []
        self._cls: str | None = None
        self._buf: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("p", "td") and self._cls is None:
            self._cls, self._buf = cls, []
        elif tag == "br" and self._cls is not None:
            self._buf.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in ("p", "td") and self._cls is not None:
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if text:
                self.out.append((self._cls, text))
            self._cls = None

    def handle_data(self, data):
        if self._cls is not None and not self._skip:
            self._buf.append(data)


def parse_act(html: str, celex: str) -> list[dict]:
    """The act's records: preambolo, one per article, one per annex."""
    p = Paragraphs()
    p.feed(html)
    paras = [(c, t) for c, t in p.out
             if not any(k in c for k in SKIP_CLASSES) and not BOILERPLATE.match(t)]
    if not paras:
        return []
    # The title paragraphs before the first article: annexes reuse the class.
    first = next((i for i, (_, t) in enumerate(paras) if ARTICLE.match(t)), len(paras))
    titles = [t for c, t in paras[:first] if any(k in c.split() for k in TITLE_CLASSES)]
    longer = [t for _, t in paras if len(t) >= 30]
    act_title = " ".join(titles) if titles else (longer or [paras[0][1]])[0]
    # The old pages follow the title with where it was published: not the act.
    act_title = re.split(r"\s+Gazzetta [Uu]fficiale\b", act_title)[0][:400]

    records: list[dict] = []
    num, heading, title, body = "preambolo", "", None, []

    def close():
        text = "\n".join(body).strip()
        if text:
            head = f"{act_title}\n{heading}".strip()
            records.append({"article_num": num, "article_title": title,
                            "text": f"{head}\n{text}".strip()})

    i = 0
    while i < len(paras):
        cls, text = paras[i]
        art, annex = ARTICLE.match(text), ANNEX.match(text)
        if art or annex:
            close()
            if art:
                num = re.sub(r"\s+", "", art.group(1).lower())
            else:
                num = "allegato" + (f"_{annex.group(1)}" if annex.group(1) else "")
            heading, title, body = text, None, []
            nxt = paras[i + 1] if i + 1 < len(paras) else None
            if art and nxt and any(k in nxt[0].split() for k in SUBTITLE_CLASSES):
                title = nxt[1]
                heading = f"{text}\n{title}"
                i += 1
        else:
            body.append(text)
        i += 1
    close()
    # A number seen twice (an article quoted in an annex, say) keeps the first.
    seen, out = set(), []
    for r in records:
        if r["article_num"] not in seen:
            seen.add(r["article_num"])
            out.append(r)
    return out


def iter_acts(src: Path):
    """(code, records) for every act in ``src``, one at a time.

    One at a time because the 17,563 acts are ~2 GB compressed: held all at
    once they would ask a login node for tens of GB.
    """
    files = sorted(src.glob("*.html.gz")) + sorted(src.glob("*.html"))
    for path in files:
        celex = path.name.split(".")[0]
        try:
            raw = gzip.open(path).read() if path.suffix == ".gz" else path.read_bytes()
            records = parse_act(raw.decode("utf-8", errors="replace"), celex)
        except Exception as exc:  # one bad file must not stop seventeen thousand
            print(f"[warn] {path.name}: {exc}", file=sys.stderr)
            continue
        if records:
            yield f"eurlex_{celex}", records
        else:
            print(f"[warn] {path.name}: no text", file=sys.stderr)


def write_acts(src: Path, output: Path, name: str, cfg: ChunkConfig,
               dry_run: bool) -> tuple[int, int, int]:
    """Every act's chunks into legislazione_<name>.chunks.jsonl, streamed;
    (acts, articles, chunks). Written under a temporary name and renamed."""
    dst = output / f"legislazione_{name}.chunks.jsonl"
    tmp = dst.with_name(dst.name + ".tmp")
    n_acts = n_articles = n_chunks = 0
    out = None if dry_run else tmp.open("w", encoding="utf-8")
    try:
        for code, records in iter_acts(src):
            n_acts += 1
            n_articles += len(records)
            for rec in chunk_records(code, records, cfg, kind="legge_ue"):
                if out is not None:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1
            if n_acts % 1000 == 0:
                print(f"  {n_acts:,} acts, {n_chunks:,} chunks", file=sys.stderr, flush=True)
    finally:
        if out is not None:
            out.close()
    if not dry_run and n_acts:
        tmp.replace(dst)
    return n_acts, n_articles, n_chunks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path, help="directory of <CELEX>.html.gz files")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--name", default="eurlex", help="legislazione_<name>.chunks.jsonl")
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.src.is_dir():
        ap.error(f"{args.src} is not a directory")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = ChunkConfig(max_chars=args.max_chars, overlap=args.overlap, min_chars=args.min_chars)
    n_acts, n_articles, n_chunks = write_acts(args.src, args.output, args.name, cfg, args.dry_run)
    print(f"Total: {n_acts:,} acts, {n_articles:,} articles → {n_chunks:,} chunks")
    return 0 if n_acts else 1


if __name__ == "__main__":
    sys.exit(main())
