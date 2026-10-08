#!/usr/bin/env python3
"""The statutes of the LEGAL_PACK for RAG Enterprise, one article per file.

    python forge/scripts/build_legal_pack.py \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --vigente 2026-09-15 --out $WORK/legal-pack

Writes, under ``--out``:

* ``articoli/<citation>.txt``: one file per article, named as the article is
  cited -- ``Codice penale, art. 54 - Stato di necessità.txt``. RAG
  Enterprise cites its sources as ``[file name]``, so with these files every
  citation it makes is an article citation, which can be checked
  (eullm-priv docs/legal-pack-spec.md, sections 3.1 and 6). A file holds the
  whole article, reassembled from however Normattiva's export chunked it,
  under a one-line heading;
* ``legal-pack.jsonl``: the same articles as records (code, number,
  rubrica, text, date in force), for Pro's structured knowledge provider.

The articles are split and reassembled by
`eullm_forge.eval.norm_exam.articles_from_records`, the same code the exams
are drawn with, so the pack holds exactly the articles the exams ask about.
Statutes are public (art. 5, l. n. 633/1941); no ruling goes in here.

It prints counts only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval.norm_exam import articles_from_records  # noqa: E402

# How each code is named in a citation. "/" cannot be in a file name: the
# two laws are written with "-" in the year.
CODE_NAMES = {
    "codice_civile": "Codice civile",
    "codice_penale": "Codice penale",
    "codice_procedura_civile": "Codice di procedura civile",
    "codice_procedura_penale": "Codice di procedura penale",
    "codice_consumo": "Codice del consumo",
    "costituzione": "Costituzione",
    "codice_processo_amministrativo": "Codice del processo amministrativo",
    "legge_procedimento_amministrativo": "Legge n. 241-1990",
    "ricorsi_amministrativi": "D.P.R. n. 1199-1971",
}

# Not allowed in a Windows file name, where RAG Enterprise also runs.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
MAX_NAME = 150


def citation(code: str, number: str, rubrica: str) -> str:
    """How an article is cited and its file named: "Codice penale, art. 54 - Stato di necessità"."""
    head = f"{CODE_NAMES.get(code, code)}, art. {number}"
    return f"{head} - {rubrica}" if rubrica else head


def file_name(cite: str) -> str:
    """The citation made safe for a file name on every platform, not too long."""
    name = re.sub(r"\s+", " ", _UNSAFE.sub(" ", cite)).strip().rstrip(".")
    return name[:MAX_NAME].rstrip() + ".txt"


def read_jsonl(paths) -> list[dict]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True,
                    help="legislation chunks (prepare_legislation.py output)")
    ap.add_argument("--vigente", required=True,
                    help="date the texts are in force at (the Normattiva download), YYYY-MM-DD")
    ap.add_argument("--codes", nargs="*", default=[],
                    help="only these codes (default: every code in --norms)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    records = read_jsonl(args.norms)
    if args.codes:
        records = [r for r in records if r.get("code") in args.codes]
    articles = articles_from_records(records)
    if not articles:
        print("[legal-pack] no articles in --norms", file=sys.stderr)
        return 1

    folder = args.out / "articoli"
    folder.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    counts: dict[str, int] = {}
    tmp = args.out / "legal-pack.jsonl.partial"
    with tmp.open("w", encoding="utf-8") as jf:
        for (code, number), art in sorted(articles.items()):
            rubrica = art.heading
            cite = citation(code, number, rubrica)
            name = file_name(cite)
            stem, n = name[:-4], 2
            while name.lower() in used:          # two rubriche cut to the same 150 characters
                name = f"{stem} ({n}).txt"
                n += 1
            used.add(name.lower())
            head = f"{CODE_NAMES.get(code, code)}, art. {number}"
            head += f" ({rubrica})" if rubrica else ""
            (folder / name).write_text(f"{head}\n\n{art.text.strip()}\n", encoding="utf-8")
            jf.write(json.dumps({"id": f"{code}-{number}", "code": code,
                                 "codice": CODE_NAMES.get(code, code), "articolo": number,
                                 "rubrica": rubrica, "citazione": cite, "file": name,
                                 "testo": art.text.strip(), "vigente_al": args.vigente},
                                ensure_ascii=False) + "\n")
            counts[code] = counts.get(code, 0) + 1
    tmp.replace(args.out / "legal-pack.jsonl")
    for code, n in sorted(counts.items()):
        print(f"[legal-pack] {CODE_NAMES.get(code, code)}: {n:,} articles")
    print(f"[legal-pack] {sum(counts.values()):,} articles in force at {args.vigente} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
