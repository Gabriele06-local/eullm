#!/usr/bin/env python3
"""The legislation corpus as distill.py reads it: train.jsonl and val.jsonl.

On JUPITER this distillation runs on legislation only (Normattiva, EUR-Lex:
public domain).
Its source is Leonardo's ``$WORK/norms/legislazione_*.chunks.jsonl``, one
chunk a line with ``code``, ``article_num`` and ``text``, copied over as it
is. Not ``$WORK/datasets/``: those corpora hold court decisions.

An article goes whole to one side: the validation set is the articles whose
``code`` and ``article_num`` hash below ``--val-share``, so no chunk of a
held-out article is trained on, and the split is the same every time it is
built from the same files. distill.py reads ``text``; the other fields are
kept for tracing a line back to its article.

    python forge/scripts/jupiter/make_norms_split.py \\
        --out "$FORGE_DATA_DIR/norms" "$FORGE_DATA_DIR/norms-src"/legislazione_*.chunks.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def held_out(code: str, article: str, share: float) -> bool:
    """Whether this article is in the validation set: stable across runs."""
    digest = hashlib.sha256(f"{code}\x00{article}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < share


def split(paths: list[Path], out: Path, share: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "skipped": 0, "chars_train": 0, "chars_val": 0}
    seen = set()
    with open(out / "train.jsonl.tmp", "w", encoding="utf-8") as train, \
         open(out / "val.jsonl.tmp", "w", encoding="utf-8") as val:
        for path in paths:
            with open(path, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        sys.exit(f"[err] {path}:{n} is not JSON")
                    text = (row.get("text") or "").strip()
                    key = hashlib.sha1(text.encode()).digest()
                    if not text or key in seen:
                        counts["skipped"] += 1
                        continue
                    seen.add(key)
                    code = str(row.get("code", ""))
                    article = str(row.get("article_num", ""))
                    side = "val" if held_out(code, article, share) else "train"
                    record = {"text": text, "code": code, "article_num": article,
                              "source": path.name}
                    (val if side == "val" else train).write(
                        json.dumps(record, ensure_ascii=False) + "\n")
                    counts[side] += 1
                    counts[f"chars_{side}"] += len(text)
    # Renamed only once both are whole: a job never reads half a corpus.
    (out / "train.jsonl.tmp").replace(out / "train.jsonl")
    (out / "val.jsonl.tmp").replace(out / "val.jsonl")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sources", nargs="+", type=Path,
                    help="legislazione_*.chunks.jsonl files")
    ap.add_argument("--out", type=Path, required=True,
                    help="directory for train.jsonl and val.jsonl")
    ap.add_argument("--val-share", type=float, default=0.02,
                    help="share of articles held out (default 0.02)")
    args = ap.parse_args()
    missing = [p for p in args.sources if not p.is_file()]
    if missing:
        sys.exit(f"[err] not found: {' '.join(map(str, missing))}")
    if not 0 < args.val_share < 0.5:
        sys.exit("[err] --val-share must be between 0 and 0.5")
    c = split(args.sources, args.out, args.val_share)
    total = c["chars_train"] + c["chars_val"]
    print(f"[ok] {args.out}/train.jsonl: {c['train']} chunks, "
          f"~{c['chars_train'] / 4 / 1e6:.1f}M tokens")
    print(f"[ok] {args.out}/val.jsonl:   {c['val']} chunks, "
          f"~{c['chars_val'] / 4 / 1e6:.1f}M tokens")
    print(f"     skipped (empty or repeated): {c['skipped']}; "
          f"~{total / 4 / 1e6:.1f}M tokens in all (4 characters a token, a rough count)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
