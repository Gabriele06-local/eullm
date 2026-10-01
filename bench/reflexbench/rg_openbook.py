#!/usr/bin/env python3
"""An Italian RAG gate set, from Forge's open-book pairs or from the law alone.

Forge's open-book set (forge/eullm_forge/datasets/openbook_gen.py) asks
questions about articles of Italian law by topic, and shows the model the
articles retrieval finds for each, its own article made sure of; the RAFT
pairs (forge/scripts/make_raft_absent.py) show the same question without
it. Those are a gate's two cases: the passages that hold the article the
question was written from (`answer`), and what retrieval returns once that
article is left out (`abstain`). The passages are retrieved again here, as
Forge retrieves them, from the same legislation records.

    python3 bench/reflexbench/rg_openbook.py $WORK/eullm_runs/stage3/openbook-v04.jsonl \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl --out rag-legal-it.jsonl
    python3 bench/reflexbench/ragbench.py --sets '' --data rag-legal-it.jsonl ...

Only questions asked by topic are used: one that names its article gets it
by lookup, and there is nothing for a gate to judge. The set is written
where the pairs are; nothing of it belongs in the repository.

Each case names the article its question was written from (`document`,
`code/number`): the pairs ask up to four questions of one article, and a
model trained on one of them and tested on another has read the answer.
`eullm-forge decisions import-rag` keeps every question about an article
on one side of its split.

Without the pairs, which a large model writes on the cluster, `--by-heading`
asks by an article's rubrica instead — "Che cosa prevede la legge in materia
di risarcimento per fatto illecito?" — for the articles whose rubrica is
their own and says more than "Definizioni". Plainer questions than the
pairs', from the legislation records alone:

    python3 bench/reflexbench/rg_openbook.py --by-heading --limit 1000 \
        --norms ~/work/corpus/legislazione_*.chunks.jsonl --out rag-legal-it.jsonl
"""

import argparse
import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "forge"))
from eullm_forge.eval.retrieval import (  # noqa: E402
    NormIndex,
    label,
    record_articles,
    record_heading,
)

MAX_CHARS = 3000  # as Forge's open-book prompt shows an article


def passage(record):
    """A record as Forge's prompt shows it: where it is from, then its text."""
    text = record.get("text", "")
    body = text[:MAX_CHARS].rstrip() + (" […]" if len(text) > MAX_CHARS else "")
    return f"{label(record)}\n{body}"


def article(key):
    """The article a question was written from, `(code, number)`, read from
    its key: `ob-g-<code>-<number>` for a pair, `-v<n>` after it for each
    further question about the same article, or `h-<code>-<number>` for one
    asked by rubrica. None for a key of another shape."""
    for prefix in ("ob-g-", "h-"):
        if key.startswith(prefix):
            code, _, number = key[len(prefix) :].partition("-")
            if prefix == "ob-g-":
                number = re.sub(r"-v\d+$", "", number)
            return (code, number) if code and number else None
    return None


def document(key):
    """A case's `document`, the article of the question `key` as
    `code/number`, or None. For a set written before cases carried it."""
    found = article(key)
    return f"{found[0]}/{found[1]}" if found else None


def cases(pair, index, k=3):
    """The two cases of one by-topic grounded pair, or none."""
    key = str(pair.get("key", ""))
    found = article(key) if key.startswith("ob-g-") else None
    if pair.get("task") != "openbook_grounded" or pair.get("named", True) or not found:
        return []
    code, number = found
    question = pair["instruction"].rsplit("Domanda: ", 1)[-1].strip()
    return contexts(key, question, code, number, index, k)


def contexts(key, question, code, number, index, k):
    """The question with the article `code` `number` among the passages
    retrieved for it, and without."""

    def own(record):
        return record.get("code") == code and number in record_articles(record)

    mine = [r for r in index.records if own(r)]
    if not mine:
        return []
    # As Forge builds the grounded context: what retrieval returns, the
    # article put in at a place of its own when retrieval missed it.
    found = index.search(question, k)
    if not any(own(r) for r in found):
        found = found[: k - 1]
        found.insert(random.Random(key).randrange(len(found) + 1), mine[0])
    # As Forge builds the absent one: retrieval, without the article.
    without = [r for r in index.search(question, k + 6) if not own(r)][:k]
    if not without:
        return []
    return [
        {
            "id": f"{key}:{name}",
            "group": key,
            "document": f"{code}/{number}",
            "question": question,
            "passages": [passage(r) for r in records],
            "label": name,
        }
        for name, records in (("answer", found), ("abstain", without))
    ]


# Rubriche that name no topic of their own.
GENERIC = {
    "abrogazione",
    "abrogazioni",
    "ambito di applicazione",
    "definizioni",
    "disposizioni finali",
    "disposizioni generali",
    "disposizioni transitorie",
    "entrata in vigore",
    "finalità",
    "norme transitorie",
    "oggetto",
}


def heading_cases(index, k=3, limit=0, seed=1):
    """Cases asked by rubrica, for the articles whose rubrica is theirs
    alone in the collection and names a topic."""
    headings = {}
    for record in index.records:
        heading, articles = record_heading(record), record_articles(record)
        if heading and len(articles) == 1:
            headings.setdefault((record.get("code") or "", articles[0]), heading)
    seen = {}
    for heading in headings.values():
        name = " ".join(heading.lower().split())
        seen[name] = seen.get(name, 0) + 1
    chosen = sorted(
        (code, number, heading)
        for (code, number), heading in headings.items()
        if seen[" ".join(heading.lower().split())] == 1
        and heading.lower().strip(" .") not in GENERIC
        and len(heading) >= 12
    )
    random.Random(seed).shuffle(chosen)
    out, questions = [], 0
    for code, number, heading in chosen:
        if limit and questions >= limit:
            break
        topic = heading.rstrip(" .")
        question = f"Che cosa prevede la legge in materia di {topic[:1].lower() + topic[1:]}?"
        drawn = contexts(f"h-{code}-{number}", question, code, number, index, k)
        if drawn:
            questions += 1
            out.extend(drawn)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "pairs", type=pathlib.Path, nargs="?", help="Forge's open-book pairs (JSONL)"
    )
    parser.add_argument("--norms", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument(
        "--by-heading", action="store_true", help="ask by rubrica, without the pairs"
    )
    parser.add_argument("--k", type=int, default=3, help="passages a case (Forge shows 3)")
    parser.add_argument("--limit", type=int, default=0, help="questions at most (0: all)")
    parser.add_argument("--seed", type=int, default=1, help="which questions, with --limit")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if bool(args.pairs) == args.by_heading:
        parser.error("give the open-book pairs, or --by-heading without them")
    index = NormIndex.from_files(args.norms)
    if args.by_heading:
        drawn = heading_cases(index, args.k, args.limit, args.seed)
    else:
        pairs = [json.loads(line) for line in args.pairs.open(encoding="utf-8") if line.strip()]
        topic = [
            p for p in pairs if p.get("task") == "openbook_grounded" and not p.get("named", True)
        ]
        random.Random(args.seed).shuffle(topic)
        drawn = []
        for pair in topic:
            if args.limit and len(drawn) >= 2 * args.limit:
                break
            drawn.extend(cases(pair, index, args.k))
    with args.out.open("w", encoding="utf-8") as f:
        for case in drawn:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"{len(drawn) // 2} questions, {len(drawn)} cases -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
