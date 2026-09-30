#!/usr/bin/env python3
"""An Italian RAG gate set, from Forge's open-book pairs.

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
"""

import argparse
import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "forge"))
from eullm_forge.eval.retrieval import NormIndex, label, record_articles  # noqa: E402

MAX_CHARS = 3000  # as Forge's open-book prompt shows an article


def passage(record):
    """A record as Forge's prompt shows it: where it is from, then its text."""
    text = record.get("text", "")
    body = text[:MAX_CHARS].rstrip() + (" […]" if len(text) > MAX_CHARS else "")
    return f"{label(record)}\n{body}"


def cases(pair, index, k=3):
    """The two cases of one by-topic grounded pair, or none."""
    key = str(pair.get("key", ""))
    if (
        pair.get("task") != "openbook_grounded"
        or pair.get("named", True)
        or not key.startswith("ob-g-")
    ):
        return []
    code, _, number = key[len("ob-g-") :].partition("-")
    number = re.sub(r"-v\d+$", "", number)
    question = pair["instruction"].rsplit("Domanda: ", 1)[-1].strip()

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
            "question": question,
            "passages": [passage(r) for r in records],
            "label": name,
        }
        for name, records in (("answer", found), ("abstain", without))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pairs", type=pathlib.Path, help="Forge's open-book pairs (JSONL)")
    parser.add_argument("--norms", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--k", type=int, default=3, help="passages a case (Forge shows 3)")
    parser.add_argument("--limit", type=int, default=0, help="questions at most (0: all)")
    parser.add_argument("--seed", type=int, default=1, help="which questions, with --limit")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    pairs = [json.loads(line) for line in args.pairs.open(encoding="utf-8") if line.strip()]
    topic = [p for p in pairs if p.get("task") == "openbook_grounded" and not p.get("named", True)]
    random.Random(args.seed).shuffle(topic)
    index = NormIndex.from_files(args.norms)
    written = questions = 0
    with args.out.open("w", encoding="utf-8") as f:
        for pair in topic:
            if args.limit and questions >= args.limit:
                break
            drawn = cases(pair, index, args.k)
            if drawn:
                questions += 1
                for case in drawn:
                    f.write(json.dumps(case, ensure_ascii=False) + "\n")
                    written += 1
    print(
        f"{len(pairs)} pairs, {len(topic)} by topic: {questions} questions, "
        f"{written} cases -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
