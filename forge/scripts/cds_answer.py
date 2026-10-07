#!/usr/bin/env python3
"""The case-law exam: development rulings' questions, answered with case-law retrieval.

    python forge/scripts/cds_answer.py <merged model> --label grpo-v04 \\
        --questions $WORK/eval/cds/dev-cards.jsonl --dev-ids $WORK/eval/cds/cds-dev-ids.txt \\
        --chunks ... --openga ... --cards $WORK/eullm_runs/cds/schede*.jsonl \\
        --answers $WORK/eval/cds-dev-answers/answers-grpo-v04.jsonl

The acceptance test of step 4 (research report of 2026-10-05). The exam
questions of the development rulings' cards -- written by Qwen3.6-27B, with
a reference answer and a rubric -- are asked the way the model will be
used: the question with the passages the case-law index retrieves
(`caselaw.prompts.caselaw_prompt`). The answers file has the fields
judge_answers.py grades (question, reference, rubric, answer), so the
paired comparison is compare_graded.py's, as for the statutes.

Two checks need no judge, and are printed and written per answer:

* ``cited_ok``: every ruling the answer cites by number is among the
  passages it was given. A citation outside them is invented -- the failure
  legal tools are sanctioned for (TAR Lombardia n. 3348/2025);
* ``source_cited``: the answer cites the ruling the question came from.

With ``--gguf`` the answers come from that file through llama-server, the
model directory giving the tokenizer, as legal_eval.py does.

Counts only, never a question or an answer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.caselaw import attach_meta, load_openga, load_rulings  # noqa: E402
from eullm_forge.caselaw.index import RulingIndex, SparseBM25, build_units  # noqa: E402
from eullm_forge.caselaw.prompts import caselaw_prompt, ruling_label  # noqa: E402

_CITED = re.compile(r"n\.\s*(\d{9})\b|n\.\s*(\d{1,5})\s*/\s*((?:19|20)\d\d)\b")


def cited_numbers(text: str) -> set[str]:
    """Ruling numbers an answer cites, in OpenGA's form: "n. 202301234" or "n. 1234/2023"."""
    out = set()
    for full, num, year in _CITED.findall(text or ""):
        out.add(full if full else f"{year}{int(num):05d}")
    return out


def exam_items(rows: list[dict], dev: set[str], limit: int) -> list[dict]:
    items = []
    for r in rows:
        if r["id"] not in dev:
            continue
        for i, q in enumerate(r.get("domande_esame", [])):
            items.append({"id": f"cds-{r['id'].split('/')[-1]}-{i}", "ruling": r["id"],
                          "question": q["domanda"], "reference": q["risposta"],
                          "rubric": q["rubrica"]})
    return items[:limit] if limit else items


def _legal_eval():
    spec = importlib.util.spec_from_file_location(
        "legal_eval", Path(__file__).resolve().parent / "legal_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate(args, contents: list[str]) -> list[tuple[str, bool]]:
    """(answer, ended_its_turn) per user turn, from the merged model or its GGUF."""
    le = _legal_eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    end_ids = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
               if isinstance(i, int) and i >= 0]
    prompts = [le.chat_prompt(tok, c) for c in contents]
    if args.gguf:
        from eullm_forge.eval.gguf import LlamaServer, generate_answers_gguf
        with LlamaServer(args.gguf, binary=args.llama_server, parallel=args.batch_size,
                         log_path=args.answers.with_suffix(".server.log")) as server:
            return generate_answers_gguf(server, tok, prompts, max_new_tokens=args.max_new_tokens,
                                         end_ids=end_ids, parallel=args.batch_size)
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    model = le.load_model(AutoModelForCausalLM, args.model, torch.cuda.device_count(),
                          fallback_cls=AutoModelForImageTextToText)
    model.eval()
    return le.generate_answers(model, tok, prompts, batch_size=args.batch_size,
                               max_new_tokens=args.max_new_tokens, end_ids=end_ids)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="merged model directory (or the tokenizer's, with --gguf)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--questions", nargs="+", type=Path, required=True)
    ap.add_argument("--dev-ids", type=Path, required=True)
    ap.add_argument("--chunks", nargs="+", type=Path, required=True)
    ap.add_argument("--openga", nargs="*", type=Path, default=[])
    ap.add_argument("--cards", nargs="*", type=Path, default=[],
                    help="cards of the indexed rulings: the chunks carry their prefix")
    ap.add_argument("--index", choices=["prefix", "prefix+cards"], default="prefix+cards",
                    help="units of the case-law index: chunks with their card prefix, and the "
                         "cards as units of their own besides (the retrieval check of "
                         "2026-10-07: recall@3 +5.4 and +7.2 points over plain chunks)")
    ap.add_argument("--embedder")
    ap.add_argument("--reranker")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=1200, help="questions asked (0: all)")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="answers decoded together (llama-server slots, with --gguf)")
    ap.add_argument("--gguf")
    ap.add_argument("--llama-server")
    ap.add_argument("--answers", type=Path, required=True)
    args = ap.parse_args(argv)

    dev = {ln.strip() for ln in args.dev_ids.open(encoding="utf-8") if ln.strip()}
    rows = []
    for p in args.questions:
        rows += [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    items = exam_items(rows, dev, args.limit)
    if not items:
        print("[cds-exam] no exam questions of development rulings", file=sys.stderr)
        return 1

    rulings = load_rulings(args.chunks)
    if args.openga:
        attach_meta(rulings, load_openga(args.openga))
    cards = {}
    for p in args.cards:
        for line in p.open(encoding="utf-8"):
            if line.strip():
                c = json.loads(line)
                cards[c["id"]] = c
    chunks = []
    for p in args.chunks:
        for line in p.open(encoding="utf-8"):
            if line.strip():
                c = json.loads(line)
                if (c.get("kind") or str(c.get("sentence_id", "")).split("/")[0]) == "cds":
                    chunks.append(c)
    units = build_units(rulings, chunks, cards=cards, prefix_chunks=bool(cards),
                        card_units=bool(cards) and args.index == "prefix+cards")
    index = RulingIndex(units, bm25=SparseBM25([u.text for u in units]))
    if args.embedder:
        from eullm_forge.caselaw.index import shard_vectors, units_key
        from eullm_forge.eval.dense import Embedder, Reranker
        emb = Embedder(args.embedder, batch_size=32, max_length=768)
        index.vectors = shard_vectors([u.text[:3000] for u in units], emb.encode,
                                      args.cache_dir, units_key(units, args.embedder))
        index.query_fn = lambda q: emb.encode([q], query=True)[0]
        index.reranker = Reranker(args.reranker) if args.reranker else None

    def label_of(rid: str) -> str:
        r = rulings.get(rid)
        return ruling_label({"sezione": getattr(r, "section", ""),
                             "numero": rid.split("/", 1)[-1]})

    contents, contexts = [], []
    for it in items:
        passages, ids = [], []
        for i in index.ranked_units(it["question"]):
            u = units[i]
            passages.append((label_of(u.ruling), u.text))
            ids.append(u.ruling)
            if len(passages) >= args.k:
                break
        contents.append(caselaw_prompt(it["question"], passages))
        contexts.append(ids)

    results = generate(args, contents)

    ok = src = found = cut = 0
    args.answers.parent.mkdir(parents=True, exist_ok=True)
    with args.answers.open("w", encoding="utf-8") as f:
        for it, ctx, (answer, ended) in zip(items, contexts, results):
            cited = cited_numbers(answer)
            in_ctx = {r.split("/", 1)[-1] for r in ctx}
            row = {**it, "answer": answer, "context": ctx, "cited": sorted(cited),
                   "cited_ok": cited <= in_ctx,
                   "source_cited": it["ruling"].split("/", 1)[-1] in cited,
                   "source_retrieved": it["ruling"] in ctx, "ended": ended}
            ok += row["cited_ok"]
            src += row["source_cited"]
            found += row["source_retrieved"]
            cut += not ended
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(items)
    print(f"[cds-exam] {args.label}: {n} questions; source ruling retrieved {found / n:.3f}, "
          f"cited {src / n:.3f}; answers citing only rulings they were given {ok / n:.3f}; "
          f"cut at --max-new-tokens {cut / n:.3f} -> {args.answers}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
