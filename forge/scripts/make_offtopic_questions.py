#!/usr/bin/env python3
"""Questions on matters the legal collection does not cover, written by a teacher.

    python forge/scripts/make_offtopic_questions.py $TEACHER \\
        --gguf $WORK/gguf/qwen3-30b-a3b-instruct-2507/qwen3-30b-a3b-instruct-2507-q8_0.gguf \\
        --out $WORK/eullm_runs/gate/offtopic.jsonl

$TEACHER is the Hugging Face directory of Qwen3-30B-A3B-Instruct-2507 (its tokenizer).

The context gate (`eullm_forge.datasets.context_gate`) has to learn that a
question on taxes, traffic fines or pensions finds no answer in a collection
of codes that does not hold those laws, however close the retrieved texts
look. A teacher writes the questions, as a person would ask them, for each
matter in `TOPICS`; the label needs no teacher, since the matter decides it.

A question that names a code the collection holds is dropped: it is not off
topic. Counts only, never a question.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval.retrieval import named_code  # noqa: E402

# Matters governed by laws outside the collection (the codes, the Constitution,
# l. 241/1990, d.P.R. 1199/1971, the consumer code, the GDPR and the AI Act).
# Matters the civil code covers in part (condominium, companies, employment
# contracts and dismissal, copyright) are left out: their label would be wrong.
TOPICS = [
    "imposte sui redditi delle persone fisiche (IRPEF), detrazioni e dichiarazione dei redditi",
    "IVA, fatturazione elettronica e regime forfettario delle partite IVA",
    "IMU, TARI e altri tributi locali",
    "imposta di successione e di donazione",
    "accertamento fiscale, cartelle esattoriali e rottamazione",
    "contenzioso tributario davanti alle corti di giustizia tributaria",
    "codice della strada: multe, punti della patente, ricorso al prefetto",
    "assicurazione RC auto obbligatoria e risarcimento diretto",
    "pensioni INPS: requisiti, contributi, riscatto della laurea",
    "indennità di disoccupazione NASpI e cassa integrazione",
    "sicurezza sul lavoro (d.lgs. 81/2008)",
    "locazioni abitative: canone concordato, cedolare secca, sfratto per morosità",
    "permesso di soggiorno, cittadinanza italiana e ricongiungimento familiare",
    "crisi d'impresa e insolvenza: liquidazione giudiziale e concordato preventivo",
    "edilizia: permesso di costruire, SCIA, CILA, condono e sanatoria",
    "appalti pubblici e codice dei contratti pubblici",
    "bonus edilizi e superbonus",
    "assegno unico per i figli e prestazioni sociali",
    "invalidità civile, legge 104 e permessi retribuiti",
    "tutela dell'ambiente, rifiuti e bonifiche",
    "antiriciclaggio e limiti all'uso del contante",
    "caccia, pesca e porto d'armi",
    "ordinamento penitenziario: permessi premio, semilibertà, affidamento in prova",
    "stupefacenti: uso personale e sanzioni amministrative",
    "sanità: consenso informato, testamento biologico, responsabilità dei medici",
]
ASK = (
    "Scrivi {n} domande diverse che un cittadino o un professionista potrebbe fare a un "
    "assistente legale su questa materia: {topic}. Domande concrete, ciascuna su un aspetto "
    "diverso, in italiano corrente, una per riga, senza numeri, senza elenchi puntati e senza "
    "citare articoli di legge.")
_BULLET = re.compile(r"^\s*(?:[-*•–]|\d+[.)]|\(\d+\))\s*")


def parse_questions(text: str) -> list[str]:
    """The questions of a teacher's reply: one per line, bullets and numbers removed."""
    out = []
    for line in text.splitlines():
        q = _BULLET.sub("", line).strip().strip('"«»').strip()
        if q.endswith("?") and 25 <= len(q) <= 300:
            out.append(q)
    return out


def keep(questions: list[tuple[str, str]]) -> tuple[list[dict], Counter]:
    """Distinct questions that name no code of the collection, with counts."""
    seen, rows, stats = set(), [], Counter()
    for topic, q in questions:
        norm = re.sub(r"\W+", " ", q.lower()).strip()
        if norm in seen:
            stats["duplicate"] += 1
            continue
        seen.add(norm)
        if named_code(q):
            stats["names_a_code_we_hold"] += 1
            continue
        rows.append({"question": q, "topic": topic,
                     "id": f"off-{zlib.crc32(norm.encode()):08x}"})
    return rows, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the teacher's HF directory: its tokenizer and template")
    ap.add_argument("--gguf", required=True, help="the teacher, served by llama-server")
    ap.add_argument("--llama-server", help="llama-server binary (default: LCPP_DIR's)")
    ap.add_argument("--per-call", type=int, default=10, help="questions asked for per call")
    ap.add_argument("--calls", type=int, default=3, help="calls per matter, each its own seed")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--server-parallel", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from eullm_forge.eval.gguf import LlamaServer, generate_answers_gguf

    tok = AutoTokenizer.from_pretrained(args.model)
    end_ids = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
               if isinstance(i, int) and i >= 0]
    jobs = [(t, s) for t in TOPICS for s in range(args.calls)]
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": ASK.format(n=args.per_call, topic=t)}],
        tokenize=False, add_generation_prompt=True) for t, _ in jobs]
    seeds = [zlib.crc32(f"{t}-{s}".encode()) % 2**31 for t, s in jobs]
    print(f"[offtopic] {len(TOPICS)} matters x {args.calls} calls x {args.per_call} questions",
          flush=True)
    with LlamaServer(args.gguf, binary=args.llama_server, parallel=args.server_parallel,
                     ctx_per_slot=4096, log_path=args.out.with_suffix(".server.log")) as server:
        replies = generate_answers_gguf(server, tok, prompts, max_new_tokens=1200,
                                        end_ids=end_ids, parallel=args.server_parallel,
                                        temperature=args.temperature, seeds=seeds)
    asked = [(t, q) for (t, _), (text, _) in zip(jobs, replies) for q in parse_questions(text)]
    rows, stats = keep(asked)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print(f"[offtopic] {len(asked)} questions written, {len(rows)} kept -> {args.out}"
          + "".join(f", {k} {v}" for k, v in sorted(stats.items())), flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
