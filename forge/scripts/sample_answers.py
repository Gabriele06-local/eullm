#!/usr/bin/env python3
"""Several sampled answers per verifiable question, graded by program.

    python forge/scripts/sample_answers.py <merged-dir> --gguf <its q4_k_m.gguf> \\
        --prompts $WORK/eullm_runs/premise/grpo.jsonl $WORK/eullm_runs/grpo/prompts-hyb06.jsonl \\
        --n 8 --out $WORK/eullm_runs/premise/samples-8b.jsonl

The prompts are GRPO rows (make_grpo_prompts.py, make_premise_pairs.py):
the open-book prompt, its kind and the keywords `eullm_forge.rl.rewards`
grades an answer on. Each is asked ``--n`` times at ``--temperature``,
through the GGUF that ships (`eullm_forge.eval.gguf`), and every answer is
scored 1 or 0. The output feeds two things:

* make_correction_pairs.py: the model's own right and wrong answers to the
  same question, for DPO and for correction exercises;
* the next GRPO round, with ``--informative-out``: only the questions the
  model gets sometimes right and sometimes wrong, the ones whose answers
  differ in reward and so carry a gradient (DAPO's dynamic sampling, done
  once before the round instead of during it).

Kinds no program can grade (judged questions) are left out. Prints counts
and the pass-rate spread only, never a question or an answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.correction_gen import VERIFIABLE, informative  # noqa: E402
from eullm_forge.rl import score_answer  # noqa: E402


def templated(tok, messages: list[dict], system: str | None) -> str:
    """The chat prompt as the exam builds it (legal_eval.chat_prompt), thinking off."""
    head = [{"role": "system", "content": system}] if system else []
    return tok.apply_chat_template(head + messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="merged HF directory: the tokenizer of the GGUF")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompts", nargs="+", type=Path, required=True, help="GRPO prompt files")
    ap.add_argument("--n", type=int, default=8, help="answers per question")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="at most this many questions (0: all)")
    ap.add_argument("--system-file", type=Path,
                    help="ask with this system prompt instead of the template's default")
    ap.add_argument("--llama-server", help="llama-server binary (default $LCPP_SERVER)")
    ap.add_argument("--server-parallel", type=int, default=8)
    ap.add_argument("--server-ctx", type=int, default=8192)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--informative-out", type=Path,
                    help="also write the questions answered both right and wrong, as GRPO rows")
    args = ap.parse_args(argv)

    rows = []
    for p in args.prompts:
        with p.open(encoding="utf-8") as f:
            rows += [json.loads(ln) for ln in f if ln.strip()]
    rows = [r for r in rows if r.get("tipo") in VERIFIABLE]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        ap.error("no verifiable questions in the prompt files")

    from transformers import AutoTokenizer

    from eullm_forge.eval.gguf import LlamaServer, generate_answers_gguf

    tok = AutoTokenizer.from_pretrained(args.model)
    end_ids = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
               if isinstance(i, int) and i >= 0]
    system = args.system_file.read_text(encoding="utf-8").strip() if args.system_file else None
    prompts, seeds = [], []
    for r in rows:
        text = templated(tok, r["prompt"], system)
        base = zlib.crc32(r["id"].encode())
        for i in range(args.n):
            prompts.append(text)
            seeds.append((base + i) % 2**31)
    print(f"[sample] {len(rows)} questions x {args.n} answers at T={args.temperature}",
          flush=True)
    log = args.out.with_suffix(".server.log")
    with LlamaServer(args.gguf, binary=args.llama_server, parallel=args.server_parallel,
                     ctx_per_slot=args.server_ctx, log_path=log) as server:
        answers = generate_answers_gguf(server, tok, prompts, max_new_tokens=args.max_new_tokens,
                                        end_ids=end_ids, parallel=args.server_parallel,
                                        temperature=args.temperature, seeds=seeds)

    out_rows = []
    for j, r in enumerate(rows):
        mine = answers[j * args.n:(j + 1) * args.n]
        samples = [{"answer": a, "ended": e,
                    "score": score_answer(a, r["tipo"], r.get("keywords"))} for a, e in mine]
        out_rows.append({**r, "samples": samples,
                         "pass_rate": sum(s["score"] for s in samples) / len(samples)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    spread = Counter()
    for r in out_rows:
        spread[(r["tipo"], "0" if r["pass_rate"] == 0 else "1" if r["pass_rate"] == 1
                else "mixed")] += 1
    for tipo in sorted({t for t, _ in spread}):
        n = sum(v for (t, _), v in spread.items() if t == tipo)
        mean = sum(r["pass_rate"] for r in out_rows if r["tipo"] == tipo) / n
        print(f"[sample] {tipo:<22} {n:>5} questions, pass rate {mean:.2f}: always wrong "
              f"{spread[(tipo, '0')]}, mixed {spread[(tipo, 'mixed')]}, always right "
              f"{spread[(tipo, '1')]}", flush=True)
    if args.informative_out:
        keep = [{k: v for k, v in r.items() if k not in ("samples", "pass_rate")}
                for r in out_rows if informative(r)]
        args.informative_out.parent.mkdir(parents=True, exist_ok=True)
        with args.informative_out.open("w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[sample] {len(keep)} of {len(out_rows)} questions informative for GRPO "
              f"-> {args.informative_out}", flush=True)
    print(f"[sample] -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
