#!/usr/bin/env python3
"""Generate open-book stage-3 pairs: a question, the retrieved norms, the answer.

See `eullm_forge.datasets.openbook_gen` for what the pairs are and why. The
teacher (Qwen3-30B-A3B-Instruct-2507) writes a question and a grounded answer
for one article at a time; the pair is then assembled in the exact prompt the
evaluation and the engine use. Questions about absent articles need no
teacher and are written directly.

    python forge/scripts/generate_openbook_pairs.py \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --exclude-exam $WORK/eval/norm-exam.jsonl \\
        --out $WORK/eullm_runs/stage3/openbook-pairs.jsonl --limit 4000

--exclude-exam is REQUIRED when an exam exists: its articles are left out, and
it prints how many, never which. Resumable like generate_instructions.py:
jobs come from the seed, and keys already written are skipped.

When no job is left it writes ``<out>.done``, so a watcher
(submit_when_ready.sh --need) can start stage 3 on complete data instead of
someone checking the queue. The marker is never empty: the watcher counts
only a non-empty file as there (a copy that stopped at zero bytes is not a
file), and on 2026-09-28 an empty marker kept stage 3 waiting all night.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_instructions import DEFAULT_MODEL, TransformersGenerator, done_keys  # noqa: E402

from eullm_forge.datasets.instruct_gen import GenConfig, Rejected  # noqa: E402
from eullm_forge.datasets.openbook_gen import (  # noqa: E402
    build_messages,
    exam_exclusions,
    make_openbook_jobs,
    missing_pair,
    parse_openbook,
)
from eullm_forge.eval import NormIndex, load_eval_set  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--exclude-exam", type=Path,
                    help="held-out exam JSONL whose articles must not be trained on")
    ap.add_argument("--no-exam", action="store_true",
                    help="confirm there is no exam to exclude (otherwise refused)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--stop-after-min", type=float, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="derive jobs, write the teacher-free pairs, load no model")
    args = ap.parse_args(argv)

    if not args.exclude_exam and not args.no_exam:
        print("[ob] give --exclude-exam <exam.jsonl> (or --no-exam if there is none): "
              "training on the exam's articles would make its number meaningless",
              file=sys.stderr)
        return 2
    start = time.time()
    index = NormIndex.from_files(args.norms)
    exclude = exam_exclusions(load_eval_set(args.exclude_exam)) if args.exclude_exam else set()
    jobs = make_openbook_jobs(index, args.limit, seed=args.seed, exclude=exclude)
    rejected_path = args.out.with_name(args.out.name + ".rejected.jsonl")
    done = done_keys(args.out, rejected_path)
    todo = [j for j in jobs if j.key not in done]
    mix = collections.Counter(j.kind for j in jobs)
    print(f"[ob] {len(index.records):,} records, {len(exclude)} exam articles left out, "
          f"{len(jobs)} jobs {dict(mix)}, {len(jobs) - len(todo)} already done", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    reasons: collections.Counter[str] = collections.Counter()
    accepted = 0

    def write(pairs, bads):
        with open(args.out, "a", encoding="utf-8") as ok, \
             open(rejected_path, "a", encoding="utf-8") as bad:
            for p in pairs:
                ok.write(json.dumps(p, ensure_ascii=False) + "\n")
            for b in bads:
                bad.write(json.dumps(b, ensure_ascii=False) + "\n")

    # The teacher-free half first: it costs nothing and cannot fail on a GPU.
    pairs, bads = [], []
    for job in [j for j in todo if j.kind == "missing"]:
        try:
            pairs.append(missing_pair(job, index))
        except Rejected as exc:
            reasons[exc.reason] += 1
            bads.append({"key": job.key, "reason": exc.reason, "detail": str(exc)})
    write(pairs, bads)
    accepted += len(pairs)
    todo = [j for j in todo if j.kind == "grounded"]
    print(f"[ob] {accepted} missing-article pairs written; {len(todo)} grounded to go",
          flush=True)
    done_marker = args.out.with_name(args.out.name + ".done")

    def mark_done() -> None:
        done_marker.write_text(f"done {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    if args.dry_run:
        if todo:
            print("--- one teacher prompt ---\n" + build_messages(todo[0])[1]["content"][:1200])
        return 0
    if not todo:
        mark_done()
        print(f"[ob] nothing left to do -> {done_marker}", flush=True)
        return 0

    cfg = GenConfig()
    gen = TransformersGenerator(args.model, args.max_new_tokens)
    todo.sort(key=lambda j: len(j.text), reverse=True)
    tokens, gen_seconds, last_batch = 0, 0.0, 0.0
    finished = True
    for i in range(0, len(todo), args.batch_size):
        elapsed = (time.time() - start) / 60
        if args.stop_after_min and elapsed + 1.5 * last_batch / 60 > args.stop_after_min:
            print(f"[ob] stopping at {elapsed:.0f} min, before the walltime", flush=True)
            finished = False
            break
        batch = todo[i:i + args.batch_size]
        t0 = time.time()
        texts, n = gen.generate([build_messages(j) for j in batch])
        last_batch = time.time() - t0
        gen_seconds += last_batch
        tokens += n
        pairs, bads = [], []
        for job, raw in zip(batch, texts):
            try:
                pairs.append(parse_openbook(raw, job, index, cfg=cfg))
            except Rejected as exc:
                reasons[exc.reason] += 1
                bads.append({"key": job.key, "reason": exc.reason, "detail": str(exc),
                             "raw": raw[:3000]})
        write(pairs, bads)
        accepted += len(pairs)
        print(f"[ob] {i + len(batch)}/{len(todo)}  accepted {accepted}  rejected "
              f"{dict(reasons)}  {tokens / max(gen_seconds, 1e-9):.0f} tok/s", flush=True)
    print(f"[ob] done: {accepted} accepted, rejected {dict(reasons)}", flush=True)
    if finished:
        mark_done()
        print(f"[ob] every job processed -> {done_marker}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
