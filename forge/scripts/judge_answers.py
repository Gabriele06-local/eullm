#!/usr/bin/env python3
"""Grade eval answers against their references with a large model.

`legal_eval.py` writes one answers file per model; its keyword score is a
first sort and no more — v0.2 scored 0.67 on the ricorso straordinario while
putting the deadline at 30 days instead of 120. This reads every answers file
given, asks Qwen3-30B-A3B-Instruct-2507 to grade each answer against the
reference (correct / partial / wrong, see `ReferenceGrader`), and writes:

  * ``<answers>.graded.jsonl`` next to each input, one line per item with
    the grade and the grader's one-sentence reason, for a person to check;
  * one summary row per model to ``--csv``.

Greedy decoding, so the same answers get the same grades. The grader is the
same model that generated the stage-3 pairs; it grades against the written
reference, not its own memory, which is what the prompt insists on. The
references are still "da validare" until a lawyer has read them.

    python forge/scripts/judge_answers.py eval/answers-*.jsonl --csv eval/graded.csv

Runs inside a GPU job (sbatch_judge.slurm): 61 GB of weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import Grade, ReferenceGrader  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
CSV_HEADER = ["timestamp", "label", "items", "correct", "partial", "wrong",
              "unparsed", "score"]


def label_of(path: Path) -> str:
    """answers-v0.2-open.jsonl -> v0.2-open"""
    name = path.name
    for prefix, suffix in (("answers-", ".jsonl"),):
        if name.startswith(prefix) and name.endswith(suffix):
            return name[len(prefix):-len(suffix)]
    return path.stem


def summary_row(label: str, grades: list[Grade]) -> list:
    """Counts per grade and the mean score (correct 1, partial 0.5)."""
    counts = {k: sum(g.label == k for g in grades)
              for k in ("correct", "partial", "wrong", "unparsed")}
    score = sum(g.score for g in grades) / len(grades) if grades else float("nan")
    return [time.strftime("%Y-%m-%dT%H:%M:%S"), label, len(grades),
            counts["correct"], counts["partial"], counts["wrong"],
            counts["unparsed"], f"{score:.3f}"]


class Greedy:
    """The grader model on the GPUs of the job, answering one prompt at a time."""

    def __init__(self, model_id: str, max_new_tokens: int = 120):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("no GPU visible — this runs inside a GPU job")
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id)
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="balanced",
        )
        self.model.eval()
        print(f"[judge] loaded {model_id} in {time.time() - t0:.0f}s", flush=True)

    def __call__(self, prompt: str) -> str:
        return self.batch([prompt])[0]

    def batch(self, prompts: list[str]) -> list[str]:
        """Grade several prompts in one generate call (left-padded)."""
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}],
                                              tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)


def grade_file(path: Path, grader: ReferenceGrader, *, batch_size: int = 1,
               quiet: bool = False) -> list[Grade]:
    """Grade every line of one answers file; write the .graded.jsonl beside it.

    With a ``chat_fn`` that has a ``batch`` method, ``batch_size`` prompts
    go through it at once. ``quiet`` prints nothing per item: an item id of
    the held-out exam names the article it asks about.
    """
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    prompts = [grader.prompt(r["question"], r.get("reference", ""), r["answer"],
                             r.get("rubric", "")) for r in rows]
    batch = getattr(grader.chat_fn, "batch", None)
    raw: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        raw.extend(batch(chunk) if batch and batch_size > 1 else
                   [grader.chat_fn(p) for p in chunk])
    grades = [ReferenceGrader.parse(t) for t in raw]
    out = path.with_suffix(".graded.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for r, g in zip(rows, grades):
            if not quiet:
                print(f"[judge] {label_of(path):<22} {r['id']:<20} {g.label}", flush=True)
            f.write(json.dumps({**r, "grade": g.label, "why": g.rationale},
                               ensure_ascii=False) + "\n")
    return grades


def append_csv_row(path: Path, row: list[str]) -> None:
    """Append one summary row, writing the header if the file has none yet.

    The header is written on existence alone, which is not enough: `open("a")`
    creates the file, and the writer's buffer is only flushed after the first
    `grade_file` — 61 GB of weights and a full pass over the answers on a GPU
    node. A job killed in that window leaves a 0-byte CSV, the re-submitted
    job sees a file that exists, and the first model's row lands where the
    header belongs: `csv.DictReader` then reads no rows at all and the
    decision table is silently empty. So the test is size, not existence,
    and the header is flushed before the expensive work rather than after it.
    """
    needs_header = not (path.exists() and path.stat().st_size > 0)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if needs_header:
            w.writerow(CSV_HEADER)
            f.flush()
        w.writerow(row)
        f.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="+", type=Path, help="answers-*.jsonl from legal_eval.py")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--quiet", action="store_true",
                    help="totals only — for the held-out exam (see legal_eval.py)")
    args = ap.parse_args()

    files = [p for p in args.answers if not p.name.endswith(".graded.jsonl")]
    grader = ReferenceGrader(Greedy(args.model))
    for path in files:
        row = summary_row(label_of(path), grade_file(
            path, grader, batch_size=args.batch_size, quiet=args.quiet))
        append_csv_row(args.csv, row)
        print(f"[judge] {row[1]}: score {row[-1]} — correct {row[3]}, "
              f"partial {row[4]}, wrong {row[5]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
