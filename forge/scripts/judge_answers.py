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
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def grade_file(path: Path, grader: ReferenceGrader) -> list[Grade]:
    """Grade every line of one answers file; write the .graded.jsonl beside it."""
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    grades = []
    out = path.with_suffix(".graded.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            g = grader.grade(r["question"], r.get("reference", ""), r["answer"],
                             r.get("rubric", ""))
            grades.append(g)
            print(f"[judge] {label_of(path):<22} {r['id']:<20} {g.label}", flush=True)
            f.write(json.dumps({**r, "grade": g.label, "why": g.rationale},
                               ensure_ascii=False) + "\n")
    return grades


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="+", type=Path, help="answers-*.jsonl from legal_eval.py")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    files = [p for p in args.answers if not p.name.endswith(".graded.jsonl")]
    grader = ReferenceGrader(Greedy(args.model))
    new_file = not args.csv.exists()
    with args.csv.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        for path in files:
            row = summary_row(label_of(path), grade_file(path, grader))
            w.writerow(row)
            f.flush()
            print(f"[judge] {row[1]}: score {row[-1]} — correct {row[3]}, "
                  f"partial {row[4]}, wrong {row[5]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
