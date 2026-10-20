#!/usr/bin/env python3
"""Correction exercises and DPO pairs from a model's own sampled answers.

    python forge/scripts/make_correction_pairs.py $WORK/eullm_runs/premise/samples-8b.jsonl \\
        --out-sft $WORK/eullm_runs/premise/sft-correzione.jsonl \\
        --out-dpo $WORK/eullm_runs/premise/dpo.jsonl

Reads sample_answers.py output (`eullm_forge.datasets.correction_gen`): for
each verifiable question answered both right and wrong, one DPO pair (its
right answer over its wrong one) and one correction exercise (check the
proposed answer against the texts, say what is wrong, write the right one).
``--right-share`` of the exercises propose a right answer, to be confirmed.

Every answer is graded again by `eullm_forge.rl.rewards` as it stands now,
not as it stood at sampling, so a reward fixed later needs no new sampling.
``--informative-out`` writes, as GRPO rows, the questions answered both
right and wrong under that grading.

Counts only, never a question or an answer.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.correction_gen import build, informative  # noqa: E402
from eullm_forge.rl import score_answer  # noqa: E402


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", type=Path, help="sample_answers.py output")
    ap.add_argument("--right-share", type=float, default=0.3,
                    help="share of exercises whose proposed answer is right")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-sft", type=Path)
    ap.add_argument("--out-dpo", type=Path)
    ap.add_argument("--informative-out", type=Path,
                    help="the questions answered both right and wrong, as GRPO rows")
    args = ap.parse_args(argv)
    if not (args.out_sft or args.out_dpo or args.informative_out):
        ap.error("give --out-sft, --out-dpo or --informative-out")

    rows = []
    for p in args.samples:
        with p.open(encoding="utf-8") as f:
            rows += [json.loads(ln) for ln in f if ln.strip()]
    rng = random.Random(args.seed)
    corrections, prefs, stats = build(rows, rng, right_share=args.right_share)
    rng.shuffle(corrections)
    rng.shuffle(prefs)
    if args.out_sft:
        write(args.out_sft, corrections)
    if args.out_dpo:
        write(args.out_dpo, prefs)
    if args.informative_out:
        write(args.informative_out, [{k: v for k, v in r.items() if k not in ("samples",
                                                                            "pass_rate")}
                                     for r in rows if informative(r)])
    for tipo in sorted({r["tipo"] for r in rows}):
        mine = [r for r in rows if r["tipo"] == tipo]
        rates = [sum(score_answer(s["answer"], tipo, r.get("keywords")) for s in r["samples"])
                 / max(1, len(r["samples"])) for r in mine]
        print(f"[corr] {tipo:<22} {len(mine):>5} questions, pass rate "
              f"{sum(rates) / len(rates):.2f}: always wrong {rates.count(0.0)}, mixed "
              f"{sum(0 < x < 1 for x in rates)}, always right {rates.count(1.0)}")
    print(f"[corr] {len(rows)} questions: {len(corrections)} correction exercises, "
          f"{len(prefs)} DPO pairs; " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
    return 0 if corrections or prefs else 1


if __name__ == "__main__":
    sys.exit(main())
