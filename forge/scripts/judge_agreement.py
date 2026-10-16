#!/usr/bin/env python3
"""How two judges agree on the same answers, answer by answer.

    python forge/scripts/judge_agreement.py \\
        $WORK/eval/devbig-graded-v3 $WORK/eval/devbig-graded-v3-mistral

Step 0.4 of the legal-it roadmap (2026-10-14): every grade so far comes from a
Qwen judge, and the students are Qwen too. A second judge of another family
(Mistral-Small-3.2-24B) graded a sample of the same answers under the same
rubric; this says whether the two read them the same way.

Reads the ``*.graded.jsonl`` of both directories (judge_answers.py output) and,
for every model graded in both, over the questions graded in both, prints:

* exact agreement and Cohen's kappa on the three grades (correct / partial /
  wrong), and the same with the grades made binary (correct or not), which is
  what every result is read on;
* each judge's count of correct answers, so a judge that is simply stricter
  shows as a gap here and not only as a lower kappa;
* the same per question kind (contenuto, inesistente, termine, ...);
* on the deadline and absent-article questions, how often each judge agrees
  with the check that needs no judge at all (`paired.verifiable`).

A kappa above about 0.6 is usually read as substantial agreement, under 0.4 as
weak. Counts only, never an item, so it is safe on held-out answers.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval.paired import cohen_kappa, kind_of, load_graded  # noqa: E402


def _binary(g: str) -> str:
    return "correct" if g == "correct" else "not"


def agreement(a: dict[str, str], b: dict[str, str]) -> dict:
    """Agreement of two judges' grades over the items both graded."""
    ids = sorted(a.keys() & b.keys())
    pairs = [(a[i], b[i]) for i in ids]
    bin_pairs = [(_binary(x), _binary(y)) for x, y in pairs]
    n = len(pairs)
    return {"n": n,
            "exact": sum(x == y for x, y in pairs) / n if n else float("nan"),
            "kappa": cohen_kappa(pairs),
            "exact_bin": sum(x == y for x, y in bin_pairs) / n if n else float("nan"),
            "kappa_bin": cohen_kappa(bin_pairs),
            "correct_a": sum(x == "correct" for x, _ in pairs),
            "correct_b": sum(y == "correct" for _, y in pairs)}


def _line(name: str, r: dict) -> str:
    return (f"{name:<32} n={r['n']:<4} exact {r['exact']:.2f} kappa {r['kappa']:.2f} | "
            f"correct/not: exact {r['exact_bin']:.2f} kappa {r['kappa_bin']:.2f} | "
            f"correct {r['correct_a']} vs {r['correct_b']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first", type=Path, help="graded directory of the first judge")
    ap.add_argument("second", type=Path, help="graded directory of the second judge")
    args = ap.parse_args(argv)

    def load(d: Path) -> dict:
        return {g.label: g for g in (load_graded(p) for p in sorted(d.glob("*.graded.jsonl")))}

    first, second = load(args.first), load(args.second)
    common = sorted(first.keys() & second.keys())
    if not common:
        print(f"[agree] no model graded in both: {sorted(first)} / {sorted(second)}")
        return 1
    print(f"[agree] {args.first.name} vs {args.second.name}: {len(common)} models in both")
    pooled_a: dict[str, str] = {}
    pooled_b: dict[str, str] = {}
    by_kind_a: dict[str, dict[str, str]] = defaultdict(dict)
    by_kind_b: dict[str, dict[str, str]] = defaultdict(dict)
    verif = {"a": [0, 0], "b": [0, 0]}
    for label in common:
        ga, gb = first[label], second[label]
        print(_line(label, agreement(ga.grades, gb.grades)))
        for i in ga.grades.keys() & gb.grades.keys():
            key = f"{label}|{i}"
            pooled_a[key], pooled_b[key] = ga.grades[i], gb.grades[i]
            by_kind_a[kind_of(i)][key], by_kind_b[kind_of(i)][key] = ga.grades[i], gb.grades[i]
            v = ga.verif.get(i)
            if v is not None:
                for side, g in (("a", ga.grades[i]), ("b", gb.grades[i])):
                    verif[side][0] += (g == "correct") == bool(v)
                    verif[side][1] += 1
    print()
    print(_line("ALL", agreement(pooled_a, pooled_b)))
    for kind in sorted(by_kind_a):
        print(_line(f"  {kind}", agreement(by_kind_a[kind], by_kind_b[kind])))
    if verif["a"][1]:
        print(f"\nagreement with the no-judge check (deadlines, absent articles; "
              f"n={verif['a'][1]}): {args.first.name} {verif['a'][0] / verif['a'][1]:.2f}, "
              f"{args.second.name} {verif['b'][0] / verif['b'][1]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
