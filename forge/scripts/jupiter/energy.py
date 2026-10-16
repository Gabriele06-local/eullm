#!/usr/bin/env python3
"""The energy of a JUPITER job's GPUs, from the power its job sampled.

Slurm keeps no energy on JUPITER Booster (``ConsumedEnergy`` is empty), so
every Forge job samples ``nvidia-smi --query-gpu=timestamp,index,power.draw,
utilization.gpu,memory.used --format=csv,noheader,nounits -l N`` into a CSV,
and this sums it: each sample stands for ``--interval`` seconds of its GPU.
Deliverable 9 (energy and carbon footprint) is the sum over the jobs.

    python forge/scripts/jupiter/energy.py "$FORGE_RUN_DIR"/power/distill-*.csv --interval 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def energy(path: Path, interval: float) -> tuple[dict, float, int]:
    """(Wh per GPU, mean utilisation %, seconds covered) of one CSV."""
    wh = defaultdict(float)
    util, n, per_gpu = 0.0, 0, defaultdict(int)
    with open(path, newline="") as f:
        for row in csv.reader(f):
            try:
                gpu, watts, use = row[1].strip(), float(row[2]), float(row[3])
            except (IndexError, ValueError):
                continue  # a header, or a line cut by the end of the job
            wh[gpu] += watts * interval / 3600
            util += use
            n += 1
            per_gpu[gpu] += 1
    seconds = int(max(per_gpu.values(), default=0) * interval)
    return dict(wh), (util / n if n else 0.0), seconds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds between samples (nvidia-smi -l)")
    args = ap.parse_args()
    total = 0.0
    print("=== GPU energy, from nvidia-smi power samples ===")
    for path in args.csv:
        if not path.is_file():
            print(f"  {path}: not found", file=sys.stderr)
            continue
        wh, util, seconds = energy(path, args.interval)
        s = sum(wh.values())
        total += s
        gpus = "  ".join(f"GPU{g} {w:.0f}" for g, w in sorted(wh.items()))
        print(f"  {path.name}: {s / 1000:.2f} kWh in {seconds / 3600:.2f} h "
              f"(mean {s / max(seconds, 1) * 3600:.0f} W, GPU use {util:.0f}%); Wh: {gpus}")
    if len(args.csv) > 1:
        print(f"  total: {total / 1000:.2f} kWh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
