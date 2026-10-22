#!/usr/bin/env python3
"""How each point's VRAM moved while it ran: for every point a campaign job
started, the VRAM of its devices from the node's samples (`results/<job>.node.jsonl`,
one reading every two seconds), after the load settled and until the point
ended or its server died: first, peak and last reading, and the slope of a
straight line through them in GiB an hour.

Written for the soaks whose servers died where their GCD was full
(c12, 19-20-10-2026): llama.cpp's legacy pool on ROCm grew 1.3-2.5 GiB an
hour. A flat line, or a saw that never reaches the top, is a server that
keeps its memory.

    source tools/lumi/campaign_env.sh
    $PY tools/lumi/vram_slopes.py 22981849 [22982132 ...]
"""

import argparse
import ast
import glob
import json
import os
import re
import sys
from datetime import datetime

START = re.compile(r"^\[([^\]]+)\] start (\S+) on (\[[^\]]*\])")
END = re.compile(r"^\[([^\]]+)\] (done|failed|todo|released|blocked)\s+(\S+) after ")
GIB = 2**30


def when(stamp: str) -> float:
    return datetime.fromisoformat(stamp).timestamp()


def points_of(out_path: str) -> dict:
    """{point id: {"devices": [...], "start": t, "end": t or None, "state": ...}}
    from the job's output."""
    points = {}
    with open(out_path, errors="replace") as f:
        for line in f:
            m = START.match(line)
            if m:
                points[m.group(2)] = {"devices": [str(d) for d in ast.literal_eval(m.group(3))],
                                      "start": when(m.group(1)), "end": None, "state": "running"}
                continue
            m = END.match(line)
            if m and m.group(3) in points:
                points[m.group(3)].update(end=when(m.group(1)), state=m.group(2))
    return points


def samples_of(node_path: str) -> list:
    """[(t, {device: bytes used})] from the node's samples."""
    out = []
    with open(node_path, errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
                out.append((r["t"], {d: v.get("used", 0) for d, v in r["d"].items()}))
            except (ValueError, KeyError, AttributeError):
                continue
    return out


def slope_per_hour(series: list) -> float:
    """Least-squares slope of [(t, GiB)] in GiB an hour."""
    n = len(series)
    if n < 2:
        return 0.0
    mt = sum(t for t, _ in series) / n
    mv = sum(v for _, v in series) / n
    var = sum((t - mt) ** 2 for t, _ in series)
    if var == 0:
        return 0.0
    return sum((t - mt) * (v - mv) for t, v in series) / var * 3600


def report(points: dict, samples: list, settle_s: float, dead_gib: float) -> list:
    rows = []
    for pid, p in sorted(points.items(), key=lambda kv: (kv[1]["start"], kv[0])):
        for dev in p["devices"]:
            lo = p["start"] + settle_s
            hi = p["end"] if p["end"] is not None else float("inf")
            series = [(t, d[dev] / GIB) for t, d in samples if lo <= t <= hi and dev in d]
            # A server that died leaves its device empty before the point ends.
            # A normal point ends the other way round: the runner stops its
            # servers inside point.run, before the END line is printed, so the
            # device is already empty by the point's own end timestamp. Do not
            # read that as a death -- truncate the trailing zeros and keep the
            # state clean. A drop earlier than the last minute of a done or
            # released point is still a death.
            normal_stop = p["state"] in ("done", "released")
            died_at = None
            for i, (t, v) in enumerate(series):
                if v < dead_gib and i > 0 and series[i - 1][1] >= dead_gib:
                    if normal_stop and p["end"] is not None and t >= p["end"] - 60:
                        series = series[:i]
                    else:
                        died_at = t
                        series = series[:i]
                    break
            if not series:
                rows.append((pid, dev, 0.0, None, None, None, 0.0, p["state"]))
                continue
            hours = (series[-1][0] - series[0][0]) / 3600
            state = p["state"]
            if died_at is not None:
                state += f", emptied after {(died_at - p['start']) / 3600:.1f} h"
            rows.append((pid, dev, hours, series[0][1], max(v for _, v in series), series[-1][1],
                         slope_per_hour(series), state))
    return rows


def job_output(queue: str, job: str) -> str:
    """The job's output, where the runner logs each point's start and end:
    LUMI-G's `eullm-campaign-<job>.out`, or another site's `<name>-<job>.out`
    in the same `logs/` (JUPITER's `eullm-gh200-<job>.out`)."""
    logs = os.path.join(queue, "logs")
    first = os.path.join(logs, f"eullm-campaign-{job}.out")
    if os.path.exists(first):
        return first
    for path in sorted(glob.glob(os.path.join(logs, f"*-{job}.out"))):
        with open(path, errors="replace") as f:
            if any(START.match(line) for line in f):
                return path
    return first


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--queue", default=os.environ.get("CAMPAIGN_DIR"),
                    help="the campaign directory (default: $CAMPAIGN_DIR)")
    ap.add_argument("--settle-s", type=float, default=900,
                    help="seconds after a point's start left out, for its load (default 900)")
    ap.add_argument("--dead-gib", type=float, default=1.0,
                    help="below this a device that held a model is taken as emptied (default 1)")
    args = ap.parse_args(argv)
    if not args.queue:
        print("set CAMPAIGN_DIR or --queue", file=sys.stderr)
        return 2
    for job in args.jobs:
        out = job_output(args.queue, job)
        node = os.path.join(args.queue, "results", f"{job}.node.jsonl")
        missing = [p for p in (out, node) if not os.path.exists(p)]
        if missing:
            print(f"== job {job}: missing {', '.join(missing)}")
            continue
        rows = report(points_of(out), samples_of(node), args.settle_s, args.dead_gib)
        # A point done before its settle time has no line to draw: a load or
        # throughput point of minutes. One line says how many, not eight
        # rows of dashes each.
        short = {r[0] for r in rows if r[3] is None and r[7] == "done"}
        rows = [r for r in rows if r[0] not in short]
        print(f"== job {job}: GiB used, after the first {args.settle_s:.0f} s of each point")
        if short:
            print(f"  ({len(short)} points done within their first {args.settle_s:.0f} s "
                  f"are left out)")
        if not rows:
            continue
        print(f"  {'point':34} {'dev':>3} {'hours':>5} {'first':>6} {'peak':>6} "
              f"{'last':>6} {'GiB/h':>6}  end")
        for pid, dev, hours, first, peak, last, slope, state in rows:
            if first is None:
                print(f"  {pid:34} {dev:>3} {hours:5.1f} {'-':>6} {'-':>6} {'-':>6} {'-':>6}"
                      f"  {state}")
            else:
                print(f"  {pid:34} {dev:>3} {hours:5.1f} {first:6.1f} {peak:6.1f} {last:6.1f}"
                      f" {slope:6.2f}  {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
