"""How much of the allocation is spent, and how far behind the calendar.

LUMI-G billing, from docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/:

    standard-g        GPU-hours = 4 x nodes x hours           (whole nodes)
    small-g, dev-g    GPU-hours = 0.5 x hours x
                        max(ceil(cores / 8), ceil(memory / 64 GB), GCDs)

and one node-hour is 4 GPU-hours. `lumi-allocations` is the authority; this
recomputes the same figure from `sacct` so the pace can be checked against
the calendar in one place, and so the two can be compared when they differ.
"""

from __future__ import annotations

import datetime as dt
import math
import re
import subprocess

GPU_HOURS_PER_NODE_HOUR = 4
SMALL_G_MEM_UNIT_MIB = 64 * 1024

SACCT_FIELDS = "JobID,JobName,Partition,State,ElapsedRaw,NNodes,AllocTRES"


def mem_mib(text: str) -> float:
    """'480G', '64000M', '1T', '512000K' → MiB."""
    m = re.fullmatch(r"([\d.]+)([KMGT]?)", text.strip())
    if not m:
        return 0.0
    value, unit = float(m.group(1)), m.group(2) or "M"
    return value * {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]


def parse_tres(text: str) -> dict:
    out = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def billed_gpu_hours(partition: str, elapsed_s: float, nodes: int, tres: dict) -> float:
    hours = elapsed_s / 3600
    partition = partition.split(",")[0]
    if partition == "standard-g":
        return GPU_HOURS_PER_NODE_HOUR * nodes * hours
    if partition in ("small-g", "dev-g"):
        gcds = int(tres.get("gres/gpu", 0) or 0)
        cores = int(tres.get("cpu", 0) or 0)
        mem = mem_mib(tres.get("mem", "0"))
        units = max(math.ceil(cores / 8), math.ceil(mem / SMALL_G_MEM_UNIT_MIB), gcds)
        return 0.5 * hours * units
    return 0.0  # CPU partitions spend a different budget


def parse_sacct(text: str) -> list:
    """`sacct -X -n -P -o SACCT_FIELDS` → one dict per allocation."""
    jobs = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) != 7:
            continue
        jobid, name, partition, state, elapsed, nodes, tres = parts
        if not elapsed.isdigit():
            continue
        t = parse_tres(tres)
        jobs.append({
            "job": jobid,
            "name": name,
            "partition": partition,
            "state": state,
            "elapsed_s": int(elapsed),
            "gpu_hours": billed_gpu_hours(partition, int(elapsed), int(nodes or 0), t),
        })
    return jobs


def read_sacct(start: str, account=None) -> list:
    cmd = ["sacct", "-X", "-n", "-P", "-S", start, "-E", "now", "-o", SACCT_FIELDS]
    if account:
        cmd += ["-A", account, "--allusers"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"sacct failed: {out.stderr.strip()}")
    return parse_sacct(out.stdout)


def pace(budget_node_hours: float, start: dt.date, end: dt.date, now: dt.datetime,
         spent_node_hours: float) -> dict:
    """Where the spend stands against a straight line from start to end."""
    total_h = (dt.datetime.combine(end, dt.time()) - dt.datetime.combine(start, dt.time()))
    total_h = total_h.total_seconds() / 3600
    elapsed_h = (now - dt.datetime.combine(start, dt.time())).total_seconds() / 3600
    elapsed_h = min(max(elapsed_h, 0.0), total_h)
    left_h = total_h - elapsed_h
    target = budget_node_hours * elapsed_h / total_h if total_h else 0.0
    remaining = budget_node_hours - spent_node_hours
    per_day = remaining / (left_h / 24) if left_h > 0 else None
    return {
        "budget_node_hours": budget_node_hours,
        "spent_node_hours": round(spent_node_hours, 1),
        "spent_pct": round(100 * spent_node_hours / budget_node_hours, 1),
        "calendar_pct": round(100 * elapsed_h / total_h, 1) if total_h else None,
        "target_to_date": round(target, 1),
        "behind_node_hours": round(target - spent_node_hours, 1),
        "days_left": round(left_h / 24, 1),
        "needed_node_hours_per_day": round(per_day, 1) if per_day is not None else None,
        # Nodes kept busy around the clock from now on to spend the rest.
        "needed_nodes_continuous": round(per_day / 24, 2) if per_day is not None else None,
    }
