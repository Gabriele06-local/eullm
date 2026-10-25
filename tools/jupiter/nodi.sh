#!/usr/bin/env bash
# Node-hours at a glance, as on Leonardo: each of the last 7 days, the week,
# what is left and what a day must spend to use it, what runs and waits now.
# One node-hour = 4 GPU-hours (booster jobs hold whole nodes).
#
#     bash tools/jupiter/nodi.sh            # or `nodi`, with the alias:
#     echo "alias nodi='bash $PWD/tools/jupiter/nodi.sh'" >> ~/.bashrc
#
# The days come from Slurm's accounting (sacct, today included); what is
# left from JSC's count (jutil user cpuquota, a day behind) and from
# Slurm's. budget.sh has the long form.

set -uo pipefail
ACCOUNT="${SBATCH_ACCOUNT:-e-aif-2026pg01-1434}"
TOTAL="${JUPITER_GPU_HOURS:-5000}"
PLAN_END="${JUPITER_PLAN_END:-2026-12-15}"
START="${JUPITER_START:-2026-10-14}"

ROWS=$(sacct -X -n -P -A "$ACCOUNT" -S "$START" -E now -o State,AllocTRES,Start,End 2>/dev/null)
JSC=$(jutil user cpuquota 2>/dev/null | awk -F= -v acct="$ACCOUNT" '
    $1 == "budget-account" { mine = ($2 == acct) }
    mine && $1 == "coreh-consumed-total" { c = $2 }
    mine && $1 == "acct-period-end" { e = $2 }
    END { if (c != "") print c / 72, e }')
RUN=$(squeue --me -h -t RUNNING -o '%j' 2>/dev/null | sort | uniq -c | awk '{ printf "%s%s x%d", (NR > 1 ? ", " : ""), $2, $1 }')
WAIT=$(squeue --me -h -t PENDING -o '%j' 2>/dev/null | sort | uniq -c | awk '{ printf "%s%s x%d", (NR > 1 ? ", " : ""), $2, $1 }')

ROWS="$ROWS" JSC="$JSC" RUN="$RUN" WAIT="$WAIT" python3 - "$TOTAL" "$PLAN_END" <<'PY'
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

total, plan_end = float(sys.argv[1]), sys.argv[2]
now = datetime.now()
day_h, slurm = defaultdict(float), 0.0
for line in os.environ["ROWS"].splitlines():
    f = line.split("|")
    if len(f) < 4 or "PENDING" in f[0] or f[2] in ("", "Unknown", "None"):
        continue
    gpus = sum(int(t.split("=")[1]) for t in f[1].split(",") if t.startswith("gres/gpu="))
    t0 = datetime.fromisoformat(f[2])
    t1 = now if f[3] in ("", "Unknown", "None") else datetime.fromisoformat(f[3])
    while t0 < t1:
        cut = min(t1, datetime(t0.year, t0.month, t0.day) + timedelta(days=1))
        h = (cut - t0).total_seconds() * gpus / 3600 / 4      # node-hours
        day_h[t0.date()] += h
        slurm += h
        t0 = cut

today = now.date()
print("Node-hours (1 = 4 GPU-hours), last 7 days:")
week = 0.0
for i in range(6, -1, -1):
    d = today - timedelta(days=i)
    h = day_h.get(d, 0.0)
    week += h
    label = "today, so far" if i == 0 else d.strftime("%a %d-%m")
    print(f"  {d}  {label:<14} {h:6.1f}  {'#' * round(h)}")
print(f"  {'week':<25} {week:6.1f}  ({week / 7:.1f} a day)")

jsc = os.environ.get("JSC", "").split()
left_slurm = total / 4 - slurm
print()
if jsc:
    left_jsc = total / 4 - float(jsc[0]) / 4
    print(f"Left: {left_jsc:,.0f} node-hours by JSC (a day behind), {left_slurm:,.0f} by Slurm, of {total / 4:,.0f}")
else:
    print(f"Left: {left_slurm:,.0f} node-hours by Slurm, of {total / 4:,.0f} (jutil gave no figure)")
jsc_end = jsc[1][:10] if len(jsc) > 1 and jsc[1] else "2026-12-31"
for label, end in (("to the plan's end", plan_end), ("to JSC's end", jsc_end)):
    days = (datetime.fromisoformat(end) + timedelta(days=1) - now).total_seconds() / 86400
    print(f"  {label} {end}: {days:4.1f} days, {left_slurm / days:4.1f} node-hours a day "
          f"(one node busy {100 * left_slurm / days / 24:.0f}% of the time)")
print()
print(f"Running: {os.environ.get('RUN') or 'nothing'}")
print(f"Waiting: {os.environ.get('WAIT') or 'nothing'}")
PY
