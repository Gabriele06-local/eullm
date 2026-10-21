#!/usr/bin/env bash
# GPU-hours spent on JUPITER Booster and the daily rate that spends the rest
# of the allocation by its end, from Slurm's own accounting (sacct):
#
#     bash tools/jupiter/budget.sh [END_DATE] [TOTAL_GPU_HOURS]
#
# END_DATE defaults to 2026-12-15 and TOTAL_GPU_HOURS to 5000 (the award).
# The allocation runs two months from its activation (14-10-2026), so the
# plan ends mid-December; JuDoor shows 31-12-2026, kept as the margin
# (bash tools/jupiter/budget.sh 2026-12-31 for the rate to that date).
# A booster job holds whole nodes and pays for every GPU of them, so a job's
# GPU-hours are its elapsed time times the GPUs it was given (gres/gpu in
# AllocTRES): 4 an hour per node. Running jobs count up to now; queued ones
# are shown at their time limit, the most they can cost. JuDoor has the
# official figure, and `jutil user cpuquota` prints it on a login node, in
# core-hours: 72 a GPU-hour (a GH200's Grace cores; the award's 5,000
# GPU-hours are its 360,000), and it lags Slurm's accounting. Both are shown
# at the end; this script's own figure is for steering. The last table says
# whether the spending is on pace, to the plan's end and to JSC's.

set -uo pipefail
END="${1:-2026-12-15}"
TOTAL="${2:-5000}"
ACCOUNT="${SBATCH_ACCOUNT:-e-aif-2026pg01-1434}"
START="${JUPITER_START:-2026-10-14}"

now=$(date +%s)
end=$(date -d "$END 23:59:59" +%s) || { echo "[err] not a date: $END"; exit 1; }

# JobID|JobName|State|ElapsedRaw|AllocTRES, one line a job (allocations only).
ROWS=$(sacct -X -n -P -A "$ACCOUNT" -S "$START" -E now \
    -o JobID,JobName,State,ElapsedRaw,AllocTRES,Start,End 2>/dev/null)
[ -n "$ROWS" ] || { echo "[err] sacct returned no jobs for $ACCOUNT since $START"; exit 1; }

awk -F'|' -v now="$now" -v end="$end" -v enddate="$END" -v total="$TOTAL" -v start="$START" '
function gpus(tres,   m) { return match(tres, /gres\/gpu=[0-9]+/) ? substr(tres, RSTART + 9, RLENGTH - 9) + 0 : 0 }
{
    g = gpus($5); h = $4 * g / 3600
    if ($3 ~ /PENDING/) next
    used += h; n++
    byname[$2] += h
    if ($3 ~ /RUNNING/) running += h
}
END {
    days = (end - now) / 86400
    left = total - used
    printf "JUPITER allocation since %s, to the end of %s\n", start, enddate
    printf "  used         %8.1f GPU-hours (%.1f%%) in %d jobs, %.1f of them still running\n", used, 100 * used / total, n, running
    printf "  left         %8.1f GPU-hours\n", left
    printf "  days left    %8.1f\n", days
    printf "  to spend it  %8.1f GPU-hours a day = %.1f node-hours a day (one node busy %.0f%% of the time)\n", \
        left / days, left / days / 4, 100 * left / days / 96
    printf "\n  by job name (GPU-hours):\n"
    for (k in byname) printf "    %-24s %8.1f\n", k, byname[k]
}' <<<"$ROWS"

# GPU-hours a calendar day, each job's time split at midnight, against the
# rate the rest of the allocation needs.
ROWS="$ROWS" python3 - "$now" "$end" "$TOTAL" <<'PY'
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

now, end, total = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
day_h, used = defaultdict(float), 0.0
for line in os.environ["ROWS"].splitlines():
    f = line.split("|")
    if len(f) < 7 or "PENDING" in f[2] or f[5] in ("", "Unknown", "None"):
        continue
    gpus = sum(int(t.split("=")[1]) for t in f[4].split(",") if t.startswith("gres/gpu="))
    if not gpus:
        continue
    t0 = datetime.fromisoformat(f[5])
    t1 = (datetime.fromtimestamp(now) if f[6] in ("", "Unknown", "None")
          else datetime.fromisoformat(f[6]))
    while t0 < t1:
        cut = min(t1, datetime(t0.year, t0.month, t0.day) + timedelta(days=1))
        h = (cut - t0).total_seconds() * gpus / 3600
        day_h[t0.date()] += h
        used += h
        t0 = cut
need = (total - used) / max(1e-9, (end - now) / 86400)
print(f"\n  by day (GPU-hours; {need:.1f} a day spends the rest by the end):")
for d in sorted(day_h)[-10:]:
    print(f"    {d}  {day_h[d]:6.1f}  {day_h[d] / 4:5.1f} node-h  {'#' * round(day_h[d] / 4)}")
PY

QUEUED=$(squeue --me -h -t PENDING -o '%i|%j|%l|%D' 2>/dev/null)
if [ -n "$QUEUED" ]; then
    echo
    echo "  queued (at most their time limit):"
    awk -F'|' '
    function secs(l,   d, p, n) {          # [D-]HH:MM:SS, MM:SS or MM
        d = 0
        if (index(l, "-")) { d = substr(l, 1, index(l, "-") - 1); l = substr(l, index(l, "-") + 1) }
        n = split(l, p, ":")
        if (n == 3) return d * 86400 + p[1] * 3600 + p[2] * 60 + p[3]
        if (n == 2) return d * 86400 + p[1] * 60 + p[2]
        return d * 86400 + p[1] * 60
    }
    { h = secs($3) * $4 * 4 / 3600; q += h
      printf "    %-10s %-22s %10s  <= %5.1f GPU-hours\n", $1, $2, $3, h }
    END { printf "    %-45s <= %5.1f GPU-hours\n", "together", q }' <<<"$QUEUED"
fi

# JSC's own count of the budget, core-hours turned into GPU-hours.
JSC=$(jutil user cpuquota 2>/dev/null | awk -F= -v acct="$ACCOUNT" '
    $1 == "budget-account" { mine = ($2 == acct) }
    mine && $1 == "coreh-quota-total" { q = $2 }
    mine && $1 == "coreh-consumed-total" { c = $2 }
    mine && $1 == "acct-period-start" { s = $2 }
    mine && $1 == "acct-period-end" { e = $2 }
    END { if (q > 0) print c / 72, q / 72, s, e }')
[ -n "$JSC" ] || echo -e "\n  [..] jutil user cpuquota gave no figure for $ACCOUNT: JSC's rows left out (try again; JSC's count lags a day anyway)"
[ -n "$JSC" ] && awk '{ printf "\n  JSC (jutil user cpuquota): %.1f of %.0f GPU-hours used (%.1f%%), %.1f left, period %s to %s\n", $1, $2, 100 * $1 / $2, $2 - $1, ($3 == "" ? "?" : $3), $4 }' <<<"$JSC"

# Whether the spending is on pace, as on Leonardo: billed against a uniform
# pace from the start of the period, what is left by JSC's count and by
# Slurm's (which includes today, JSC's lags), the daily rate that spends it
# by the plan's end and by JSC's, and the last 24 hours against them.
ROWS="$ROWS" JSC="$JSC" python3 - "$now" "$TOTAL" "$START" "$END" <<'PY2'
import os
import sys
from datetime import datetime

now, total, start, plan_end = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
jsc = os.environ.get("JSC", "").split()
billed = float(jsc[0]) if jsc else None
period_start = jsc[2] if len(jsc) > 3 and jsc[2] else start
jsc_end = jsc[-1] if len(jsc) > 2 else "2026-12-31"


def ts(day, end_of_day=False):
    d = datetime.fromisoformat(day[:10])
    return d.timestamp() + (86400 - 1 if end_of_day else 0)


slurm = last24 = 0.0
for line in os.environ["ROWS"].splitlines():
    f = line.split("|")
    if len(f) < 7 or "PENDING" in f[2] or f[5] in ("", "Unknown", "None"):
        continue
    gpus = sum(int(t.split("=")[1]) for t in f[4].split(",") if t.startswith("gres/gpu="))
    t0 = datetime.fromisoformat(f[5]).timestamp()
    t1 = now if f[6] in ("", "Unknown", "None") else datetime.fromisoformat(f[6]).timestamp()
    slurm += (t1 - t0) * gpus / 3600
    last24 += max(0.0, min(t1, now) - max(t0, now - 86400)) * gpus / 3600

p0, p1 = ts(period_start), ts(jsc_end, True)
pace = (now - p0) / (p1 - p0)
rows = [("Allocation", f"{total:,.0f}", f"{total / 4:,.0f}")]
if billed is not None:
    ahead = (billed - pace * total) / 4   # node-hours spent beyond a uniform pace
    rows += [("Billed (jutil, lags a day)", f"{billed:,.1f} ({100 * billed / total:.1f}%)", f"~{billed / 4:,.0f}"),
             (f"Uniform pace today (from {period_start[:10]})", f"{100 * pace:.1f}%",
              f"{ahead:+,.0f} node-hours ({'ahead' if ahead >= 0 else 'behind: unspent hours lapse'})"),
             ("Left by JSC's count", f"{total - billed:,.1f}", f"~{(total - billed) / 4:,.0f}")]
rows.append(("Left by Slurm's count (today included)", f"{total - slurm:,.1f}", f"~{(total - slurm) / 4:,.0f}"))
need = {}
for label, end in ((f"plan, to {plan_end}", plan_end), (f"JSC's period, to {jsc_end[:10]}", jsc_end)):
    days = (ts(end, True) - now) / 86400
    lo = (total - slurm) / days
    hi = (total - billed) / days if billed is not None else lo
    lo, hi = min(lo, hi), max(lo, hi)
    need[end] = lo
    rows.append((f"Needed a day, {label} ({days:.1f} days)", f"{lo:,.0f}-{hi:,.0f}", f"{lo / 4:.1f}-{hi / 4:.1f}"))
mark = "  [!!] below the pace" if last24 < min(need.values()) else ""
rows.append(("Last 24 hours", f"{last24:,.1f}", f"{last24 / 4:.1f}{mark}"))
w = max(len(r[0]) for r in rows)
print(f"\n  {'':{w}}  {'GPU-hours':>18}  node-hours")
for a, b, c in rows:
    print(f"  {a:{w}}  {b:>18}  {c}")
PY2
