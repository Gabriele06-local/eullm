#!/usr/bin/env bash
# GPU-hours spent on JUPITER Booster and the daily rate that spends the rest
# of the allocation by its end, from Slurm's own accounting (sacct):
#
#     bash tools/jupiter/budget.sh [END_DATE] [TOTAL_GPU_HOURS]
#
# END_DATE defaults to 2026-12-31 and TOTAL_GPU_HOURS to 5000 (the award).
# A booster job holds whole nodes and pays for every GPU of them, so a job's
# GPU-hours are its elapsed time times the GPUs it was given (gres/gpu in
# AllocTRES): 4 an hour per node. Running jobs count up to now; queued ones
# are shown at their time limit, the most they can cost. JuDoor has the
# official figure; this one is for steering.

set -uo pipefail
END="${1:-2026-12-31}"
TOTAL="${2:-5000}"
ACCOUNT="${SBATCH_ACCOUNT:-e-aif-2026pg01-1434}"
START="${JUPITER_START:-2026-10-14}"

now=$(date +%s)
end=$(date -d "$END 23:59:59" +%s) || { echo "[err] not a date: $END"; exit 1; }

# JobID|JobName|State|ElapsedRaw|AllocTRES, one line a job (allocations only).
ROWS=$(sacct -X -n -P -A "$ACCOUNT" -S "$START" -E now \
    -o JobID,JobName,State,ElapsedRaw,AllocTRES 2>/dev/null)
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
