#!/usr/bin/env bash
# One command, every morning: what is running, what died, what is waiting on
# what — and a line starting with [!!] for anything that needs a human.
#
#   bash $WORK/eullm/forge/scripts/leonardo/status.sh            # last 24 h
#   bash $WORK/eullm/forge/scripts/leonardo/status.sh 2026-09-27T21:00
#
# Written after a night (2026-09-27/28) lost to a watcher that said "not yet"
# every twenty minutes about a file that was there but empty. Everything
# needed to see it was on disk; nobody had one place to look. The problems
# looked for are the ones that have actually cost runs here:
#
#   * a job that ended FAILED / OUT_OF_MEMORY / NODE_FAIL, and a training
#     link that ended within ten minutes of starting — the shape of a gate
#     refusing the data, a missing file, or the quota (2026-09-26);
#   * a job pending for a reason that never resolves by itself
#     (DependencyNeverSatisfied, a hold);
#   * a watcher whose file exists but is empty. One that simply waits long
#     is shown, not flagged: the package watcher waits a whole training run;
#   * $WORK more than 90% full. On 2026-10-05 it reached 109% of its 1 TB
#     quota unnoticed, and a 27B conversion failed writing its first tensor;
#     every job that writes a checkpoint would have been next;
#   * a base model moved to the fast scratch (move_to_scratch.sh) whose copy
#     has been purged: the symlink on $WORK points at nothing;
#   * no GPU job running or queued at all. On 2026-10-04 the allocation spent
#     1.8 node-hours in a day with ~530 left for 29 days, and it was the user
#     who noticed. Idle GPUs are flagged, unspent hours are not: the rule is
#     never to run out of useful work, not to burn the budget on filler.
#     What is left of the allocation, and the daily pace that would use it,
#     is shown next to the pace of the last day.
#
# Read-only: it submits, cancels and writes nothing. It prints counts and
# log lines of the pipeline, never the held-out exam.

set -uo pipefail

# $WORK is Leonardo's. Run on another machine, the line below used to stop
# the script with "WORK: unbound variable" and nothing else (5 October).
if [ -z "${EULLM_RUNS:-}" ] && [ -z "${WORK:-}" ]; then
    echo "status.sh runs on Leonardo, where \$WORK is set: ssh <user>@login.leonardo.cineca.it first" >&2
    exit 2
fi
RUNS="${EULLM_RUNS:-$WORK/eullm_runs}"
# The checkout's own logs/: where a job submitted without -o writes
# (sbatch scripts say --output=logs/%x-%j.out, relative to where they were
# submitted). GRPO chains submitted from $WORK/eullm log there, and links
# that did their work were flagged as having done none (2026-10-15).
REPO_LOGS="${EULLM_REPO_LOGS:-$(cd "$(dirname "$0")/../../.." 2>/dev/null && pwd)/logs}"
SINCE="${1:-$(date -d '-24 hours' +%Y-%m-%dT%H:%M)}"
PROBLEMS=0
flag() { echo "[!!] $*"; PROBLEMS=$((PROBLEMS + 1)); }

secs() {  # [D-]HH:MM:SS or MM:SS -> seconds
    local t="$1" d=0 h=0 m=0 s=0
    case "$t" in *-*) d="${t%%-*}"; t="${t#*-}" ;; esac
    IFS=: read -r a b c <<< "$t"
    if [ -n "$c" ]; then h=$a; m=$b; s=$c; else m=$a; s=$b; fi
    echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#${s:-0} ))
}

echo "== queue now =="
while IFS='|' read -r name state reason; do
    case "$reason" in
        DependencyNeverSatisfied*|JobHeld*|*launch*failed*|BadConstraints*|InvalidAccount*|InvalidQOS*)
            flag "$name is $state and will not start by itself: $reason" ;;
    esac
done < <(squeue --me -h -o "%j|%T|%r")
squeue --me -h -o "%j %T" | sort | uniq -c | awk '{printf "   %-26s %-8s %d\n", $2, $3, $1}'

echo
echo "== ended since $SINCE =="
while IFS='|' read -r id name state elapsed; do
    [ -n "$id" ] || continue
    case "$state" in
        FAILED*|OUT_OF_ME*|NODE_FAIL*|BOOT_FAIL*|DEADLINE*|PREEMPTED*)
            flag "$id $name ended $state after $elapsed — log: logs/$name-$id.out" ;;
        COMPLETED*|TIMEOUT*)
            case "$name" in
                # Two jobs the eullm-p* glob below would take, and both are
                # short on purpose rather than stalled. sbatch_quantize.slurm
                # submits eullm-p3-gguf on lrd_all_serial with no --gres at
                # all, a CPU quantize that is done in minutes; and
                # sbatch_backfill_probe.slurm's own header says eullm-probe
                # "does nothing but report where it landed and exit", so it
                # ends in seconds every time it is run. Neither resumes a
                # chain, so there is no work for them to have failed to do.
                eullm-p3-gguf|eullm-probe) ;;
                # eullm-opd-prompts builds the OPD prompts file and trains
                # nothing: a few minutes is all it ever takes.
                eullm-opd-prompts) ;;
                # eullm-package-NAME: a GRPO round's merge and GGUF, one job
                # and no chain; a 4B is packaged in nine minutes (jpn17,
                # 2026-10-25), and a failed one ends FAILED, flagged above.
                eullm-package-*) ;;
                # eullm-s3-*: stage-3 chains submitted under a name of their
                # own (-J), one per experiment.
                eullm-p*|eullm-stage3|eullm-s3-*|eullm-gen-*|eullm-grpo*|eullm-opd*)
                    if [ "$(secs "$elapsed")" -lt 600 ]; then
                        # The last link of a chain finds nothing left and
                        # exits in seconds, which is correct. So does a
                        # stage-3 link that resumed from the final checkpoint
                        # and only wrote the adapter out (before stage3_sft.py
                        # learnt to stop when the adapter is already there).
                        # grpo_train.py prints the same line with its own
                        # prefix when it saves, so a GRPO link that resumed
                        # at the last step is the same case and was flagged
                        # as one that did no work. opd_train.py is the third
                        # of the shape: its save line wears the [opd] prefix.
                        log="$(ls "$RUNS"/*/logs/"$name-$id".out "$REPO_LOGS/$name-$id.out" 2>/dev/null | head -1)"
                        if [ -n "$log" ] && grep -q "nothing left to do" "$log"; then
                            echo "   $id $name: ended in $elapsed with nothing left to do (fine)"
                        elif [ -n "$log" ] && grep -qE "^\[(stage3|grpo|opd)\] adapter /" "$log"; then
                            echo "   $id $name: ended in $elapsed, training finished (fine)"
                        else
                            flag "$id $name ended $state after only $elapsed — a link that short did no work"
                        fi
                    fi ;;
            esac ;;
    esac
done < <(sacct -X -n -P -S "$SINCE" -o JobID,JobName,State,Elapsed 2>/dev/null |
         grep -v -E '\|(RUNNING|PENDING)')
sacct -X -n -P -S "$SINCE" -o JobName,State 2>/dev/null | grep -v '^wait-' |
    awk -F'|' '{split($2, s, " "); n[$1" "s[1]]++} END {for (k in n) printf "   %-40s %d\n", k, n[k]}' | sort

echo
echo "== watchers =="
found=0
for name in $(squeue --me -h -o "%j" | grep '^wait-' | sort -u); do
    found=1
    log="$(ls -t "$RUNS"/*/logs/"$name"-[0-9]*.out 2>/dev/null | head -1)"
    if [ -z "$log" ]; then
        echo "   $name: queued, no log found under $RUNS/*/logs"
        continue
    fi
    grep '^\[wait\]' "$log" | sed 's/^/   /'
    # What it waits for is judged NOW, not from its log: the log is as old as
    # its last wake, and the file may have arrived since. An empty file never
    # counts, so that is the one state that needs a human.
    ready=1
    for f in $(sed -n -e 's/.*not yet: //p' -e 's/.*counts as missing): //p' "$log"); do
        if [ -e "$f" ] && [ ! -s "$f" ]; then
            flag "$name waits on $f, which exists but is EMPTY — it will never start by itself"
            ready=0
        elif [ ! -s "$f" ]; then
            ready=0
        fi
    done
    if [ "$ready" -eq 1 ]; then
        echo "   -> everything it needs is there now: it submits at its next wake"
    fi
    # Waiting long is normal when the input is a long job's output (the
    # package watcher waits for a whole training run), so this is a note.
    left="$(sed -n 's/.*(job [0-9]*, \([0-9]*\) tries left).*/\1/p' "$log" | tail -1)"
    if [ -n "$left" ]; then
        echo "   -> $left tries left"
    fi
done
[ "$found" -eq 1 ] || echo "   none queued"

echo
echo "== disk =="
# df, not cindata: cindata's figure lags by hours, df is the filesystem now.
pct="$(df -P "${WORK:-$RUNS}" 2>/dev/null | awk 'NR == 2 {sub("%", "", $5); print $5}')"
if [ -n "$pct" ]; then
    echo "   \$WORK ${pct}% full"
    if [ "$pct" -ge 90 ]; then
        flag "\$WORK is ${pct}% full: free space before a job fails writing (du -sh \$WORK/*)"
    fi
fi
# The fast scratch has a 1 TB quota of its own: on 2026-10-24 it was full
# with the merged weights of 33 GRPO rounds and every merge failed there.
SCRATCH="${EULLM_SCRATCH:-/leonardo_scratch/fast/$(basename "${WORK:-/nonexistent}")}"
spct=""
[ -d "$SCRATCH" ] && spct="$(df -P "$SCRATCH" 2>/dev/null | awk 'NR == 2 {sub("%", "", $5); print $5}')"
if [ -n "$spct" ]; then
    echo "   scratch ${spct}% full"
    if [ "$spct" -ge 90 ]; then
        flag "the scratch is ${spct}% full: merges write there (du -sh $SCRATCH/*)"
    fi
fi
# Base models live on the fast scratch behind symlinks (move_to_scratch.sh),
# and the scratch is purged when unused. A symlink left pointing at nothing is
# a model every job that needs it will fail to load -- compute nodes cannot
# download it again -- so say which, and how to get it back.
for link in "${WORK:-/nonexistent}"/models/* "${HF_HOME:-/nonexistent}/hub"; do
    if [ -L "$link" ] && [ ! -e "$link" ]; then
        flag "$link points to $(readlink "$link"), which is gone (scratch purged?): download it again from a login node"
    fi
done

echo
echo "== GPU work =="
# Anything on the GPU partition counts, running or waiting for its turn or
# for a dependency: a chain queued behind a serial job is work lined up.
gpu_jobs="$(squeue --me -h -p boost_usr_prod -o "%i" 2>/dev/null | grep -c .)"
# Billed the way saldo bills: 8 local hours per GPU-hour, 32 a node-hour, so
# a one-GPU exam costs a quarter of a node. Counting every job as a whole
# node (as this did until 2026-10-06) made a day of one-GPU jobs look four
# times as expensive. Checked against saldo: October to the 6th, 2,275 here
# against saldo's 1,685 -- saldo is a day behind, the rest is today's jobs.
local_hours() {  # local hours billed on the GPU partition since $1
    sacct -X -n -S "$1" -r boost_usr_prod -o ElapsedRaw,AllocTRES -P 2>/dev/null |
        awk -F'|' '{g = 0; if (match($2, /gres\/gpu=[0-9]+/)) g = substr($2, RSTART + 9, RLENGTH - 9)
                    s += $1 * g * 8} END {printf "%.1f", s / 3600}'
}
used="$(local_hours "$SINCE")"
echo "   $gpu_jobs GPU job(s) running or queued; ${used:-0.0} local hours" \
     "($(awk -v u="${used:-0}" 'BEGIN {printf "%.1f", u / 32}') node-hours) since $SINCE"
if [ "$gpu_jobs" -eq 0 ]; then
    flag "no GPU job running or queued: the allocation is idle -- decide the next useful GPU work now"
fi
# The allocation: how much is left and the daily pace that would use it by
# its end, against the pace of the window above. Shown, never flagged: the
# aim is better models, not a spent budget (forge/CLAUDE.md, rule 8).
B_TOTAL="${EULLM_BUDGET_HOURS:-40000}"
B_START="${EULLM_BUDGET_START:-2026-09-02}"
B_END="${EULLM_BUDGET_END:-2026-11-02}"
spent="$(local_hours "$B_START")"
now="$(date +%s)"
end="$(date -d "$B_END 23:59" +%s 2>/dev/null || echo "$now")"
since_s="$(date -d "${SINCE/T/ }" +%s 2>/dev/null || echo $((now - 86400)))"
awk -v total="$B_TOTAL" -v spent="${spent:-0}" -v used="${used:-0}" -v now="$now" \
    -v end="$end" -v since="$since_s" -v start="$B_START" -v stop="$B_END" 'BEGIN {
    left = total - spent; days = (end - now) / 86400
    printf "   allocation: %.0f of %.0f local hours used since %s, %.0f left", spent, total, start, left
    if (days <= 0) { print "; it has ended"; exit }
    printf " for %.1f days = %.0f a day (%.1f node-hours)\n", days, left / days, left / days / 32
    window = (now - since) / 86400
    if (window <= 0) exit
    rate = used / window
    printf "   pace of the window above: %.0f local hours a day", rate
    if (rate * days < left)
        printf "; at that pace %.0f would be left unused on %s", left - rate * days, stop
    print ""
    print "   (saldo -b is the bill; it is a day behind these figures)"
}'
# The bill itself, where saldo is (Leonardo's login nodes). The figures above
# are this script's own count, a day ahead of saldo and not the same thing:
# on 2026-10-13 they said 63% used where saldo said 59.0%, and a catch-up
# plan was first worked out on the wrong one. The even pace is the share of
# the allocation's days gone by: below it, hours are being left behind.
if command -v saldo >/dev/null 2>&1; then
    D8='[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'
    read -r s_acct s_start s_end s_total _ s_used s_pct _ < <(saldo -b 2>/dev/null |
        awk -v d="^$D8\$" '$2 ~ d && $3 ~ d {print; exit}')
    if [ -n "${s_total:-}" ]; then
        s0="$(date -d "$s_start" +%s)"; s1="$(date -d "$s_end 23:59" +%s)"
        awk -v acct="$s_acct" -v total="$s_total" -v used="$s_used" -v pct="$s_pct" \
            -v s0="$s0" -v s1="$s1" -v now="$now" -v stop="$s_end" 'BEGIN {
            share = (now - s0) / (s1 - s0); if (share > 1) share = 1; even = total * share
            gap = used - even; word = gap < 0 ? "behind" : "ahead"; if (gap < 0) gap = -gap
            printf "   bill (saldo -b, %s): %.0f of %.0f local hours, %s%%; the even pace is %.1f%%:" \
                   " %.0f local hours (%.0f node-hours) %s\n", acct, used, total, pct, share * 100,
                   gap, gap / 32, word
            days = (s1 - now) / 86400
            if (days > 0)
                printf "   to use the rest by %s: %.0f local hours a day (%.1f node-hours)\n",
                       stop, (total - used) / days, (total - used) / days / 32
        }'
    fi
fi

echo
echo "== GRPO =="
# grpo_train.py prints a progress line every ten steps and a STOP line when
# it stops itself (a NaN, or no group disagreeing any more). The last two
# progress lines are enough to see whether reward is climbing.
found=0
for log in $(ls -t "$RUNS"/grpo/logs/eullm-grpo-*.out "$REPO_LOGS"/eullm-grpo-*.out 2>/dev/null | head -3); do
    [ -n "$(find "$log" -newermt "${SINCE/T/ }" 2>/dev/null)" ] || continue
    found=1
    echo "   ${log##*/}"
    # -o, not ^: tqdm's bar shares the log and the line can follow it.
    grep -ao '\[grpo\] step .*' "$log" | tail -2 | sed 's/^/     /'
    stop="$(grep -ao '\[grpo\] STOP.*' "$log" | tail -1)"
    [ -z "$stop" ] || flag "${log##*/}: ${stop#\[grpo\] }"
done
[ "$found" -eq 1 ] || echo "   none since $SINCE"

echo
echo "== OPD =="
# opd_train.py prints a line per step: the first and the last of each recent
# run show whether the KL is coming down, and the step reached; "adapter"
# when the run is done.
found=0
for log in $(ls -t "$RUNS"/opd/logs/eullm-opd-*.out 2>/dev/null | head -4); do
    [ -n "$(find "$log" -newermt "${SINCE/T/ }" 2>/dev/null)" ] || continue
    found=1
    echo "   ${log##*/}"
    grep -ao '\[opd\] \(step .*\|adapter .*\|time is up.*\)' "$log" | sed -n '1p;$p' | uniq |
        sed 's/^/     /'
done
[ "$found" -eq 1 ] || echo "   none since $SINCE"

echo
echo "== case-law exam =="
# cds_answer.py's summary line per model; the judges' grades are in the
# graded.csv beside the answers.
found=0
for log in $(ls -t "$REPO_LOGS"/eullm-cds-exam-*.out 2>/dev/null | head -6); do
    [ -n "$(find "$log" -newermt "${SINCE/T/ }" 2>/dev/null)" ] || continue
    line="$(grep -ao '\[cds-exam\] [^:]*: [0-9]* questions; .*->' "$log" | tail -1)"
    [ -z "$line" ] || { found=1; echo "   ${line%% ->}"; }
done
[ "$found" -eq 1 ] || echo "   none since $SINCE"

echo
echo "== latest measurements =="
for f in "$RUNS"/stage3/perplexity-cds.csv "$RUNS"/stage3/chat-smoke.csv "$RUNS"/*/exports/perplexity.csv; do
    [ -f "$f" ] || continue
    echo "   ${f#"$RUNS"/}"
    tail -n 2 "$f" | sed 's/^/     /'
done

echo
if [ "$PROBLEMS" -eq 0 ]; then
    echo "[ok] nothing wrong found"
else
    echo "[!!] $PROBLEMS thing(s) above need a look"
fi
