#!/usr/bin/env bash
# Submit a chain the moment its input exists, and not before.
#
# On 2026-09-26 the administrative arm was submitted twice before its corpus
# had reached Leonardo: nine links died in three seconds each on "no such
# data directory", and a second chain of twenty had to be cancelled by hand.
# The alternative on offer was worse — someone checking every few minutes
# whether a file had arrived. This does the checking.
#
#   cd $EULLM_RUN_DIR
#   bash submit_when_ready.sh --need $WORK/datasets/legal_it_amm/train.jsonl \
#        --need $WORK/datasets/legal_it_amm/val.jsonl \
#        -- sbatch_phase2_amm.slurm 20
#
# Everything after `--` is handed to submit_chain.sh unchanged.
#
# WHAT IT DOES. Every --need file present: submit the chain, done. Anything
# missing: say what, and queue ITSELF on the serial partition to look again
# in --every minutes (default 30), up to --tries times (default 144, three
# days). A job waiting on its begin time costs nothing, and a login-node cron
# would not do: there are many login nodes and a crontab lives on one.
#
# SAFE TO RUN AGAIN, which is the point of it. It will not submit a chain
# whose job name is already in the queue, and it will not queue a second
# watcher beside one already waiting — so the command can be pasted early,
# twice, or out of order, and the worst it does is say so.
#
# Files are complete when they appear: rsync writes to a temporary name and
# renames at the end, so a present train.jsonl is a whole one. Needing both
# train and val means a copy that stopped halfway does not count.

set -euo pipefail

NEED=()
EVERY=30
TRIES=144
while [ $# -gt 0 ]; do
    case "$1" in
        --need)  NEED+=("${2:?--need takes a path}"); shift 2 ;;
        --every) EVERY="${2:?--every takes minutes}"; shift 2 ;;
        --tries) TRIES="${2:?--tries takes a count}"; shift 2 ;;
        --)      shift; break ;;
        *)       echo "[wait] unknown option $1 (the chain's arguments go after --)" >&2; exit 1 ;;
    esac
done
[ "${#NEED[@]}" -gt 0 ] || { echo "[wait] nothing to wait for: give at least one --need" >&2; exit 1; }
SCRIPT="${1:?[wait] after -- give the chain: <script.slurm> [count] [sbatch args]}"
case "$EVERY$TRIES" in *[!0-9]*) echo "[wait] --every and --tries are integers" >&2; exit 1 ;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
# Paths relative to where the user stood, so the watcher job finds the same
# files: Slurm starts it in the submission directory, but say it anyway.
[ -f "$SCRIPT" ] || SCRIPT="$HERE/$SCRIPT"
[ -f "$SCRIPT" ] || { echo "[wait] no such sbatch script: $1" >&2; exit 1; }

CHAIN="$(sed -n 's/^#SBATCH --job-name=//p' "$SCRIPT" | head -1)"
[ -n "$CHAIN" ] || { echo "[wait] $SCRIPT has no --job-name to check the queue by" >&2; exit 1; }
WATCHER="wait-$CHAIN"
now() { date '+%F %T'; }

# Already queued? Then there is nothing to do, whoever queued it.
if [ -n "$(squeue --me -h -n "$CHAIN" -o %i)" ]; then
    echo "[wait] $(now) $CHAIN is already in the queue — not submitting it again"
    exit 0
fi

missing=()
for f in "${NEED[@]}"; do [ -s "$f" ] || missing+=("$f"); done

if [ "${#missing[@]}" -eq 0 ]; then
    echo "[wait] $(now) everything is there — submitting $CHAIN"
    shift
    exec bash "$HERE/submit_chain.sh" "$SCRIPT" "$@"
fi

for f in "${missing[@]}"; do echo "[wait] $(now) not yet: $f"; done

# A watcher already waiting? Then this call only reported. The running
# watcher (this job, when it is one) is not "another" one.
others="$(squeue --me -h -t PENDING -n "$WATCHER" -o %i | grep -vx "${SLURM_JOB_ID:-none}" || true)"
if [ -n "$others" ]; then
    echo "[wait] a watcher is already waiting ($others) — it will look again by itself"
    exit 0
fi

if [ "$TRIES" -le 0 ]; then
    echo "[wait] $(now) gave up: still missing after the last try. Run this again to restart." >&2
    exit 1
fi

mkdir -p logs
again=()
for f in "${NEED[@]}"; do again+=(--need "$f"); done
jid=$(sbatch --parsable -J "$WATCHER" -p lrd_all_serial -c 1 --mem=1G -t 00:05:00 \
      --begin="now+${EVERY}minutes" -o "logs/$WATCHER-%j.out" \
      "$SELF" "${again[@]}" --every "$EVERY" --tries $((TRIES - 1)) -- "$@")
echo "[wait] $(now) will look again in $EVERY minutes (job ${jid%%;*}, $((TRIES - 1)) tries left)"
echo "[wait]   cancel with: scancel -n $WATCHER"
