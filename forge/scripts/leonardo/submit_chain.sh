#!/usr/bin/env bash
# Submit N chained copies of an sbatch script (afterany dependencies).
#
# Leonardo's boost_usr_prod partition caps jobs at 24 h, so a multi-day
# phase is a chain: when job k dies (walltime or otherwise), job k+1
# starts and the launchers (train.sh / distill.sh) resume from the
# latest checkpoint. A count of 1 is a plain submission with the logs
# dir prepared.
#
# Usage:
#   bash submit_chain.sh [--after <jobid> | --after-any <jobid>] \
#                        <script.slurm> [count] [extra sbatch args...]
#
# Examples:
#   bash submit_chain.sh sbatch_smoke.slurm
#   bash submit_chain.sh sbatch_phase2.slurm 7
#   bash submit_chain.sh --after 56912622 sbatch_phase2.slurm 7
#   bash submit_chain.sh --after-any 57353624 sbatch_phase2.slurm 4
#
# `--after <jobid>` holds the FIRST link until that job finishes cleanly
# (afterok). That is right when the predecessor is a DIFFERENT phase: a
# failed Phase 1 wrote no checkpoint, so a Phase 2 that started anyway would
# resume from nothing.
#
# `--after-any <jobid>` holds it until the predecessor ends any which way
# (afterany), and is what EXTENDING AN EXISTING CHAIN needs. A 24 h link is
# meant to end in TIMEOUT — that is the design, not a fault — and TIMEOUT
# does not satisfy afterok. Appending with `--after` therefore produces a
# chain that never runs at all: it sits in DependencyNeverSatisfied until
# someone notices, which on a 61-day allocation is exactly the silent idle
# the queue-stats job exists to measure. Picking the wrong one of these two
# is easy, so `--after` now says which it used on submission.
#
# Both exist because passing
# --dependency through the extra args does NOT work and fails dangerously:
# those args are appended after the chain's own --dependency, sbatch takes the
# last occurrence, and every link would then wait on the same external job
# instead of on its predecessor. Seven Phase-2 jobs would start at once, all
# resuming from the same checkpoint. Extra args containing --dependency are
# therefore rejected outright.
#
# Run from $EULLM_RUN_DIR (the scripts' #SBATCH --output is relative to
# the submission directory). The account is picked up from
# SBATCH_ACCOUNT — export EULLM_ACCOUNT and source env.sh first.

set -euo pipefail

AFTER=""
AFTER_KIND=""
case "${1:-}" in
    --after)
        AFTER="${2:?--after needs a job id}"
        AFTER_KIND="afterok"
        shift 2
        ;;
    --after-any)
        AFTER="${2:?--after-any needs a job id}"
        AFTER_KIND="afterany"
        shift 2
        ;;
esac

SCRIPT="${1:?Usage: $0 [--after <jobid> | --after-any <jobid>] <script.slurm> [count] [extra sbatch args...]}"
COUNT="${2:-1}"
shift
if [ $# -gt 0 ]; then shift; fi

[ -f "$SCRIPT" ] || { echo "[err] sbatch script not found: $SCRIPT" >&2; exit 1; }

# See the header: a --dependency in the extra args overrides the chain's own
# and every link would wait on the same job rather than on its predecessor.
for arg in "$@"; do
    case "$arg" in
        -d|-d=*|--dependency|--dependency=*)
            echo "[err] do not pass $arg — it would override each link's own" >&2
            echo "[err] dependency and start the whole chain at once." >&2
            echo "[err] Use: $0 --after <jobid> $SCRIPT $COUNT" >&2
            exit 1
            ;;
    esac
done
case "$COUNT" in
    ''|*[!0-9]*) echo "[err] count must be a positive integer, got '$COUNT'" >&2; exit 1;;
esac

if [ -z "${SBATCH_ACCOUNT:-}" ]; then
    echo "[warn] SBATCH_ACCOUNT not set — export EULLM_ACCOUNT and source" >&2
    echo "[warn] forge/scripts/leonardo/env.sh, or pass -A <account>" >&2
fi

mkdir -p logs

prev=""
for i in $(seq 1 "$COUNT"); do
    dep=()
    if [ -z "$prev" ]; then
        # afterok when the predecessor is another phase (a FAILED one wrote
        # no checkpoint to resume from), afterany when this is an extension
        # of a chain whose links end in TIMEOUT by design. See the header:
        # the wrong one here does not fail, it queues forever.
        [ -n "$AFTER" ] && dep=(--dependency="$AFTER_KIND":"$AFTER")
    else
        # afterany within the chain: TIMEOUT is how a 24 h link is meant to
        # end, and afterok would stop the chain on every one of them.
        dep=(--dependency=afterany:"$prev")
    fi
    # ${dep[@]+...} rather than a bare "${dep[@]}": under `set -u` an empty
    # array counts as unbound on bash before 4.4, and this should not depend
    # on which node's shell it happens to run under.
    jid=$(sbatch --parsable ${dep[@]+"${dep[@]}"} "$@" "$SCRIPT")
    jid="${jid%%;*}"   # --parsable may append ';cluster'
    echo "[ok] submitted $jid ($i/$COUNT)${prev:+ — after $prev}${prev:+}"
    if [ -z "$prev" ] && [ -n "$AFTER" ]; then
        if [ "$AFTER_KIND" = "afterany" ]; then
            echo "[ok]   held until $AFTER ends, however it ends (afterany)"
        else
            echo "[ok]   held until $AFTER succeeds (afterok) — a TIMEOUT"
            echo "[ok]   predecessor will NOT release it; use --after-any"
            echo "[ok]   to extend a chain of 24 h links."
        fi
    fi
    prev="$jid"
done

echo "[ok] monitor with: squeue --me   |   logs in $(pwd)/logs/"
