#!/usr/bin/env bash
# Keep N whole nodes on the campaign: N independent chains of jobs, each job
# starting when the previous one in its chain ends (afterany), all of them
# draining the same queue.
#
#   bash tools/lumi/submit_campaign.sh [nodes=2] [jobs-per-chain=3] [sbatch args...]
#   TOP_UP=1 bash tools/lumi/submit_campaign.sh 4 2
#
# TOP_UP=1 makes N the number of chains to have, not to add: the chains
# already in the queue (a campaign job running, or waiting for anything but
# another job) are counted and only the missing ones submitted, so the same
# command can be run again, or after an earlier one, without doubling them.
#
# Each job asks for the time its label's waiting points need, spread over the
# N chains (`campaign.py walltime`), not the 48 hours of
# sbatch_campaign.slurm: a job asking 48 hours for a few hours of work waits
# on priority, one asking a few starts in the gaps Slurm leaves before the
# nodes it holds for others (24-10-2026, within the minute). The runner reads
# the time it got and starts only the points that fit. --time among the
# sbatch args overrides it; nothing is submitted when no point waits for the
# label.
#
# Two nodes in parallel is the default because one is not enough any more:
# with the first month of the window gone unused, spending the rest by
# 12-03-2027 takes about 1.2 nodes around the clock, and queue waits come off
# that. Each chain of three 48-hour jobs covers six days; re-run this (or
# status.sh says so) before the chains run out.
#
# Before the first submission, on a login node:
#   bash tools/lumi/campaign_setup.sh      # models, workload sets, the queue

set -euo pipefail

NODES="${1:-2}"
LINKS="${2:-3}"
shift $(( $# > 2 ? 2 : $# ))

# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "$(dirname "$0")/campaign_env.sh"
: "${EULLM_BIN:?set EULLM_BIN to the ROCm eullm binary}"
: "${SBATCH_ACCOUNT:?set SBATCH_ACCOUNT (and SALLOC_ACCOUNT, SLURM_ACCOUNT: docs/lumi/lumi-g.md)}"
export EULLM_BIN

mkdir -p "$CAMPAIGN_DIR/logs"
# Its first lines only: `head` closes the pipe before a long status is out,
# and under pipefail the status command dying of that stopped the script
# before any job was submitted (09-10-2026).
$CAMPAIGN status --queue "$CAMPAIGN_DIR" 2>/dev/null | head -3 || true
# A job runs only the points planned with its label (plan --engine-label),
# or only the unlabelled ones without one.
echo "engine $EULLM_BIN, label ${EULLM_ENGINE_LABEL:-(none: unlabelled points only)}"

# The devices each job asks for: a whole node's 8 unless --gpus-per-node says
# fewer (small-g, for points of one GCD).
GCDS=8
PREV=""
for a in "$@"; do
    case "$a" in --gpus-per-node=*) GCDS="${a#*=}" ;; esac
    [ "$PREV" = "--gpus-per-node" ] && GCDS="$a"
    PREV="$a"
done
TIME=()
case " $* " in
    *" --time"* | *" -t "*) ;;
    *)
        WALL=$($CAMPAIGN walltime --queue "$CAMPAIGN_DIR" --nodes "$NODES" --gcds "$GCDS" \
            --engine-label "${EULLM_ENGINE_LABEL:-}") || { echo "nothing submitted"; exit 0; }
        echo "time per job: $WALL (campaign.py walltime; --time overrides)"
        TIME=(--time="$WALL")
        ;;
esac

if [ "${TOP_UP:-0}" = 1 ]; then
    # A chain's later links wait on a dependency; its head runs or waits
    # for anything else (priority, resources).
    HEADS=$(squeue --me -h -n eullm-campaign -o '%t %r' |
        awk '$1=="R" || ($1=="PD" && $2 !~ /Dependency/)' | grep -c . || true)
    echo "chains in the queue: $HEADS, wanted: $NODES"
    NODES=$(( NODES > HEADS ? NODES - HEADS : 0 ))
    [ "$NODES" -eq 0 ] && { echo "nothing to add"; exit 0; }
fi

for n in $(seq 1 "$NODES"); do
    prev=""
    for i in $(seq 1 "$LINKS"); do
        dep=()
        [ -n "$prev" ] && dep=(--dependency=afterany:"$prev")
        jid=$(sbatch --parsable "${dep[@]}" \
            --output="$CAMPAIGN_DIR/logs/eullm-campaign-%j.out" \
            --export=ALL "${TIME[@]}" "$@" "$EULLM_REPO/tools/lumi/sbatch_campaign.slurm")
        jid="${jid%%;*}"
        echo "chain $n, job $i/$LINKS: $jid${prev:+ (after $prev)}"
        prev="$jid"
    done
done
echo "watch: bash $EULLM_REPO/tools/lumi/status.sh"
