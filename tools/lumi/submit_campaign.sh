#!/usr/bin/env bash
# Keep N whole nodes on the campaign: N independent chains of jobs, each job
# starting when the previous one in its chain ends (afterany), all of them
# draining the same queue.
#
#   bash tools/lumi/submit_campaign.sh [nodes=2] [jobs-per-chain=3] [sbatch args...]
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

for n in $(seq 1 "$NODES"); do
    prev=""
    for i in $(seq 1 "$LINKS"); do
        dep=()
        [ -n "$prev" ] && dep=(--dependency=afterany:"$prev")
        jid=$(sbatch --parsable "${dep[@]}" \
            --output="$CAMPAIGN_DIR/logs/eullm-campaign-%j.out" \
            --export=ALL "$@" "$EULLM_REPO/tools/lumi/sbatch_campaign.slurm")
        jid="${jid%%;*}"
        echo "chain $n, job $i/$LINKS: $jid${prev:+ (after $prev)}"
        prev="$jid"
    done
done
echo "watch: bash $EULLM_REPO/tools/lumi/status.sh"
