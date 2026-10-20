#!/usr/bin/env bash
# Submit a chain of campaign links (sbatch_campaign.slurm) on Leonardo, after
# setup_campaign.sh has planned the points:
#
#     bash tools/leonardo/submit_campaign.sh [LINKS]    # default 2
#
# Each link waits for the one before it to end in any way (afterany: a link
# that reaches its walltime ends in TIMEOUT, which is the design), and starts
# from what is left in the queue; a link that finds the queue empty ends at
# once. Two links hold c13-sites' 14 points (docs/sites-comparison.md).
#
# Output: $CAMPAIGN_DIR/logs/eullm-campaign-<job>.out and summary-<job>.txt.

set -euo pipefail
# shellcheck source=campaign_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/campaign_env.sh"
LINKS="${1:-2}"
[ -x "$EULLM_BIN" ] || { echo "no engine at $EULLM_BIN: run setup_campaign.sh" >&2; exit 1; }
if [ -n "$(squeue --me -h -n eullm-campaign -o %i 2>/dev/null)" ]; then
    echo "a campaign chain is already queued: nothing submitted"
    squeue --me -n eullm-campaign
    exit 0
fi
export EULLM_BIN EULLM_REPO
# The checkout's commit, read here: compute nodes have no git.
EULLM_BENCH_REV=$(git -C "$EULLM_REPO" rev-parse --short HEAD)
[ -z "$(git -C "$EULLM_REPO" status --porcelain --untracked-files=no)" ] ||
    EULLM_BENCH_REV="$EULLM_BENCH_REV-dirty"
export EULLM_BENCH_REV
mkdir -p "$CAMPAIGN_DIR/logs"
$CAMPAIGN status --queue "$CAMPAIGN_DIR" 2>/dev/null | head -3 || true

dep=""
for i in $(seq 1 "$LINKS"); do
    jid=$(sbatch --parsable --account "$SBATCH_ACCOUNT" ${dep:+--dependency=afterany:$dep} \
        --output="$CAMPAIGN_DIR/logs/eullm-campaign-%j.out" --export=ALL \
        "$EULLM_REPO/tools/leonardo/sbatch_campaign.slurm")
    dep="${jid%%;*}"
    echo "link $i/$LINKS: job $dep"
done
echo "logs: $CAMPAIGN_DIR/logs/eullm-campaign-<job>.out, summary-<job>.txt"
