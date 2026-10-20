#!/usr/bin/env bash
# What a campaign job measured, written to $CAMPAIGN_DIR/logs/summary-<job>.txt:
# its points by group (`campaign.py report --job`, failures included) and how
# each point's VRAM moved while it ran (vram_slopes.py). sbatch_campaign.slurm
# runs it when its runner ends; summary_after.sh queues it behind jobs that
# were submitted before that.
#
#   bash tools/lumi/job_summary.sh 22981849 [...]

set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "$(dirname "$0")/campaign_env.sh"
[ $# -gt 0 ] || { echo "usage: $0 JOB [JOB ...]" >&2; exit 2; }

for JOB in "$@"; do
    SUMMARY="$CAMPAIGN_DIR/logs/summary-$JOB.txt"
    {
        echo "=== what job $JOB measured ==="
        $CAMPAIGN report --queue "$CAMPAIGN_DIR" --job "$JOB"
        echo
        echo "=== VRAM of each point while it ran ==="
        "$PY" "$EULLM_REPO/tools/lumi/vram_slopes.py" --queue "$CAMPAIGN_DIR" "$JOB"
    } > "$SUMMARY" 2>&1
    echo "summary: $SUMMARY"
done
