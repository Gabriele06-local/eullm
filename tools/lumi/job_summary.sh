#!/usr/bin/env bash
# What a campaign job measured, written to $CAMPAIGN_DIR/logs/summary-<job>.txt:
# its points by group (`campaign.py report --job`, failures included), how
# each point's VRAM moved while it ran (vram_slopes.py), and, for servers with
# llama.cpp's log on, how long reading and copying their weights took (patch
# 0012's load_all_data line, c11's load-* groups). sbatch_campaign.slurm
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
        LOADS=$(grep -H "load_all_data: .* weights" "$CAMPAIGN_DIR/results/logs/$JOB"/*.log 2>/dev/null |
            sed -E 's|^.*/([^/]+)\.server([0-9]+)\.log:.*load_all_data: |\1 server\2: |')
        if [ -n "$LOADS" ]; then
            echo
            echo "=== how each server read and copied its weights (llama.cpp's load_all_data) ==="
            echo "$LOADS"
        fi
    } > "$SUMMARY" 2>&1
    echo "summary: $SUMMARY"
done
