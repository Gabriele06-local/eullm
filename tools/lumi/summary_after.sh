#!/usr/bin/env bash
# Queue the summary of campaign jobs to be written when each ends: a small
# CPU job per campaign job, started by Slurm after it (afterany), which runs
# job_summary.sh. For jobs submitted before sbatch_campaign.slurm wrote its
# own summary; nobody has to watch for them to end.
#
#   bash tools/lumi/summary_after.sh 22981849 22982573 22982132

set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "$(dirname "$0")/campaign_env.sh"
[ $# -gt 0 ] || { echo "usage: $0 JOB [JOB ...]" >&2; exit 2; }
: "${SBATCH_ACCOUNT:=project_465003366}"

for JOB in "$@"; do
    if [ -z "$(squeue -h -j "$JOB" -o %i 2>/dev/null)" ]; then
        echo "job $JOB: no longer queued, summarising now"
        bash "$EULLM_REPO/tools/lumi/job_summary.sh" "$JOB"
        continue
    fi
    sbatch --parsable --account="$SBATCH_ACCOUNT" --partition=small --ntasks=1 \
        --cpus-per-task=1 --mem=2G --time=00:20:00 --job-name="eullm-summary-$JOB" \
        --dependency="afterany:$JOB" --output="$CAMPAIGN_DIR/logs/eullm-summary-%j.out" \
        --export=ALL,EULLM_REPO="$EULLM_REPO",CAMPAIGN_DIR="$CAMPAIGN_DIR" \
        --wrap="bash '$EULLM_REPO/tools/lumi/job_summary.sh' $JOB" |
        sed "s/^/job $JOB: summary job /"
done
