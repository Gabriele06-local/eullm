# sbatch that waits out what passes: the partition down for maintenance
# ("Required partition not available", 21-10-2026) or a controller that does
# not answer ("Connection reset by peer", "Zero Bytes were transmitted"). It
# tries again every 15 minutes, FORGE_SUBMIT_WAIT_H hours at most (default
# 24; run the submitting script under tmux), so that nobody has to come back
# and submit. Any other refusal is returned at once. Source it, then:
#
#     jid=$(sbatch_retry --parsable ... script.slurm) || exit 1
#
# The job id (or sbatch's own output) goes to stdout, the waiting to stderr.

SBATCH_RETRY_UNTIL=$(( $(date +%s) + ${FORGE_SUBMIT_WAIT_H:-24} * 3600 ))
sbatch_retry() {
    local out
    while :; do
        if out=$(sbatch "$@" 2>&1); then
            echo "$out"
            return 0
        fi
        case "$out" in
            *"partition not available"*|*"Connection reset"*|*"Zero Bytes"*|*"Socket timed out"*) ;;
            *) echo "$out" >&2; return 1 ;;
        esac
        if [ "$(date +%s)" -ge "$SBATCH_RETRY_UNTIL" ]; then
            echo "$out" >&2
            echo "[!!] sbatch still refused after ${FORGE_SUBMIT_WAIT_H:-24} h" >&2
            return 1
        fi
        echo "[..] $(date '+%d-%m %H:%M') sbatch: ${out##*error: }; again in 15 minutes" >&2
        sleep 900
    done
}
