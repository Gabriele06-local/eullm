#!/usr/bin/env bash
# Submit N links of the distillation chain (sbatch_distill.slurm), on a
# JUPITER login node, from the checkout:
#
#     bash forge/scripts/jupiter/submit_distill.sh N
#
# The links follow one another (--dependency=singleton on the job name), and
# each resumes the runs from their newest checkpoint. Checked first, and
# nothing is submitted if one check fails (forge/CLAUDE.md, rule 7): the
# corpus, the models in the offline cache, the configs, and that no link of
# the chain is already queued or running (FORGE_CHAIN_MORE=1 adds N links
# behind the ones there). FORGE_CHAIN_AFTER=<job> makes the chain wait for
# that job to end, whatever its outcome (a probe, say), so the allocation
# never runs more than two nodes at once. FORGE_CHAIN_NAME names the chain
# (default forge-distill): a chain of another name runs beside this one
# instead of waiting behind it.

set -euo pipefail
N="${1:?give the number of two-hour links}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
cd "$EULLM_REPO"
DATA="${FORGE_DISTILL_DATA:-$FORGE_DATA_DIR/norms}"
CFG_DIR="$EULLM_REPO/forge/training/configs/jupiter"
ARMS="${FORGE_DISTILL_ARMS:-norms-4b:$CFG_DIR/distill_qwen3_30b_a3b_to_4b_norms.yaml norms-8b:$CFG_DIR/distill_qwen3_30b_a3b_to_8b_norms.yaml}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }

[ -z "${SLURM_JOB_ID:-}" ] || err "submit from a login node, not from a job"
[ -s "$DATA/train.jsonl" ] && [ -s "$DATA/val.jsonl" ] ||
    err "no corpus at $DATA: make_norms_split.py first"
ok "corpus $DATA: $(wc -l <"$DATA/train.jsonl") train, $(wc -l <"$DATA/val.jsonl") val"

MODELS=()
for arm in $ARMS; do
    cfg="${arm#*:}"
    [ -f "$cfg" ] || err "no config $cfg"
    MODELS+=("$(sed -n 's/^teacher_model:[[:space:]]*//p' "$cfg")"
             "$(sed -n 's/^student_model:[[:space:]]*//p' "$cfg")")
done
# The way the jobs will load them: offline, from $HF_HOME.
HF_HUB_OFFLINE=1 python - "${MODELS[@]}" <<'PY' || err "a model is not in \$HF_HOME: prefetch_models.py --models ..."
import sys
from transformers import AutoConfig
for m in sorted(set(sys.argv[1:])):
    AutoConfig.from_pretrained(m, local_files_only=True)
    print(f"[ok]  {m} in the offline cache")
PY

NAME="${FORGE_CHAIN_NAME:-forge-distill}"
QUEUED=$(squeue --me -h -n "$NAME" -o %i | wc -l)
if [ "$QUEUED" -gt 0 ] && [ "${FORGE_CHAIN_MORE:-0}" != 1 ]; then
    squeue --me -n "$NAME"
    err "$QUEUED link(s) of the chain already queued or running: FORGE_CHAIN_MORE=1 to add $N more"
fi

DEP=singleton
[ -z "${FORGE_CHAIN_AFTER:-}" ] || DEP="afterany:$FORGE_CHAIN_AFTER,singleton"
mkdir -p "$FORGE_RUN_DIR/logs"
# sbatch refuses while the partition is down for maintenance ("Required
# partition not available", 21-10-2026) or the controller does not answer
# ("Connection reset by peer", "Zero Bytes were transmitted"): those are
# waited out, FORGE_SUBMIT_WAIT_H hours at most (default 24; run it under
# tmux), a link at a time, so that nobody has to come back and submit.
# Any other refusal stops at once.
WAIT_UNTIL=$(( $(date +%s) + ${FORGE_SUBMIT_WAIT_H:-24} * 3600 ))
submit_link() {
    local out
    while :; do
        if out=$(sbatch --account "$SBATCH_ACCOUNT" --job-name "$NAME" --dependency="$DEP" \
                --output="$FORGE_RUN_DIR/logs/distill-%j.out" "$HERE/sbatch_distill.slurm" 2>&1); then
            echo "$out"; return 0
        fi
        case "$out" in
            *"partition not available"*|*"Connection reset"*|*"Zero Bytes"*|*"Socket timed out"*) ;;
            *) echo "$out" >&2; return 1 ;;
        esac
        [ "$(date +%s)" -lt "$WAIT_UNTIL" ] || { echo "$out" >&2; err "still refused after ${FORGE_SUBMIT_WAIT_H:-24} h"; }
        echo "[..] $(date '+%d-%m %H:%M') sbatch: ${out##*error: }; again in 15 minutes"
        sleep 900
    done
}
for _ in $(seq 1 "$N"); do
    submit_link || err "sbatch refused a link"
done
squeue --me -n "$NAME"
echo "The output of each link: $FORGE_RUN_DIR/logs/distill-<job>.out"
