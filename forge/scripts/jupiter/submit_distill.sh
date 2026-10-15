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
# behind the ones there).

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

QUEUED=$(squeue --me -h -n forge-distill -o %i | wc -l)
if [ "$QUEUED" -gt 0 ] && [ "${FORGE_CHAIN_MORE:-0}" != 1 ]; then
    squeue --me -n forge-distill
    err "$QUEUED link(s) of the chain already queued or running: FORGE_CHAIN_MORE=1 to add $N more"
fi

mkdir -p "$FORGE_RUN_DIR/logs"
for _ in $(seq 1 "$N"); do
    sbatch --account "$SBATCH_ACCOUNT" --dependency=singleton \
        --output="$FORGE_RUN_DIR/logs/distill-%j.out" \
        "$HERE/sbatch_distill.slurm"
done
squeue --me -n forge-distill
echo "The output of each link: $FORGE_RUN_DIR/logs/distill-<job>.out"
