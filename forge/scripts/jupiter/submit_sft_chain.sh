#!/usr/bin/env bash
# Instruction tuning as a chain that finishes by itself: LINKS links of
# sbatch_sft.slurm, each starting when the one before ends (afterany; a link
# resumes the runs, skips what is done and only adds what is missing), then
# the judge (sbatch_judge.slurm), which grades every exam answered so far.
# From a login node, with the FORGE_SFT_* settings of sbatch_sft.slurm:
#
#     eullm && FORGE_SFT_TAG=ob1 FORGE_SFT_PAIRS=... FORGE_SFT_ARMS="..." \
#         bash forge/scripts/jupiter/submit_sft_chain.sh [LINKS]
#
# Checked first, and nothing submitted when one fails: the pairs, the exam,
# each arm's model in the offline cache (compute nodes have no network), the
# system prompt, and no chain of the same tag already queued.

set -uo pipefail
# shellcheck source=env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LINKS="${1:-2}"
TAG="${FORGE_SFT_TAG:?FORGE_SFT_TAG: the name of this set of runs}"
PAIRS="${FORGE_SFT_PAIRS:?FORGE_SFT_PAIRS: the pairs to train on}"
ARMS="${FORGE_SFT_ARMS:?FORGE_SFT_ARMS: name=model ...}"
EXAM="${FORGE_SFT_EXAM:-$FORGE_DATA_DIR/eval/norm-exam.jsonl}"
ACCOUNT="${SBATCH_ACCOUNT:-e-aif-2026pg01-1434}"
LOGS="$FORGE_RUN_DIR/logs"
NAME="forge-sft-$TAG"
bad=0

[ -s "$PAIRS" ] && echo "[ok] pairs $PAIRS ($(wc -l <"$PAIRS"))" || { echo "[!!] no pairs at $PAIRS"; bad=1; }
[ -s "$EXAM" ] && echo "[ok] exam $EXAM ($(wc -l <"$EXAM") questions)" || { echo "[!!] no exam at $EXAM"; bad=1; }
[ -z "${FORGE_SFT_SYSTEM_FILE:-}" ] || [ -s "$FORGE_SFT_SYSTEM_FILE" ] ||
    { echo "[!!] no system prompt at $FORGE_SFT_SYSTEM_FILE"; bad=1; }
[ -n "${FORGE_SFT_NORMS:-}" ] && echo "[ok] open book over $(wc -w <<<"$FORGE_SFT_NORMS") norm files" ||
    echo "[..] no FORGE_SFT_NORMS: closed book only"
for arm in $ARMS; do
    model="${arm#*=}"
    if [ -d "$model" ]; then
        [ -f "$model/config.json" ] && echo "[ok] ${arm%%=*}: $model" || { echo "[!!] ${arm%%=*}: no model in $model"; bad=1; }
    elif python -c "import sys; from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)" "$model" >/dev/null 2>&1; then
        echo "[ok] ${arm%%=*}: $model in the offline cache"
    else
        echo "[!!] ${arm%%=*}: $model is not in the offline cache (download it on this login node)"; bad=1
    fi
done
if [ -n "$(squeue -h -u "$USER" -n "$NAME" -o %i 2>/dev/null)" ]; then
    echo "[!!] a $NAME chain is already queued: squeue -u $USER -n $NAME"; bad=1
fi
[ "$bad" = 0 ] || { echo "[!!] nothing submitted"; exit 1; }

mkdir -p "$LOGS"
prev="" jobs=()
for i in $(seq 1 "$LINKS"); do
    j=$(cd "$REPO" && sbatch --parsable --account "$ACCOUNT" --job-name "$NAME" \
        ${prev:+--dependency=afterany:$prev} --output="$LOGS/sft-%j.out" \
        forge/scripts/jupiter/sbatch_sft.slurm) || { echo "[!!] sbatch failed at link $i"; exit 1; }
    prev="${j%%;*}" jobs+=("$prev")
done
judge=$(cd "$REPO" && sbatch --parsable --account "$ACCOUNT" --dependency="afterany:$prev" \
    --output="$LOGS/judge-%j.out" forge/scripts/jupiter/sbatch_judge.slurm) ||
    { echo "[!!] sbatch of the judge failed"; exit 1; }
echo "[ok] $NAME: links ${jobs[*]}, then the judge ${judge%%;*}"
echo "     results: $FORGE_RUN_DIR/sft-$TAG/norm-exam.csv, $FORGE_RUN_DIR/logs/judge-${judge%%;*}.out"
