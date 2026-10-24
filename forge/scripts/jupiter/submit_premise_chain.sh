#!/usr/bin/env bash
# The false-premise round, unattended, from a login node:
#
#   1. the false-premise pairs (sbatch_premise_pairs.slurm);
#   2. instruction tuning on the open-book pairs with them, Leonardo's
#      open-book recipe (submit_sft_chain.sh), the held-out exam open book
#      with and without the note, then its judge;
#   3. the private traps and practical exam asked of each arm
#      (sbatch_teacher_eval.slurm), then their judge on JUPITER
#      (sbatch_judge.slurm with JUDGE_ANSWERS_DIR).
#
#     eullm && bash forge/scripts/jupiter/submit_premise_chain.sh
#
# The arms default to the Qwen3.5 4B and 9B distilled on norms-v2, the
# 9B first on the practical exam and last on the traps (dossier §5.11).
# PREMISE_TAG names the round (default dqp), PREMISE_ARMS its arms
# ("name=merged dir ..."). Each step waits for the one before (afterany):
# a step whose input is missing stops at its own check.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
# shellcheck source=sbatch_retry.sh
source "$HERE/sbatch_retry.sh"
REPO="$(cd "$HERE/../../.." && pwd)"
R="$FORGE_RUN_DIR" D="$FORGE_DATA_DIR" LOGS="$FORGE_RUN_DIR/logs"
P="${FORGE_PRIVATE_EVAL:-$FORGE_HOME/private-eval}"
TAG="${PREMISE_TAG:-dqp}"
ARMS="${PREMISE_ARMS:-d-q4b=$R/q35v2-4b/checkpoints/qwen35_4b_normsv2_distilled/merged d-q9b=$R/q35v2-9b/checkpoints/qwen35_9b_normsv2_distilled/merged}"
LIBS="${FORGE_EXTRA_LIBS:-$FORGE_HOME/libs-qwen35}"
PREMISES="$D/premise/sft-steps.jsonl"
bad=0
for f in "$D/openbook/pairs.jsonl" "$P/lexia-trappole.jsonl" "$P/practical-dev.jsonl"; do
    [ -s "$f" ] || { echo "[!!] missing $f"; bad=1; }
done
[ -d "$LIBS" ] || { echo "[!!] no $LIBS: setup_qwen35_libs.sh"; bad=1; }
[ "$bad" = 0 ] || { echo "[!!] nothing submitted"; exit 1; }
wait_for_slurm || exit 1
mkdir -p "$LOGS"
cd "$REPO" || exit 1

pp=$(sbatch_retry --parsable --account "$SBATCH_ACCOUNT" \
    --output="$LOGS/premise-%j.out" forge/scripts/jupiter/sbatch_premise_pairs.slurm) ||
    { echo "[!!] sbatch of the premise pairs failed"; exit 1; }
pp="${pp%%;*}"
echo "[ok] premise pairs: job $pp"

N="$(ls "$D"/norms-src/legislazione_*.chunks.jsonl | grep -v eurlex | tr '\n' ' ')"
FORGE_CHAIN_AFTER="$pp" FORGE_LINK_H="${FORGE_LINK_H:-6}" FORGE_EXTRA_LIBS="$LIBS" \
FORGE_SFT_NORMS="$N" FORGE_SFT_TAG="$TAG" \
FORGE_SFT_PAIRS="$D/openbook/pairs.jsonl $PREMISES" \
FORGE_SFT_ARGS="--epochs 1 --lr 1e-4 --max-length 4096 --batch-size 1 --grad-accum 16" \
FORGE_SFT_NO_NOTE=1 FORGE_SFT_ARMS="$ARMS" \
    bash forge/scripts/jupiter/submit_sft_chain.sh 1 || { echo "[!!] the instruction-tuning chain was not submitted"; exit 1; }
link=$(squeue -h -u "$USER" -n "forge-sft-$TAG" -o %i | sort -n | tail -1)
[ -n "$link" ] || { echo "[!!] no forge-sft-$TAG link in the queue"; exit 1; }

ids=""
for arm in $ARMS; do
    name="${arm%%=*}"
    for set in lexia-trappole practical-dev; do
        j=$(export EVAL_MODEL="$R/sft-$TAG/$name/merged" EVAL_LABEL="$TAG-$name" \
                   EVAL_ITEMS="$P/$set.jsonl" EVAL_SET="$set" FORGE_EXTRA_LIBS="$LIBS"
            sbatch_retry --parsable --account "$SBATCH_ACCOUNT" --time=01:00:00 \
            --dependency="afterany:$link" --output="$LOGS/teacher-eval-%j.out" \
            forge/scripts/jupiter/sbatch_teacher_eval.slurm) || { echo "[!!] sbatch of $name on $set failed"; exit 1; }
        ids="$ids:${j%%;*}"
        echo "[ok] $name on $set: job ${j%%;*}"
    done
done
j=$(export JUDGE_ANSWERS_DIR="$R/teacher-eval"; sbatch_retry --parsable --account "$SBATCH_ACCOUNT" \
    --dependency="afterany$ids" --output="$LOGS/judge-%j.out" forge/scripts/jupiter/sbatch_judge.slurm) ||
    { echo "[!!] sbatch of the private judge failed"; exit 1; }
echo "[ok] the traps and the practical exam graded by job ${j%%;*}"
echo "     results: $R/sft-$TAG/norm-exam.csv and $R/teacher-eval/graded-v3/graded-gpu*.csv"
