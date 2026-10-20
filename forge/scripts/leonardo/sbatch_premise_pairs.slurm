#!/bin/bash
# Questions on a false premise (make_premise_pairs.py), retrieved the way the
# released models retrieve, on one GPU: SFT pairs in both answer formats and
# GRPO prompts, on different articles.
#
#   sbatch forge/scripts/leonardo/sbatch_premise_pairs.slurm
#
# Writes, in PP_DIR (default $WORK/eullm_runs/premise):
#   sft-steps.jsonl   SFT pairs answered in three checked steps
#   sft-plain.jsonl   the same pairs answered in one paragraph (the ablation)
#   grpo.jsonl        GRPO prompts on the other articles
# Files already written are kept: a second submission exits at once.
#
# No article of the PP_EXCLUDE sets is used: the exams, the practical exam
# and the LexIA traps stay tests.
#
# Settable:
#   PP_DIR       output directory
#   PP_EXCLUDE   sets whose articles are never used (default every
#                norm-exam-*.jsonl and practical-*.jsonl in $WORK/eval, and
#                $WORK/eval/lexia/lexia-trappole.jsonl)
#   PP_PER_CODE  pairs of each kind per code (default 80)
#   PP_SEED      draw (default 0)
#   PP_NORMS     legislation files (default $WORK/norms/legislazione_*.chunks.jsonl)
#   PP_CACHE     embedding cache (default $WORK/eval/retrieval-cache)
#   PP_REPO      checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-premise
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${PP_REPO:-}" ]; then
    REPO="$PP_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

DIR="${PP_DIR:-$WORK/eullm_runs/premise}"
if [ -s "$DIR/sft-steps.jsonl" ] && [ -s "$DIR/sft-plain.jsonl" ] && [ -s "$DIR/grpo.jsonl" ]; then
    echo "[pp] $DIR holds all three files, kept"
    exit 0
fi
# shellcheck disable=SC2206
NORMS=(${PP_NORMS:-$WORK/norms/legislazione_*.chunks.jsonl})
# shellcheck disable=SC2206
EXCLUDE=(${PP_EXCLUDE:-$WORK/eval/norm-exam-*.jsonl $WORK/eval/practical-*.jsonl $WORK/eval/lexia/lexia-trappole.jsonl})
fail=0
for f in "${NORMS[@]}" "${EXCLUDE[@]}"; do
    [ -s "$f" ] || { echo "[pp] missing $f" >&2; fail=1; }
done
[ "$fail" -eq 0 ] || exit 1
mkdir -p "$DIR"

echo "[pp] repo $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "[pp] excluding: ${EXCLUDE[*]}"
ARGS=(--norms "${NORMS[@]}" --exclude-exam "${EXCLUDE[@]}" --per-code "${PP_PER_CODE:-80}"
      --seed "${PP_SEED:-0}" --embedder Qwen/Qwen3-Embedding-0.6B
      --reranker Qwen/Qwen3-Reranker-0.6B
      --retrieval-cache "${PP_CACHE:-$WORK/eval/retrieval-cache}")
# The same seed gives the same pairs: the plain run writes the SFT pairs of
# the steps run in the other format.
python "$REPO/forge/scripts/make_premise_pairs.py" "${ARGS[@]}" --format steps \
    --out-sft "$DIR/sft-steps.jsonl" --out-grpo "$DIR/grpo.jsonl" &&
python "$REPO/forge/scripts/make_premise_pairs.py" "${ARGS[@]}" --format plain \
    --out-sft "$DIR/sft-plain.jsonl"
