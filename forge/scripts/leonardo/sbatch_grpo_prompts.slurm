#!/bin/bash
# GRPO prompts with the texts retrieved the way the released models retrieve
# them (make_grpo_prompts.py --embedder/--reranker), on one GPU.
#
#   GP_OUT=$WORK/eullm_runs/grpo/prompts-hyb06.jsonl \
#       sbatch forge/scripts/leonardo/sbatch_grpo_prompts.slurm
#
# The embedding model and the reranker are what need the GPU; the document
# embeddings are read from the exam's cache (GP_CACHE), so only the questions
# are embedded here. A file already written is kept: a second submission
# exits at once, and a GRPO chain behind it with afterok starts on the same
# prompts.
#
# No article of the GP_EXCLUDE exams is drawn: the held-out exams and the
# development set stay unseen by training.
#
# Settable:
#   GP_OUT       prompts file (required)
#   GP_EMBEDDER  embedding model (default Qwen/Qwen3-Embedding-0.6B)
#   GP_RERANKER  reranker (default Qwen/Qwen3-Reranker-0.6B; "none" for none)
#   GP_EXCLUDE   exams whose articles are never drawn (default v3, v4 and
#                every norm-exam-dev*.jsonl in $WORK/eval)
#   GP_SEED      draw (default 1: the first round's prompts were seed 0)
#   GP_PER_CODE  questions of each kind per code (default 400)
#   GP_NORMS     legislation files (default $WORK/norms/legislazione_*.chunks.jsonl)
#   GP_CACHE     embedding cache (default $WORK/eval/retrieval-cache)
#   GP_REPO      checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-grpo-prompts
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
if [ -n "${GP_REPO:-}" ]; then
    REPO="$GP_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

OUT="${GP_OUT:-}"
[ -n "$OUT" ] || { echo "[gp] GP_OUT is required" >&2; exit 1; }
if [ -s "$OUT" ]; then
    echo "[gp] $OUT exists ($(grep -c . "$OUT") prompts), kept"
    exit 0
fi
# shellcheck disable=SC2206
NORMS=(${GP_NORMS:-$WORK/norms/legislazione_*.chunks.jsonl})
# shellcheck disable=SC2206
EXCLUDE=(${GP_EXCLUDE:-$WORK/eval/norm-exam-v3.jsonl $WORK/eval/norm-exam-v4.jsonl $WORK/eval/norm-exam-dev*.jsonl})
fail=0
for f in "${NORMS[@]}" "${EXCLUDE[@]}"; do
    [ -s "$f" ] || { echo "[gp] missing $f" >&2; fail=1; }
done
[ "$fail" -eq 0 ] || exit 1

RR=()
[ "${GP_RERANKER:-}" = "none" ] || RR=(--reranker "${GP_RERANKER:-Qwen/Qwen3-Reranker-0.6B}")
echo "[gp] repo $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "[gp] excluding: ${EXCLUDE[*]}"
# Written beside the target and moved into place: an interrupted job leaves
# no half file for the chain behind it to train on.
python "$REPO/forge/scripts/make_grpo_prompts.py" --norms "${NORMS[@]}" \
    --exclude-exam "${EXCLUDE[@]}" --per-code "${GP_PER_CODE:-400}" --seed "${GP_SEED:-1}" \
    --embedder "${GP_EMBEDDER:-Qwen/Qwen3-Embedding-0.6B}" "${RR[@]}" \
    --retrieval-cache "${GP_CACHE:-$WORK/eval/retrieval-cache}" --out "$OUT.partial" &&
    mv -f "$OUT.partial" "$OUT"
