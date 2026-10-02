#!/bin/bash
# How often each retriever finds the question's own article (check_retrieval.py).
#
#   RET_ITEMS=$WORK/eval/norm-exam-devbig.jsonl RET_EMBEDDER=Qwen/Qwen3-Embedding-8B \
#   RET_RERANKER=Qwen/Qwen3-Reranker-4B \
#       sbatch forge/scripts/leonardo/sbatch_retrieval.slurm
#
# Measures BM25 (the retriever every exam so far used), the embedding model
# alone, the two fused, and the fused list reordered by the reranker, on a
# DEVELOPMENT set: tuning retrieval on the held-out exam would make it one
# more development set. Counts only, never a question.
#
# One embedding model per job: the document embeddings of the whole
# legislation are the long part, and are cached in RET_CACHE, so a second job
# (or the exam that then uses the index) does not compute them again. Two
# models are two jobs, side by side on separate GPUs.
#
# The models must be in $HF_HOME already (compute nodes are offline):
#   python forge/scripts/leonardo/prefetch_models.py <ids>   (login node)
#
# Settable:
#   RET_ITEMS     development set(s), space-separated (required)
#   RET_EMBEDDER  embedding model (required), e.g. Qwen/Qwen3-Embedding-8B
#   RET_RERANKER  reranker model (optional), e.g. Qwen/Qwen3-Reranker-4B
#   RET_PAIRS     open-book pairs: also measure on the teacher's topic
#                 questions, which paraphrase instead of quoting the rubrica
#   RET_PAIRS_LIMIT  how many of those (default 500)
#   RET_NORMS     legislation files (default $WORK/norms/legislazione_*.chunks.jsonl)
#   RET_CACHE     embedding cache (default $WORK/eval/retrieval-cache)
#   RET_CSV       results (default $WORK/eval/retrieval.csv)
#   RET_REPO      checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-retrieval
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
if [ -n "${RET_REPO:-}" ]; then
    REPO="$RET_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

# shellcheck disable=SC2206
NORMS=(${RET_NORMS:-$WORK/norms/legislazione_*.chunks.jsonl})
CACHE="${RET_CACHE:-$WORK/eval/retrieval-cache}"
CSV="${RET_CSV:-$WORK/eval/retrieval.csv}"
echo "[ret] repo     $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "[ret] items    ${RET_ITEMS:-}"
echo "[ret] embedder ${RET_EMBEDDER:-}"
echo "[ret] reranker ${RET_RERANKER:-none}"

fail=0
[ -n "${RET_ITEMS:-}" ] || { echo "[ret] RET_ITEMS is required" >&2; fail=1; }
for f in ${RET_ITEMS:-}; do [ -s "$f" ] || { echo "[ret] no items at $f" >&2; fail=1; }; done
case "${RET_ITEMS:-}" in *dev*) ;; *) echo "[ret] RET_ITEMS must be development sets (*dev*)" >&2; fail=1 ;; esac
[ -n "${RET_EMBEDDER:-}" ] || { echo "[ret] RET_EMBEDDER is required" >&2; fail=1; }
[ -f "${NORMS[0]}" ] || { echo "[ret] no legislation files" >&2; fail=1; }
[ "$fail" -eq 0 ] || exit 1
mkdir -p "$CACHE"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

ARGS=(--norms "${NORMS[@]}" --boost 0 3 --embedder "$RET_EMBEDDER" --cache-dir "$CACHE"
      --csv "$CSV")
[ -z "${RET_RERANKER:-}" ] || ARGS+=(--reranker "$RET_RERANKER")
# shellcheck disable=SC2206
[ -z "${RET_PAIRS:-}" ] || ARGS+=(--pairs $RET_PAIRS --pairs-limit "${RET_PAIRS_LIMIT:-500}")
# shellcheck disable=SC2086
python "$REPO/forge/scripts/check_retrieval.py" $RET_ITEMS "${ARGS[@]}"
