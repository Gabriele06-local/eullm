#!/bin/bash
# Step 1's acceptance test: recall@1/3/10 of the development rulings, per
# index setting (cds_retrieval.py), on one GPU.
#
#   CR_QUESTIONS=$WORK/eval/cds/dev-cards.jsonl \
#       sbatch forge/scripts/leonardo/sbatch_cds_retrieval.slurm
#
# The unit embeddings are cached in shards (CR_CACHE), so a link that runs
# out of its two hours leaves what it computed and the next one carries on:
# chain two or three links the first time, one after that.
#
# Settable:
#   CR_QUESTIONS  cards whose questions are asked (required: the development
#                 rulings' cards by a model other than the index's)
#   CR_SETTINGS   index settings (default "chunks prefix prefix+cards")
#   CR_CARDS      cards of the indexed rulings (default $WORK/eullm_runs/cds/schede*.jsonl,
#                 rejects excluded)
#   CR_EMBEDDER / CR_RERANKER  (default the released models' Qwen3-Embedding-0.6B and
#                 Qwen3-Reranker-0.6B; CR_EMBEDDER=none for BM25 alone)
#   CR_CHUNKS     chunk files (default $WORK/datasets/legal_it_amm/{train,val}.jsonl)
#   CR_CACHE      embedding cache (default $WORK/eval/cds/retrieval-cache)
#   CR_CSV        results (default $WORK/eval/cds/retrieval.csv); a setting already in
#                 it is skipped, so a later link measures only what is left
#   CR_LIMIT      questions asked, a fixed random sample (default 1500; 0: all ~8,000,
#                 more than two hours a setting with the reranker)
#   CR_REPO       checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-cds-retrieval
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
if [ -n "${CR_REPO:-}" ]; then
    REPO="$CR_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

# shellcheck disable=SC2206
CHUNKS=(${CR_CHUNKS:-$WORK/datasets/legal_it_amm/train.jsonl $WORK/datasets/legal_it_amm/val.jsonl})
CARDS=()
for f in ${CR_CARDS:-$WORK/eullm_runs/cds/schede*.jsonl}; do
    case "$f" in *rejects*) ;; *) [ -s "$f" ] && CARDS+=("$f") ;; esac
done
fail=0
[ -s "${CR_QUESTIONS:-}" ] || { echo "[cr] no questions at '${CR_QUESTIONS:-}'" >&2; fail=1; }
for f in "${CHUNKS[@]}"; do [ -s "$f" ] || { echo "[cr] missing $f" >&2; fail=1; }; done
[ "$fail" -eq 0 ] || exit 1

EMB=()
if [ "${CR_EMBEDDER:-}" != "none" ]; then
    EMB=(--embedder "${CR_EMBEDDER:-Qwen/Qwen3-Embedding-0.6B}"
         --reranker "${CR_RERANKER:-Qwen/Qwen3-Reranker-0.6B}"
         --cache-dir "${CR_CACHE:-$WORK/eval/cds/retrieval-cache}")
fi
CARD_ARGS=()
[ "${#CARDS[@]}" -eq 0 ] || CARD_ARGS=(--cards "${CARDS[@]}")
# shellcheck disable=SC2206
SETTINGS=(${CR_SETTINGS:-chunks prefix prefix+cards})

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

python "$REPO/forge/scripts/cds_retrieval.py" --chunks "${CHUNKS[@]}" \
    --openga $WORK/eval/openga/cds-sentenze-2017-2024.csv "${CARD_ARGS[@]}" \
    --questions "$CR_QUESTIONS" --dev-ids "$WORK/eval/cds/cds-dev-ids.txt" \
    --setting "${SETTINGS[@]}" "${EMB[@]}" --csv "${CR_CSV:-$WORK/eval/cds/retrieval.csv}" \
    --limit "${CR_LIMIT:-1500}"
