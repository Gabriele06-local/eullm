#!/bin/bash
# Step 4's acceptance test: the development rulings' exam questions, answered
# with the case-law index (cds_answer.py), one model per job, on one GPU.
#
#   CE_LABEL=v04-4b CE_MODEL=$WORK/eullm_runs/grpo/merged-4b \
#   CE_GGUF=$WORK/gguf/legal-it-4b-Q4_K_M.gguf \
#       sbatch forge/scripts/leonardo/sbatch_cds_exam.slurm
#
# Writes $CE_DIR/answers-$CE_LABEL.jsonl; grade the directory afterwards
# with sbatch_judge.slurm (JUDGE_DIR=$CE_DIR) and compare pairs with
# compare_graded.py. The embeddings are the retrieval check's cache, so with
# the same cards no unit is embedded again.
#
# Settable:
#   CE_LABEL      name of the answers file (required)
#   CE_MODEL      merged model directory; with CE_GGUF only its tokenizer is used (required)
#   CE_GGUF       answer with this GGUF through llama-server instead of the bf16 weights
#   CE_QUESTIONS  cards whose exam questions are asked (default $WORK/eval/cds/dev-cards.jsonl,
#                 written by a model other than the index's)
#   CE_CARDS      cards of the indexed rulings (default $WORK/eullm_runs/cds/schede*.jsonl,
#                 rejects excluded)
#   CE_CHUNKS     chunk files (default $WORK/datasets/legal_it_amm/{train,val}.jsonl)
#   CE_DIR        answers directory (default $WORK/eval/cds-exam)
#   CE_ARGS       more options for cds_answer.py, e.g. "--limit 0"
#   CE_REPO       checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-cds-exam
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
if [ -n "${CE_REPO:-}" ]; then
    REPO="$CE_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

QUESTIONS="${CE_QUESTIONS:-$WORK/eval/cds/dev-cards.jsonl}"
DIR="${CE_DIR:-$WORK/eval/cds-exam}"
# shellcheck disable=SC2206
CHUNKS=(${CE_CHUNKS:-$WORK/datasets/legal_it_amm/train.jsonl $WORK/datasets/legal_it_amm/val.jsonl})
CARDS=()
for f in ${CE_CARDS:-$WORK/eullm_runs/cds/schede*.jsonl}; do
    case "$f" in *rejects*) ;; *) [ -s "$f" ] && CARDS+=("$f") ;; esac
done
SERVER="${LCPP_SERVER:-$LCPP_DIR/build-cuda/bin/llama-server}"
fail=0
[ -n "${CE_LABEL:-}" ] || { echo "[ce] CE_LABEL is required" >&2; fail=1; }
[ -d "${CE_MODEL:-}" ] || { echo "[ce] no model directory at '${CE_MODEL:-}'" >&2; fail=1; }
[ -s "$QUESTIONS" ] || { echo "[ce] no questions at $QUESTIONS" >&2; fail=1; }
for f in "${CHUNKS[@]}"; do [ -s "$f" ] || { echo "[ce] missing $f" >&2; fail=1; }; done
GGUF_ARGS=()
if [ -n "${CE_GGUF:-}" ]; then
    [ -f "$CE_GGUF" ] || { echo "[ce] no GGUF at $CE_GGUF" >&2; fail=1; }
    [ -x "$SERVER" ] || { echo "[ce] no llama-server at $SERVER -- sbatch_build_llama_server.slurm" >&2; fail=1; }
    GGUF_ARGS=(--gguf "$CE_GGUF" --llama-server "$SERVER")
fi
[ "$fail" -eq 0 ] || exit 1
# The CUDA build records the modules it was compiled with; its libraries
# come from them.
if [ -n "${CE_GGUF:-}" ] && [ -s "$(dirname "$SERVER")/../modules.txt" ] &&
   command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2046
    module load $(cat "$(dirname "$SERVER")/../modules.txt") 2>/dev/null || true
fi
CARD_ARGS=()
[ "${#CARDS[@]}" -eq 0 ] || CARD_ARGS=(--cards "${CARDS[@]}")
mkdir -p "$DIR"
echo "[ce] repo   $REPO"
echo "[ce] commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[ce] model  $CE_LABEL"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

# shellcheck disable=SC2086
python "$REPO/forge/scripts/cds_answer.py" "$CE_MODEL" --label "$CE_LABEL" \
    --questions "$QUESTIONS" --dev-ids "$WORK/eval/cds/cds-dev-ids.txt" \
    --chunks "${CHUNKS[@]}" --openga "$WORK/eval/openga/cds-sentenze-2017-2024.csv" \
    "${CARD_ARGS[@]}" --embedder Qwen/Qwen3-Embedding-0.6B --reranker Qwen/Qwen3-Reranker-0.6B \
    --cache-dir "$WORK/eval/cds/retrieval-cache" "${GGUF_ARGS[@]}" \
    --answers "$DIR/answers-$CE_LABEL.jsonl" ${CE_ARGS:-}
