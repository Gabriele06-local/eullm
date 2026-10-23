#!/bin/bash
# The context gate, one step per job, on one GPU; submit_gate.sh chains them.
#
#   GATE_STEP=offtopic|pairs|train sbatch forge/scripts/leonardo/sbatch_gate.slurm
#
#   offtopic  the teacher (Qwen3-30B-A3B-Instruct, served by llama-server)
#             writes questions on matters the collection does not cover
#             (make_offtopic_questions.py) -> $GATE_DIR/offtopic.jsonl
#   pairs     labelled examples, texts retrieved as the released models
#             retrieve them (make_gate_pairs.py) -> train.jsonl, dev.jsonl
#   train     Qwen3-Reranker-0.6B fine-tuned into the gate
#             (train_context_gate.py) -> $GATE_DIR/model with gate.json
#
# Each step keeps what is already written and exits at once, so the chain
# can be submitted again. Logs hold counts and scores, never a question.
#
# Settable:
#   GATE_STEP      the step (required)
#   GATE_DIR       outputs (default $WORK/eullm_runs/gate)
#   GATE_OPENBOOK  open-book SFT pairs (pairs; default openbook-v04.jsonl)
#   GATE_PREMISE   false-premise GRPO prompts (pairs; default premise5/grpo.jsonl)
#   GATE_PRACTICAL practical-exam GRPO prompts (pairs; default
#                  grpo/prompts-practical-abst-nn.jsonl, when it exists)
#   GATE_TEACHER_GGUF  the teacher (offtopic; default the judge's Q8_0)
#   GATE_REPO      checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-gate
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
if [ -n "${GATE_REPO:-}" ]; then
    REPO="$GATE_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

S="$REPO/forge/scripts"
GD="${GATE_DIR:-$WORK/eullm_runs/gate}"
mkdir -p "$GD"
echo "[gate] step ${GATE_STEP:-?}, repo $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"

case "${GATE_STEP:-}" in
offtopic)
    [ -s "$GD/offtopic.jsonl" ] && { echo "[gate] $GD/offtopic.jsonl exists, kept"; exit 0; }
    TEACHER="$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/* 2>/dev/null | head -1)"
    TG="${GATE_TEACHER_GGUF:-$WORK/gguf/qwen3-30b-a3b-instruct-2507/qwen3-30b-a3b-instruct-2507-q8_0.gguf}"
    SERVER="${LCPP_SERVER:-$LCPP_DIR/build-cuda/bin/llama-server}"
    for f in "$TEACHER/config.json" "$TG" "$SERVER"; do
        [ -e "$f" ] || { echo "[gate] missing $f" >&2; exit 1; }
    done
    if [ -s "$(dirname "$SERVER")/../modules.txt" ] && command -v module >/dev/null 2>&1; then
        # shellcheck disable=SC2046
        module load $(cat "$(dirname "$SERVER")/../modules.txt") 2>/dev/null || true
    fi
    python "$S/make_offtopic_questions.py" "$TEACHER" --gguf "$TG" --llama-server "$SERVER" \
        --out "$GD/offtopic.jsonl"
    ;;
pairs)
    if [ -s "$GD/train.jsonl" ] && [ -s "$GD/dev.jsonl" ]; then
        echo "[gate] $GD/train.jsonl and dev.jsonl exist, kept"; exit 0
    fi
    OB="${GATE_OPENBOOK:-$WORK/eullm_runs/stage3/openbook-v04.jsonl}"
    PM="${GATE_PREMISE:-$WORK/eullm_runs/premise5/grpo.jsonl}"
    for f in "$OB" "$PM" "$GD/offtopic.jsonl"; do
        [ -s "$f" ] || { echo "[gate] missing $f" >&2; exit 1; }
    done
    # shellcheck disable=SC2206
    EXCLUDE=($WORK/eval/norm-exam-*.jsonl $WORK/eval/practical-*.jsonl $WORK/eval/lexia/lexia-trappole.jsonl)
    PR="${GATE_PRACTICAL:-$WORK/eullm_runs/grpo/prompts-practical-abst-nn.jsonl}"
    PRACTICAL=()
    [ -s "$PR" ] && PRACTICAL=(--practical "$PR")
    python "$S/make_gate_pairs.py" --norms "$WORK"/norms/legislazione_*.chunks.jsonl \
        --openbook "$OB" --premise "$PM" "${PRACTICAL[@]}" --offtopic "$GD/offtopic.jsonl" \
        --exclude-exam "${EXCLUDE[@]}" --embedder Qwen/Qwen3-Embedding-0.6B \
        --reranker Qwen/Qwen3-Reranker-0.6B --retrieval-cache "$WORK/eval/retrieval-cache" \
        --out-train "$GD/train.jsonl" --out-dev "$GD/dev.jsonl"
    ;;
train)
    python "$S/train_context_gate.py" --train "$GD/train.jsonl" --dev "$GD/dev.jsonl" \
        --out "$GD/model"
    ;;
*)
    echo "[gate] GATE_STEP must be offtopic, pairs or train" >&2; exit 2 ;;
esac
