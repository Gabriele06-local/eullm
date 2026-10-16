#!/bin/bash
# GRPO prompts of practical questions (make_practical_prompts.py, roadmap
# 1.2): one teacher llama-server per GPU, three GPUs, one two-hour job.
#
#   PP_TEACHER=$WORK/gguf/qwen3.6-27b/qwen3.6-27b-q8_0.gguf \
#       sbatch forge/scripts/leonardo/sbatch_practical_prompts.slurm
#
# Writes PP_OUT once, the practical prompts mixed with PP_MIX: a job that
# finds it exits at once. No article of an exam in $WORK/eval (the practical
# exam's groups included) is in a prompt. Retrieval is the released models'
# hybrid one, as the exams ask. Counts only.
#
# Settable:
#   PP_TEACHER   teacher GGUF (default Qwen3.6-27B Q8_0: not the judge's model)
#   PP_OUT       prompts file (default $WORK/eullm_runs/grpo/prompts-practical.jsonl)
#   PP_MIX       prompts files whose rows come first (default prompts-judge.jsonl)
#   PP_NORMS     legislation files (default $WORK/norms/legislazione_*.chunks.jsonl)
#   PP_ARGS      more make_practical_prompts.py options, e.g. "--per-code 200"
#   PP_REPO      checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-practical-prompts
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=340G
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

TEACHER="${PP_TEACHER:-$WORK/gguf/qwen3.6-27b/qwen3.6-27b-q8_0.gguf}"
OUT="${PP_OUT:-$WORK/eullm_runs/grpo/prompts-practical.jsonl}"
# shellcheck disable=SC2206
MIX=(${PP_MIX:-$WORK/eullm_runs/grpo/prompts-judge.jsonl})
# shellcheck disable=SC2206
NORMS=(${PP_NORMS:-$WORK/norms/legislazione_*.chunks.jsonl})
EXAMS=()
for f in "$WORK"/eval/practical-dev.jsonl "$WORK"/eval/practical-heldout.jsonl \
         "$WORK"/eval/norm-exam*.jsonl; do
    [ -s "$f" ] && EXAMS+=("$f")
done
SERVER="${LCPP_SERVER:-$LCPP_DIR/build-cuda/bin/llama-server}"
GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"

if [ -s "$OUT" ]; then
    echo "[pp] $OUT is written: nothing left to do"
    exit 0
fi
fail=0
[ -s "$TEACHER" ] || { echo "[pp] no teacher GGUF at $TEACHER" >&2; fail=1; }
[ -x "$SERVER" ] || { echo "[pp] no llama-server at $SERVER" >&2; fail=1; }
for f in "${NORMS[@]}" "${MIX[@]}"; do [ -s "$f" ] || { echo "[pp] missing $f" >&2; fail=1; }; done
[ -s "$WORK/eval/practical-dev.jsonl" ] ||
    { echo "[pp] no practical exam yet: its articles must be left out first" >&2; fail=1; }
[ "$GPUS" -ge 1 ] || { echo "[pp] no GPU visible" >&2; fail=1; }
[ "$fail" -eq 0 ] || exit 1
echo "[pp] ${#EXAMS[@]} exam file(s) whose articles are left out"

if [ -s "$(dirname "$SERVER")/../modules.txt" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2046
    module load $(cat "$(dirname "$SERVER")/../modules.txt") 2>/dev/null || true
fi
DEVS="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"
echo "[pp] repo $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')), GPUs $DEVS"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

# shellcheck disable=SC2086
python "$REPO/forge/scripts/make_practical_prompts.py" --norms "${NORMS[@]}" \
    --exclude-exam "${EXAMS[@]}" --mix "${MIX[@]}" \
    --embedder Qwen/Qwen3-Embedding-0.6B --reranker Qwen/Qwen3-Reranker-0.6B \
    --retrieval-cache "$WORK/eval/retrieval-cache" \
    --serve "$TEACHER" --gpus "$DEVS" --llama-server "$SERVER" --out "$OUT" ${PP_ARGS:-}
