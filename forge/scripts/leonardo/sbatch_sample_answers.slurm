#!/bin/bash
# Several sampled answers per verifiable question (sample_answers.py), from a
# GGUF through llama-server, on one GPU; then the correction exercises and
# DPO pairs made from them (make_correction_pairs.py).
#
#   SA_MODEL=$WORK/eullm_runs/grpo/ministral-v04/merged \
#   SA_GGUF=$WORK/eullm_runs/stage3/gguf/legal-it-8b-grpo-ministral-v04/legal-it-8b-grpo-ministral-v04-q4_k_m.gguf \
#   SA_PROMPTS=$WORK/eullm_runs/premise/grpo.jsonl SA_OUT=$WORK/eullm_runs/premise/samples-8b \
#       sbatch forge/scripts/leonardo/sbatch_sample_answers.slurm
#
# Writes $SA_OUT.jsonl (every answer with its score), $SA_OUT-informative.jsonl
# (the questions answered both right and wrong, for the next GRPO round),
# $SA_OUT-correzione.jsonl and $SA_OUT-dpo.jsonl. A finished sampling is
# kept: a second submission only remakes the pairs.
#
# Settable:
#   SA_MODEL     merged directory: the GGUF's tokenizer (required)
#   SA_GGUF      the GGUF answering (required)
#   SA_PROMPTS   GRPO prompt files, space-separated (required)
#   SA_OUT       output path without .jsonl (required)
#   SA_N         answers per question (default 8)
#   SA_LIMIT     at most this many questions (default all)
#   SA_SYSTEM    system prompt file (default: the template's own)
#   SA_REPO      checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-sample
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
if [ -n "${SA_REPO:-}" ]; then
    REPO="$SA_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

: "${SA_MODEL:?SA_MODEL: the merged directory}" "${SA_GGUF:?SA_GGUF: the GGUF}"
: "${SA_PROMPTS:?SA_PROMPTS: GRPO prompt files}" "${SA_OUT:?SA_OUT: output path}"
SERVER="${LCPP_SERVER:-$LCPP_DIR/build-cuda/bin/llama-server}"
fail=0
[ -f "$SA_MODEL/config.json" ] || { echo "[sa] no model at $SA_MODEL" >&2; fail=1; }
[ -f "$SA_GGUF" ] || { echo "[sa] no GGUF at $SA_GGUF" >&2; fail=1; }
[ -x "$SERVER" ] || { echo "[sa] no llama-server at $SERVER" >&2; fail=1; }
# shellcheck disable=SC2086
for f in $SA_PROMPTS; do [ -s "$f" ] || { echo "[sa] missing $f" >&2; fail=1; }; done
[ "$fail" -eq 0 ] || exit 1
# The CUDA build records the modules it was compiled with; its libraries
# come from them.
if [ -s "$(dirname "$SERVER")/../modules.txt" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2046
    module load $(cat "$(dirname "$SERVER")/../modules.txt") 2>/dev/null || true
fi
echo "[sa] repo $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"

if [ -s "$SA_OUT.jsonl" ]; then
    echo "[sa] $SA_OUT.jsonl exists, sampling skipped"
else
    SYS=()
    [ -z "${SA_SYSTEM:-}" ] || SYS=(--system-file "$SA_SYSTEM")
    # shellcheck disable=SC2086
    python "$REPO/forge/scripts/sample_answers.py" "$SA_MODEL" --gguf "$SA_GGUF" \
        --llama-server "$SERVER" --prompts $SA_PROMPTS --n "${SA_N:-8}" \
        --limit "${SA_LIMIT:-0}" --server-parallel 16 "${SYS[@]}" \
        --out "$SA_OUT.jsonl.partial" --informative-out "$SA_OUT-informative.jsonl" &&
        mv -f "$SA_OUT.jsonl.partial" "$SA_OUT.jsonl" || exit 1
fi
python "$REPO/forge/scripts/make_correction_pairs.py" "$SA_OUT.jsonl" \
    --out-sft "$SA_OUT-correzione.jsonl" --out-dpo "$SA_OUT-dpo.jsonl"
