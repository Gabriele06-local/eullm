#!/bin/bash
# Stage 3, second half — merge the adapter, build the GGUF, measure it.
# All CPU work, so the serial partition, same resources as the curve job.
#
#   sbatch --export=ALL,S3_BASE=<merged distilled dir> \
#       forge/scripts/leonardo/sbatch_stage3_package.slurm
#
# The adapter is looked for where sbatch_stage3.slurm wrote it, from the same
# S3_BASE/S3_OUT. The result is a Q4_K_M GGUF named S3_NAME (default
# legal-it-4b-v0.1), measured against the untouched base on the Consiglio di
# Stato held-out corpus, like every point of the transfer curve.
#
# It also runs forge/scripts/chat_smoke.py on the merged weights: three
# questions, greedy, and whether each answer ENDS its turn. The verdict is the
# last "[smoke]" line of the log and a row of $WORK/eullm_runs/stage3/
# chat-smoke.csv, so "is it usable" no longer waits for someone to open a chat.
#
# That number answers one question: did instruction tuning DAMAGE what
# distillation taught? Perplexity on raw rulings is expected to move a little,
# since the model is now shaped to answer rather than continue; a large loss
# against the distilled step is the warning. Whether it answers WELL is a
# separate evaluation, done on the GGUF.
#
# Settable: S3_BASE (required), S3_OUT, S3_NAME, S3_REPO, CURVE_CORPUS,
# CURVE_BASE, CURVE_CHUNKS — the last three as in sbatch_transfer_curve.slurm.
#
#SBATCH --job-name=eullm-stage3-pkg
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${S3_REPO:-}" ]; then
    REPO="$S3_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

BASE="${S3_BASE:-}"
OUT="${S3_OUT:-$WORK/eullm_runs/stage3/sft-$(basename "${BASE:-none}")}"
NAME="${S3_NAME:-legal-it-4b-v0.1}"
ADAPTER="$OUT/adapter"
MERGED="$OUT/merged"
GGUF_DIR="$WORK/eullm_runs/stage3/gguf/$NAME"
Q4="$GGUF_DIR/${NAME}-q4_k_m.gguf"
CORPUS="${CURVE_CORPUS:-$WORK/eval/cds-corpus.txt}"
EVAL_BASE="${CURVE_BASE:-$WORK/gguf/qwen3-4b-base/qwen3-4b-base-q4_k_m.gguf}"
CHUNKS="${CURVE_CHUNKS:-120}"
CSV="$WORK/eullm_runs/stage3/perplexity-cds.csv"

echo "[s3p] repo    $REPO"
echo "[s3p] commit  $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[s3p] adapter $ADAPTER"
echo "[s3p] output  $Q4"

LCPP_BIN="${LCPP_DIR:-$WORK/llama.cpp}/build/bin"
fail=0
[ -n "$BASE" ] && [ -f "$BASE/config.json" ] || { echo "[s3p] no merged base at '$BASE'" >&2; fail=1; }
[ -f "$ADAPTER/adapter_config.json" ] || { echo "[s3p] no adapter at $ADAPTER — has sbatch_stage3 finished?" >&2; fail=1; }
[ -x "$LCPP_BIN/llama-quantize" ] || { echo "[s3p] llama-quantize not built in $LCPP_BIN" >&2; fail=1; }
[ -x "$LCPP_BIN/llama-perplexity" ] || { echo "[s3p] llama-perplexity not built in $LCPP_BIN" >&2; fail=1; }
[ -f "$CORPUS" ] || { echo "[s3p] no corpus at $CORPUS" >&2; fail=1; }
[ -f "$EVAL_BASE" ] || { echo "[s3p] no base GGUF at $EVAL_BASE" >&2; fail=1; }
[ "$fail" -eq 0 ] || exit 1

if [ ! -f "$Q4" ]; then
    echo "[s3p] === merging the adapter into the weights ==="
    PYTHONPATH="$REPO/forge:${PYTHONPATH:-}" \
        python - "$BASE" "$ADAPTER" "$MERGED" <<'EOF' || exit 1
import sys
from eullm_forge.identity import merge_identity_adapter
merge_identity_adapter(sys.argv[1], sys.argv[2], sys.argv[3])
EOF
    echo "[s3p] === converting and quantizing ==="
    GGUF_NAME="$NAME" LCPP_SKIP_UPDATE=1 \
        bash "$REPO/forge/scripts/quantize_to_gguf.sh" "$MERGED" "$GGUF_DIR" || exit 1
    rm -f "$GGUF_DIR"/*-f16.gguf
fi

echo "[s3p] === measuring against $(basename "$EVAL_BASE") ==="
bash "$REPO/forge/scripts/perplexity_compare.sh" \
    --base "$EVAL_BASE" --student "$Q4" --corpus "$CORPUS" \
    --chunks "$CHUNKS" --base-ppl-cache "$WORK/eval/.base_ppl_cds.cache" \
    --csv "$CSV" --label "$NAME-cds" || echo "[s3p] measurement failed — the GGUF is still good" >&2

# Perplexity cannot tell whether the model ends its turn — v0.1 scored like
# the checkpoint under it and never stopped. Ask the model itself, on the
# merged weights, and keep the answers in this log for a human to read.
if [ -f "$MERGED/config.json" ]; then
    echo "[s3p] === chat smoke test: does it answer, and does it stop? ==="
    PYTHONPATH="$REPO/forge:${PYTHONPATH:-}" \
        python "$REPO/forge/scripts/chat_smoke.py" "$MERGED" \
            --csv "$WORK/eullm_runs/stage3/chat-smoke.csv" --label "$NAME" ||
        echo "[s3p] smoke test failed to run — the GGUF is still good" >&2
fi

echo "[s3p] done: $Q4"
