#!/bin/bash
# One checkpoint of the transfer curve: export, quantize, measure, clean up.
#
# The question this answers is whether the quality gain is still growing, and
# a single number cannot answer it. At step 28,000 of 70,457 — 40 % of one
# epoch, with the cosine schedule still at two thirds of peak learning rate —
# the student scored 23 % below its base on Consiglio di Stato text, a court
# its training never touched. Good, but good compared to what? The checkpoints
# from 1,000 to 28,000 are all still on disk, so the curve can be measured
# rather than argued about: if the gap is still opening at 28,000, finishing
# the epoch pays and by how much; if it flattened at 21,000, the remaining
# weeks belong somewhere else. That is worth knowing while there is still
# allocation left to act on it.
#
# ON THE SERIAL PARTITION. Merging is elementwise arithmetic, GGUF conversion
# and llama-quantize are CPU-only, and perplexity runs on CPU. None of it
# needs a GPU and none of it touches the training chain's allocation.
#
# One job per checkpoint, submitted once each, rather than one job for all of
# them: each takes an hour and a half against a four-hour walltime, and a
# failure costs one point of the curve instead of the whole run.
#
#   sbatch --export=ALL,STEP=7000  forge/scripts/leonardo/sbatch_transfer_curve.slurm
#   sbatch --export=ALL,STEP=14000 forge/scripts/leonardo/sbatch_transfer_curve.slurm
#   sbatch --export=ALL,STEP=21000 forge/scripts/leonardo/sbatch_transfer_curve.slurm
#
# Settable, all with defaults that match the measurement already taken so the
# points are comparable with it:
#
#   STEP            checkpoint to package (required)
#   CURVE_RUN_DIR   run directory holding checkpoints/ and exports/
#   CURVE_CKPT_DIR  where the checkpoint-N directories are
#   CURVE_CORPUS    evaluation text
#   CURVE_BASE      base GGUF to compare against
#   CURVE_CHUNKS    perplexity chunks (default 120)
#   CURVE_KEEP      1 to keep the merged dir and F16 (default 0, delete both)
#
#SBATCH --job-name=eullm-curve
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Matched to the other serial jobs. lrd_all_serial enforces
# QOSMaxMemoryPerUser and rejected 64G outright; 30G is known to pass and is
# ample for a 4 B model in bf16 plus the copy save_pretrained writes.
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

if [ -z "${STEP:-}" ]; then
    echo "[curve] STEP is required: sbatch --export=ALL,STEP=14000 $0" >&2
    exit 1
fi

SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${EULLM_REPO:-}" ]; then
    REPO="$EULLM_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    echo "[curve] cannot locate the repository — set EULLM_REPO" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

RUN_DIR="${CURVE_RUN_DIR:-/leonardo_work/AIFAC_P02_1147/eullm_runs/legal_it}"
CKPT_DIR="${CURVE_CKPT_DIR:-$RUN_DIR/checkpoints/qwen3_4b_legal_it_distilled}"
CORPUS="${CURVE_CORPUS:-$WORK/eval/cds-corpus.txt}"
BASE="${CURVE_BASE:-$WORK/gguf/qwen3-4b-base/qwen3-4b-base-q4_k_m.gguf}"
CHUNKS="${CURVE_CHUNKS:-120}"
KEEP="${CURVE_KEEP:-0}"

NAME="legal-it-4b-step${STEP}"
CKPT="$CKPT_DIR/checkpoint-${STEP}"
MERGED="$RUN_DIR/exports/merged/$NAME"
GGUF_DIR="$RUN_DIR/exports/gguf/$NAME"
Q4="$GGUF_DIR/${NAME}-q4_k_m.gguf"
CSV="${CURVE_CSV:-$RUN_DIR/exports/perplexity-cds.csv}"
CACHE="${CURVE_CACHE:-$WORK/eval/.base_ppl_cds.cache}"

echo "[curve] repo    $REPO"
echo "[curve] commit  $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[curve] step    $STEP"
echo "[curve] corpus  $CORPUS"

# Check every input before doing an hour of work on a missing one — the same
# lesson as the quantizer pre-flight in sbatch_export_gguf.slurm, which was
# added after a ten-minute merge died at conversion for want of a binary.
#
# The checkpoint is required ONLY when the GGUF still has to be built. A
# point whose Q4 is already on disk needs nothing but the Q4, the corpus and
# the base — and it is the common case, not the exception: the training
# configs keep `save_total_limit: 3`, so by the time anyone measures step
# 7,000 the trainer has long since rotated `checkpoint-7000` away. Checking
# for it unconditionally killed five measurements on 2026-09-24 in their
# pre-flight, each for a directory it had no use for, while the GGUFs they
# were meant to measure sat on disk next to them.
#
# Each binary is asked for by the path that uses it: llama-quantize only when
# building, llama-perplexity always, since measuring is the one thing every
# run of this job does.
LCPP_BIN="${LCPP_DIR:-$WORK/llama.cpp}/build/bin"
fail=0
if [ ! -f "$Q4" ]; then
    [ -d "$CKPT" ] || {
        echo "[curve] no checkpoint at $CKPT, and no GGUF at $Q4 to measure" >&2
        echo "[curve] instead — the trainer keeps only the last few checkpoints," >&2
        echo "[curve] so a step this old can only be measured if it was packaged" >&2
        echo "[curve] while its checkpoint still existed" >&2
        fail=1
    }
    [ -x "$LCPP_BIN/llama-quantize" ] || {
        echo "[curve] llama-quantize not built in $LCPP_BIN —" >&2
        echo "[curve] build llama.cpp from a LOGIN node (it needs network)" >&2
        fail=1
    }
fi
[ -x "$LCPP_BIN/llama-perplexity" ] || {
    echo "[curve] llama-perplexity not built in $LCPP_BIN" >&2
    fail=1
}
[ -f "$CORPUS" ]  || { echo "[curve] no corpus at $CORPUS" >&2; fail=1; }
[ -f "$BASE" ]    || { echo "[curve] no base GGUF at $BASE" >&2; fail=1; }
[ "$fail" -eq 0 ] || exit 1

# Resume rather than redo: a rerun after a walltime kill should cost only
# what is missing, and the pieces are expensive enough to be worth skipping.
if [ -f "$Q4" ]; then
    echo "[curve] $Q4 already exists — skipping export and quantization"
else
    echo "[curve] === merging checkpoint-$STEP ==="
    if ! python "$REPO/forge/scripts/export_checkpoint.py" \
            --checkpoint "$CKPT" --output "$MERGED" --force; then
        echo "[curve] merge FAILED for step $STEP" >&2
        exit 1
    fi

    echo "[curve] === converting and quantizing ==="
    # LCPP_SKIP_UPDATE, because the points of a curve have to be measured with
    # the same quantizer: a git pull here would change it mid-series.
    if ! GGUF_NAME="$NAME" LCPP_SKIP_UPDATE=1 \
            bash "$REPO/forge/scripts/quantize_to_gguf.sh" "$MERGED" "$GGUF_DIR"; then
        echo "[curve] GGUF conversion FAILED for step $STEP" >&2
        exit 1
    fi
fi

echo "[curve] === measuring against $(basename "$BASE") ==="
if ! bash "$REPO/forge/scripts/perplexity_compare.sh" \
        --base "$BASE" --student "$Q4" --corpus "$CORPUS" \
        --chunks "$CHUNKS" --base-ppl-cache "$CACHE" \
        --csv "$CSV" --label "step-${STEP}-cds"; then
    echo "[curve] measurement FAILED for step $STEP — the GGUF is still good" >&2
    exit 1
fi

# The Q4 is the artefact worth keeping; the other two are build intermediates
# of 7.5 GB and 8 GB that regenerate in minutes. Four points of this curve
# would otherwise be 60 GB of a shared $WORK.
if [ "$KEEP" != "1" ]; then
    rm -f "$GGUF_DIR"/*-f16.gguf && echo "[curve] dropped the F16 intermediate"
    rm -rf "$MERGED" && echo "[curve] dropped the merged HF directory"
fi

echo
echo "[curve] the curve so far ($CSV):"
awk -F, 'NR>1 {printf "[curve]   %-18s PPL %-9s  %+.2f%% vs base\n", $2, $10, $11}' \
    "$CSV" 2>/dev/null | tail -10
