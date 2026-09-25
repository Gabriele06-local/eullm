#!/bin/bash
# Stage 3 — train the distilled student on domain instruction pairs + identity.
# LoRA on ONE GPU: a 4 B student in bf16 with an adapter fits one A100 with
# room to spare, and a one-card request is the easiest there is to place.
#
# Input is a MERGED distilled checkpoint, not a checkpoint-N directory. The
# curve job already produces one when asked to keep it:
#
#   sbatch --export=ALL,STEP=12600,CURVE_KEEP=1,\
#   CURVE_RUN_DIR=$WORK/eullm_runs/legal_it_v12,\
#   CURVE_CKPT_DIR=$WORK/eullm_runs/legal_it_v12/checkpoints/qwen3_4b_legal_it_distilled_split \
#       forge/scripts/leonardo/sbatch_transfer_curve.slurm
#
# STEP must be one of the split arm's last three checkpoints: it keeps only
# three, one every 300 steps. The job leaves the merged model at
# $CURVE_RUN_DIR/exports/merged/legal-it-4b-step<STEP> and measures that step
# on the way. Then:
#
#   sbatch --export=ALL,S3_BASE=<that dir> forge/scripts/leonardo/sbatch_stage3.slurm
#
# and, when it has finished, package it (serial partition, no GPU):
#
#   sbatch --export=ALL,S3_BASE=<that dir> forge/scripts/leonardo/sbatch_stage3_package.slurm
#
# Settable:
#   S3_BASE    merged distilled student (required)
#   S3_PAIRS   instruction pairs (default $WORK/eullm_runs/stage3/instruct-pairs-v2.jsonl)
#   S3_OUT     checkpoints + adapter (default $WORK/eullm_runs/stage3/sft-<basename of S3_BASE>)
#   S3_REPO    checkout to run (default: the one this file was submitted from)
#
# Resumable: submitting it again with the same S3_OUT continues from the last
# checkpoint, so a walltime kill costs at most --save-steps (100) steps.
#
#SBATCH --job-name=eullm-stage3
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
PAIRS="${S3_PAIRS:-$WORK/eullm_runs/stage3/instruct-pairs-v2.jsonl}"
OUT="${S3_OUT:-$WORK/eullm_runs/stage3/sft-$(basename "${BASE:-none}")}"

echo "[s3] repo   $REPO"
echo "[s3] commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[s3] base   $BASE"
echo "[s3] pairs  $PAIRS ($(wc -l < "$PAIRS" 2>/dev/null || echo 0) lines)"
echo "[s3] out    $OUT"

[ -n "$BASE" ] || { echo "[s3] S3_BASE is required" >&2; exit 1; }
[ -f "$BASE/config.json" ] || {
    echo "[s3] no merged model at $BASE — a checkpoint-N directory is an" >&2
    echo "[s3] adapter, not a model; produce the merged one with the curve" >&2
    echo "[s3] job and CURVE_KEEP=1 (see the header)" >&2
    exit 1
}
[ -s "$PAIRS" ] || { echo "[s3] no instruction pairs at $PAIRS" >&2; exit 1; }

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

python "$REPO/forge/scripts/stage3_sft.py" --model "$BASE" --pairs "$PAIRS" --out "$OUT"
