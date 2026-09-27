#!/bin/bash
# The held-out exam (make_norm_exam.py) put to several models, on one GPU.
#
#   sbatch --export=ALL,EXAM_MODELS="v0.2=/path/merged qwen3-4b-instruct=/path/snapshot" \
#       forge/scripts/leonardo/sbatch_norm_exam.slurm
#
# One answers file per model in EXAM_OUT, then grade them with
#   sbatch --dependency=afterok:<this job> \
#       --export=ALL,JUDGE_DIR=$EXAM_OUT,JUDGE_QUIET=1 sbatch_judge.slurm
#
# QUIET THROUGHOUT. legal_eval.py runs with --quiet, so this log holds totals
# and never a question: the exam is only held out as long as nobody who
# improves the models reads it, and logs get pasted into conversations.
#
# One GPU, like stage-3 training: a 4B model in bf16 answers the whole exam,
# 16 questions at a time, in minutes, where the CPU took an hour for ten.
#
# Settable:
#   EXAM_MODELS  space-separated label=path pairs (required)
#   EXAM_ITEMS   the exam (default $WORK/eval/norm-exam.jsonl)
#   EXAM_OUT     answers directory (default $WORK/eval/norm-exam-answers)
#   EXAM_NORMS   legislation files: set to ALSO ask each model open book,
#                as "<label>-open" (default: closed book only)
#   EXAM_REPO    checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-exam
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
if [ -n "${EXAM_REPO:-}" ]; then
    REPO="$EXAM_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

ITEMS="${EXAM_ITEMS:-$WORK/eval/norm-exam.jsonl}"
OUT="${EXAM_OUT:-$WORK/eval/norm-exam-answers}"
echo "[exam] repo   $REPO"
echo "[exam] commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[exam] items  $ITEMS ($(wc -l < "$ITEMS" 2>/dev/null || echo 0) lines)"
echo "[exam] out    $OUT"

[ -s "$ITEMS" ] || { echo "[exam] no exam at $ITEMS — run make_norm_exam.py" >&2; exit 1; }
[ -n "${EXAM_MODELS:-}" ] || { echo "[exam] EXAM_MODELS is required" >&2; exit 1; }
fail=0
for pair in $EXAM_MODELS; do
    path="${pair#*=}"
    [ -f "$path/config.json" ] || { echo "[exam] no model at $path (${pair%%=*})" >&2; fail=1; }
done
[ "$fail" -eq 0 ] || exit 1
mkdir -p "$OUT"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

EVAL=(python "$REPO/forge/scripts/legal_eval.py" --items "$ITEMS" --quiet
      --csv "$OUT/keywords.csv")
status=0
for pair in $EXAM_MODELS; do
    label="${pair%%=*}"; path="${pair#*=}"
    echo "[exam] === $label ==="
    "${EVAL[@]}" "$path" --label "$label" --answers "$OUT/answers-$label.jsonl" || status=1
    if [ -n "${EXAM_NORMS:-}" ]; then
        # shellcheck disable=SC2086
        "${EVAL[@]}" "$path" --label "$label-open" --norms $EXAM_NORMS \
            --answers "$OUT/answers-$label-open.jsonl" || status=1
    fi
done
exit "$status"
