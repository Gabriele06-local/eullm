#!/bin/bash
# One exam round, start to finish: draw the exam, answer it, grade it.
#
# Meant to be queued by a watcher the moment a model is packaged, so nobody
# has to time three submissions by hand:
#
#   export ROUND_MODELS="v0.3=$WORK/eullm_runs/stage3/sft-v03-instruct-ob/merged qwen3-4b-instruct=<snapshot>"
#   export ROUND_PAIRS=$WORK/eullm_runs/stage3/openbook-pairs.jsonl
#   bash submit_when_ready.sh --need <the model's q4_k_m.gguf> -- sbatch_exam_round.slurm 1
#
# What it does, on the serial partition (drawing the exam is CPU work):
#
#   1. Draws the exam into ROUND_ITEMS with make_norm_exam.py, leaving out
#      every article the training pairs in ROUND_PAIRS ask about — unless
#      ROUND_ITEMS already exists, in which case it is reused, never redrawn:
#      a round run twice must grade the same exam.
#   2. Submits sbatch_norm_exam.slurm (closed and open book for each model),
#      then sbatch_judge.slurm with afterok on it.
#
# Everything stays quiet: counts only, never a question (see make_norm_exam.py).
#
# Settable:
#   ROUND_MODELS  space-separated label=path pairs (required)
#   ROUND_PAIRS   training pairs whose articles are not drawn (required;
#                 "none" only if the models were trained on none)
#   ROUND_ITEMS   exam file (default $WORK/eval/norm-exam-v2.jsonl)
#   ROUND_OUT     answers directory (default $WORK/eval/norm-exam-answers-v2)
#   ROUND_NORMS   legislation files (default $WORK/norms/legislazione_*.chunks.jsonl)
#   ROUND_EXCLUDE_EXAM  another exam whose articles are not drawn — set it to the
#                 held-out exam when drawing a readable development set
#   ROUND_GPUS    GPUs for the exam job (default 1). 2 or more for models that
#                 do not fit one 64 GB A100 in bf16 (27B dense, 35B MoE)
#   ROUND_PER_CODE  items of each kind per code (default 10, make_norm_exam.py's)
#   ROUND_CODES   only these codes, space-separated (default: all)
#   ROUND_OPEN_ONLY  1 = ask open book only (EXAM_OPEN_ONLY)
#   ROUND_EMBEDDER / ROUND_RERANKER / ROUND_OPEN_TAG  retrieve with BM25 +
#                embeddings + reranker, labelled "<model>-<tag>" (see
#                sbatch_norm_exam.slurm's EXAM_EMBEDDER)
#   ROUND_EXAM_LINKS / ROUND_JUDGE_LINKS  2 h links for the exam and for the
#                 judge (default 1 each). A large development set does not fit
#                 one link: each link skips what an earlier one finished
#                 (sbatch_norm_exam.slurm, judge_answers.py), so a chain of
#                 short jobs does the work of one long one and gets placed
#   ROUND_QUIET   1 (default) grades without printing items; 0 only for a
#                 development set, never for the held-out exam
#   ROUND_REPO    checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-exam-round
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${ROUND_REPO:-}" ]; then
    REPO="$ROUND_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

LEO="$REPO/forge/scripts/leonardo"
ITEMS="${ROUND_ITEMS:-$WORK/eval/norm-exam-v2.jsonl}"
OUT="${ROUND_OUT:-$WORK/eval/norm-exam-answers-v2}"
PAIRS="${ROUND_PAIRS:-}"
# shellcheck disable=SC2206
NORMS=(${ROUND_NORMS:-$WORK/norms/legislazione_*.chunks.jsonl})

echo "[round] repo   $REPO"
echo "[round] commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[round] models ${ROUND_MODELS:-}"
echo "[round] items  $ITEMS"
echo "[round] out    $OUT"

fail=0
[ -n "${ROUND_MODELS:-}" ] || { echo "[round] ROUND_MODELS is required" >&2; fail=1; }
for pair in ${ROUND_MODELS:-}; do
    path="${pair#*=}"
    [ -f "$path/config.json" ] || { echo "[round] no model at $path (${pair%%=*})" >&2; fail=1; }
done
[ -f "${NORMS[0]}" ] || { echo "[round] no legislation files" >&2; fail=1; }
if [ -z "$PAIRS" ]; then
    echo "[round] ROUND_PAIRS is required: the exam must leave out what training asked" >&2
    echo "[round] (ROUND_PAIRS=none only if the models were trained on no open-book pairs)" >&2
    fail=1
elif [ "$PAIRS" != "none" ]; then
    for f in $PAIRS; do [ -s "$f" ] || { echo "[round] no training pairs at $f" >&2; fail=1; }; done
fi
[ "$fail" -eq 0 ] || exit 1

if [ -s "$ITEMS" ]; then
    echo "[round] exam already drawn — reusing it, not redrawing"
else
    EXCLUDE=()
    # shellcheck disable=SC2206
    [ "$PAIRS" = "none" ] || EXCLUDE=(--exclude-pairs $PAIRS)
    # shellcheck disable=SC2206
    [ -z "${ROUND_EXCLUDE_EXAM:-}" ] || EXCLUDE+=(--exclude-exam $ROUND_EXCLUDE_EXAM)
    DRAW=(--per-code "${ROUND_PER_CODE:-10}")
    # shellcheck disable=SC2206
    [ -z "${ROUND_CODES:-}" ] || DRAW+=(--codes $ROUND_CODES)
    python "$REPO/forge/scripts/make_norm_exam.py" "${NORMS[@]}" --out "$ITEMS" \
        "${EXCLUDE[@]}" "${DRAW[@]}" --check-retrieval || { echo "[round] drawing the exam failed" >&2; exit 1; }
fi

mkdir -p "$OUT" logs
export EXAM_MODELS="$ROUND_MODELS" EXAM_ITEMS="$ITEMS" EXAM_OUT="$OUT"
export EXAM_NORMS="${NORMS[*]}" EXAM_REPO="$REPO" EXAM_OPEN_ONLY="${ROUND_OPEN_ONLY:-0}"
export EXAM_EMBEDDER="${ROUND_EMBEDDER:-}" EXAM_RERANKER="${ROUND_RERANKER:-}"
export EXAM_OPEN_TAG="${ROUND_OPEN_TAG:-open}"
GPUS="${ROUND_GPUS:-1}"
case "$GPUS" in ''|*[!0-9]*) echo "[round] ROUND_GPUS must be a number" >&2; exit 1 ;; esac
EXAM_LINKS="${ROUND_EXAM_LINKS:-1}"
JUDGE_LINKS="${ROUND_JUDGE_LINKS:-1}"
for n in "$EXAM_LINKS" "$JUDGE_LINKS"; do
    case "$n" in ''|*[!0-9]*|0) echo "[round] ROUND_*_LINKS must be a positive number" >&2; exit 1 ;; esac
done

# Links of one chain wait on each other with afterany: a link that runs out
# of its two hours ends TIMEOUT, which is the design, and the next one picks
# up. The judge waits on the LAST exam link with afterok: that one finds
# every finished answers file and exits 0, unless a model really failed --
# and then nothing should be graded as if the round were whole.
prev=""
for i in $(seq 1 "$EXAM_LINKS"); do
    DEP=()
    [ -z "$prev" ] || DEP=(--dependency="afterany:$prev")
    exam=$(sbatch --parsable --export=ALL --gres="gpu:$GPUS" --mem="$((GPUS * 120))G" \
           --cpus-per-task="$((GPUS * 8))" "${DEP[@]}" "$LEO/sbatch_norm_exam.slurm") ||
        { echo "[round] could not submit exam link $i${prev:+ (link $((i - 1)) is $prev)}" >&2; exit 1; }
    prev="${exam%%;*}"
    echo "[round] exam link $i/$EXAM_LINKS submitted: $prev"
done

export JUDGE_DIR="$OUT" JUDGE_REPO="$REPO"
if [ "${ROUND_QUIET:-1}" = "0" ]; then unset JUDGE_QUIET; else export JUDGE_QUIET=1; fi
kind="afterok"
for i in $(seq 1 "$JUDGE_LINKS"); do
    judge=$(sbatch --parsable --export=ALL --dependency="$kind:$prev" "$LEO/sbatch_judge.slurm") ||
        { echo "[round] could not submit judge link $i (after $prev)" >&2; exit 1; }
    echo "[round] judge link $i/$JUDGE_LINKS submitted: ${judge%%;*} ($kind $prev)"
    prev="${judge%%;*}"
    kind="afterany"
done
echo "[round] results will be in $OUT/graded.csv"
