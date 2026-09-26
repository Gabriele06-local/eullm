#!/bin/bash
# Grade legal-eval answers with Qwen3-30B-A3B-Instruct-2507 (judge_answers.py).
#
#   sbatch --export=ALL,JUDGE_DIR=$WORK/eullm_runs/stage3/eval \
#       forge/scripts/leonardo/sbatch_judge.slurm
#
# Grades every answers-*.jsonl in JUDGE_DIR and appends one row per model to
# JUDGE_DIR/graded.csv. Same geometry as the pair generation — 3 GPUs, 24
# cores, 340 GB, two hours — because it is the same 61 GB model and the
# geometry measured to be placed quickly. Ten items for five models is a few
# minutes of generation; most of the job is loading the weights.
#
# Settable: JUDGE_DIR (required), JUDGE_GLOB (default answers-*.jsonl),
# JUDGE_REPO (default: the checkout this file was submitted from).
#
#SBATCH --job-name=eullm-judge
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
if [ -n "${JUDGE_REPO:-}" ]; then
    REPO="$JUDGE_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

DIR="${JUDGE_DIR:-}"
GLOB="${JUDGE_GLOB:-answers-*.jsonl}"
echo "[judge] repo   $REPO"
echo "[judge] commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[judge] dir    $DIR"

[ -n "$DIR" ] && [ -d "$DIR" ] || { echo "[judge] JUDGE_DIR is required and must exist" >&2; exit 1; }
# The graded files this job writes match the same glob; never grade those.
FILES=()
for f in "$DIR"/$GLOB; do
    case "$f" in *.graded.jsonl) ;; *) [ -f "$f" ] && FILES+=("$f") ;; esac
done
[ "${#FILES[@]}" -gt 0 ] || { echo "[judge] no $GLOB in $DIR" >&2; exit 1; }
echo "[judge] files  ${FILES[*]}"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

python "$REPO/forge/scripts/judge_answers.py" "${FILES[@]}" --csv "$DIR/graded.csv"
