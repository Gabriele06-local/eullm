#!/bin/bash
# Stage-3 data, open book: question + retrieved norms -> grounded answer.
# What and why: forge/eullm_forge/datasets/openbook_gen.py. Same teacher and
# the same node geometry as sbatch_gen_instruct.slurm (3 GPUs, 24 cores,
# 340 GB, two hours), so the same placement.
#
#   sbatch --export=ALL,OB_LIMIT=4000 forge/scripts/leonardo/sbatch_gen_openbook.slurm
#
# The held-out exam's articles are excluded: OB_EXAM must exist, or the job
# refuses (set OB_EXAM=none only if there is really no exam). Resumable:
# submit it again with the same settings and it carries on.
#
# Settable:
#   OB_LIMIT   pairs attempted (default 4000; ~15% are absent-article pairs)
#   OB_OUT     output JSONL (default $WORK/eullm_runs/stage3/openbook-pairs.jsonl)
#   OB_NORMS   legislation files (default $WORK/norms/legislazione_*.chunks.jsonl)
#   OB_EXAM    held-out exam(s) and dev sets, space-separated (default
#              $WORK/eval/norm-exam.jsonl); all their articles are left out
#   OB_ARGS    extra generate_openbook_pairs.py options, e.g.
#              "--per-article 4 --shard 0/3 --seed 1"
#   OB_REPO    checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-gen-openbook
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=340G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

# The checkout is the one this file was submitted from, not $EULLM_REPO —
# the lesson of the curve job, which once ran another tree's code.
SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${OB_REPO:-}" ]; then
    REPO="$OB_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

LIMIT="${OB_LIMIT:-4000}"
OUT="${OB_OUT:-$WORK/eullm_runs/stage3/openbook-pairs.jsonl}"
EXAM="${OB_EXAM:-$WORK/eval/norm-exam.jsonl}"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
# shellcheck disable=SC2206
NORMS=(${OB_NORMS:-$WORK/norms/legislazione_*.chunks.jsonl})

echo "[ob] repo   $REPO"
echo "[ob] commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[ob] limit  $LIMIT"
echo "[ob] out    $OUT"
echo "[ob] norms  ${#NORMS[@]} files"

[ -f "${NORMS[0]}" ] || { echo "[ob] no legislation files" >&2; exit 1; }
if [ "$EXAM" = "none" ]; then
    EXAM_ARGS=(--no-exam)
else
    for f in $EXAM; do
        [ -s "$f" ] || { echo "[ob] no exam at $f (OB_EXAM=none if there is none)" >&2; exit 1; }
    done
    # shellcheck disable=SC2206
    EXAM_ARGS=(--exclude-exam $EXAM)
fi
mkdir -p "$(dirname "$OUT")"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

# shellcheck disable=SC2086
python "$REPO/forge/scripts/generate_openbook_pairs.py" \
    --norms "${NORMS[@]}" "${EXAM_ARGS[@]}" --out "$OUT" --model "$MODEL" \
    --limit "$LIMIT" --stop-after-min 110 ${OB_ARGS:-}
