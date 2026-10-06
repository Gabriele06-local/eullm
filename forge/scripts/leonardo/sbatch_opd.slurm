#!/bin/bash
# Privileged-context distillation (opd_train.py): the student on the first
# GPU, the teacher on the other two, three GPUs in two-hour links.
#
#   OPD_STUDENT=$WORK/eullm_runs/grpo/v04/merged \
#   OPD_TEACHER=<Qwen3-30B-A3B-Instruct-2507 snapshot, or the student itself for Ministral> \
#   OPD_PROMPTS=$WORK/eullm_runs/opd/prompts.jsonl OPD_OUT=$WORK/eullm_runs/opd/v04 \
#       sbatch forge/scripts/leonardo/sbatch_opd.slurm
#
# Each link saves the adapter, optimizer and step before its two hours are
# up and the next one carries on from them; a link that finds the adapter
# written exits at once. Chain with afterany, then merge the adapter into
# OPD_STUDENT like any other (merge_identity_adapter).
#
# Settable: OPD_STUDENT, OPD_TEACHER, OPD_PROMPTS, OPD_OUT (all required),
# OPD_ARGS (extra opd_train.py options, e.g. "--steps 300 --lr 5e-5"),
# OPD_REPO (checkout to run; default the one this file was submitted from).
#
#SBATCH --job-name=eullm-opd
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
if [ -n "${OPD_REPO:-}" ]; then
    REPO="$OPD_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

fail=0
[ -f "${OPD_STUDENT:-}/config.json" ] || { echo "[opd] no student at '${OPD_STUDENT:-}'" >&2; fail=1; }
[ -f "${OPD_TEACHER:-}/config.json" ] || { echo "[opd] no teacher at '${OPD_TEACHER:-}'" >&2; fail=1; }
[ -s "${OPD_PROMPTS:-}" ] || { echo "[opd] no prompts at '${OPD_PROMPTS:-}'" >&2; fail=1; }
[ -n "${OPD_OUT:-}" ] || { echo "[opd] OPD_OUT is required" >&2; fail=1; }
[ "$fail" -eq 0 ] || exit 1
GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
# The student on the first GPU this job was given, the teacher on the rest,
# by their index in this job (CUDA_VISIBLE_DEVICES renumbers them from 0).
TEACHER_GPUS="$(seq -s, 1 $((GPUS - 1)))"
[ "$GPUS" -ge 2 ] || { echo "[opd] needs at least two GPUs, has $GPUS" >&2; exit 1; }
echo "[opd] repo $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')), student cuda:0, teacher $TEACHER_GPUS"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

# shellcheck disable=SC2086
python "$REPO/forge/scripts/opd_train.py" --student "$OPD_STUDENT" --teacher "$OPD_TEACHER" \
    --prompts "$OPD_PROMPTS" --out "$OPD_OUT" --student-device cuda:0 \
    --teacher-devices "$TEACHER_GPUS" --stop-after 6600 ${OPD_ARGS:-}
