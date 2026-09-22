#!/bin/bash
# Does a SHORT job slot into a backfill window that a long one cannot?
#
# This is a probe, not work. It asks one question and costs half a node-hour
# to answer it.
#
# The background. On 2026-09-22 the allocation stood at 20 % of its node hours
# against 34 % of its calendar, with three eligible jobs on boost_usr_prod and
# no start in over 26 hours. The hypothesis was that a shorter job would fit
# between the big ones — so the queued links were cut from 24 h to 4 h. After
# 28 hours none of the three heads had even been given a planned START_TIME,
# which says backfill is not placing them. But that is not a clean result: we
# never had a 4 h chain and a 24 h chain queued at the same time, so what was
# compared was two weeks of contention, not two walltimes.
#
# THE ONE VARIABLE IS DURATION. Everything else is copied from
# sbatch_phase2_split.slurm on purpose — partition, one node, four GPUs, 32
# cores, 450 GB, QOS `normal`. If this job starts and a 4 h job does not, the
# difference is the walltime and the hypothesis is alive. If it sits pending
# beside them, the hypothesis is dead and the remaining chains belong back at
# 24 h where they book four times the node hours per grant.
#
# NOT boost_qos_dbg, although it is the obvious place for a 30-minute job.
# That QOS carries priority 80 against `normal`'s 40, so a job that started
# there would not tell us whether duration or QOS did it — it would be the one
# experiment less informative than the one already run.
#
# It does nothing but report where it landed and exit, so it never holds a
# node it is not using. What to look for, from a login node:
#
#   squeue --me -n eullm-probe -o "%.10i %.2t %.10L %.20R %.19S"
#
# START_TIME filling in is already the answer — the job need not run. A
# planned start means backfill found it a window; N/A after several hours,
# while the 4 h heads are also N/A, means short jobs are not the lever.
#
#SBATCH --job-name=eullm-probe
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=450G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

echo "[probe] job     ${SLURM_JOB_ID:-?}"
echo "[probe] node    $(hostname)"
echo "[probe] started $(date -Is)"

# How long it waited, which is the number this exists to produce. Everything
# else in the log is context.
SUBMIT="$(squeue -h -j "${SLURM_JOB_ID:-0}" -o "%V" 2>/dev/null ||
          sacct -X -n -j "${SLURM_JOB_ID:-0}" -o Submit 2>/dev/null | head -1)"
echo "[probe] submitted at $SUBMIT"
echo "[probe] waited       $(sacct -X -n -j "${SLURM_JOB_ID:-0}" \
                              -o Reserved 2>/dev/null | head -1 | xargs || echo '?')"

# Confirm the node is what the chains ask for, so a start here really is
# comparable to a start for them.
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null ||
    echo "[probe] no nvidia-smi"

echo "[probe] done — releasing the node immediately"
