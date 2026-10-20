#!/bin/bash
# One link of the campaign runner on Leonardo Booster: the points
# setup_campaign.sh planned, one per A100, on three of a node's four GPUs.
# Submit through submit_campaign.sh, which chains the links.
#
# Leonardo's shape (forge/CLAUDE.md): two hours, three GPUs, 24 cores,
# 340 GB, never the whole node -- a job asking for part of a node starts far
# sooner. The runner stops starting points ten minutes before the end; at
# the walltime the interrupted ones go back to the queue and the next link
# takes them.
#
# The job finishes its own reading: what it measured and how each point's
# VRAM moved (`campaign.py report`, vram_slopes.py) in
# $CAMPAIGN_DIR/logs/summary-<job>.txt, and a check for the site's trap --
# a CUDA binary whose driver is too old prints "GPU backend: CUDA" and runs
# on the CPU, and only `ggml_cuda_init: failed to initialize CUDA` in the
# server's log says so (docs/cineca/leonardo.md).

#SBATCH --job-name=eullm-campaign
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=340G
#SBATCH --time=02:00:00

set -uo pipefail
# shellcheck source=campaign_env.sh
source "${EULLM_REPO:?submit through submit_campaign.sh, which exports EULLM_REPO}/tools/leonardo/campaign_env.sh"
JOB="${SLURM_JOB_ID:-local}"
GPUS="${SLURM_GPUS_ON_NODE:-3}"

echo "=== eullm campaign on Leonardo: job $JOB on $(hostname), $(date -Is) ==="
printf '  %-12s %s\n' gpus "$GPUS" visible "${CUDA_VISIBLE_DEVICES:-(unset)}" \
    engine "$("$EULLM_BIN" --version 2>&1 | head -1)" repo "${EULLM_BENCH_REV:-?}" \
    queue "$CAMPAIGN_DIR"
# What nvidia-smi numbers, beside CUDA_VISIBLE_DEVICES: the runner reads the
# VRAM of the devices the job was given by these indices (devices.py).
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
echo

$CAMPAIGN run --queue "$CAMPAIGN_DIR" --devices "0-$((GPUS - 1))" --backend cuda \
    --bind none --site leonardo-booster --engine "$EULLM_BIN"

echo
echo "=== CUDA fallback check ==="
if grep -rl "failed to initialize CUDA" "$CAMPAIGN_DIR/results/logs/$JOB" 2>/dev/null; then
    echo "[!!] a server ran on the CPU: those points measured nothing"
else
    echo "  every server found the GPU"
fi
SUMMARY="$CAMPAIGN_DIR/logs/summary-$JOB.txt"
{
    echo "=== what job $JOB measured ==="
    $CAMPAIGN report --queue "$CAMPAIGN_DIR" --job "$JOB"
    echo
    echo "=== VRAM of each point while it ran (GiB an hour) ==="
    "$PY" "$EULLM_REPO/tools/lumi/vram_slopes.py" --queue "$CAMPAIGN_DIR" "$JOB"
    echo
    echo "=== what is left ==="
    $CAMPAIGN status --queue "$CAMPAIGN_DIR"
} > "$SUMMARY" 2>&1
echo "summary: $SUMMARY"
echo "=== done $(date -Is) ==="
