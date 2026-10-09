#!/bin/bash
# One whole LUMI-G node, all eight GCDs, measuring until the walltime runs
# out or the campaign queue is empty.
#
# WHY. The allocation is about one node busy every day until 12-03-2027, and
# a node that runs one configuration per job leaves seven GCDs idle while it
# bills all eight. This job runs bench/campaign/campaign.py, which packs
# points onto the GCDs — single-GCD points eight at a time, pairs on one
# MI250X module, halves, the whole node — and starts the next one the moment
# devices free up. Workload points stretch to fill whatever time is left, so
# the end of a 48-hour job is not billed idle either.
#
# Each server is pinned to the seven cores closest to its GCD (LUMI's
# documented binding: GCD and NUMA numbering do not match), and every result
# records which cores it had and how many other points shared the node.
#
# Do not submit this directly: submit_campaign.sh sets the output path and
# chains jobs so that the node never stops between them.
#
#     export SBATCH_ACCOUNT=project_465003366
#     export EULLM_BIN=$PWD/eullm-rocm
#     bash tools/lumi/submit_campaign.sh 2 3     # two nodes, three jobs each
#
# The whole node bills 4 GPU-hours per hour on standard-g whatever is running
# on it: the point of this script is that something always is.

#SBATCH --job-name=eullm-campaign
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=eullm-campaign-%j.out

set -uo pipefail

# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "${EULLM_REPO:?submit through submit_campaign.sh, which exports EULLM_REPO}/tools/lumi/campaign_env.sh"
: "${EULLM_BIN:?set EULLM_BIN to the ROCm eullm binary}"

GCDS="${SLURM_GPUS_ON_NODE:-8}"
echo "=== eullm campaign job $SLURM_JOB_ID on $(hostname) ==="
printf '  %-12s %s\n' gcds "$GCDS" engine "$("$EULLM_BIN" --version 2>&1 | head -1)" \
    label "${EULLM_ENGINE_LABEL:-(none: unlabelled points only)}" \
    repo "$(git -C "$EULLM_REPO" rev-parse --short HEAD 2>/dev/null)" \
    python "$PY" queue "$CAMPAIGN_DIR" models "$EULLM_MODELS_DIR"
rocm-smi --showproductname 2>&1 | grep -E 'GPU\[' | head -8 | sed 's/^/  /'
echo

# exec: Slurm's SIGTERM at the walltime goes straight to the runner, which
# kills its servers and puts the interrupted points back in the queue.
exec $CAMPAIGN run --queue "$CAMPAIGN_DIR" --devices "0-$((GCDS - 1))" \
    --backend rocm --bind lumi --site lumi-g --engine "$EULLM_BIN"
