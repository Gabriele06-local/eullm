#!/usr/bin/env bash
# Submit the GH200 comparison job (sbatch_compare.slurm) for an engine build,
# after setup.sh has planned its points:
#
#     EULLM_BIN=$PROJECT/$USER/eullm-target/LABEL/release/eullm \
#         bash tools/jupiter/submit.sh LABEL [sbatch args...]
#
# Its output: $CAMPAIGN_DIR/logs/eullm-gh200-<job>.out; the results of the
# points: $CAMPAIGN_DIR/results, read with
#     $CAMPAIGN report --queue "$CAMPAIGN_DIR" --campaign c07-runtimes

set -euo pipefail
# shellcheck source=env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LABEL="${1:?give the label of the engine build, as given to build_engine.sh}"
shift
: "${EULLM_BIN:?set EULLM_BIN to the CUDA eullm binary}"
export EULLM_BIN EULLM_REPO
export EULLM_ENGINE_LABEL="$LABEL"
mkdir -p "$CAMPAIGN_DIR/logs"
$CAMPAIGN status --queue "$CAMPAIGN_DIR" 2>/dev/null | head -3 || true
jid=$(sbatch --parsable --account "$SBATCH_ACCOUNT" \
    --output="$CAMPAIGN_DIR/logs/eullm-gh200-%j.out" \
    --export=ALL "$@" "$EULLM_REPO/tools/jupiter/sbatch_compare.slurm")
echo "job ${jid%%;*}, engine $EULLM_BIN, label $LABEL"
echo "output: $CAMPAIGN_DIR/logs/eullm-gh200-${jid%%;*}.out"
