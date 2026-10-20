#!/usr/bin/env bash
# JUPITER's part of the three sites' comparison (docs/sites-comparison.md,
# tools/lumi/campaigns/c13-sites.json), from a login node and unattended:
# the engine built from this checkout, the models and workload sets fetched,
# the points planned in round r-sites, the job submitted. Under tmux, since
# the build takes a while:
#
#     tmux new -d -s sites 'bash tools/jupiter/sites.sh'
#
# LABEL (default sites) names the build and the points, so that only this
# job runs them. The job writes $CAMPAIGN_DIR/logs/summary-<job>.txt itself;
# this script's own log is $CAMPAIGN_DIR/logs/sites-<label>.log.

set -uo pipefail
# shellcheck source=env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LABEL="${1:-sites}"
LOG="$CAMPAIGN_DIR/logs/sites-$LABEL.log"
mkdir -p "$CAMPAIGN_DIR/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== c13-sites on JUPITER, label $LABEL, $(date -Is), checkout $(git -C "$EULLM_REPO" rev-parse --short HEAD) ==="

bash "$EULLM_REPO/tools/jupiter/build_engine.sh" "$LABEL" || { echo "[!!] build failed"; exit 1; }
export EULLM_BIN="$EULLM_TARGETS/$LABEL/release/eullm"

SETUP="$CAMPAIGN_DIR/logs/sites-$LABEL.setup"
EULLM_SPEC=tools/lumi/campaigns/c13-sites.json ROUND=r-sites \
    EULLM_GROUPS='graphs speed' EULLM_PULL='qwen3-8b qwen3-32b qwen3.6-35b-a3b' \
    bash "$EULLM_REPO/tools/jupiter/setup.sh" "$LABEL" 2>&1 | tee "$SETUP"
# A model or a workload set missing blocks its points: no job for them.
if grep -qE '^\[!!\] (not pulled|prefetch failed)' "$SETUP"; then
    echo "[!!] setup incomplete, nothing submitted"
    exit 1
fi

# c07's conversations are not part of this comparison.
EULLM_SKIP_CONVERSATIONS=1 bash "$EULLM_REPO/tools/jupiter/submit.sh" "$LABEL" --time=02:30:00
echo "=== submitted $(date -Is); the job writes logs/summary-<job>.txt when it ends ==="
