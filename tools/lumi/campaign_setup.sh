#!/usr/bin/env bash
# Everything a campaign needs that only a login node can do — compute nodes
# have no network — then the plan. Idempotent: re-run it after adding a spec,
# or with ROUND=<label> to queue the same specs again for a new engine build.
#
#   export SBATCH_ACCOUNT=project_465003366 EULLM_BIN=$PWD/eullm-rocm
#   tmux new -s setup                    # the large MoE pulls take hours
#   bash tools/lumi/campaign_setup.sh [SPEC.json ...]
#
# Default specs: tools/lumi/campaigns/c01-node-baseline.json (catalog models,
# starts at once) and c02-quant-large-moe.json (~1.2 TB from Hugging Face).
#
#   SKIP_PULLS=1      plan without pulling (their points block until pulled;
#                     `campaign.py unblock` after pulling puts them back)
#   ROUND=v0.7.21     a new round: every point of the specs measured again

set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "$(dirname "$0")/campaign_env.sh"
: "${EULLM_BIN:?set EULLM_BIN to the ROCm eullm binary}"

if [ $# -gt 0 ]; then
    SPECS=("$@")
else
    SPECS=("$EULLM_REPO/tools/lumi/campaigns/c01-node-baseline.json"
           "$EULLM_REPO/tools/lumi/campaigns/c02-quant-large-moe.json")
fi
mkdir -p "$CAMPAIGN_DIR" "$EULLM_MODELS_DIR"
echo "queue  $CAMPAIGN_DIR"
echo "models $EULLM_MODELS_DIR ($(df -h "$EULLM_MODELS_DIR" 2>/dev/null | awk 'NR==2{print $4}') free)"

if [ "${SKIP_PULLS:-0}" != "1" ]; then
    echo
    echo "=== models the specs use and the store lacks ==="
    mapfile -t REFS < <($CAMPAIGN pulls "${SPECS[@]}" --engine "$EULLM_BIN")
    FAILED=()
    for ref in "${REFS[@]}"; do
        echo "--- $EULLM_BIN pull $ref"
        "$EULLM_BIN" pull "$ref" || FAILED+=("$ref")
    done
    if [ ${#FAILED[@]} -gt 0 ]; then
        echo
        echo "[!!] could not pull (their points will block until they are in the store):"
        printf '       %s\n' "${FAILED[@]}"
    fi
fi

echo
echo "=== workload sets (GSM8K, ARC-Easy, ARC-Challenge), frozen for offline nodes ==="
$CAMPAIGN prefetch --queue "$CAMPAIGN_DIR" || echo "[!!] prefetch failed: workload points will block"

echo
echo "=== plan ==="
$CAMPAIGN plan "${SPECS[@]}" --queue "$CAMPAIGN_DIR" --engine "$EULLM_BIN" \
    ${ROUND:+--round "$ROUND"}
$CAMPAIGN unblock --queue "$CAMPAIGN_DIR" >/dev/null

cat <<EOF

Next:
  bash $EULLM_REPO/tools/lumi/submit_campaign.sh 2 3   # two nodes, three 48 h jobs each
  bash $EULLM_REPO/tools/lumi/status.sh                # budget, jobs, queue
EOF
