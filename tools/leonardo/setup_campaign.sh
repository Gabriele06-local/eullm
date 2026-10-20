#!/usr/bin/env bash
# The cross-site campaign (c13-sites, docs/sites-comparison.md) set up on
# Leonardo from a login node: compute nodes have no network, so everything a
# job needs is fetched here.
#
#     bash tools/leonardo/setup_campaign.sh
#
# 1. the engine: the published data-center CUDA build of EULLM_VERSION
#    (eullm-linux-x64-cuda-12.4-datacenter: sm_80, CUDA 12.4, for RHEL 8's
#    glibc), unless EULLM_BIN is already there;
# 2. the three models of c13's speed group into the scratch store;
# 3. the workload sets of the graphs group, frozen (`campaign.py prefetch`);
# 4. the plan: c13-sites in round r-sites, as the other two sites.
#
# Idempotent: what is already there is kept, and planning twice adds nothing.
# Then: bash tools/leonardo/submit_campaign.sh

set -uo pipefail
# shellcheck source=campaign_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/campaign_env.sh"
SPEC="$EULLM_REPO/tools/lumi/campaigns/c13-sites.json"
read -r -a MODELS <<<"${EULLM_PULL:-qwen3-8b qwen3-32b qwen3.6-35b-a3b}"
ASSET="eullm-linux-x64-cuda-12.4-datacenter"

mkdir -p "$CAMPAIGN_DIR/logs" "$EULLM_MODELS_DIR" "$(dirname "$EULLM_BIN")"
echo "queue  $CAMPAIGN_DIR"
echo "models $EULLM_MODELS_DIR"

echo
echo "=== engine ==="
if [ ! -x "$EULLM_BIN" ]; then
    url="https://github.com/eullm/eullm/releases/download/EuLLM-v$EULLM_VERSION/$ASSET"
    echo "--- $url"
    curl -fL --retry 3 -o "$EULLM_BIN.partial" "$url" && mv "$EULLM_BIN.partial" "$EULLM_BIN" &&
        chmod +x "$EULLM_BIN" || { echo "[!!] download failed" >&2; exit 1; }
fi
VERSION=$("$EULLM_BIN" --version 2>&1 | head -1)
echo "engine $EULLM_BIN: $VERSION"
case "$VERSION" in
    *"$EULLM_VERSION"*) ;;
    *) echo "[!!] the binary does not say $EULLM_VERSION (GLIBC error? docs/cineca/leonardo.md)" >&2
       exit 1 ;;
esac

echo
echo "=== models ==="
FAILED=()
# Installed is a row of `eullm list`'s table that starts with the id, its file
# present (tools/jupiter/setup.sh says why not any word it prints).
LISTING=$("$EULLM_BIN" list 2>/dev/null)
installed() {  # MODEL
    awk -v m="$1" 'index($0, m " ") == 1 && !/\(file missing\)/ { found = 1 }
        END { exit !found }' <<<"$LISTING"
}
for m in "${MODELS[@]}"; do
    if installed "$m"; then
        echo "  $m: already there"
        continue
    fi
    echo "--- $EULLM_BIN pull $m"
    "$EULLM_BIN" pull "$m" || FAILED+=("$m")
done
[ ${#FAILED[@]} -eq 0 ] || echo "[!!] not pulled (their points block): ${FAILED[*]}"
df -h "$EULLM_MODELS_DIR" | tail -1 | sed 's/^/  scratch: /'

echo
echo "=== workload sets (GSM8K and the ARC sets), frozen for offline nodes ==="
$CAMPAIGN prefetch --queue "$CAMPAIGN_DIR" ||
    echo "[!!] prefetch failed: the graphs points will block"

echo
echo "=== plan ==="
$CAMPAIGN plan "$SPEC" --queue "$CAMPAIGN_DIR" --engine "$EULLM_BIN" --round "${ROUND:-r-sites}"
$CAMPAIGN unblock --queue "$CAMPAIGN_DIR" >/dev/null
$CAMPAIGN status --queue "$CAMPAIGN_DIR" 2>/dev/null | head -5
echo
echo "Next: bash tools/leonardo/submit_campaign.sh"
