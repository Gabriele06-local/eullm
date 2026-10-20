#!/usr/bin/env bash
# The cross-site campaign (c13-sites, docs/sites-comparison.md) set up on
# Leonardo from a login node: compute nodes have no network, so everything a
# job needs is fetched here.
#
#     bash tools/leonardo/setup_campaign.sh
#
# 1. the engine: the published data-center CUDA build of EULLM_VERSION
#    (eullm-linux-x64-cuda-12.4-datacenter: sm_80, CUDA 12.4, for RHEL 8's
#    glibc) for the jobs, and the CPU build of the same version for this
#    node, which has no GPU driver: the CUDA binary does not even print its
#    version here (libcuda.so.1 missing, 20-10-2026);
# 2. the three models of c13's speed group into the scratch store;
# 3. the workload sets of the graphs group, frozen (`campaign.py prefetch`);
# 4. the plan: c13-sites in round r-sites, as the other two sites.
#
# Idempotent: what is already there is kept, and planning twice adds nothing.
# It stops at the first step that fails, and only a setup that got to the end
# leaves $CAMPAIGN_DIR/setup-ok, without which submit_campaign.sh submits
# nothing. Then: bash tools/leonardo/submit_campaign.sh

set -uo pipefail
# shellcheck source=campaign_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/campaign_env.sh"
SPEC="$EULLM_REPO/tools/lumi/campaigns/c13-sites.json"
read -r -a MODELS <<<"${EULLM_PULL:-qwen3-8b qwen3-32b qwen3.6-35b-a3b}"
RELEASE="https://github.com/eullm/eullm/releases/download/EuLLM-v$EULLM_VERSION"

rm -f "$CAMPAIGN_DIR/setup-ok"
mkdir -p "$CAMPAIGN_DIR/logs" "$EULLM_MODELS_DIR" "$(dirname "$EULLM_BIN")"
echo "repo   $EULLM_REPO"
echo "queue  $CAMPAIGN_DIR"
echo "models $EULLM_MODELS_DIR"

fetch() {  # ASSET DEST: the release asset, unless DEST is already there
    [ -x "$2" ] && return 0
    echo "--- $RELEASE/$1"
    curl -fL --retry 3 -o "$2.partial" "$RELEASE/$1" && mv "$2.partial" "$2" && chmod +x "$2"
}
echo
echo "=== engine ==="
fetch eullm-linux-x64-cuda-12.4-datacenter "$EULLM_BIN" &&
    fetch eullm-linux-x64 "$EULLM_CPU_BIN" || { echo "[!!] download failed" >&2; exit 1; }
# The version must come from the binary itself: an error message names its
# path, which holds the version too.
VERSION=$("$EULLM_CPU_BIN" --version 2>&1) && [[ "$VERSION" == eullm* ]] ||
    { echo "[!!] the CPU binary does not start: $VERSION (docs/cineca/leonardo.md)" >&2; exit 1; }
echo "engine $VERSION (CPU build here; the CUDA build runs in the jobs)"
[[ "$VERSION" == *"$EULLM_VERSION"* ]] ||
    { echo "[!!] expected $EULLM_VERSION" >&2; exit 1; }

echo
echo "=== models ==="
FAILED=()
# Installed is a row of `eullm list`'s table that starts with the id, its file
# present (tools/jupiter/setup.sh says why not any word it prints).
LISTING=$("$EULLM_CPU_BIN" list 2>/dev/null)
installed() {  # MODEL
    awk -v m="$1" 'index($0, m " ") == 1 && !/\(file missing\)/ { found = 1 }
        END { exit !found }' <<<"$LISTING"
}
for m in "${MODELS[@]}"; do
    if installed "$m"; then
        echo "  $m: already there"
        continue
    fi
    echo "--- eullm pull $m"
    "$EULLM_CPU_BIN" pull "$m" || FAILED+=("$m")
done
[ ${#FAILED[@]} -eq 0 ] || { echo "[!!] not pulled: ${FAILED[*]}" >&2; exit 1; }
df -h "$EULLM_MODELS_DIR" | tail -1 | sed 's/^/  scratch: /'

echo
echo "=== workload sets (GSM8K and the ARC sets), frozen for offline nodes ==="
$CAMPAIGN prefetch --queue "$CAMPAIGN_DIR" || { echo "[!!] prefetch failed" >&2; exit 1; }

echo
echo "=== plan ==="
$CAMPAIGN plan "$SPEC" --queue "$CAMPAIGN_DIR" --engine "$EULLM_CPU_BIN" \
    --round "${ROUND:-r-sites}" || { echo "[!!] plan failed" >&2; exit 1; }
$CAMPAIGN unblock --queue "$CAMPAIGN_DIR" >/dev/null
$CAMPAIGN status --queue "$CAMPAIGN_DIR" 2>/dev/null | head -5
date -Is > "$CAMPAIGN_DIR/setup-ok"
echo
echo "Next: bash tools/leonardo/submit_campaign.sh"
