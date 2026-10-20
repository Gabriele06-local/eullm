#!/usr/bin/env bash
# The engine comparison on JUPITER, set up from a login node: the models in
# the store, the workload sets frozen for nodes that may have no network, and
# the c07 points planned for this engine build.
#
#     EULLM_BIN=$PROJECT/$USER/eullm-target/LABEL/release/eullm \
#         bash tools/jupiter/setup.sh LABEL
#
# What it plans: c07's rt27-* groups (Qwen3.6-27B, the hybrid model, against
# Qwen3-32B, EuLLM against llama-server: 1 to 32 requests at once, a 16,000-
# token prompt, 16 clients over 400 gsm8k questions), in round r-gh200,
# labelled LABEL so that only jobs of this build run them. They are the
# points measured on LUMI's MI250X on 14-10-2026 (r-27b-next10), so the two
# machines compare row by row. EULLM_GROUPS='rt27-* rt-throughput' widens
# it, EULLM_PULL='…' changes the models pulled. EULLM_SPEC plans another
# campaign file in place of c07's, ROUND another round: the three sites'
# comparison (docs/sites-comparison.md) is
#
#     EULLM_SPEC=tools/lumi/campaigns/c13-sites.json ROUND=r-sites \
#         EULLM_GROUPS='graphs speed' EULLM_PULL='qwen3-8b qwen3-32b qwen3.6-35b-a3b' \
#         EULLM_BIN=... bash tools/jupiter/setup.sh LABEL
#
# Then: bash tools/jupiter/submit.sh LABEL

set -uo pipefail
# shellcheck source=env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LABEL="${1:?give the label of the engine build, as given to build_engine.sh}"
: "${EULLM_BIN:?set EULLM_BIN to the CUDA eullm binary (build_engine.sh prints it)}"
SPEC="${EULLM_SPEC:-$EULLM_REPO/tools/lumi/campaigns/c07-runtimes.json}"
[ -f "$SPEC" ] || SPEC="$EULLM_REPO/$SPEC"
[ -f "$SPEC" ] || { echo "[!!] no campaign file $SPEC"; exit 1; }
read -r -a GROUP_PATTERNS <<<"${EULLM_GROUPS:-rt27-*}"
# The models of the planned groups, and the conversations' 35B-A3B.
read -r -a MODELS <<<"${EULLM_PULL:-qwen3.6-27b qwen3-32b qwen3.6-35b-a3b}"

mkdir -p "$CAMPAIGN_DIR/logs" "$EULLM_MODELS_DIR"
echo "queue  $CAMPAIGN_DIR"
echo "models $EULLM_MODELS_DIR"
echo "engine $("$EULLM_BIN" --version 2>&1 | head -1)"
[ -x "${LLAMA_SERVER_BIN:-}" ] ||
    echo "[!!] no llama-server yet: bash tools/jupiter/install_runtimes.sh (its points block until then)"

echo
echo "=== models ==="
FAILED=()
# Installed is a row of `eullm list`'s table, which starts with the id, its
# file present: not any word it prints. With an empty store it prints the EU
# catalog, indented, where these three ids are too; read as words, the first
# job on 14-10-2026 found none of them on the node (campaign.py's
# installed_models reads it the same way).
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

echo
echo "=== workload sets (GSM8K and the ARC sets), frozen for offline nodes ==="
$CAMPAIGN prefetch --queue "$CAMPAIGN_DIR" ||
    echo "[!!] prefetch failed: the graded points will block"

echo
echo "=== plan ==="
GROUP_ARGS=()
for g in "${GROUP_PATTERNS[@]}"; do GROUP_ARGS+=(--group "$g"); done
$CAMPAIGN plan "$SPEC" --queue "$CAMPAIGN_DIR" --engine "$EULLM_BIN" \
    --round "${ROUND:-r-gh200}" --engine-label "$LABEL" "${GROUP_ARGS[@]}"
$CAMPAIGN unblock --queue "$CAMPAIGN_DIR" >/dev/null
echo
echo "Next: EULLM_BIN=$EULLM_BIN bash tools/jupiter/submit.sh $LABEL [--time=...]"
