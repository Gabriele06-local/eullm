# Shared by sbatch_campaign.slurm, submit_campaign.sh and status.sh:
# source it, don't run it.
#
#   EULLM_REPO      this checkout (found from this file when unset)
#   CAMPAIGN_DIR    the campaign queue + results, on scratch, shared by every job
#   EULLM_BIN       the ROCm engine binary
#   EULLM_MODELS_DIR, SBATCH_ACCOUNT as for the other LUMI scripts
#
# and sets PY to a Python the campaign code can run on.

if [ -z "${EULLM_REPO:-}" ]; then
    EULLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export EULLM_REPO
export CAMPAIGN_DIR="${CAMPAIGN_DIR:-/scratch/${SBATCH_ACCOUNT:-project_465003366}/${USER}/campaign}"
export EULLM_MODELS_DIR="${EULLM_MODELS_DIR:-/scratch/${SBATCH_ACCOUNT:-project_465003366}/${USER}/eullm-models}"
export REFLEXBENCH_CACHE="${REFLEXBENCH_CACHE:-$CAMPAIGN_DIR/reflexbench-cache}"
# The runtimes the engine is compared with (install_runtimes.sh), when there.
if [ -z "${LLAMA_SERVER_BIN:-}" ] && [ -x "$CAMPAIGN_DIR/runtimes/llama-server" ]; then
    export LLAMA_SERVER_BIN="$CAMPAIGN_DIR/runtimes/llama-server"
fi
if [ -z "${OLLAMA_BIN:-}" ] && [ -x "$CAMPAIGN_DIR/runtimes/ollama/bin/ollama" ]; then
    export OLLAMA_BIN="$CAMPAIGN_DIR/runtimes/ollama/bin/ollama"
fi
export OLLAMA_MODELS="${OLLAMA_MODELS:-$CAMPAIGN_DIR/ollama-models}"

# SLES 15's /usr/bin/python3 is 3.6, too old for the campaign code (3.8+).
# Take the newest interpreter on PATH, then LUMI's cray-python module.
_campaign_python() {
    local c
    for c in python3.13 python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
        if command -v "$c" >/dev/null 2>&1 &&
           "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
            command -v "$c"
            return 0
        fi
    done
    return 1
}
if [ -z "${PY:-}" ]; then
    PY="$(_campaign_python)" || {
        module load cray-python >/dev/null 2>&1
        PY="$(_campaign_python)"
    } || {
        echo "[err] no Python >= 3.8 found (tried python3.x on PATH and 'module load cray-python')" >&2
        # shellcheck disable=SC2317  # exit is reached when this file is run, not sourced
        return 1 2>/dev/null || exit 1
    }
fi
export PY
# Used unquoted by the scripts that source this file: two words, interpreter and script.
# shellcheck disable=SC2034
CAMPAIGN="$PY $EULLM_REPO/bench/campaign/campaign.py"
