# The campaign runner's environment on Leonardo Booster (CINECA): source it,
# don't run it. The Leonardo twin of tools/lumi/campaign_env.sh and
# tools/jupiter/env.sh, used by setup_campaign.sh, submit_campaign.sh and
# sbatch_campaign.slurm here.
#
#   source tools/leonardo/campaign_env.sh
#
# Where things go:
#
#   $CAMPAIGN_DIR     $WORK/eullm-campaign: the queue, the results and the
#                     logs, small and kept ($WORK is the project's 1 TB)
#   $EULLM_MODELS_DIR the fast scratch: the models can be pulled again, and
#                     $WORK reached 96% with two downloads (forge/CLAUDE.md)
#   $EULLM_BIN        default $WORK/eullm-engine/<version>/eullm, the
#                     published data-center CUDA build setup_campaign.sh
#                     downloads (sm_80, CUDA 12.4, built on Rocky Linux 8 for
#                     RHEL 8's glibc: docs/cineca/leonardo.md)

LEONARDO_ACCOUNT="${LEONARDO_ACCOUNT:-AIFAC_P02_1147}"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-$LEONARDO_ACCOUNT}"
export SALLOC_ACCOUNT="${SALLOC_ACCOUNT:-$SBATCH_ACCOUNT}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-$SBATCH_ACCOUNT}"
: "${WORK:?no \$WORK: this is for Leonardo}"

if [ -z "${EULLM_REPO:-}" ]; then
    EULLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export EULLM_REPO
export EULLM_VERSION="${EULLM_VERSION:-0.7.90}"
export EULLM_BIN="${EULLM_BIN:-$WORK/eullm-engine/$EULLM_VERSION/eullm}"
export EULLM_MODELS_DIR="${EULLM_MODELS_DIR:-/leonardo_scratch/fast/$LEONARDO_ACCOUNT/eullm-models}"
export CAMPAIGN_DIR="${CAMPAIGN_DIR:-$WORK/eullm-campaign}"
export REFLEXBENCH_CACHE="${REFLEXBENCH_CACHE:-$CAMPAIGN_DIR/reflexbench-cache}"

# The campaign code wants Python 3.8+; RHEL 8's own python3 is 3.6, so the
# site's module first, quietly (Lmod's output does not survive `set -u`).
if [ -z "${PY:-}" ] && command -v module >/dev/null 2>&1; then
    _nounset=""; [[ $- == *u* ]] && _nounset=1; set +u
    module load python/3.11.7 >/dev/null 2>&1 || true
    [ -n "$_nounset" ] && set -u; unset _nounset
fi
if [ -z "${PY:-}" ]; then
    for c in python3.11 python3.12 python3.10 python3.9 python3.8 python3; do
        if command -v "$c" >/dev/null 2>&1 &&
           "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
            PY="$(command -v "$c")"
            break
        fi
    done
fi
: "${PY:?no Python 3.8+ on PATH}"
export PY
# Used unquoted by the scripts that source this file: two words.
# shellcheck disable=SC2034
CAMPAIGN="$PY $EULLM_REPO/bench/campaign/campaign.py"
