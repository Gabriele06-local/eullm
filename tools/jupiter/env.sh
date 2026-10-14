# The engine's environment on JUPITER Booster (JSC): source it, don't run it.
# Shared by build_engine.sh, install_runtimes.sh and the sbatch scripts here,
# as tools/lumi/campaign_env.sh is on LUMI.
#
#   source tools/jupiter/env.sh
#
# What the login probe found on 14-10-2026 (forge/scripts/jupiter/probe_env.sh):
# aarch64 Neoverse-V2 login nodes with a GH200 of their own (97,871 MiB, 900 W),
# the `booster` partition with whole nodes only (OverSubscribe=EXCLUSIVE: a job
# pays for all four GPUs of its node), modules GCC/14.3.0, CUDA/13,
# Python/3.13.5 and PyTorch/2.9.1, huggingface.co reachable from the login node,
# and the project's directories:
#
#   $PROJECT  /e/project1/e-aif-2026pg01-1434   code, binaries, models, the queue
#   $SCRATCH  /e/scratch/e-aif-2026pg01-1434    unused here: files idle for 90
#             days are deleted, and early access gave compute nodes $HOME and
#             $PROJECT only
#
# Everything of the engine's goes under $PROJECT/$USER: the checkout, the
# Rust toolchain (rustup's, not a module: the crates need a newer Rust than
# sites carry), the build directories, the models and the campaign queue.

JUPITER_PROJECT="${JUPITER_PROJECT:-e-aif-2026pg01-1434}"
if [ -z "${PROJECT:-}" ] || [[ "$PROJECT" != *"$JUPITER_PROJECT"* ]]; then
    jutil env activate -p "$JUPITER_PROJECT" >/dev/null 2>&1 || true
fi
: "${PROJECT:?no \$PROJECT: run 'jutil env activate -p $JUPITER_PROJECT' first}"

export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-$JUPITER_PROJECT}"
export SALLOC_ACCOUNT="${SALLOC_ACCOUNT:-$SBATCH_ACCOUNT}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-$SBATCH_ACCOUNT}"

if [ -z "${EULLM_REPO:-}" ]; then
    EULLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export EULLM_REPO
EULLM_HOME="${EULLM_HOME:-$PROJECT/$USER}"
export EULLM_HOME
export RUSTUP_HOME="${RUSTUP_HOME:-$EULLM_HOME/.rustup}"
export CARGO_HOME="${CARGO_HOME:-$EULLM_HOME/.cargo}"
[ -d "$CARGO_HOME/bin" ] && case ":$PATH:" in
    *":$CARGO_HOME/bin:"*) ;;
    *) export PATH="$CARGO_HOME/bin:$PATH" ;;
esac
export EULLM_TARGETS="${EULLM_TARGETS:-$EULLM_HOME/eullm-target}"
export EULLM_MODELS_DIR="${EULLM_MODELS_DIR:-$EULLM_HOME/eullm-models}"
export CAMPAIGN_DIR="${CAMPAIGN_DIR:-$EULLM_HOME/campaign}"
export REFLEXBENCH_CACHE="${REFLEXBENCH_CACHE:-$CAMPAIGN_DIR/reflexbench-cache}"
if [ -z "${LLAMA_SERVER_BIN:-}" ] && [ -x "$CAMPAIGN_DIR/runtimes/llama-server" ]; then
    export LLAMA_SERVER_BIN="$CAMPAIGN_DIR/runtimes/llama-server"
fi

# Lmod's `module`, in scripts too: a non-interactive shell may not have the
# function, and Lmod's output does not survive `set -u`.
if ! command -v module >/dev/null 2>&1 && [ -n "${LMOD_CMD:-}" ]; then
    module() { eval "$("$LMOD_CMD" bash "$@")"; }
fi
eullm_module() {  # ARGS...: `module ARGS`, quietly, whatever `set -u` says
    local nounset=""
    [[ $- == *u* ]] && nounset=1
    set +u
    module "$@" >/dev/null 2>&1
    local rc=$?
    [ -n "$nounset" ] && set -u
    return $rc
}

# The compilers and CUDA the build and the servers need. Quietly: a module
# this stage does not have is reported where it matters (build_engine.sh).
if command -v module >/dev/null 2>&1; then
    eullm_module load GCC CUDA || true
fi
if [ -z "${CUDA_PATH:-}" ]; then
    for d in "${CUDA_HOME:-}" "${EBROOTCUDA:-}" /usr/local/cuda; do
        if [ -n "$d" ] && [ -x "$d/bin/nvcc" ]; then
            export CUDA_PATH="$d"
            break
        fi
    done
fi

# The campaign code wants Python 3.8+; the system's is 3.9 on JUPITER.
if [ -z "${PY:-}" ]; then
    for c in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
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
