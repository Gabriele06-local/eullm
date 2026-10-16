#!/usr/bin/env bash
# What the Qwen3.5 runs need on top of the Forge venv, kept out of it. On a
# login node (compute nodes have no internet):
#
#     bash forge/scripts/jupiter/setup_qwen35_libs.sh
#
# flash-linear-attention (in the venv) refuses Triton 3.4.0 to 3.7.0 on
# Hopper: they give wrong gradients in chunk_bwd_dqkwg (fla issue #640), and
# the module's Triton is 3.5.1 (jobs 2349818-2349820 stopped there,
# 16-10-2026). Triton 3.8.0, the latest on PyPI (checked 16-10-2026, wheel
# for aarch64 and Python 3.13), goes to $FORGE_HOME/libs-qwen35, which a job
# puts first on PYTHONPATH with FORGE_EXTRA_LIBS (env.sh). The Qwen3 jobs
# keep the module's Triton: nothing they run changes.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
[ -z "${SLURM_JOB_ID:-}" ] || { echo "[err] run this on a login node" >&2; exit 1; }
LIBS="$FORGE_HOME/libs-qwen35"
mkdir -p "$LIBS"
python -m pip install --quiet --no-deps --upgrade --target "$LIBS" triton==3.8.0
PYTHONPATH="$LIBS:$PYTHONPATH" python - "$LIBS" <<'PY'
import sys, triton, torch
libs = sys.argv[1]
if not triton.__file__.startswith(libs):
    sys.exit(f"[err] triton comes from {triton.__file__}, not {libs}")
print(f"[ok]  triton {triton.__version__} from {libs}; torch {torch.__version__} still the module's")
PY
echo "In a Qwen3.5 job: FORGE_EXTRA_LIBS=$LIBS"
