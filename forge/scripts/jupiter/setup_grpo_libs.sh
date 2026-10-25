#!/usr/bin/env bash
# TRL for the GRPO jobs (sbatch_grpo.slurm), kept out of the Forge venv, as on
# Leonardo (forge/scripts/leonardo/setup_grpo_libs.sh). On a login node
# (compute nodes have no internet):
#
#     eullm && bash forge/scripts/jupiter/setup_grpo_libs.sh
#
# TRL 1.16.0, the latest on PyPI (checked 25-10-2026; Leonardo's rounds ran
# 1.14.1), pure Python, installed with --no-deps into $FORGE_HOME/libs-grpo so
# pip cannot bring a torch or a transformers along: everything it needs is in
# the venv already (datasets 5.1.0 >= 4.7, accelerate, transformers 5.19,
# pandas, pyarrow). Only sbatch_grpo.slurm puts the directory on PYTHONPATH,
# after libs-qwen35 (Triton 3.8.0 for Qwen3.5's kernels). The last step
# imports GRPOTrainer against the venv's torch and transformers, with
# libs-qwen35 in front as the job has it: if that fails, nothing else changed.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
[ -z "${SLURM_JOB_ID:-}" ] || { echo "[err] run this on a login node" >&2; exit 1; }
TRL_VERSION="${TRL_VERSION:-1.16.0}"
LIBS="${GRPO_LIBS:-$FORGE_HOME/libs-grpo}"
QWEN35="${FORGE_EXTRA_LIBS:-$FORGE_HOME/libs-qwen35}"
mkdir -p "$LIBS"
echo "[grpo-libs] trl==$TRL_VERSION into $LIBS (no dependencies)"
python -m pip install --quiet --no-deps --upgrade --target "$LIBS" "trl==$TRL_VERSION"
PYTHONPATH="$LIBS:$QWEN35:$PYTHONPATH" python - "$LIBS" <<'PY'
import sys
import datasets, torch, transformers, trl
from trl import GRPOConfig, GRPOTrainer  # noqa: F401
if not trl.__file__.startswith(sys.argv[1]):
    sys.exit(f"[err] trl comes from {trl.__file__}, not {sys.argv[1]}")
print(f"[ok]  trl {trl.__version__}, datasets {datasets.__version__}, "
      f"torch {torch.__version__}, transformers {transformers.__version__}")
PY
echo "In a GRPO job: GRPO_LIBS=$LIBS (the default)"
