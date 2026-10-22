#!/usr/bin/env bash
# The vertical (sbatch_vertical.slurm) and the judge behind it, from a login
# node: checked first, nothing submitted when a check fails, and a partition
# down for maintenance or a silent controller waited out (sbatch_retry.sh).
#
#     eullm && [VERT_ARMS="p-m8i p-q4b"] bash forge/scripts/jupiter/submit_vertical.sh
#
# FORGE_LINK_H sets the job's hours (default 4).

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
# shellcheck source=sbatch_retry.sh
source "$HERE/sbatch_retry.sh"
REPO="$(cd "$HERE/../../.." && pwd)"
SFT="${VERT_SFT:-$FORGE_RUN_DIR/sft-ob1}"
EULLM_BIN="${EULLM_BIN:-$PROJECT/$USER/eullm-target/sites/release/eullm}"
RT="${VERT_RUNTIMES:-$PROJECT/$USER/campaign/runtimes}"
LCPP="$REPO/engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp"
bad=0
for arm in ${VERT_ARMS:-p-m8i p-q4b}; do
    [ -f "$SFT/$arm/merged/config.json" ] && echo "[ok] $arm: $SFT/$arm/merged" ||
        { echo "[!!] no merged model at $SFT/$arm/merged"; bad=1; }
done
for t in llama-quantize llama-server; do
    [ -x "$RT/$t" ] && echo "[ok] $RT/$t" ||
        { echo "[!!] no $RT/$t: bash tools/jupiter/install_runtimes.sh"; bad=1; }
done
[ -x "$EULLM_BIN" ] && echo "[ok] engine $("$EULLM_BIN" --version 2>&1 | head -1)" ||
    { echo "[!!] no engine at $EULLM_BIN"; bad=1; }
[ -f "$LCPP/convert_hf_to_gguf.py" ] && echo "[ok] llama.cpp $(git -C "$LCPP" rev-parse --short HEAD 2>/dev/null)" ||
    { echo "[!!] no llama.cpp submodule: git submodule update --init --recursive"; bad=1; }
python -c "import sys; sys.path.insert(1, sys.argv[1]); import gguf, sentencepiece" "$LCPP/gguf-py" 2>/dev/null &&
    echo "[ok] gguf-py imports" || { echo "[!!] llama.cpp's gguf-py or sentencepiece does not import in the venv: setup_venv.sh"; bad=1; }
[ "$bad" = 0 ] || { echo "[!!] nothing submitted"; exit 1; }
wait_for_slurm || exit 1
[ -z "$(squeue -h -u "$USER" -n forge-vertical -o %i)" ] ||
    { echo "[!!] a forge-vertical job is already queued"; exit 1; }

mkdir -p "$FORGE_RUN_DIR/logs"
v=$(cd "$REPO" && sbatch_retry --parsable --account "$SBATCH_ACCOUNT" \
    --time="${FORGE_LINK_H:-4}:00:00" --output="$FORGE_RUN_DIR/logs/vertical-%j.out" \
    forge/scripts/jupiter/sbatch_vertical.slurm) || { echo "[!!] sbatch failed"; exit 1; }
j=$(cd "$REPO" && sbatch_retry --parsable --account "$SBATCH_ACCOUNT" --dependency="afterany:${v%%;*}" \
    --output="$FORGE_RUN_DIR/logs/judge-%j.out" forge/scripts/jupiter/sbatch_judge.slurm) ||
    { echo "[!!] sbatch of the judge failed"; exit 1; }
echo "[ok] vertical ${v%%;*}, then the judge ${j%%;*}"
echo "     $FORGE_RUN_DIR/logs/vertical-${v%%;*}.out, $FORGE_RUN_DIR/logs/judge-${j%%;*}.out"
