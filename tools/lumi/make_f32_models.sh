#!/usr/bin/env bash
# The F32 GGUFs the finetune points train, made on a login node: compute
# nodes have no network, and `eullm finetune` trains F32 weights only (no
# store or catalog holds those).
#
#   export SBATCH_ACCOUNT=project_465003366
#   bash tools/lumi/make_f32_models.sh [SPEC.json ...]
#
# Default spec: tools/lumi/campaigns/c05-finetune.json. For every model its
# finetune points name and $CAMPAIGN_DIR/f32 lacks (`campaign.py f32s`), the
# Hugging Face repo the spec's `f32` map names is downloaded — weights and
# tokenizer only — and converted by the converter of the engine's own
# llama.cpp submodule with --outtype f32, to a .partial renamed on success: a
# conversion killed half-way leaves no file a point could mistake for a model.
#
# The converter's Python packages go into a venv of their own,
# $CAMPAIGN_DIR/convert-venv, from the requirements file the submodule ships
# for it — never into a shared venv (forge/scripts/quantize_to_gguf.sh tells
# what an upgraded torch did to the one on Leonardo).
#
# Sizes in F32: Qwen3-0.6B-Base 2.4 GB, 1.7B 6.9 GB, 4B 16.1 GB. The converter
# streams tensors, but a login node may still kill the 4B. The download is
# kept when a conversion fails, so the same command finishes the job on a
# compute node, which needs no network for it:
#
#   srun -p small-g -A "$SBATCH_ACCOUNT" -t 1:00:00 -c 8 --mem=60G --gpus=1 \
#       bash tools/lumi/make_f32_models.sh
#
#   KEEP_HF=1   keep the downloaded checkpoints after converting them

set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "$(dirname "$0")/campaign_env.sh"

if [ $# -gt 0 ]; then
    SPECS=("$@")
else
    SPECS=("$EULLM_REPO/tools/lumi/campaigns/c05-finetune.json")
fi
F32_DIR="$CAMPAIGN_DIR/f32"
HF_DIR="$CAMPAIGN_DIR/hf"
VENV="$CAMPAIGN_DIR/convert-venv"
LCPP="$EULLM_REPO/engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp"
CONVERT="$LCPP/convert_hf_to_gguf.py"
REQUIREMENTS="$LCPP/requirements/requirements-convert_hf_to_gguf.txt"
# Hugging Face's client reports usage to its servers unless told not to.
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$CAMPAIGN_DIR/hf-home"

[ -f "$CONVERT" ] || {
    echo "[err] $CONVERT not found: git submodule update --init in $EULLM_REPO" >&2
    exit 1
}
mkdir -p "$F32_DIR" "$HF_DIR"

mapfile -t TODO < <($CAMPAIGN f32s "${SPECS[@]}" --queue "$CAMPAIGN_DIR")
if [ ${#TODO[@]} -eq 0 ]; then
    echo "every F32 model the specs train is in $F32_DIR"
    exit 0
fi

# The submodule pins a torch that needs Python 3.10 or newer.
if [ ! -x "$VENV/bin/python" ]; then
    VPY=""
    for c in python3.13 python3.12 python3.11 python3.10 "$PY"; do
        if command -v "$c" >/dev/null 2>&1 &&
           "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            VPY="$c"
            break
        fi
    done
    [ -n "$VPY" ] || { echo "[err] no Python >= 3.10 for the converter's venv" >&2; exit 1; }
    echo "=== converter venv: $VENV ($("$VPY" --version 2>&1)) ==="
    "$VPY" -m venv "$VENV" || exit 1
    if ! { "$VENV/bin/python" -m pip install --quiet --upgrade pip &&
           "$VENV/bin/python" -m pip install --quiet -r "$REQUIREMENTS"; }; then
        echo "[err] installing $REQUIREMENTS failed (a compute node has no network: run" \
             "this once on a login node)" >&2
        rm -rf "$VENV"
        exit 1
    fi
fi

FAILED=()
for line in "${TODO[@]}"; do
    read -r file repo <<<"$line"
    if [ "$repo" = "-" ]; then
        echo "[!!] $file: the spec's f32 map names no repo for it"
        FAILED+=("$file")
        continue
    fi
    src="$HF_DIR/${repo//\//__}"
    out="$F32_DIR/$file"
    echo
    echo "=== $file ← $repo ==="
    if [ ! -f "$src/.complete" ]; then
        "$VENV/bin/python" - "$repo" "$src" <<'EOF' || { FAILED+=("$file"); continue; }
import sys
from huggingface_hub import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo, local_dir=dest, allow_patterns=[
    "*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "tokenizer*"])
EOF
        touch "$src/.complete"
    fi
    rm -f "$out.partial"
    if "$VENV/bin/python" "$CONVERT" "$src" --outtype f32 --outfile "$out.partial"; then
        mv -f "$out.partial" "$out"
        echo "[ok] $out ($(du -h "$out" | cut -f1))"
        [ "${KEEP_HF:-0}" = "1" ] || rm -rf "$src"
    else
        rm -f "$out.partial"
        echo "[!!] converting $repo failed; the download stays in $src"
        FAILED+=("$file")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo
    echo "[!!] not made (their points block until they are; then: $CAMPAIGN unblock" \
         "--queue $CAMPAIGN_DIR):"
    printf '       %s\n' "${FAILED[@]}"
    exit 1
fi
$CAMPAIGN unblock --queue "$CAMPAIGN_DIR" >/dev/null 2>&1 || true
