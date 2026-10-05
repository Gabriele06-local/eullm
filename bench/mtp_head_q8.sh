#!/usr/bin/env bash
# Does the MTP head draft better at Q8_0? Two Q4_K_M quantizations of one MTP
# model, the same but for the MTP layer, and bench/mtp_sweep.sh on each.
#
#   bench/mtp_head_q8.sh EULLM_BINARY SOURCE.gguf [MORE SERVE FLAGS...]
#
# SOURCE is the MTP model at Q8_0 or BF16 (unsloth's Qwen3.5-9B-MTP Q8_0
# GGUF, say). llama-quantize writes Q4_K_M from it twice: as unsloth's
# Q4_K_M has it, with only the head's own tensors (nextn.*) at Q8_0 and the
# MTP layer's attention and FFN at Q4_K/Q6_K — llama-quantize alone would put
# the head's projection at Q4_K too — and with every tensor of the MTP layer
# at Q8_0: the last nextn_predict_layers blocks, read from the GGUF. Both
# files are measured by bench/mtp_sweep.sh with the same
# settings (MTP_SETTINGS, default "0 1 2 3") at TEMPERATURE (default 0); the
# drafts kept and the speed say whether a Q8_0 layer is worth the size it
# adds, printed first. The two GGUFs stay in $OUT for a rerun.
#
# LLAMA_QUANTIZE: llama-quantize, by default the one on PATH, else
# ~/llama.cpp/build/bin/llama-quantize.
set -u
export LC_ALL=C

BIN=$1
SOURCE=$2
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/work/mtp-head-q8}
export MTP_SETTINGS=${MTP_SETTINGS:-"0 1 2 3"}
export SPEED_CHECK=${SPEED_CHECK:-$HERE/speed_check.py}
QUANTIZE=${LLAMA_QUANTIZE:-$(command -v llama-quantize || echo "$HOME/llama.cpp/build/bin/llama-quantize")}

if [[ ! -x $QUANTIZE ]]; then
    echo "llama-quantize not found: set LLAMA_QUANTIZE" >&2
    exit 1
fi

# The MTP blocks: the last nextn_predict_layers of block_count, read with
# Forge's GGUF reader, which needs nothing beyond the standard library.
blocks=$(PYTHONPATH="$HERE/../forge" python3 - "$SOURCE" <<'EOF'
import sys
from eullm_forge.gguf_metadata import read_metadata
m = read_metadata(sys.argv[1])
arch = m["general.architecture"]
n, nextn = m.get(f"{arch}.block_count"), m.get(f"{arch}.nextn_predict_layers")
if not n or not nextn:
    sys.exit(f"{sys.argv[1]}: no MTP layer ({arch}.nextn_predict_layers)")
print("|".join(str(b) for b in range(n - nextn, n)))
EOF
) || exit 1
echo "MTP layer: blk.($blocks)"

mkdir -p "$OUT"
stem=$(basename "$SOURCE" .gguf)
plain="$OUT/$stem-Q4_K_M.gguf"
head="$OUT/$stem-Q4_K_M-mtp-Q8_0.gguf"
# --allow-requantize: a Q8_0 source is quantized already.
[[ -s $plain ]] || "$QUANTIZE" --allow-requantize --tensor-type "nextn\.=q8_0" \
    "$SOURCE" "$plain" Q4_K_M >"$OUT/quantize-plain.log" 2>&1 ||
    { echo "quantizing failed: $OUT/quantize-plain.log" >&2; exit 1; }
[[ -s $head ]] || "$QUANTIZE" --allow-requantize --tensor-type "blk\.($blocks)\.=q8_0" \
    "$SOURCE" "$head" Q4_K_M >"$OUT/quantize-head.log" 2>&1 ||
    { echo "quantizing failed: $OUT/quantize-head.log" >&2; exit 1; }
printf '%s: %s MiB\n%s: %s MiB\n' "$(basename "$plain")" $(($(stat -c %s "$plain") >> 20)) \
    "$(basename "$head")" $(($(stat -c %s "$head") >> 20))

for model in "$plain" "$head"; do
    echo
    echo "== $(basename "$model")"
    OUT="$OUT/sweep-$(basename "$model" .gguf)" "$HERE/mtp_sweep.sh" "$BIN" "$model" "$@"
done
