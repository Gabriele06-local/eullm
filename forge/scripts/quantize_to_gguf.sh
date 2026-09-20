#!/usr/bin/env bash
# Phase 3 — Convert the distilled student to GGUF Q4_K_M and smoke-test it.
#
# Inputs:
#   1) HF model directory (the final --output-dir of distill.py),
#      containing config.json, tokenizer.json, and a .safetensors weight
#      file. Must be in standard HuggingFace format (not LoRA adapter).
#   2) Output directory for the GGUF file.
#
# Steps:
#   * clone llama.cpp (CPU build, no GPU needed for conversion +
#     quantization) into ~/llama.cpp on first run
#   * build the conversion + quantize binaries
#   * convert HF → GGUF F16 (full precision), ~14 GB for a 7B model
#   * quantize F16 → Q4_K_M, ~4.5 GB (Q4_K_M is the sweet spot for
#     legal text quality vs file size)
#   * run a smoke prompt through llama-cli to confirm the GGUF loads
#
# Usage:
#   bash forge/scripts/quantize_to_gguf.sh \
#       <hf-model-dir> [output-dir]
#
# Example:
#   bash forge/scripts/quantize_to_gguf.sh \
#       ~/checkpoints/qwen3_7b_legal_it_distilled \
#       ~/gguf/legal-it-7b
#
# After this completes, the file at <output-dir>/${GGUF_NAME}-${QUANT_TYPE}.gguf
# can be loaded by the EULLM Engine, Ollama, or any llama.cpp-compatible
# runtime.

set -euo pipefail

HF_DIR="${1:?Usage: $0 <hf-model-dir> [output-dir]}"
OUT_DIR="${2:-$HF_DIR/gguf}"
# $WORK when it exists, because $HOME on Leonardo is 50 GB and a llama.cpp
# checkout plus its build objects is several of them. Running this without
# sourcing env.sh first put the clone in $HOME, which is a quota away from
# failing halfway through a link step.
if [ -n "${WORK:-}" ] && [ -d "${WORK:-}" ]; then
    LCPP_DIR="${LCPP_DIR:-$WORK/llama.cpp}"
else
    LCPP_DIR="${LCPP_DIR:-$HOME/llama.cpp}"
fi
LCPP_REPO="${LCPP_REPO:-https://github.com/ggerganov/llama.cpp.git}"
# Named after the directory being converted unless told otherwise. The old
# default was the literal string "legal-it-7b", which outlived the 7 B target
# and would have stamped the same wrong name on every model in a series.
GGUF_NAME="${GGUF_NAME:-$(basename "$HF_DIR")}"
QUANT_TYPE="${QUANT_TYPE:-q4_k_m}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }

# std::filesystem moved into libstdc++ proper in GCC 9. Before that it lives in
# a separate -lstdc++fs that llama.cpp does not link, and the build gets all
# the way to the final link before dying in a wall of "undefined reference to
# std::filesystem::__cxx11::path::parent_path()". That is five minutes of
# compiling to learn something `g++ -dumpversion` answers instantly, and the
# error names templates and symbols rather than the actual problem.
check_compiler() {
    command -v g++ >/dev/null 2>&1 || err "g++ not found — try 'module load gcc'"
    local v
    v="$(g++ -dumpversion 2>/dev/null | cut -d. -f1)"
    case "$v" in
        ''|*[!0-9]*) return 0;;   # unreadable: let the build speak for itself
    esac
    if [ "$v" -lt 9 ]; then
        err "g++ $v is too old: std::filesystem is not in libstdc++ before GCC 9,
      and the build fails at the final link with hundreds of undefined
      references to std::filesystem. Load a newer compiler first:
          module avail gcc
          module load gcc/<11 or newer>
      then run this again."
    fi
}
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

# Before the clone, not after it: a 36 MB checkout and five minutes of
# compiling are a poor way to discover something `g++ -dumpversion` answers
# instantly, and the link error that follows names templates rather than the
# problem. Skipped when the binaries already exist — a rebuilt toolchain is
# not needed to reuse one.
if [ ! -x "$LCPP_DIR/build/bin/llama-quantize" ]; then
    check_compiler
fi

[ -d "$HF_DIR" ]                || err "HF model dir not found: $HF_DIR"
[ -f "$HF_DIR/config.json" ]    || err "missing $HF_DIR/config.json"

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# 1. Clone or update llama.cpp
# ---------------------------------------------------------------------------

if [ -d "$LCPP_DIR/.git" ]; then
    if [ "${LCPP_SKIP_UPDATE:-0}" = "1" ]; then
        # Offline mode (e.g. HPC compute nodes without outbound network):
        # use the checkout as-is. Clone/update it on a login node first.
        log "llama.cpp at $LCPP_DIR — LCPP_SKIP_UPDATE=1, using as-is"
    else
        log "llama.cpp already at $LCPP_DIR — pulling latest"
        git -C "$LCPP_DIR" pull --quiet --ff-only
    fi
else
    [ "${LCPP_SKIP_UPDATE:-0}" != "1" ] || \
        err "LCPP_SKIP_UPDATE=1 but no checkout at $LCPP_DIR — clone it on a login node first"
    log "cloning llama.cpp into $LCPP_DIR"
    git clone --depth 1 "$LCPP_REPO" "$LCPP_DIR"
fi

# ---------------------------------------------------------------------------
# 2. Build (CPU only is enough for conversion + quantization + smoke)
# ---------------------------------------------------------------------------

# llama-perplexity is in the list because comparing perplexity against the
# untouched base model is the only quantitative read on whether distillation
# moved anything. Eyeballing a completion is not a measurement, and a build
# that omits the tool makes the measurement something nobody does.
if [ ! -f "$LCPP_DIR/build/bin/llama-quantize" ] || \
   [ ! -f "$LCPP_DIR/build/bin/llama-cli" ] || \
   [ ! -f "$LCPP_DIR/build/bin/llama-perplexity" ]; then
    # CMake caches the compiler it configured with, so a build directory left
    # behind by a failed attempt keeps using that compiler no matter what is
    # loaded now. `module load gcc/12` then re-running would have rebuilt with
    # the GCC 8 recorded in CMakeCache.txt and failed identically — the second
    # time looking like the fix did not work, rather than like a stale cache.
    cache="$LCPP_DIR/build/CMakeCache.txt"
    if [ -f "$cache" ]; then
        cached_cxx="$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$cache" | head -1)"
        current_cxx="$(command -v g++ 2>/dev/null)"
        if [ -n "$cached_cxx" ] && [ -n "$current_cxx" ] &&
           [ "$cached_cxx" != "$current_cxx" ]; then
            log "compiler changed ($cached_cxx -> $current_cxx) — clearing the"
            log "stale CMake cache so the new one is actually used"
            rm -rf "$LCPP_DIR/build"
        fi
    fi
    log "building llama.cpp (this takes 2-5 min on first run)"
    cmake -S "$LCPP_DIR" -B "$LCPP_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF \
        >/dev/null
    cmake --build "$LCPP_DIR/build" --config Release \
        --target llama-quantize llama-cli llama-perplexity \
        -j "${LCPP_BUILD_JOBS:-8}" \
        >/dev/null
    ok "llama.cpp built"
fi

# Make sure the conversion script has its Python deps
log "ensuring HF→GGUF Python deps installed"
python3 -m pip install --quiet --upgrade \
    "transformers>=4.45" "sentencepiece" "gguf>=0.10" "protobuf>=4" \
    "torch>=2.1" "numpy"

# ---------------------------------------------------------------------------
# 3. Convert HF → GGUF F16 (full precision)
# ---------------------------------------------------------------------------

F16_FILE="$OUT_DIR/${GGUF_NAME}-f16.gguf"
QUANT_FILE="$OUT_DIR/${GGUF_NAME}-${QUANT_TYPE}.gguf"

# "Already there, skipping" is the right default and also the way a fix
# silently fails to apply. Re-exporting the HF directory and re-running this
# script reused the F16 built from the *previous* export, so the corrected
# model was never converted and the smoke test reproduced the old output
# exactly — which reads as "the fix did nothing" rather than "nothing ran".
# Compare mtimes: if anything in the HF directory is newer than the GGUF, the
# GGUF describes a model that no longer exists.
newest_in() { find "$1" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1; }
if [ -f "$F16_FILE" ]; then
    src_t="$(newest_in "$HF_DIR")"
    gguf_t="$(stat -c %Y "$F16_FILE" 2>/dev/null || echo 0)"
    if [ -n "$src_t" ] && awk "BEGIN{exit !($src_t > $gguf_t)}"; then
        log "$HF_DIR is newer than $F16_FILE — the existing GGUFs were built"
        log "from an older export; reconverting rather than reusing them"
        rm -f "$F16_FILE" "$QUANT_FILE"
    fi
fi

if [ -f "$F16_FILE" ]; then
    log "F16 GGUF already at $F16_FILE — skipping conversion"
else
    log "converting HF → GGUF F16 ($F16_FILE)"
    python3 "$LCPP_DIR/convert_hf_to_gguf.py" "$HF_DIR" \
        --outfile "$F16_FILE" \
        --outtype f16
    ok "F16 GGUF written ($(du -h "$F16_FILE" | cut -f1))"
fi

# ---------------------------------------------------------------------------
# 4. Quantize F16 → Q4_K_M
# ---------------------------------------------------------------------------

if [ -f "$QUANT_FILE" ]; then
    log "$QUANT_TYPE GGUF already at $QUANT_FILE — skipping quantization"
else
    log "quantizing F16 → ${QUANT_TYPE} ($QUANT_FILE)"
    "$LCPP_DIR/build/bin/llama-quantize" \
        "$F16_FILE" "$QUANT_FILE" "${QUANT_TYPE}"
    ok "${QUANT_TYPE} GGUF written ($(du -h "$QUANT_FILE" | cut -f1))"
fi

# ---------------------------------------------------------------------------
# 5. Smoke prompt — confirm the model loads and replies in Italian
# ---------------------------------------------------------------------------

log "smoke prompt to verify the GGUF loads correctly"

# The prompt has to reach the model verbatim, and by default it does not.
#
# Recent llama.cpp switches llama-cli into conversation mode on its own as
# soon as the model carries a chat template, and then wraps whatever you
# passed to -p in <|im_start|>user … <|im_end|><|im_start|>assistant. A
# completion model distilled from Qwen3-4B-*Base* has never seen that format,
# so it answers with loops, echoes and stray subword tokens — and the smoke
# test reports a broken model when the model is fine.
#
# That is exactly what happened on the step-8400 export: through the template
# the output was "…: ictures" followed by timestamps; fed the same prompt
# verbatim, the same file continued into correct Italian legal prose. So the
# smoke test overrides the template with a passthrough one rather than
# trusting the default. (-no-cnv and --in-prefix are not accepted by every
# build; --chat-template-file is.)
SMOKE_TMPL="$(mktemp "${TMPDIR:-/tmp}/eullm-passthrough-XXXXXX.jinja")"
trap 'rm -f "$SMOKE_TMPL"' EXIT
printf '%s' '{% for m in messages %}{{ m.content }}{% endfor %}' > "$SMOKE_TMPL"

"$LCPP_DIR/build/bin/llama-cli" \
    -m "$QUANT_FILE" \
    --chat-template-file "$SMOKE_TMPL" \
    -p "Articolo 2086 del codice civile italiano: " \
    -n 128 -t 4 --temp 0.7 --top-p 0.95 --no-display-prompt \
    < /dev/null 2>/dev/null | head -20 \
    || err "smoke prompt failed — the GGUF is malformed"
# stdin from /dev/null, and that is load-bearing. The passthrough template
# fixes what the model is *fed*, but llama-cli still enters conversation mode
# and, after generating, sits at a "> " prompt waiting for a second turn —
# which in a script means hanging forever, and in a batch job means burning
# the walltime on an idle prompt. EOF on stdin ends the turn on every build,
# where the flag that does it (-no-cnv, -st, --single-turn) has changed names
# more than once and is not accepted by all of them.

cat <<EOF

================================================================================
 Phase 3 done.
   F16 GGUF:    $F16_FILE
   ${QUANT_TYPE} GGUF:  $QUANT_FILE

 Measure it before believing it. Perplexity alone is not a result — the same
 number on the untouched base model, same corpus, same --chunks, is what says
 whether distillation moved anything:
   $LCPP_DIR/build/bin/llama-perplexity -m "$QUANT_FILE" \\
       -f <corpus.txt> --chunks 40 -t 4

 Next: load into the EULLM Engine, or push to HuggingFace Hub:
   huggingface-cli upload eullm/${GGUF_NAME} "$QUANT_FILE" \\
       ${GGUF_NAME}-${QUANT_TYPE}.gguf
================================================================================
EOF
