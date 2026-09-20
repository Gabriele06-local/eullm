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
# On a cluster, run it through the batch system rather than on a login node.
# "CPU only" means no GPU; it does not mean small. Converting a 4 B model
# holds gigabytes of tensors in memory and writes an 8 GB file, and a login
# node's per-user memory is whatever is left after everyone else — the
# conversion was OOM-killed there at 91%, and the merge before it at the same
# place. A serial allocation is a request, not a share of the leftovers:
#
#   srun --account=<acct> --partition=lrd_all_serial \
#        --cpus-per-task=4 --mem=30G --time=02:00:00 \
#        bash forge/scripts/quantize_to_gguf.sh <hf-model-dir> [output-dir]
#
# 30G and 4 cores are not arbitrary and should not be raised casually:
# lrd_all_serial enforces QOSMaxMemoryPerUser, and --mem=64G was rejected
# outright. These are the values sbatch_quantize.slurm already uses because
# they are known to be accepted.
#
# forge/scripts/leonardo/sbatch_export_gguf.slurm does this unattended on a
# cadence and is the better answer for anything recurring.
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

# Both heavy steps write to a .partial and rename only on success, because
# "the file exists" is the only thing the skip logic above can see and a file
# can exist while being wrong.
#
# Measured: the F16 conversion was OOM-killed on a login node at 91%, leaving
# 7.3 GB of an 8.05 GB file behind. Nothing reported an error afterwards — the
# next run would have said "F16 GGUF already at …, skipping conversion" and
# quantized a truncated model. A rename is atomic on the same filesystem, so
# what the skip logic sees is either a complete file or no file.
if [ -f "$F16_FILE" ]; then
    log "F16 GGUF already at $F16_FILE — skipping conversion"
else
    rm -f "$F16_FILE.partial"
    log "converting HF → GGUF F16 ($F16_FILE)"
    log "8 GB of tensors through a CPU: this wants the serial partition, not a"
    log "login node. See the header of this script if it gets killed."
    python3 "$LCPP_DIR/convert_hf_to_gguf.py" "$HF_DIR" \
        --outfile "$F16_FILE.partial" \
        --outtype f16
    mv -f "$F16_FILE.partial" "$F16_FILE"
    ok "F16 GGUF written ($(du -h "$F16_FILE" | cut -f1))"
fi

# ---------------------------------------------------------------------------
# 4. Quantize F16 → Q4_K_M
# ---------------------------------------------------------------------------

if [ -f "$QUANT_FILE" ]; then
    log "$QUANT_TYPE GGUF already at $QUANT_FILE — skipping quantization"
else
    rm -f "$QUANT_FILE.partial"
    log "quantizing F16 → ${QUANT_TYPE} ($QUANT_FILE)"
    "$LCPP_DIR/build/bin/llama-quantize" \
        "$F16_FILE" "$QUANT_FILE.partial" "${QUANT_TYPE}"
    mv -f "$QUANT_FILE.partial" "$QUANT_FILE"
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

# Getting out of conversation mode, three ways, because one is not reliable.
#
# The passthrough template fixes what the model is *fed*; it does not stop
# llama-cli sitting at a "> " prompt afterwards waiting for a second turn —
# a hang in a script, and burnt walltime in a batch job. Redirecting stdin
# from /dev/null was not enough on build b1-3cf0325: it kept waiting anyway,
# which is what an interactive UI reading the terminal rather than stdin does.
#
# Worse than not exiting: with stdin at /dev/null and no single-turn flag it
# spins. It generates the answer, returns to the "> " prompt, reads EOF,
# prints the prompt again, reads EOF again — thousands of bare "> " lines a
# second. So /dev/null alone is not a safety net, it is a busy loop, and the
# flag and the timeout below are what actually end the process.
#
# So: ask the binary which flag it has instead of guessing (the name has been
# -no-cnv, --no-conversation and -st across versions, and passing one a build
# does not know is a hard error), and put a timeout around the whole thing so
# the worst case is a slow smoke test rather than a job that never ends. A
# timeout is not a failure here — the model has already loaded and generated
# by then, which is all this step claims to check.
smoke_help="$("$LCPP_DIR/build/bin/llama-cli" --help 2>&1 || true)"
smoke_noconv=()
for flag in --single-turn --no-conversation -no-cnv; do
    if printf '%s' "$smoke_help" | grep -qe "$flag"; then
        smoke_noconv=("$flag")
        log "single-turn flag for this build: $flag"
        break
    fi
done
[ ${#smoke_noconv[@]} -gt 0 ] || \
    log "no single-turn flag in this build's --help; relying on the timeout"

# Output to a file, not through `head`. Piping into `head` closes the pipe
# after N lines, and against the prompt loop above that kills llama-cli with
# SIGPIPE — exit 141, which the check below would have reported as "the GGUF
# is malformed". A false alarm on a healthy model is the exact failure this
# whole smoke test keeps producing, so there is no pipe to raise it.
SMOKE_OUT="$(mktemp "${TMPDIR:-/tmp}/eullm-smoke-XXXXXX.txt")"
trap 'rm -f "$SMOKE_TMPL" "$SMOKE_OUT"' EXIT

smoke_rc=0
timeout "${SMOKE_TIMEOUT:-180}" \
    "$LCPP_DIR/build/bin/llama-cli" \
    -m "$QUANT_FILE" \
    --chat-template-file "$SMOKE_TMPL" \
    ${smoke_noconv[@]+"${smoke_noconv[@]}"} \
    -p "Articolo 2086 del codice civile italiano: " \
    -n 128 -t 4 --temp 0.7 --top-p 0.95 --no-display-prompt \
    < /dev/null > "$SMOKE_OUT" 2>&1 || smoke_rc=$?

# The bare prompts are the loop, not output, and there can be tens of
# thousands of them. Drop them before showing anything, or they bury the one
# thing worth reading.
grep -v '^> *$' "$SMOKE_OUT" | head -40 || true

case "$smoke_rc" in
    0) ;;
    124) log "smoke prompt hit the ${SMOKE_TIMEOUT:-180}s timeout: the model"
         log "loaded and generated, llama-cli just would not exit. The GGUF is fine." ;;
    141) log "llama-cli exited on SIGPIPE, not a model problem" ;;
    *)   # Only now is it worth calling it broken — and only if nothing was
         # generated. A non-zero exit *after* real output is llama-cli's
         # problem; an empty file is the GGUF's.
         if [ -s "$SMOKE_OUT" ] && grep -qi 'llm_load\|load_tensors\|llama_model_load' "$SMOKE_OUT"; then
             log "llama-cli exited $smoke_rc after loading the model — see above"
         else
             err "smoke prompt failed (exit $smoke_rc) — the GGUF is malformed"
         fi ;;
esac

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
