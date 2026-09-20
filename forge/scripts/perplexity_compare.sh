#!/usr/bin/env bash
# Is the distilled student better than the base it came from? Measure both.
#
# A perplexity number on its own says nothing. "5.65" is not good or bad; it
# is a property of the corpus as much as of the model, and the only question
# worth asking — did distillation move anything on Italian legal text? — needs
# the *same* number from the untouched base model, on the *same* corpus, with
# the *same* chunk count and the *same* quantization. This script exists so
# that comparison is one command instead of four, because a measurement that
# takes four commands is one nobody repeats.
#
# Two rules it enforces rather than documents:
#
#   * Identical flags on both sides. llama-perplexity's result depends on
#     --chunks and the context size, so a student measured over 340 chunks
#     against a base measured over 40 is not a comparison.
#   * Same quantization on both sides. A Q4_K_M student against an F16 base
#     mixes distillation with quantization loss and attributes the sum to
#     whichever you were hoping for.
#
# Usage:
#   bash forge/scripts/perplexity_compare.sh \
#       --student $WORK/gguf/legal-it-4b-step8400/legal-it-4b-step8400-q4_k_m.gguf \
#       --base    $WORK/gguf/qwen3-4b-base/qwen3-4b-base-q4_k_m.gguf \
#       --corpus  $WORK/eullm-data/val_sample.txt \
#       --chunks  40
#
# The base GGUF, if you do not have one yet, is the same pipeline applied to
# the untouched model — it is already in the HF cache, so this is a conversion
# and a quantization, no download:
#   bash forge/scripts/quantize_to_gguf.sh <path-to-Qwen3-4B-Base> $WORK/gguf/qwen3-4b-base

set -euo pipefail

STUDENT=""
BASE=""
CORPUS=""
CHUNKS="${EULLM_PPL_CHUNKS:-40}"
# Follow the allocation, do not assume it.
#
# This defaulted to a flat 8, and under `srun --cpus-per-task=4` that ran
# llama-perplexity with n_threads=8 on four cores for two twenty-minute
# measurements. Oversubscribing a compute-bound matmul does not share nicely;
# it just adds context switching to work that was already saturating the
# cores. SLURM_CPUS_PER_TASK is what the job was actually given, so it wins,
# and nproc is the fallback outside a job.
THREADS="${EULLM_PPL_THREADS:-${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 4)}}"
CTX="${EULLM_PPL_CTX:-512}"
LCPP_DIR="${LCPP_DIR:-${WORK:-$HOME}/llama.cpp}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*" >&2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --student) STUDENT="$2"; shift 2;;
        --base)    BASE="$2";    shift 2;;
        --corpus)  CORPUS="$2";  shift 2;;
        --chunks)  CHUNKS="$2";  shift 2;;
        --threads) THREADS="$2"; shift 2;;
        --ctx)     CTX="$2";     shift 2;;
        -h|--help) sed -n '2,30p' "$0"; exit 0;;
        *) err "unknown argument: $1";;
    esac
done

[ -n "$STUDENT" ] || err "--student <gguf> is required"
[ -n "$BASE" ]    || err "--base <gguf> is required (see --help for how to build it)"
[ -n "$CORPUS" ]  || err "--corpus <text file> is required"
[ -f "$STUDENT" ] || err "no such file: $STUDENT"
[ -f "$BASE" ]    || err "no such file: $BASE"
[ -f "$CORPUS" ]  || err "no such file: $CORPUS"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/eullm-ppl-XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

PPL_BIN="$LCPP_DIR/build/bin/llama-perplexity"
[ -x "$PPL_BIN" ] || err "llama-perplexity not built at $PPL_BIN
      Run forge/scripts/quantize_to_gguf.sh once — it builds this target."

# Quantization mismatch is silent and ruins the comparison, so say it out loud
# before spending the time rather than after reading the numbers.
quant_of() { basename "$1" | sed -n 's/.*-\(f16\|bf16\|q[0-9][^.]*\)\.gguf$/\1/p'; }
qs="$(quant_of "$STUDENT")"; qb="$(quant_of "$BASE")"
if [ -n "$qs" ] && [ -n "$qb" ] && [ "$qs" != "$qb" ]; then
    printf '\033[33m[warn]\033[0m student is %s and base is %s. The difference\n' "$qs" "$qb" >&2
    printf '       below will be distillation AND quantization mixed together.\n' >&2
fi

# llama-perplexity's last "Final estimate: PPL = X" line is the result; the
# per-chunk running values printed before it are not.
# Progress goes to the terminal while it runs, not into a variable.
#
# Capturing the output to parse the final PPL out of it also swallowed the
# running per-chunk estimates and the ETA that llama-perplexity prints. Each
# model takes ten to fifteen minutes, so that produced half an hour of total
# silence — indistinguishable from a hung job, and it got asked as exactly
# that. The output now goes through a file: progress is echoed to stderr as
# it arrives, and the file is what gets parsed afterwards.
measure() {
    local label="$1" model="$2" rc=0
    local raw="$WORKDIR/$label.log"
    log "$label: $CHUNKS chunks, ctx $CTX — $(basename "$model")"
    "$PPL_BIN" -m "$model" -f "$CORPUS" \
        --chunks "$CHUNKS" -c "$CTX" -t "$THREADS" \
        > "$raw" 2>&1 &
    local pid=$!
    tail -f --pid="$pid" -n +1 "$raw" >&2 &
    local tailpid=$!
    wait "$pid" || rc=$?
    wait "$tailpid" 2>/dev/null || true
    [ "$rc" = 0 ] || err "$label: llama-perplexity failed (exit $rc) — see above"
    sed -n 's/.*Final estimate: PPL = \([0-9.]*\).*/\1/p' "$raw" | tail -1
}

ppl_base="$(measure base "$BASE")"
ppl_student="$(measure student "$STUDENT")"

[ -n "$ppl_base" ]    || err "could not parse a final PPL for the base model"
[ -n "$ppl_student" ] || err "could not parse a final PPL for the student"

cat <<EOF

================================================================================
 Perplexity — $(basename "$CORPUS"), $CHUNKS chunks, ctx $CTX
   base     $(basename "$BASE")
              PPL = $ppl_base
   student  $(basename "$STUDENT")
              PPL = $ppl_student
EOF

awk -v b="$ppl_base" -v s="$ppl_student" '
BEGIN {
    d = (b - s) / b * 100
    printf "   delta    %+.2f%% ", d
    if      (d >  0.005) print "(student is better on this corpus)"
    else if (d < -0.005) print "(student is worse on this corpus)"
    else                 print "(no measurable difference)"
}'

cat <<'EOF'

 Read it carefully: lower is better, and a few percent on one corpus is not a
 verticalized model. What this answers is whether the run is going the right
 way — which is the question a checkpoint is exported to answer.
================================================================================
EOF
