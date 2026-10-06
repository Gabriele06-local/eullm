#!/usr/bin/env bash
# Test D: does llama.cpp's own MTP pay on an MoE whose experts are in RAM,
# now that they are pinned and the expert cache holds the ones used most?
#
#   bench/mtp_test_d.sh LLAMA_SERVER MODEL.gguf [MORE LLAMA-SERVER FLAGS...]
#
# MODEL is an MTP GGUF of an MoE (unsloth's Qwen3.6-35B-A3B-MTP, say), and
# LLAMA_SERVER a llama-server built from the llama.cpp EuLLM pins, which
# carries the expert cache (PR #29887): engine/vendor/llama-cpp-rs/
# llama-cpp-sys-2/llama.cpp, built with -DGGML_CUDA=ON. The script starts it
# once per entry of DRAFTS (default "0 1 2": no drafts, then 1 and 2 MTP drafts
# per step), every time with all the experts in RAM (--cpu-moe), read into
# memory rather than mapped (--load-mode none, which puts them in pinned
# memory) and a VRAM cache of CACHE_MIB of them (default 8000), and measures
# each with bench/speed_check.py: its story and a piece of code, at
# TEMPERATURE (default 0). One line per setting, as bench/mtp_sweep.sh prints
# for EuLLM; each server's log stays in $OUT.
#
# What it decides (docs/roadmap-engine-0.7-1.0.md, 0.8-Z2): if drafting gains
# here, MTP on an MoE with its experts in RAM is worth measuring in EuLLM
# (bench/mtp_sweep.sh with --moe-cache auto); if it loses, it waits for phase
# 3 of docs/moe-offload-plan.md, where the CPU computes the experts a check
# adds instead of copying them.
set -u
export LC_ALL=C

SERVER=$1
MODEL=$2
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-11520}
SPEED_CHECK=${SPEED_CHECK:-$HERE/speed_check.py}
OUT=${OUT:-$HOME/work/mtp-test-d}
DRAFTS=${DRAFTS:-"0 1 2"}
CACHE_MIB=${CACHE_MIB:-8000}
CTX=${CTX:-8192}
TEMPERATURE=${TEMPERATURE:-0}
CODE="Write a Python function that parses an ISO 8601 date string into a datetime, with a docstring, type hints and three unit tests."

if ! python3 "$SPEED_CHECK" --help 2>/dev/null | grep -q -- --temperature; then
    echo "$SPEED_CHECK is missing or too old (no --temperature): use bench/speed_check.py" >&2
    exit 1
fi
if [[ ! -f $MODEL ]]; then
    echo "no model at $MODEL" >&2
    exit 1
fi
if ! "$SERVER" --help 2>/dev/null | grep -q -- --moe-cache-mib; then
    echo "$SERVER has no --moe-cache-mib: build llama-server from the llama.cpp EuLLM pins" >&2
    exit 1
fi
pid=
trap '[[ -n $pid ]] && kill "$pid" 2>/dev/null' EXIT
trap 'exit 130' INT TERM

mkdir -p "$OUT"
speed() { # a speed_check.py run: "tokens/s kept drafted answer"; extra arguments go to it
    python3 "$SPEED_CHECK" --url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
        --prompt-tokens 1000 --temperature "$TEMPERATURE" "$@" |
        awk '/writes answers/ {w = $3} /answer text/ {h = $3}
             /drafts kept/ {k = substr($4, 2); d = substr($6, 1, length($6) - 1)}
             END {print w, k + 0, d + 0, h}'
}

# The last column is the two answers' text, hashed: two servers, or two
# settings, wrote the same answers when it is the same.
printf '%-16s %12s %12s %12s  %s\n' setting story_tok/s code_tok/s drafts_kept answers
for n in $DRAFTS; do
    spec=()
    [[ $n != 0 ]] && spec=(--spec-type draft-mtp --spec-draft-n-max "$n")
    log="$OUT/llama-server-mtp$n.log"
    # One slot, as EuLLM's default: speculative checks then never share a batch.
    "$SERVER" -m "$MODEL" --port "$PORT" -c "$CTX" -np 1 -ngl 99 \
        --cpu-moe --moe-cache-mib "$CACHE_MIB" --load-mode none "${spec[@]}" "$@" \
        >"$log" 2>&1 &
    pid=$!
    # Reading 20-35 GB of experts into memory takes a while.
    for _ in $(seq 1 600); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
        echo "llama-server did not come up with $n drafts; the end of $log:" >&2
        tail -n 15 "$log" >&2
        exit 1
    fi
    read -r story story_kept story_drafted story_text <<<"$(speed)"
    read -r code code_kept code_drafted code_text <<<"$(speed --write-prompt "$CODE")"
    kill "$pid"
    wait "$pid" 2>/dev/null
    pid=
    kept=$(awk -v k=$((story_kept + code_kept)) -v d=$((story_drafted + code_drafted)) \
        'BEGIN {if (d) printf "%.0f%%", 100 * k / d; else print "-"}')
    printf '%-16s %12s %12s %12s  %s\n' "drafts $n" "${story:-?}" "${code:-?}" "$kept" \
        "${story_text:-?}/${code_text:-?}"
done
