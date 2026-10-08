#!/usr/bin/env bash
# Stock llama-server against one carrying EuLLM's prefetch patches (0003, 0004),
# on the same model, the same flags and the same long prompt.
#
#   bench/llama_server_compare.sh MODEL.gguf VARIANT=SERVER [VARIANT=SERVER ...]
#
# A VARIANT is NAME or NAME:SLOTS or NAME:SLOTS:bus; SERVER is a llama-server
# binary built from the same llama.cpp commit (the stock one, and one with the
# patches and the `--moe-prefetch` flag). SLOTS is passed as `--moe-prefetch`
# (0 or absent: no flag, which is all a stock server has); `:bus` sets
# LLAMA_MOE_PREFETCH_FROM_CACHE=0, so the experts the cache holds are copied
# over the bus as in patch 0003 alone. ORDER (default: the variants as given)
# lists the variants to run, so that running the same list in two orders tells
# an effect of the patches from one of running second.
#
# Every server runs with the experts in RAM (--cpu-moe), read into memory
# (--load-mode none, which pins them), a VRAM cache of CACHE_MIB of them, and
# micro-batches of N_UBATCH tokens. It is asked the same 33,200-token question
# twice at temperature 0 (the answers must match), then measured with
# bench/speed_check.py: reading that prompt, and writing an answer. One line
# per variant; each server's log and answers stay in $OUT.
set -u
export LC_ALL=C

MODEL=$1
shift
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-11540}
SPEED_CHECK=${SPEED_CHECK:-$HERE/speed_check.py}
OUT=${OUT:-$HOME/work/llama-server-compare}
CTX=${CTX:-40960}
CACHE_MIB=${CACHE_MIB:-5500}
N_UBATCH=${N_UBATCH:-2048}
PROMPT_TOKENS=${PROMPT_TOKENS:-33200}

if ! python3 "$SPEED_CHECK" --help 2>/dev/null | grep -q -- --temperature; then
    echo "$SPEED_CHECK is missing or too old (no --temperature): use bench/speed_check.py" >&2
    exit 1
fi
declare -A server
order=()
for arg in "$@"; do
    name=${arg%%=*}
    server[$name]=${arg#*=}
    order+=("$name")
done
if [[ ${#order[@]} -eq 0 ]]; then
    echo "usage: $0 MODEL.gguf VARIANT=SERVER ..." >&2
    exit 1
fi
pid=
trap '[[ -n $pid ]] && kill "$pid" 2>/dev/null' EXIT
trap 'exit 130' INT TERM

mkdir -p "$OUT"
PYTHONPATH="$HERE" python3 - "$PROMPT_TOKENS" "$MODEL" >"$OUT/question.json" <<'EOF' || exit 1
import json
import sys

from speed_check import document

tokens, model = int(sys.argv[1]), sys.argv[2]
question = "Which items did the auditor postpone? List the first ten item numbers."
print(json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": document(tokens, 20261004) + "\n\n" + question}],
    "max_tokens": 64,
    "temperature": 0,
    "cache_prompt": False,
    "stream": False,
}))
EOF

printf '%-18s %12s %12s %18s  %s\n' variant read_tok/s write_tok/s answers server_said
for v in ${ORDER:-${order[*]}}; do
    IFS=: read -r name slots mode <<<"$v"
    bin=${server[$name]:-}
    if [[ -z $bin ]]; then
        echo "no server for variant $name" >&2
        continue
    fi
    vars=()
    flags=()
    [[ ${slots:-0} != 0 ]] && flags=(--moe-prefetch "$slots")
    [[ ${mode:-} == bus ]] && vars=(LLAMA_MOE_PREFETCH_FROM_CACHE=0)
    tag=${v//:/-}
    log="$OUT/llama-server-$tag.log"
    env "${vars[@]}" "$bin" -m "$MODEL" --port "$PORT" -c "$CTX" -np 1 -ngl 99 -b "$N_UBATCH" -ub "$N_UBATCH" \
        --cpu-moe --moe-cache-mib "$CACHE_MIB" --load-mode none "${flags[@]}" >"$log" 2>&1 &
    pid=$!
    for _ in $(seq 1 900); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
        echo "$v: llama-server did not come up; the end of $log:" >&2
        tail -n 12 "$log" >&2
        pid=
        continue
    fi
    for i in 1 2; do
        curl -s --max-time 1800 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" -d @"$OUT/question.json" |
            python3 -c 'import json, sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' \
                >"$OUT/answer-$tag-$i.txt" 2>/dev/null
    done
    read -r reads writes <<<"$(python3 "$SPEED_CHECK" --url "http://127.0.0.1:$PORT/v1" \
        --model "$MODEL" --prompt-tokens "$PROMPT_TOKENS" |
        awk '/writes answers/ {w = $3} /reads a prompt/ {r = $4} END {print r, w}')"
    kill "$pid"
    wait "$pid" 2>/dev/null
    pid=
    said=$(grep -o 'moe prefetch: .*' "$log" | sort -u | paste -sd ';' -)
    sums=$(for i in 1 2; do
        if [[ -s $OUT/answer-$tag-$i.txt ]]; then md5sum <"$OUT/answer-$tag-$i.txt" | cut -c1-8; else echo none; fi
    done | paste -sd / -)
    printf '%-18s %12s %12s %18s  %s\n' "$v" "${reads:-?}" "${writes:-?}" "$sums" "${said:--}"
done
