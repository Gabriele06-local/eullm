#!/usr/bin/env bash
# A long prompt on one llama-server per configuration, to tell whether copying the
# experts ahead beats what a plain llama-server does with the experts in RAM, mapped
# (--mmap, the default: the page cache holds them) or pinned.
#
#   bench/llama_server_mmap_compare.sh MODEL.gguf NAME=SERVER'|'FLAGS [NAME=SERVER'|'FLAGS ...]
#
# Each argument is a configuration: a name, the llama-server binary and the flags that
# make it (the flags every server shares are below). ORDER (default: as given) lists
# the names to run, so that the same list in two orders tells an effect from running
# second. Each server is asked the same 33,200-token question twice at temperature 0
# (the answers must match, and the pages are warm for the measurement) and then
# measured with bench/speed_check.py: reading that prompt, and writing an answer.
# One line per configuration; each server's log and answers stay in $OUT.
set -u
export LC_ALL=C

MODEL=$1
shift
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-11580}
SPEED_CHECK=${SPEED_CHECK:-$HERE/speed_check.py}
OUT=${OUT:-$HOME/work/llama-server-mmap-compare}
CTX=${CTX:-40960}
N_UBATCH=${N_UBATCH:-2048}
PROMPT_TOKENS=${PROMPT_TOKENS:-33200}

if ! python3 "$SPEED_CHECK" --help 2>/dev/null | grep -q -- --temperature; then
    echo "$SPEED_CHECK is missing or too old (no --temperature): use bench/speed_check.py" >&2
    exit 1
fi
declare -A server flags
names=()
for arg in "$@"; do
    name=${arg%%=*}
    rest=${arg#*=}
    server[$name]=${rest%%|*}
    flags[$name]=${rest#*|}
    names+=("$name")
done
if [[ ${#names[@]} -eq 0 ]]; then
    echo "usage: $0 MODEL.gguf NAME=SERVER|FLAGS ..." >&2
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

printf '%-18s %12s %12s %18s\n' config read_tok/s write_tok/s answers
for name in ${ORDER:-${names[*]}}; do
    bin=${server[$name]:-}
    if [[ -z $bin ]]; then
        echo "no configuration $name" >&2
        continue
    fi
    log="$OUT/llama-server-$name.log"
    # shellcheck disable=SC2086
    "$bin" -m "$MODEL" --port "$PORT" -c "$CTX" -np 1 -b "$N_UBATCH" -ub "$N_UBATCH" ${flags[$name]} >"$log" 2>&1 &
    pid=$!
    for _ in $(seq 1 1200); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
        echo "$name: llama-server did not come up; the end of $log:" >&2
        tail -n 12 "$log" >&2
        pid=
        continue
    fi
    for i in 1 2; do
        curl -s --max-time 3600 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" -d @"$OUT/question.json" |
            python3 -c 'import json, sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' \
                >"$OUT/answer-$name-$i.txt" 2>/dev/null
    done
    read -r reads writes <<<"$(python3 "$SPEED_CHECK" --url "http://127.0.0.1:$PORT/v1" \
        --model "$MODEL" --prompt-tokens "$PROMPT_TOKENS" |
        awk '/writes answers/ {w = $3} /reads a prompt/ {r = $4} END {print r, w}')"
    kill "$pid"
    wait "$pid" 2>/dev/null
    pid=
    sums=$(for i in 1 2; do
        if [[ -s $OUT/answer-$name-$i.txt ]]; then md5sum <"$OUT/answer-$name-$i.txt" | cut -c1-8; else echo none; fi
    done | paste -sd / -)
    printf '%-18s %12s %12s %18s\n' "$name" "${reads:-?}" "${writes:-?}" "$sums"
done
