#!/usr/bin/env bash
# Phase 6 of docs/moe-offload-plan.md, measured: does copying an MoE's experts
# ahead (LLAMA_MOE_PREFETCH=1, llama.cpp patch 0003) read a long prompt
# faster, and to the same answer?
#
#   bench/prefetch_check.sh EULLM_BINARY MODEL.gguf [MORE SERVE FLAGS...]
#
# MODEL is an MoE whose experts do not all fit in VRAM (the reference PC's
# Qwen3.8-Flash-Next IQ2_XS, say), EULLM_BINARY a CUDA build carrying the
# patch. The script starts `eullm serve` with LLAMA_MOE_PREFETCH=0, then =1,
# both with `--ctx-size CTX --moe-cache MOE_CACHE --n-ubatch N_UBATCH` (40960,
# auto and 4096 by default; flags after MODEL are added), and on each:
#
# 1. asks one long question twice, greedy and without the prompt cache
#    (`cache_prompt: false`), about a document of PROMPT_TOKENS tokens
#    (default 33200) that is the same for both servers. The four answers must
#    be equal: the prefetch changes when the experts are copied, not what the
#    GPU computes from them;
# 2. measures it with bench/speed_check.py over a fresh document as long.
#
# One line per setting: reading and writing speeds, a checksum of each answer
# and what the server said about the prefetch on stderr (`on, 2 slots of ...`
# or `off, <why>`); then whether the answers match. LLAMA_MOE_PREFETCH_SLOTS
# and LLAMA_MOE_PREFETCH_MIN_TOKENS, if set, reach the second server. Each
# server's log and answers stay in $OUT. A prefetch that stays off for want
# of VRAM says how much it needed: MOE_CACHE=<MiB> below what --fit chose
# makes the room.
set -u
export LC_ALL=C

BIN=$1
MODEL=$2
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-11530}
SPEED_CHECK=${SPEED_CHECK:-$HERE/speed_check.py}
OUT=${OUT:-$HOME/work/prefetch-check}
CTX=${CTX:-40960}
N_UBATCH=${N_UBATCH:-4096}
MOE_CACHE=${MOE_CACHE:-auto}
PROMPT_TOKENS=${PROMPT_TOKENS:-33200}

if ! python3 "$SPEED_CHECK" --help 2>/dev/null | grep -q -- --temperature; then
    echo "$SPEED_CHECK is missing or too old (no --temperature): use bench/speed_check.py" >&2
    exit 1
fi
# A server left running by Ctrl+C would keep the port, and the GPU busy.
pid=
trap '[[ -n $pid ]] && kill "$pid" 2>/dev/null' EXIT
trap 'exit 130' INT TERM

mkdir -p "$OUT"
rm -f "$OUT"/answer-prefetch*.txt
# The long question, the same for both servers: speed_check.py's document, from a fixed seed.
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
    "think": False,
}))
EOF

said_on=
printf '%-12s %12s %12s %18s  %s\n' setting read_tok/s write_tok/s answers server_said
for p in 0 1; do
    log="$OUT/serve-prefetch$p.log"
    LLAMA_MOE_PREFETCH=$p "$BIN" serve --port "$PORT" --default-model "$MODEL" \
        --ctx-size "$CTX" --moe-cache "$MOE_CACHE" --n-ubatch "$N_UBATCH" "$@" >"$log" 2>&1 &
    pid=$!
    for _ in $(seq 1 600); do
        curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    for i in 1 2; do
        curl -s --max-time 1800 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" -d @"$OUT/question.json" |
            python3 -c 'import json, sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' \
                >"$OUT/answer-prefetch$p-$i.txt" 2>/dev/null
    done
    read -r reads writes <<<"$(python3 "$SPEED_CHECK" --url "http://127.0.0.1:$PORT/v1" \
        --model "$MODEL" --prompt-tokens "$PROMPT_TOKENS" |
        awk '/writes answers/ {w = $3} /reads a prompt/ {r = $4} END {print r, w}')"
    kill "$pid"
    wait "$pid" 2>/dev/null
    pid=
    said=$(grep -o 'moe prefetch: .*' "$log" | sort -u | paste -sd ';' -)
    [[ $p == 1 ]] && said_on=$said
    sums=$(for i in 1 2; do
        if [[ -s $OUT/answer-prefetch$p-$i.txt ]]; then md5sum <"$OUT/answer-prefetch$p-$i.txt" | cut -c1-8; else echo none; fi
    done | paste -sd / -)
    printf '%-12s %12s %12s %18s  %s\n' "prefetch=$p" "${reads:-?}" "${writes:-?}" "$sums" "${said:--}"
done

echo
same() { cmp -s "$OUT/answer-prefetch$1.txt" "$OUT/answer-prefetch$2.txt"; }
if [[ ! -s $OUT/answer-prefetch0-1.txt || ! -s $OUT/answer-prefetch1-1.txt ]]; then
    echo "A server gave no answer: see $OUT/serve-prefetch0.log and serve-prefetch1.log"
elif [[ $said_on != *"moe prefetch: on"* ]]; then
    echo "The prefetch did not turn on (${said_on:-no line on stderr}): this compared nothing"
elif ! same 0-1 0-2 || ! same 1-1 1-2; then
    echo "A server answered the same request two ways: the comparison says nothing ($OUT/answer-prefetch*.txt)"
elif same 0-1 1-1; then
    echo "Same answer with and without the prefetch."
else
    echo "THE PREFETCH CHANGES THE ANSWER: diff $OUT/answer-prefetch0-1.txt $OUT/answer-prefetch1-1.txt"
fi
