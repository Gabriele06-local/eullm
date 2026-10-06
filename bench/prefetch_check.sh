#!/usr/bin/env bash
# Phase 6 of docs/moe-offload-plan.md, measured: does copying an MoE's experts
# ahead (`--moe-prefetch`, llama.cpp patch 0003) read a long prompt faster, to
# the same answer, and what does the VRAM its slots take cost the writing?
#
#   bench/prefetch_check.sh EULLM_BINARY MODEL.gguf [MORE SERVE FLAGS...]
#
# MODEL is an MoE whose experts do not all fit in VRAM (the reference PC's
# Qwen3.8-Flash-Next IQ2_XS, say), EULLM_BINARY a CUDA build carrying the
# patch. The script starts `eullm serve --moe-prefetch N` once for each N of
# SETTINGS ("0 4" by default: off, then the default four slots), in that
# order, every one with `--ctx-size CTX --moe-cache MOE_CACHE` (40960 and auto
# by default), `--n-ubatch N_UBATCH` when it is set (unset, the engine chooses:
# 2048 with an expert cache) and the flags after MODEL; on each:
#
# 1. asks one long question twice, greedy and without the prompt cache
#    (`cache_prompt: false`), about a document of PROMPT_TOKENS tokens
#    (default 33200) that is the same for every server. All the answers must
#    be equal: the prefetch changes when the experts are copied, not what the
#    GPU computes from them;
# 2. measures it with bench/speed_check.py over a fresh document as long.
#
# With `--moe-cache auto` each server sizes its own cache, and the slots'
# VRAM comes out of it: the cache column is where the writing cost shows.
# SETTINGS="4 0" starts the server with the prefetch first, to tell an effect
# of the prefetch from one of running second.
#
# A setting N:bus is --moe-prefetch N with LLAMA_MOE_PREFETCH_FROM_CACHE=0:
# the experts the expert cache holds are copied over the bus as well, as before
# patch 0004, so SETTINGS="4:bus 4" (and "4 4:bus") measures what copying them
# from VRAM gains (phase 6b).
#
# One line per setting: reading and writing speeds, the expert cache the
# server sized, a checksum of each answer and what llama.cpp said about the
# prefetch on stderr (`on, 4 slots of ...` or `off, <why>`); then whether the
# answers match. Each server's log and answers stay in $OUT. A prefetch that
# stays off for want of VRAM says how much it needed: MOE_CACHE=<MiB> below
# what --fit chose makes the room.
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
N_UBATCH=${N_UBATCH:-}
MOE_CACHE=${MOE_CACHE:-auto}
PROMPT_TOKENS=${PROMPT_TOKENS:-33200}
SETTINGS=${SETTINGS:-"0 4"}

if ! python3 "$SPEED_CHECK" --help 2>/dev/null | grep -q -- --temperature; then
    echo "$SPEED_CHECK is missing or too old (no --temperature): use bench/speed_check.py" >&2
    exit 1
fi
if ! "$BIN" serve --help 2>/dev/null | grep -q -- --moe-prefetch; then
    echo "$BIN has no --moe-prefetch: build the engine from this checkout" >&2
    exit 1
fi
# A server left running by Ctrl+C would keep the port, and the GPU busy.
pid=
trap '[[ -n $pid ]] && kill "$pid" 2>/dev/null' EXIT
trap 'exit 130' INT TERM

mkdir -p "$OUT"
rm -f "$OUT"/answer-prefetch*.txt
# The long question, the same for every server: speed_check.py's document, from a fixed seed.
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

ubatch=()
[[ -n $N_UBATCH ]] && ubatch=(--n-ubatch "$N_UBATCH")
not_on=
names=
printf '%-14s %12s %12s %8s %18s  %s\n' setting read_tok/s write_tok/s cache answers server_said
for setting in $SETTINGS; do
    slots=${setting%%:*}
    p=${setting/:/-} # the setting in file names
    names="$names $p"
    vars=()
    [[ $setting == *:bus ]] && vars=(LLAMA_MOE_PREFETCH_FROM_CACHE=0)
    log="$OUT/serve-prefetch$p.log"
    env "${vars[@]}" "$BIN" serve --port "$PORT" --default-model "$MODEL" --moe-prefetch "$slots" \
        --ctx-size "$CTX" --moe-cache "$MOE_CACHE" "${ubatch[@]}" "$@" >"$log" 2>&1 &
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
    [[ $slots != 0 && $said != *"moe prefetch: on"* ]] && not_on="$not_on $setting"
    # the --fit line: "expert tensors in CPU RAM, 5.75 GiB of VRAM caching ..."
    cache=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -o 'CPU RAM, [0-9.]* GiB of VRAM caching' |
        head -n 1 | awk '{print $3 "G"}')
    sums=$(for i in 1 2; do
        if [[ -s $OUT/answer-prefetch$p-$i.txt ]]; then md5sum <"$OUT/answer-prefetch$p-$i.txt" | cut -c1-8; else echo none; fi
    done | paste -sd / -)
    printf '%-14s %12s %12s %8s %18s  %s\n' "prefetch=$setting" "${reads:-?}" "${writes:-?}" "${cache:--}" "$sums" "${said:--}"
done

echo
first=${names# }
first=${first%% *}
same() { cmp -s "$OUT/answer-prefetch$1.txt" "$OUT/answer-prefetch$2.txt"; }
missing=
for p in $names; do
    [[ -s $OUT/answer-prefetch$p-1.txt ]] || missing="$missing $p"
done
if [[ -n $missing ]]; then
    echo "No answer from --moe-prefetch$missing: see $OUT/serve-prefetch*.log"
elif [[ -n $not_on ]]; then
    echo "The prefetch did not turn on with --moe-prefetch$not_on (server_said above): this compared nothing"
else
    differs=
    for p in $names; do
        if ! same "$p-1" "$p-2"; then
            echo "--moe-prefetch $p answered the same request two ways: the comparison says nothing ($OUT/answer-prefetch$p-*.txt)"
            differs=1
        elif ! same "$first-1" "$p-1"; then
            echo "THE PREFETCH CHANGES THE ANSWER: diff $OUT/answer-prefetch$first-1.txt $OUT/answer-prefetch$p-1.txt"
            differs=1
        fi
    done
    [[ -z $differs ]] && echo "Same answer with every setting: $SETTINGS."
fi
