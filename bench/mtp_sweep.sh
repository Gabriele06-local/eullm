#!/usr/bin/env bash
# How fast one model writes with each --mtp setting, on this machine.
#
#   bench/mtp_sweep.sh EULLM_BINARY MODEL.gguf [MORE SERVE FLAGS...]
#
# Starts `eullm serve` once per setting — without drafts first, then each
# entry of MTP_SETTINGS, written N or N:P for `--mtp N --mtp-p-min P` — and
# measures each with speed_check.py twice: its story, and a piece of code,
# the text an MTP head predicts best. One line per setting: both writing
# speeds, and the share of drafts the model kept. Each server's log stays in
# $OUT. The prompt read is short (1,000 tokens): drafting changes how an
# answer is written, not how a prompt is read.
set -u
export LC_ALL=C

BIN=$1
MODEL=$2
shift 2
PORT=${PORT:-11510}
SPEED_CHECK=${SPEED_CHECK:-$HOME/work/speed_check.py}
OUT=${OUT:-$HOME/work/mtp-sweep}
SETTINGS=${MTP_SETTINGS:-"0 1 2 3 3:0.5 4:0.5 6:0.5"}
CODE="Write a Python function that parses an ISO 8601 date string into a datetime, with a docstring, type hints and three unit tests."

mkdir -p "$OUT"
speed() { # a speed_check.py run's writing speed; extra arguments go to it
    python3 "$SPEED_CHECK" --url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
        --prompt-tokens 1000 "$@" | awk '/writes answers/ {print $3}'
}

printf '%-16s %12s %12s %12s\n' setting story_tok/s code_tok/s drafts_kept
for setting in $SETTINGS; do
    n=${setting%%:*}
    p=0
    [[ $setting == *:* ]] && p=${setting#*:}
    log="$OUT/serve-mtp$n-p$p.log"
    "$BIN" serve --port "$PORT" --default-model "$MODEL" --mtp "$n" --mtp-p-min "$p" "$@" \
        >"$log" 2>&1 &
    pid=$!
    for _ in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null && break
        sleep 1
    done
    story=$(speed)
    code=$(speed --write-prompt "$CODE")
    kill "$pid"
    wait "$pid" 2>/dev/null
    kept=$(grep -o 'MTP drafted [0-9]* tokens, the model kept [0-9]*' "$log" |
        awk '{d += $3; k += $8} END {if (d) printf "%.0f%%", 100 * k / d; else print "-"}')
    label="--mtp $n"
    [[ $p != 0 ]] && label="$label p$p"
    printf '%-16s %12s %12s %12s\n' "$label" "${story:-?}" "${code:-?}" "$kept"
done
