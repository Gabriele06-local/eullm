#!/usr/bin/env bash
# How fast one model writes with each --mtp setting, on this machine.
#
#   bench/mtp_sweep.sh EULLM_BINARY MODEL.gguf [MORE SERVE FLAGS...]
#
# Starts `eullm serve` once per setting — without drafts first, then each
# entry of MTP_SETTINGS, written N or N:P for `--mtp N --mtp-p-min P` — and
# measures each with speed_check.py twice: its story, and a piece of code,
# the text an MTP head predicts best. One line per setting: both writing
# speeds, and the share of drafts the model kept on the two, which the server
# reports in each answer. Each server's log stays in $OUT. The prompt read is
# short (1,000 tokens): drafting changes how an answer is written, not how a
# prompt is read. TEMPERATURE (default 0) is the answers' sampling
# temperature: a draft is kept when it is the token the model samples, so a
# higher one keeps fewer.
set -u
export LC_ALL=C

BIN=$1
MODEL=$2
shift 2
PORT=${PORT:-11510}
SPEED_CHECK=${SPEED_CHECK:-$HOME/work/speed_check.py}
OUT=${OUT:-$HOME/work/mtp-sweep}
SETTINGS=${MTP_SETTINGS:-"0 1 2 3 3:0.5 4:0.5 6:0.5"}
TEMPERATURE=${TEMPERATURE:-0}
CODE="Write a Python function that parses an ISO 8601 date string into a datetime, with a docstring, type hints and three unit tests."

# speed_check.py from before --temperature would time only greedy answers,
# and print no drafts.
if ! python3 "$SPEED_CHECK" --help 2>/dev/null | grep -q -- --temperature; then
    echo "$SPEED_CHECK is missing or too old (no --temperature): use bench/speed_check.py" >&2
    exit 1
fi
# A server left running by Ctrl+C would keep the port, and the binary busy.
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
for setting in $SETTINGS; do
    n=${setting%%:*}
    p=0
    [[ $setting == *:* ]] && p=${setting#*:}
    log="$OUT/serve-mtp$n-p$p.log"
    "$BIN" serve --port "$PORT" --default-model "$MODEL" --mtp "$n" --mtp-p-min "$p" "$@" \
        >"$log" 2>&1 &
    pid=$!
    # An MoE with --moe-cache reads 20-35 GB of experts into memory first.
    for _ in $(seq 1 600); do
        curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if ! curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null; then
        echo "the server did not come up for --mtp $n; the end of $log:" >&2
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
    label="--mtp $n"
    [[ $p != 0 ]] && label="$label p$p"
    printf '%-16s %12s %12s %12s  %s\n' "$label" "${story:-?}" "${code:-?}" "$kept" \
        "${story_text:-?}/${code_text:-?}"
done
