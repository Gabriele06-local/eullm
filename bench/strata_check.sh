#!/usr/bin/env bash
# Strata's side of docs/strata-study.md §7: where its time goes on this PC,
# and how much of its 85.8 tokens/s comes from drafting, from repeating a
# request warm, and from temperature 0.
#
#   bench/strata_check.sh STRATA_DIR
#
# STRATA_DIR is Strata's folder as its setup left it. Its run script
# (run-iq2_xs.sh, or RUN_SCRIPT) names the Python it runs and the config
# (strata-iq2_xs.json) that holds the engine's flags. Nothing in STRATA_DIR
# is changed: every run starts Strata's server on PORT with a copy of the
# config in $OUT, which adds STRATA_DECODE_TIMING and STRATA_PREFILL_TIMING to
# the engine's environment and sends the engine's log to $OUT. Each run is
# then measured with bench/speed_check.py, as Strata was measured before
# (--prompt-tokens 32000). RUNS ("A B C D" by default) picks among:
#
#   A  as installed: the 85.8 and 2,320 again, with the timings behind them
#   B  --spec-min-p 1.0: windows of one token, its plain decoding (the draft
#      layer still guesses once a window, which it then does not use)
#   C  --prompt-cache 0 --adapt-swaps 0: no checkpoint to resume the repeated
#      request from, and no expert swapped into the cache towards it
#   D  as A, with speed_check --temperature 0.7
#
# One line per run: writes (the timed run, and the first one, which warms up),
# drafts kept, reads. Below the table, each run's lines from the engine: the
# expert cache and prompt chunk it chose, its decode timing per request
# (windows, tokens a window, where a window's milliseconds go) and its prompt
# timing by phase. Each run's logs and speed_check output stay in $OUT.
set -u
export LC_ALL=C

DIR=${1:?usage: bench/strata_check.sh STRATA_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-11560}
SPEED_CHECK=${SPEED_CHECK:-$HERE/speed_check.py}
OUT=${OUT:-$HOME/work/strata-check}
RUNS=${RUNS:-"A B C D"}
RUN_SCRIPT=${RUN_SCRIPT:-$DIR/run-iq2_xs.sh}
PROMPT_TOKENS=${PROMPT_TOKENS:-32000}

[[ -f $RUN_SCRIPT ]] || { echo "no $RUN_SCRIPT: set RUN_SCRIPT to Strata's run script" >&2; exit 1; }
if ! python3 "$SPEED_CHECK" --help 2>/dev/null | grep -q -- --temperature; then
    echo "$SPEED_CHECK is missing or too old (no --temperature): use bench/speed_check.py" >&2
    exit 1
fi
mkdir -p "$OUT"

# The run script's last line: exec "<python>" "<dir>/serve/server.py" ... "--config" "<config>" ...
IFS=$'\t' read -r PY CFG < <(python3 -I - "$RUN_SCRIPT" <<'EOF'
import shlex
import sys

for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if line.startswith("exec "):
        words = shlex.split(line)[1:]
        if "--config" in words:
            print(words[0], words[words.index("--config") + 1], sep="\t")
            break
EOF
)
[[ -n ${PY:-} && -f ${CFG:-} ]] || { echo "$RUN_SCRIPT names no Python and config this script can read" >&2; exit 1; }

# A copy of the config per run; prints the engine's binary and its version.
IFS=$'\t' read -r EXE VERSION < <(python3 -I - "$CFG" "$OUT" $RUNS <<'EOF'
import json
import os
import sys

cfg_path, out, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
with open(cfg_path, encoding="utf-8") as f:
    base = json.load(f)
for run in runs:
    cfg = json.loads(json.dumps(base))
    cfg["env"] = {**(cfg.get("env") or {}), "STRATA_DECODE_TIMING": "1", "STRATA_PREFILL_TIMING": "1"}
    cfg["log"] = os.path.join(out, f"engine-{run}.log")
    args = list(cfg["args"])
    if run == "B":
        if "--spec-min-p" in args:
            args[args.index("--spec-min-p") + 1] = "1.0"
        else:
            args += ["--spec-min-p", "1.0"]
    elif run == "C":
        args += ["--prompt-cache", "0", "--adapt-swaps", "0"]
    cfg["args"] = args
    with open(os.path.join(out, f"strata-{run}.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
exe = base.get("exe", "")
try:
    with open(os.path.join(os.path.dirname(exe), "BUILD.json"), encoding="utf-8") as f:
        version = json.load(f).get("version") or "?"
except (OSError, ValueError):
    version = "?"
print(exe, version, sep="\t")
EOF
)

# Strata's server starts its engine as a child: both go, as one process group.
pid=
stop() {
    [[ -n $pid ]] || return 0
    kill -TERM -- "-$pid" 2>/dev/null
    for _ in $(seq 1 60); do
        kill -0 -- "-$pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    pid=
}
trap stop EXIT
trap 'exit 130' INT TERM

if pgrep -f -- "$EXE" >/dev/null; then
    echo "Strata's engine ($EXE) is running already: it would share the GPU. Stop it first." >&2
    exit 1
fi

echo "Strata $VERSION in $DIR"
echo "CPU: $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1);" \
    "highest clock allowed $(($(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo 0) / 1000)) MHz," \
    "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo ?)"
echo
printf '%-4s %-36s %11s %11s %8s %11s\n' run change writes first_run drafts reads
for run in $RUNS; do
    case $run in
    A) change="as installed" ;;
    B) change="--spec-min-p 1.0" ;;
    C) change="--prompt-cache 0 --adapt-swaps 0" ;;
    D) change="temperature 0.7" ;;
    *) echo "no run $run" >&2; continue ;;
    esac
    : >"$OUT/engine-$run.log"
    (cd "$DIR" && exec setsid "$PY" serve/server.py --engine strata --config "$OUT/strata-$run.json" \
        --port "$PORT") >"$OUT/server-$run.log" 2>&1 &
    pid=$!
    # Reading 33 GB of experts into pinned memory takes minutes.
    up=
    for _ in $(seq 1 1200); do
        curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"loaded": *true' && { up=1; break; }
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if [[ -z $up ]]; then
        printf '%-4s %-36s  did not start: %s\n' "$run" "$change" "$OUT/server-$run.log"
        stop
        continue
    fi
    temperature=()
    [[ $run == D ]] && temperature=(--temperature 0.7)
    python3 "$SPEED_CHECK" --url "http://127.0.0.1:$PORT/v1" --prompt-tokens "$PROMPT_TOKENS" \
        "${temperature[@]}" >"$OUT/speed-$run.txt" 2>&1
    stop
    read -r writes first drafts reads <<<"$(awk '
        /writes answers/ {w = $3; if (match($0, /warming up: [0-9.]+/)) f = substr($0, RSTART + 12, RLENGTH - 12)}
        /drafts kept/ {d = $3}
        /reads a prompt/ {r = $4}
        END {print (w ? w : "?"), (f ? f : "?"), (d ? d : "-"), (r ? r : "?")}' "$OUT/speed-$run.txt")"
    printf '%-4s %-36s %11s %11s %8s %11s\n' "$run" "$change" "$writes" "$first" "$drafts" "$reads"
done

echo
for run in $RUNS; do
    [[ -s $OUT/engine-$run.log ]] || continue
    echo "--- $run: the engine's own lines"
    sed 's/\x1b\[[0-9;]*m//g' "$OUT/engine-$run.log" | grep -E \
        'strata (generate|serve): (expert cache [0-9]+ slots|expert cache auto|prompt chunk|the prompt path borrows|PCIe probe|[0-9]+ expert-pool workers)|strata decode timing|strata decode GPU stages|strata prefill timing|decode expert cache hit rate'
done
