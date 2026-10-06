#!/usr/bin/env bash
# The GPU checks the engine roadmap still waits on, one after the other, with
# nobody watching: started in the evening, read in the morning.
#
#   nohup tools/gpu_night.sh > ~/work/gpu-night.out 2>&1 &
#
# It keeps the machine from sleeping while it runs (systemd-inhibit), waits
# until no bench/prefetch_check.sh is running, stops any server of $BIN or of
# the pinned llama-server still up (it would share the GPU with every
# measurement), then runs each step of STEPS
# under a time limit of its own, and goes on whatever a step's outcome. A
# step not started by STOP_AT (06:45) is left for another night. After every
# step $NIGHT/summary.md is written again: each step's outcome and minutes,
# the tables it printed, and the GPU's temperature and clocks so far. Each
# step's whole output is in $NIGHT/<step>.log.
#
# Steps, in this order (STEPS="..." runs some of them):
#   prefetch         bench/prefetch_check.sh once for each run of PREFETCH_RUNS
#                    ("4:bus,4 4,4:bus": the --moe-prefetch of each server of a
#                    run, in order, N:bus with every expert over the bus), with
#                    --moe-cache PREFETCH_CACHE (auto) and the engine's own
#                    micro-batch unless PREFETCH_UBATCH is set: what copying
#                    the experts the cache holds from VRAM gains on reading
#                    (phase 6b), in either order; "0,4 4,0" measures the
#                    prefetch itself
#   interleave       bench/interleave_check.py on qwen3-8b, --batch-size 2 (0.7-D)
#   mtp-t08          bench/mtp_sweep.sh at TEMPERATURE=0.8 on Qwen3.5-9B-MTP (B6)
#   llama-pin        llama-server and llama-quantize from the pinned llama.cpp
#   mtp-head-q8      bench/mtp_head_q8.sh from the Q8_0 GGUF (B5)
#   test-d           bench/mtp_test_d.sh on Qwen3.6-35B-A3B-MTP (B7)
#   residency        tools/residency_check.sh with BIG=qwen3-32b (V3, V5)
#   auto             tools/auto_check.sh, the full run (V7-V9)
#   soak             tools/auto_check.sh's hour of mixed traffic alone (V8)
#   rag              the Italian RAG gate set, built and measured (MVP 1) with
#                    RAG_DECISION as the decision model (the 2B by default)
#   docker-gpu       the CUDA image built and asked one question; skipped
#                    without Docker's NVIDIA runtime or with port 11434 taken
#
# What the steps of STEPS download (the Q8_0 GGUF for mtp-head-q8, the
# catalog models in PULLS for residency, auto, soak and docker-gpu) starts
# at once, at low priority, and a step that needs it waits for it first. A
# step whose model is not on disk is skipped and says which file it missed.
# Paths default to the reference PC's; every one can be set below.
# LIMIT_MINUTES caps every step's time limit, for a short trial of the chain.
set -u
export LC_ALL=C

SELF=$(realpath "$0")
REPO=$(cd "$(dirname "$SELF")/.." && pwd)
export NIGHT=${NIGHT:-$HOME/work/gpu-night-$(date +%Y%m%d-%H%M)}
BIN=${BIN:-$HOME/work/bin/eullm-roadmap}
MODELS=${MODELS:-$HOME/work/models}
FLASH=${FLASH:-$HOME/work/Strata/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf}
M9=${M9:-$MODELS/Qwen3.5-9B-MTP-Q4_K_M.gguf}
M9_Q8=${M9_Q8:-$MODELS/Qwen3.5-9B-MTP-Q8_0.gguf}
M9_Q8_URL=${M9_Q8_URL:-https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf}
M35=${M35:-$MODELS/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf}
LLAMA_PIN=${LLAMA_PIN:-$HOME/work/llama-pin}
STORE=${EULLM_MODELS_DIR:-$HOME/.eullm/models}
DECISION_SMALL=${DECISION_SMALL:-$STORE/jev-style-0.8b-decision-v3-gguf-q4_k_m/Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf}
DECISION_LARGE=${DECISION_LARGE:-$STORE/jev-style-2b-decision-v3-gguf-q4_k_m/Jev-Style-2B-Decision-v3-Q4_K_M.gguf}
CORPUS=${CORPUS:-$HOME/work/corpus*/legislazione_*.chunks.jsonl}
INTERLEAVE_MODEL=${INTERLEAVE_MODEL:-qwen3-8b}
RAG_DECISION=${RAG_DECISION:-jev-style-2b-decision-v3-gguf-q4_k_m}
PULLS=${PULLS-"qwen3-32b qwen3-14b qwen3-8b qwen3-4b qwen3-1.7b qwen3-0.6b"}
STEPS=${STEPS:-"prefetch interleave mtp-t08 llama-pin mtp-head-q8 test-d residency auto rag docker-gpu"}
PREFETCH_RUNS=${PREFETCH_RUNS:-"4:bus,4 4,4:bus"}
PREFETCH_CACHE=${PREFETCH_CACHE:-auto}
PREFETCH_UBATCH=${PREFETCH_UBATCH:-}
STOP_AT=${STOP_AT:-06:45}
EVENING=${EVENING:-"$HOME/work/prefetch-check $HOME/work/prefetch-check-2048 $HOME/work/prefetch-check-slots3"}

# A server of this run on $1, up within ten minutes (reading a model into
# memory takes a while): its pid in $pid, killed when the step ends.
serve_on() {
    local port=$1 log=$2
    shift 2
    "$@" >"$log" 2>&1 &
    pid=$!
    trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
    for _ in $(seq 1 600); do
        curl -sf "http://127.0.0.1:$port/api/version" >/dev/null && return 0
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    echo "the server did not come up: $log"
    tail -n 20 "$log"
    return 1
}

# Skips the step (exit 3) unless every file named exists.
need() {
    local f
    for f in "$@"; do
        [[ -f $f ]] || { echo "SKIPPED: no $f"; exit 3; }
    done
}

# One step, run by the night below under `timeout` as `gpu_night.sh step:NAME`.
# Exit 3 means skipped.
step() {
    case $1 in
    prefetch)
        need "$FLASH"
        local run
        for run in $PREFETCH_RUNS; do
            echo "== --moe-prefetch ${run//,/ then }, --moe-cache $PREFETCH_CACHE${PREFETCH_UBATCH:+, --n-ubatch $PREFETCH_UBATCH}"
            SETTINGS=${run//,/ } MOE_CACHE=$PREFETCH_CACHE N_UBATCH=$PREFETCH_UBATCH \
                OUT=$NIGHT/prefetch-${run//[,:]/-} "$REPO/bench/prefetch_check.sh" "$BIN" "$FLASH"
            echo
        done
        ;;
    interleave)
        serve_on 11550 "$NIGHT/interleave-serve.log" \
            "$BIN" serve --port 11550 --default-model "$INTERLEAVE_MODEL" --batch-size 2 \
            --ctx-size 32768 || return 1
        python3 "$REPO/bench/interleave_check.py" --url http://127.0.0.1:11550 --model "$INTERLEAVE_MODEL"
        ;;
    mtp-t08)
        need "$M9"
        TEMPERATURE=0.8 MTP_SETTINGS="0 1 2 3" SPEED_CHECK=$REPO/bench/speed_check.py \
            OUT=$NIGHT/mtp-sweep-t08 "$REPO/bench/mtp_sweep.sh" "$BIN" "$M9"
        ;;
    llama-pin)
        # -j16, not every thread: 32 nvcc at once can want more than 64 GB of RAM.
        cmake -S "$REPO/engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp" -B "$LLAMA_PIN" \
            -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON &&
            cmake --build "$LLAMA_PIN" --target llama-server llama-quantize -j16 &&
            ls -l "$LLAMA_PIN/bin/llama-server" "$LLAMA_PIN/bin/llama-quantize"
        ;;
    mtp-head-q8)
        [[ -s $M9_Q8 ]] || { echo "SKIPPED: no $M9_Q8 (see downloads.log)"; return 3; }
        LLAMA_QUANTIZE=$LLAMA_PIN/bin/llama-quantize OUT=$NIGHT/mtp-head-q8 \
            "$REPO/bench/mtp_head_q8.sh" "$BIN" "$M9_Q8"
        ;;
    test-d)
        need "$M35" "$LLAMA_PIN/bin/llama-server"
        OUT=$NIGHT/mtp-test-d "$REPO/bench/mtp_test_d.sh" "$LLAMA_PIN/bin/llama-server" "$M35"
        ;;
    residency)
        BIG=qwen3-32b OUT=$NIGHT/residency "$REPO/tools/residency_check.sh" "$BIN" "$BIN"
        ;;
    auto)
        need "$DECISION_SMALL" "$DECISION_LARGE"
        local cpu_decision
        cpu_decision=$(ls "$STORE"/qwen3-0.6b/*.gguf 2>/dev/null | head -1)
        LIMIT=100 DURATION=3600 CPU_DECISION=$cpu_decision OUT=$NIGHT/auto-check \
            "$REPO/tools/auto_check.sh" "$BIN" "$DECISION_SMALL" "$DECISION_LARGE"
        ;;
    soak)
        need "$DECISION_SMALL"
        CHECKS=4 DURATION=3600 OUT=$NIGHT/soak "$REPO/tools/auto_check.sh" "$BIN" "$DECISION_SMALL"
        ;;
    rag)
        compgen -G "$CORPUS" >/dev/null || { echo "SKIPPED: nothing matches $CORPUS"; return 3; }
        # shellcheck disable=SC2086 # CORPUS is a glob
        python3 "$REPO/bench/reflexbench/rg_openbook.py" --by-heading --limit 1000 \
            --norms $CORPUS --out "$NIGHT/rag-legal-it.jsonl" || return 1
        EULLM_AUDIT_DIR=$NIGHT/rag-audit serve_on 11540 "$NIGHT/rag-serve.log" \
            "$BIN" serve --port 11540 --decision-model "$RAG_DECISION" \
            --embedding-model qwen3-embedding-0.6b-gguf-q8_0 || return 1
        python3 "$REPO/bench/reflexbench/ragbench.py" --url http://127.0.0.1:11540 --sets '' \
            --data "$NIGHT/rag-legal-it.jsonl" --embed-model qwen3-embedding-0.6b-gguf-q8_0 \
            --embed-query-prefix 'Instruct: Given a question, retrieve passages that answer it\nQuery:' \
            --out "$NIGHT/rag-it-${RAG_DECISION%%-gguf*}.json" --details "$NIGHT/rag-it-${RAG_DECISION%%-gguf*}.jsonl"
        ;;
    docker-gpu)
        command -v docker >/dev/null || { echo "SKIPPED: no docker"; return 3; }
        docker info 2>/dev/null | grep -qi nvidia ||
            { echo "SKIPPED: Docker is not running, or has no NVIDIA runtime"; return 3; }
        if curl -s -o /dev/null http://127.0.0.1:11434/; then
            echo "SKIPPED: something answers on port 11434 already"
            return 3
        fi
        local model
        model=$(ls "$STORE"/qwen3-0.6b/*.gguf 2>/dev/null | head -1)
        [[ -n $model ]] || { echo "SKIPPED: no qwen3-0.6b in $STORE"; return 3; }
        cd "$REPO" || return 1
        # The card's architecture only: the default three take three times as long.
        docker compose --profile gpu build --build-arg CUDA_ARCHS=120 engine-gpu || return 1
        docker compose --profile gpu up -d engine-gpu || return 1
        trap 'docker compose --profile gpu down' EXIT
        for _ in $(seq 1 120); do
            curl -sf http://127.0.0.1:11434/api/version >/dev/null && break
            sleep 1
        done
        docker compose cp "$model" engine-gpu:/models/qwen3-0.6b.gguf || return 1
        curl -sf http://127.0.0.1:11434/api/generate -H 'Content-Type: application/json' \
            -d '{"model":"qwen3-0.6b","prompt":"Say hello in Italian.","stream":false,"think":false,"options":{"num_predict":24,"temperature":0}}' ||
            { docker compose logs engine-gpu | tail -n 30; return 1; }
        echo
        docker compose logs engine-gpu 2>&1 | grep -i -m5 'cuda\|offload\|gpu'
        ;;
    *)
        echo "no step $1"
        return 1
        ;;
    esac
}

if [[ ${1:-} == step:* ]]; then
    step "${1#step:}"
    exit $?
fi

# Not to sleep before morning: idle suspend would stop the GPU mid-step.
if [[ -z ${GPU_NIGHT_INHIBITED:-} ]] && command -v systemd-inhibit >/dev/null &&
    systemd-inhibit --what=sleep:idle --who=eullm --why=probe true 2>/dev/null; then
    export GPU_NIGHT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --who=EuLLM --why="GPU checks running overnight" \
        bash "$SELF" "$@"
fi

note() { echo "$(date '+%H:%M:%S') $*"; }

# One night at a time: two would share the GPU, their checks would use the
# same ports, and each would stop the other's servers after its steps.
exec 9>"${XDG_RUNTIME_DIR:-/tmp}/eullm-gpu-night.lock"
flock -n 9 || { echo "another tools/gpu_night.sh is running: one at a time" >&2; exit 1; }

# The servers a step leaves behind (a killed script cannot always stop its
# own), and any started by hand: on 5 October one held 11.7 GB of VRAM when
# the night began, and no phase 6 server could start beside it. Waits until
# they are gone, a minute at most: freeing tens of GB of pinned memory takes
# a while, and the next step sizes its models against the VRAM left.
stop_servers() {
    pkill -f -- "$BIN serve" 2>/dev/null
    pkill -f -- "$LLAMA_PIN/bin/llama-server" 2>/dev/null
    local _
    for _ in $(seq 1 60); do
        pgrep -f -- "$BIN serve" >/dev/null || pgrep -f -- "$LLAMA_PIN/bin/llama-server" >/dev/null || return 0
        sleep 1
    done
}

[[ -x $BIN ]] || { echo "no EuLLM binary at $BIN (set BIN)" >&2; exit 1; }
mkdir -p "$NIGHT"
stop=$(date -d "$STOP_AT" +%s)
((stop > $(date +%s))) || stop=$(date -d "tomorrow $STOP_AT" +%s)

limit_of() { # minutes a step may take
    local m
    case $1 in
    interleave) m=15 ;;
    mtp-t08) m=45 ;;
    prefetch) m=90 ;;
    soak) m=90 ;;
    llama-pin) m=60 ;;
    mtp-head-q8 | test-d) m=90 ;;
    rag | docker-gpu) m=120 ;;
    residency) m=150 ;;
    auto) m=330 ;;
    *) m=60 ;;
    esac
    [[ -n ${LIMIT_MINUTES:-} ]] && ((LIMIT_MINUTES < m)) && m=$LIMIT_MINUTES
    echo "$m"
}

# What of a step's output goes in the summary.
extract() {
    local log=$NIGHT/$1.log
    case $1 in
    residency) cat "$NIGHT/residency/summary.txt" 2>/dev/null; tail -n 1 "$log" ;;
    soak) cat "$NIGHT/soak/summary.txt" 2>/dev/null; tail -n 1 "$log" ;;
    prefetch) grep -v '^$' "$log" ;;
    auto)
        cat "$NIGHT/auto-check/summary.txt" 2>/dev/null
        tail -n 1 "$log"
        for f in "$NIGHT"/auto-check/3-*.md; do
            [[ -f $f ]] && { echo; echo "--- $(basename "$f")"; cat "$f"; }
        done
        ;;
    llama-pin) tail -n 4 "$log" ;;
    *) tail -n 40 "$log" ;;
    esac | sed 's/\x1b\[[0-9;]*m//g'
}

gpu_report() { # the GPU over the night, from the sampler's lines
    [[ -s $NIGHT/gpu.csv ]] || { echo "(no nvidia-smi samples)"; return; }
    awk -F', *' '{n++; if ($2 > t) t = $2; if ($5 > 50) {busy++; if (!c || $3 < c) c = $3; p += $4}}
        END {printf "%d samples a minute apart; hottest %d C; under load (%d samples) slowest SM clock %d MHz, mean power %.0f W\n",
             n, t, busy, c, busy ? p / busy : 0}' "$NIGHT/gpu.csv"
}

# The evening's phase 6 runs: the cache --fit gave each server. Read from
# their logs, as the server printed it.
evening() {
    local d p f
    for d in $EVENING; do
        for p in 0 1; do
            f=$d/serve-prefetch$p.log
            [[ -f $f ]] || continue
            printf '%s prefetch=%s: %s\n' "$(basename "$d")" "$p" \
                "$(sed 's/\x1b\[[0-9;]*m//g' "$f" | grep -o 'CPU RAM, [0-9.]* GiB of VRAM caching' | head -1)"
        done
    done
}

write_summary() {
    {
        echo "# GPU night, $(date -d "@$started" '+%Y-%m-%d %H:%M') to $(date '+%H:%M')"
        echo
        echo "$("$BIN" --version 2>/dev/null), branch $(git -C "$REPO" rev-parse --abbrev-ref HEAD) at $(git -C "$REPO" rev-parse --short HEAD)"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null
        echo
        if [[ -s $NIGHT/gpu-apps-at-start.txt ]]; then
            echo "**Other programs held VRAM when the run began, and shared the GPU with every step (listed below).**"
            echo
        fi
        echo "| step | outcome | minutes |"
        echo "|---|---|---:|"
        cat "$NIGHT/steps.tsv" 2>/dev/null | awk -F'\t' '{printf "| %s | %s | %s |\n", $1, $2, $3}'
        echo
        echo "GPU: $(gpu_report)"
        echo
        echo "Downloads: $(grep -E '^(q8|pull) ' "$NIGHT/downloads.log" 2>/dev/null | tr '\n' ';')"
        echo
        echo "Other processes on the GPU at the start (they share it with every measurement):"
        echo '```text'
        cat "$NIGHT/gpu-apps-at-start.txt" 2>/dev/null
        echo '```'
        echo "Disk at the start:"
        echo '```text'
        cat "$NIGHT/disk-at-start.txt" 2>/dev/null
        echo '```'
        echo
        echo "## The evening's phase 6: the expert cache each server had"
        echo '```text'
        evening
        echo '```'
        local name
        while IFS=$'\t' read -r name _; do
            echo
            echo "## $name"
            echo '```text'
            extract "$name"
            echo '```'
        done <"$NIGHT/steps.tsv"
    } >"$NIGHT/summary.md" 2>/dev/null
}

run() {
    local name=$1 limit t0=$SECONDS rc outcome
    limit=$(limit_of "$name")
    if (($(date +%s) >= stop)); then
        printf '%s\tnot started: past %s\t0\n' "$name" "$STOP_AT" >>"$NIGHT/steps.tsv"
        write_summary
        return
    fi
    note "$name: at most $limit minutes"
    # 9>&-: what a step starts must not hold the lock after the night ends.
    timeout --kill-after=120 "${limit}m" bash "$SELF" "step:$name" >"$NIGHT/$name.log" 2>&1 9>&-
    rc=$?
    stop_servers
    case $rc in
    0) outcome=OK ;;
    3) outcome=SKIPPED ;;
    124 | 137) outcome="TIMEOUT after $limit min" ;;
    *) outcome="FAIL (exit $rc)" ;;
    esac
    printf '%s\t%s\t%d\n' "$name" "$outcome" $(((SECONDS - t0) / 60)) >>"$NIGHT/steps.tsv"
    note "$name: $outcome"
    write_summary
}

# A step that needs a download waits for it (three hours at most).
wait_for() {
    local marker=$NIGHT/$1 waited=0
    while [[ ! -s $marker ]] && ((waited < 180)); do
        ((waited % 15 == 0)) && note "waiting for $1"
        sleep 60
        waited=$((waited + 1))
    done
}

started=$(date +%s)
: >"$NIGHT/steps.tsv"
note "results in $NIGHT/summary.md; steps: $STEPS"

# Phase 6 by hand may still be running: two checks a minute apart without it.
quiet=0
while ((quiet < 2)); do
    if pgrep -f 'bench/prefetch_check\.sh' >/dev/null; then
        quiet=0
        note "waiting for bench/prefetch_check.sh to finish"
    else
        quiet=$((quiet + 1))
    fi
    ((quiet < 2)) && sleep 60
done

if pgrep -f -- "$BIN serve" >/dev/null || pgrep -f -- "$LLAMA_PIN/bin/llama-server" >/dev/null; then
    note "stopping the servers of $BIN and llama-server still running: they would share the GPU"
    stop_servers
fi

if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used \
        --format=csv,noheader,nounits -l 60 >"$NIGHT/gpu.csv" 2>/dev/null 9>&- &
    sampler=$!
    trap 'kill $sampler 2>/dev/null' EXIT
fi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    >"$NIGHT/gpu-apps-at-start.txt" 2>/dev/null
df -h "$HOME" "$MODELS" "$STORE" 2>/dev/null | awk '!seen[$0]++' >"$NIGHT/disk-at-start.txt"

# Downloads, one after the other, behind everything else: only those a step
# of STEPS needs. On 5 October a night of three steps would have fetched the
# Q8_0 GGUF again, 9.5 GB onto a disk with 20 free, for a step it was not
# running.
in_steps() { # any of the steps named is in STEPS
    local s
    for s in "$@"; do
        [[ " $STEPS " == *" $s "* ]] && return 0
    done
    return 1
}
(
    renice -n 19 -p "$BASHPID" >/dev/null 2>&1
    ionice -c2 -n7 -p "$BASHPID" >/dev/null 2>&1
    if ! in_steps mtp-head-q8; then
        rc="0 (not needed)"
    elif [[ -s $M9_Q8 ]]; then
        rc=0
    else
        mkdir -p "$(dirname "$M9_Q8")"
        curl -fL --retry 5 -C - -o "$M9_Q8.part" "$M9_Q8_URL" && mv "$M9_Q8.part" "$M9_Q8"
        rc=$?
    fi
    echo "q8 $rc"
    echo "$rc" >"$NIGHT/q8.done"
    if in_steps residency auto soak docker-gpu; then
        for model in $PULLS; do
            "$BIN" pull "$model" </dev/null
            echo "pull $model $?"
        done
    fi
    echo done >"$NIGHT/pulls.done"
) >"$NIGHT/downloads.log" 2>&1 9>&- &

for s in $STEPS; do
    case $s in
    mtp-head-q8) wait_for q8.done ;;
    residency | auto | soak | docker-gpu) wait_for pulls.done ;;
    esac
    run "$s"
done
write_summary
note "done: $NIGHT/summary.md"
