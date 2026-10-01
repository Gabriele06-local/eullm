#!/usr/bin/env bash
# `"model": "auto"` checked on real hardware: what the engine's CPU tests
# cannot see — whether the router is fast enough on a GPU to be worth its
# place, whether it holds up under concurrency and over an hour of mixed
# traffic, and what it costs on a machine without a GPU. Each check prints
# PASS or FAIL; the logs, reports and the summary stay in $OUT.
#
#   tools/auto_check.sh <eullm built from the branch> <decision GGUF> [<second decision GGUF>]
#
# The decision models are GGUF paths (--decision-model): the Jev-Style 0.8B,
# and optionally the 2B, which the routing checks are then repeated with.
#
# 1. Routed answers (V7): a server routing between the pair, its candidates
#    warmed up at start; "model": "auto" on /api/generate, /api/chat and
#    /v1/chat/completions, streamed and not, answers with a candidate and
#    says so in the X-EuLLM-* headers and the last line or chunk.
# 2. Router latency (V7): /api/route at concurrency 1, 4 and 16. The plan's
#    gate, at concurrency 1: p50 <= 60 ms and p95 <= 150 ms of decision_ms.
#    At 4 and 16 the decisions queue on one worker: reported, with the share
#    that timed out to the fallback.
# 3. AutoBench (V7): bench/reflexbench/autobench.py on each pair with each
#    decision model — calls avoided, accuracy against always-large with its
#    95% CI, router p50/p95 — the tables in $OUT/3-*.md. Stage 1's answers
#    are kept per pair in $OUT, so the second decision model reuses them.
# 4. Soak (V8): DURATION seconds (one hour) of /api/chat, /v1/chat/completions,
#    /api/embed, /v1/systemone and "auto" at concurrency 8, VRAM sampled every
#    5 s: no failed request, VRAM flat after the first minute (within 5%), no
#    out-of-memory or GGML_ASSERT in the log, every audit line parses and
#    there is one per request of each kind.
# 5. Optional, CPU_DECISION=<Qwen3-0.6B GGUF> (V9): the same pair's smaller
#    model and CPU_SMALL beside it, routed with the GPU hidden
#    (CUDA_VISIBLE_DEVICES=), N=2: router latency at concurrency 1 on a CPU,
#    reported as measured — with OMP_WAIT_POLICY=PASSIVE, as a CPU-only
#    server should run (see docs/engine.md).
#
# Models are catalog names, pulled when missing (SKIP_PULL=1 to skip):
# PAIRS="qwen3-4b,qwen3-8b qwen3-1.7b,qwen3-14b" (pairs A and B of the plan,
# small first; check 1, 2 and 4 use the first), CTX=8192 for pair A — pair B
# runs at 4096 with --decision-ctx 4096, as the plan sizes it.
# EMBED=qwen3-embedding-0.6b-gguf-q8_0, pulled from EMBED_PULL
# (hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0), for check 4 and the kNN
# baseline of check 3. LIMIT=100 items per set, SETS=gsm8k,arc-easy,
# arc-challenge,mmlu. DURATION=3600. CPU_SMALL=qwen3-1.7b.
# PORT=11500, OUT=/tmp/auto-check. Needs curl and python3.
set -u
# Numbers with a decimal point whatever the locale.
export LC_ALL=C

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    sed -n '2,44p' "$0"
    exit 2
fi
BIN=$(realpath "$1")
DECISIONS=("$(realpath "$2")")
[ $# = 3 ] && DECISIONS+=("$(realpath "$3")")
for d in "${DECISIONS[@]}"; do
    [ -f "$d" ] || { echo "no decision model at $d"; exit 2; }
done
PAIRS=${PAIRS:-qwen3-4b,qwen3-8b qwen3-1.7b,qwen3-14b}
CTX=${CTX:-8192}
EMBED=${EMBED-qwen3-embedding-0.6b-gguf-q8_0}
EMBED_PULL=${EMBED_PULL:-hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0}
LIMIT=${LIMIT:-100}
SETS=${SETS:-gsm8k,arc-easy,arc-challenge,mmlu}
DURATION=${DURATION:-3600}
CPU_DECISION=${CPU_DECISION:-}
CPU_SMALL=${CPU_SMALL:-qwen3-1.7b}
PORT=${PORT:-11500}
OUT=$(realpath -m "${OUT:-/tmp/auto-check}")
URL=http://127.0.0.1:$PORT
BENCH=$(dirname "$(realpath "$0")")/../bench/reflexbench/autobench.py
read -r SMALL LARGE <<< "$(echo "${PAIRS%% *}" | tr ',' ' ')"
mkdir -p "$OUT"
: > "$OUT/summary.txt"
PASS=0
FAIL=0
PID=

result() {
    if [ "$1" = PASS ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
    printf '%-4s  %s\n' "$1" "$2" | tee -a "$OUT/summary.txt"
}

# Start a server (log file, flags...) and wait until it answers. It runs in
# $OUT, where no .env sets API keys or an IP allowlist, and audits into
# $OUT/audit-<log name>, not into the real audit trail.
up() {
    local log=$1
    shift
    local audit
    audit="$OUT/audit-$(basename "$log" .log)"
    rm -rf "$audit"
    (cd "$OUT" && EULLM_AUDIT_DIR="$audit" exec "$BIN" serve --port "$PORT" "$@") > "$log" 2>&1 &
    PID=$!
    for _ in $(seq 1 600); do
        curl -sf "$URL/api/version" > /dev/null && return 0
        kill -0 "$PID" 2> /dev/null || { echo "the server exited, see $log"; return 1; }
        sleep 0.5
    done
    echo "the server did not start, see $log"
    return 1
}

down() {
    if [ -n "$PID" ]; then
        kill "$PID" 2> /dev/null
        wait "$PID" 2> /dev/null
        PID=
    fi
}
trap down EXIT

# Wait until the log says the warm-up is over, up to 10 minutes.
warmed() {
    for _ in $(seq 1 1200); do
        grep -q "Auto routing warm-up:" "$1" && return 0
        sleep 0.5
    done
    return 1
}

# A server routing between `small` and `large` with `decision`, and an
# embedding model when there is one: routed <log> <small> <large> <decision> [flags...].
routed() {
    local log=$1 small=$2 large=$3 decision=$4
    shift 4
    up "$log" --max-loaded-models 2 \
        --auto-model "$small=Short everyday requests, simple facts and quick rewrites" \
        --auto-model "$large=Multi-step reasoning, maths, code, analysis and long answers" \
        --decision-model "$decision" ${EMBED:+--embedding-model "$EMBED"} "$@"
}

# /api/route at each concurrency, `n` requests each: one line per level,
# "<concurrency> <p50 ms> <p95 ms> <timeouts> <decided> <wall p50 ms>".
router_latency() {
    python3 - "$URL" "$1" "${@:2}" << 'PY'
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
url, n, levels = sys.argv[1], int(sys.argv[2]), [int(c) for c in sys.argv[3:]]
asks = [
    "What is the capital of France?",
    "Prove that the square root of 2 is irrational, step by step.",
    "Write a Python function that merges two sorted lists, with tests.",
    "Translate 'good morning' into Italian.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the journey, and what if it is 25 minutes late?",
    "Summarise the causes of the First World War in three paragraphs.",
    "Is 391 a prime number?",
    "Say hello.",
]
def one(i):
    body = {"model": "auto", "messages": [{"role": "user", "content": f"{asks[i % len(asks)]} (#{i})"}]}
    req = urllib.request.Request(url + "/api/route", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        route = json.load(r)
    return route, (time.perf_counter() - t) * 1000
def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")
for c in levels:
    with ThreadPoolExecutor(c) as pool:
        got = list(pool.map(one, range(n)))
    ms = [r["decision_ms"] for r, _ in got]
    wall = [w for _, w in got]
    timeouts = sum(r["reason"] == "timeout" for r, _ in got)
    decided = sum(r["reason"] == "decided" for r, _ in got)
    print(c, f"{pct(ms, 0.5):.1f}", f"{pct(ms, 0.95):.1f}", timeouts, decided, f"{pct(wall, 0.5):.1f}")
PY
}

if [ "${SKIP_PULL:-0}" != 1 ]; then
    for m in $(echo "$PAIRS" | tr ', ' '\n\n' | sort -u) ${CPU_DECISION:+"$CPU_SMALL"}; do
        "$BIN" list 2> /dev/null | grep -q "^$m[[:space:]]" || "$BIN" pull "$m" || exit 1
    done
    if [ -n "$EMBED" ]; then
        "$BIN" list 2> /dev/null | grep -q "^$EMBED[[:space:]]" || "$BIN" pull "$EMBED_PULL" || exit 1
    fi
fi
echo "eullm:     $("$BIN" -V)"
echo "pairs:     $PAIRS (checks 1, 2 and 4: $SMALL and $LARGE)"
echo "decisions: ${DECISIONS[*]}"
others=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2> /dev/null)
if [ -n "$others" ]; then
    echo "Other programs hold VRAM, so the numbers will not compare; stop them first:"
    echo "$others"
fi
echo

# 1 and 2, with each decision model.
n=0
for decision in "${DECISIONS[@]}"; do
    n=$((n + 1))
    dname=$(basename "$decision" .gguf)
    routed "$OUT/1-$n.log" "$SMALL" "$LARGE" "$decision" --ctx-size "$CTX" || exit 1
    warmed "$OUT/1-$n.log"
    warm=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/1-$n.log" | grep -o "Auto routing warm-up: .*" | head -1)
    answers=$(python3 - "$URL" << 'PY'
import json, sys, urllib.request
url = sys.argv[1]
bad = []
for path in ("/api/generate", "/api/chat", "/v1/chat/completions"):
    for stream in (False, True):
        body = {"model": "auto", "stream": stream, "think": False, "max_tokens": 16,
                "options": {"num_predict": 16}}
        if path == "/api/generate":
            body["prompt"] = "Name a colour."
        else:
            body["messages"] = [{"role": "user", "content": "Name a colour."}]
        req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                model, reason = r.headers.get("X-EuLLM-Model"), r.headers.get("X-EuLLM-Route")
                lines = [l.decode().strip() for l in r]
        except Exception as e:
            bad.append(f"{path} stream={stream}: {e}")
            continue
        lines = [l[5:].strip() if l.startswith("data:") else l for l in lines]
        objects = [json.loads(l) for l in lines if l and l != "[DONE]"]
        route = (objects[-1].get("eullm") or {}).get("route") if objects else None
        if not model or not route or route.get("model") != model or any(o.get("model") != model for o in objects):
            bad.append(f"{path} stream={stream}: header {model}/{reason}, route {route}")
        else:
            print(f"{path}{' (stream)' if stream else ''}: {model} ({reason})", file=sys.stderr)
print("; ".join(bad) if bad else "ok")
PY
    )
    if [ "$answers" = ok ]; then
        result PASS "1 routed answers with $dname on every endpoint, streamed and not; $warm"
    else
        result FAIL "1 routed answers with $dname: $answers (see $OUT/1-$n.log)"
    fi

    router_latency 48 1 4 16 > "$OUT/2-$n.txt"
    read -r _ p50 p95 timeouts decided _ < <(head -1 "$OUT/2-$n.txt")
    levels=$(awk '{printf " c=%s p50 %s p95 %s ms, %s/48 timed out, wall p50 %s ms;", $1, $2, $3, $4, $6}' "$OUT/2-$n.txt")
    if [ "$decided" -gt 0 ] && awk "BEGIN{exit !($p50 <= 60 && $p95 <= 150)}"; then
        result PASS "2 router latency with $dname:$levels the gate (c=1 p50 <= 60, p95 <= 150 ms) holds"
    else
        result FAIL "2 router latency with $dname:$levels the gate is c=1 p50 <= 60, p95 <= 150 ms ($decided/48 decided)"
    fi
    down
done

# 3. AutoBench on each pair with each decision model.
for pair in $PAIRS; do
    read -r small large <<< "$(echo "$pair" | tr ',' ' ')"
    flags=(--ctx-size "$CTX")
    [ "$pair" != "${PAIRS%% *}" ] && flags=(--ctx-size 4096 --decision-ctx 4096)
    n=0
    for decision in "${DECISIONS[@]}"; do
        n=$((n + 1))
        tag="$small-$large-$n"
        routed "$OUT/3-$tag.log" "$small" "$large" "$decision" "${flags[@]}" || exit 1
        warmed "$OUT/3-$tag.log"
        python3 "$BENCH" \
            --url "$URL" --small "$small" --large "$large" ${EMBED:+--embed-model "$EMBED"} \
            --sets "$SETS" --limit "$LIMIT" --answers "$OUT/3-$small-$large.answers.jsonl" \
            --out "$OUT/3-$tag.json" --details "$OUT/3-$tag.jsonl" > "$OUT/3-$tag.md" 2> "$OUT/3-$tag.err"
        status=$?
        down
        reflex=$(grep -c "| reflex" "$OUT/3-$tag.md")
        if [ $status = 0 ] && [ "$reflex" -gt 0 ]; then
            result PASS "3 autobench $small/$large with $(basename "$decision" .gguf): $OUT/3-$tag.md"
        else
            result FAIL "3 autobench $small/$large with $(basename "$decision" .gguf): exit $status (see $OUT/3-$tag.err)"
        fi
    done
done

# 4. Soak: mixed traffic at concurrency 8, VRAM sampled every 5 s.
routed "$OUT/4.log" "$SMALL" "$LARGE" "${DECISIONS[0]}" --ctx-size "$CTX" || exit 1
warmed "$OUT/4.log"
python3 - "$URL" "$DURATION" "$SMALL" "$LARGE" "$EMBED" > "$OUT/4.json" << 'PY'
import json, subprocess, sys, threading, time, urllib.request
url, duration, small, large, embed = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
chat = [{"role": "user", "content": "Give one tip for a tidy desk."}]
kinds = [
    ("chat", "/api/chat", {"model": small, "messages": chat, "stream": True, "options": {"num_predict": 48}}),
    ("chat.completions", "/v1/chat/completions", {"model": large, "messages": chat, "max_tokens": 48}),
    ("auto", "/api/chat", {"model": "auto", "messages": chat, "stream": False, "options": {"num_predict": 48}}),
    ("systemone", "/v1/systemone", {"state": "A customer asks when the shop opens on Sunday.",
        "questions": {"intent": {"type": "choice", "instructions": "What does the customer want?",
            "criteria": {"hours": "Opening hours", "refund": "A refund", "other": "Something else"}}}}),
]
if embed:
    kinds.append(("embed", "/api/embed", {"model": embed, "input": ["a lighthouse", "a harbour"]}))
done = {k[0]: 0 for k in kinds}
failed = []
vram = []
stop = time.time() + duration
lock = threading.Lock()
def worker(w):
    i = w
    while time.time() < stop:
        name, path, body = kinds[i % len(kinds)]
        i += 1
        req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                text = r.read().decode()
            if '"error"' in text:
                raise RuntimeError(text[:200])
            with lock:
                done[name] += 1
        except Exception as e:
            with lock:
                failed.append(f"{name}: {e}"[:300])
def sample():
    while time.time() < stop:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.split()
            if out:
                vram.append(int(out[0]))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        time.sleep(5)
threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)] + [threading.Thread(target=sample)]
for t in threads:
    t.start()
for t in threads:
    t.join()
settled = vram[12:] or vram
print(json.dumps({"done": done, "failed": failed[:20], "failures": len(failed),
                  "vram_min": min(settled) if settled else None, "vram_max": max(settled) if settled else None,
                  "samples": len(vram)}))
PY
down
soak=$(python3 - "$OUT/4.json" "$OUT/audit-4/audit.jsonl" "$OUT/4.log" << 'PY'
import json, re, sys
report = json.load(open(sys.argv[1]))
done = report["done"]
problems = []
if report["failures"]:
    problems.append(f"{report['failures']} failed requests, e.g. {report['failed'][:2]}")
lo, hi = report["vram_min"], report["vram_max"]
if lo is not None and hi > lo * 1.05:
    problems.append(f"VRAM moved from {lo} to {hi} MiB after the first minute")
log = re.sub(r"\x1b\[[0-9;]*m", "", open(sys.argv[3], errors="replace").read())
if re.search(r"GGML_ASSERT|out of memory|failed to allocate", log, re.I):
    problems.append("an out-of-memory or GGML_ASSERT in the log")
counts, bad = {}, 0
try:
    for line in open(sys.argv[2]):
        try:
            entry = json.loads(line)
        except ValueError:
            bad += 1
            continue
        counts[entry["request_type"]] = counts.get(entry["request_type"], 0) + 1
except OSError:
    problems.append("no audit trail")
if bad:
    problems.append(f"{bad} audit lines do not parse")
# Each routed request writes a route line and a chat line; a request
# answered while the soak ended may not be counted on the client side.
want = {"chat": done["chat"] + done["auto"], "chat.completions": done["chat.completions"],
        "route": done["auto"], "systemone": done["systemone"]}
for kind, n in want.items():
    if not n <= counts.get(kind, 0) <= n + 8:
        problems.append(f"{counts.get(kind, 0)} '{kind}' audit lines for {n} requests")
vram = "" if lo is None else f", VRAM {lo}-{hi} MiB over {report['samples']} samples"
summary = ", ".join(f"{n} {k}" for k, n in done.items()) + vram
print(("FAIL " + "; ".join(problems) + f" ({summary})") if problems else f"PASS {summary}")
PY
)
result "${soak%% *}" "4 soak, $DURATION s at concurrency 8: ${soak#* }"

# 5. Router latency on a CPU.
if [ -n "$CPU_DECISION" ]; then
    # The last check: the GPU stays hidden from here on.
    export CUDA_VISIBLE_DEVICES= OMP_WAIT_POLICY=PASSIVE
    routed "$OUT/5.log" "$CPU_SMALL" "$SMALL" "$(realpath "$CPU_DECISION")" --ctx-size 4096 || exit 1
    warmed "$OUT/5.log"
    router_latency 16 1 > "$OUT/5.txt"
    down
    read -r _ p50 p95 timeouts decided wall < "$OUT/5.txt"
    if [ "$((decided + timeouts))" -gt 0 ]; then
        result PASS "5 router on a CPU ($CPU_SMALL/$SMALL, $(basename "$CPU_DECISION" .gguf)): decision p50 $p50 ms, p95 $p95 ms, $timeouts/16 past the 1000 ms timeout, wall p50 $wall ms"
    else
        result FAIL "5 router on a CPU: no route answered (see $OUT/5.log)"
    fi
fi

echo
echo "$PASS passed, $FAIL failed. Logs and the summary are in $OUT."
[ "$FAIL" = 0 ]
