"""Measuring one point: start its servers, warm them, measure, stop them.

Two kinds of point.

`throughput` is the method docs/cineca/leonardo.md established and the LUMI
scripts kept: N concurrent /api/generate requests, one prompt, aggregate
output tokens over the wall clock of the batch, a warm-up first and outside
the timed window — repeated, so a number carries its own spread. Streaming
adds what the proposal promised and nobody recorded: time to first token and
the prefill and decode rates per request.

`workload` is sustained load with answers that can be checked: the public
sets ReflexBench pins (GSM8K, ARC, MMLU), sent at a fixed concurrency for a
duration, looped. Every interval is a row of throughput, latency and HBM —
stability under sustained load, the proposal's own words — and the first pass
is graded, so each configuration also has the accuracy its memory and speed
cost. Later passes are compared with the first: decoding is greedy and the
prompt cache is off, so an answer that changes under load is a finding.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

from devices import VISIBLE_ENV, cores_for

HERE = os.path.dirname(os.path.abspath(__file__))
REFLEXBENCH = os.path.join(os.path.dirname(HERE), "reflexbench")

READY_TIMEOUT_S = 900
REQUEST_TIMEOUT_S = 1800
WARMUP_TIMEOUT_S = 3600  # a 400 GB model read off Lustre is the slow case

# Filler for synthetic long prompts: plain prose, so it tokenizes like text
# rather than like a repeated token the tokenizer could merge.
FILLER = (
    "Il mare era calmo quella mattina e le barche dei pescatori uscivano dal porto una "
    "dopo l'altra, mentre il sole saliva lentamente sopra le colline. Sulla banchina "
    "qualcuno riparava le reti, qualcun altro contava le casse vuote. "
)
LONG_PROMPT_TAIL = "\n\nRiassumi il testo precedente in una frase."
CHARS_PER_TOKEN = 4


class PointError(Exception):
    pass


class ModelMissing(PointError):
    """The model is not in the store: nothing this point can fix by retrying."""


class Interrupted(PointError):
    pass


class DoesNotFit(PointError):
    """--fit-strict refused the load: this configuration does not fit these
    devices. A measurement — where the memory boundary lies — not a failure."""


FIT_REFUSAL = "does not fully fit"


class RequestError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def is_missing_model(err: RequestError) -> bool:
    text = str(err).lower()
    return err.code == 404 or ("not found" in text and "model" in text)


# ── servers ──────────────────────────────────────────────────────────────


def server_args(p: dict) -> list:
    k, v = p["kv"].split("/")
    return [
        "--batch-size", str(p["batch"]),
        "--ctx-size", str(p["ctx"]),
        "--cache-type-k", k,
        "--cache-type-v", v,
    ] + list(p.get("extra_args", []))


class Server:
    def __init__(self, engine, port, args, env, log_path, cores=None):
        self.engine, self.port, self.args = engine, port, args
        self.env, self.log_path, self.cores = env, log_path, cores
        self.proc = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list:
        prefix = ["taskset", "-c", self.cores] if self.cores else []
        return prefix + [self.engine, "serve", "--port", str(self.port)] + self.args

    def start(self) -> None:
        log = open(self.log_path, "ab")
        # A session of its own, so stopping it stops whatever it started too.
        self.proc = subprocess.Popen(
            self.command(), stdout=log, stderr=subprocess.STDOUT, env=self.env,
            start_new_session=True,
        )
        log.close()

    def wait_ready(self, timeout_s=READY_TIMEOUT_S, stop=None) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if stop is not None and stop.is_set():
                raise Interrupted("stopped while waiting for the server")
            if self.proc.poll() is not None:
                raise PointError(
                    f"server on port {self.port} exited ({self.proc.returncode}):\n"
                    + self.log_tail()
                )
            try:
                with urllib.request.urlopen(self.url + "/api/version", timeout=5):
                    return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(1)
        raise PointError(f"server on port {self.port} not ready in {timeout_s} s:\n"
                         + self.log_tail())

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=30)
        except ProcessLookupError:
            pass

    def kill(self) -> None:
        """Immediately, without waiting: for a job whose time is up."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def log_tail(self, lines=40) -> str:
        try:
            with open(self.log_path, errors="replace") as f:
                return "".join(f.readlines()[-lines:])
        except OSError:
            return "(no server log)"


# ── requests ─────────────────────────────────────────────────────────────


def request(url, path, body, timeout=REQUEST_TIMEOUT_S) -> dict:
    """One Ollama-style request, streamed or not, read to the end."""
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    first, pieces, last = None, [], {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if body.get("stream", True):
                for raw in r:
                    if not raw.strip():
                        continue
                    line = json.loads(raw)
                    if "error" in line:
                        raise RequestError(str(line["error"]))
                    piece = line.get("response")
                    if piece is None:
                        piece = (line.get("message") or {}).get("content", "")
                    if piece and first is None:
                        first = time.perf_counter()
                    pieces.append(piece or "")
                    last = line
            else:
                last = json.loads(r.read())
                if "error" in last:
                    raise RequestError(str(last["error"]))
                message = last.get("message") or {}
                pieces.append(last.get("response") or message.get("content", ""))
    except urllib.error.HTTPError as e:
        raise RequestError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}",
                           code=e.code) from None
    ended = time.perf_counter()
    total_ms = (ended - started) * 1000
    return {
        "text": "".join(pieces),
        "ttft_ms": ((first or ended) - started) * 1000,
        "total_ms": total_ms,
        "eval_count": last.get("eval_count") or 0,
        "prompt_eval_count": last.get("prompt_eval_count") or 0,
        "eval_duration_ns": last.get("eval_duration") or 0,
        "prompt_eval_duration_ns": last.get("prompt_eval_duration") or 0,
        "load_duration_ns": last.get("load_duration") or 0,
    }


def concurrently(n, fn) -> list:
    """fn(0..n-1) on n threads at once; each slot a result or the exception."""
    out = [None] * n

    def run(i):
        try:
            out[i] = fn(i)
        except Exception as e:  # recorded, counted, and the batch goes on
            out[i] = e

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def synthetic_prompt(tokens: int, nonce: str) -> str:
    """About `tokens` tokens of prose, opened by a nonce so no two requests
    share a prefix a cache could reuse."""
    body = (FILLER * (tokens * CHARS_PER_TOKEN // len(FILLER) + 1))[: tokens * CHARS_PER_TOKEN]
    return f"[{nonce}] {body}{LONG_PROMPT_TAIL}"


def generate_body(p: dict, prompt: str) -> dict:
    body = {
        "model": p["model"],
        "prompt": prompt,
        "stream": p["stream"],
        "think": False,
        "num_predict": p["num_predict"],
        "options": {"num_predict": p["num_predict"]},
    }
    if p["prompt_tokens"]:
        body["cache_prompt"] = False
    return body


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def r1(x):
    return None if x is None else round(x, 1)


def rate(count, duration_s):
    return count / duration_s if count and duration_s and duration_s > 0 else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return r1(sum(xs) / len(xs)) if xs else None


# ── one point ────────────────────────────────────────────────────────────


class Context:
    """What the runner gives a point: where it runs and what it may use."""

    def __init__(self, engine, backend, binding, physical, port_base, workdir,
                 sampler=None, stop=None, model_seen=None, sets_dir=None):
        self.engine, self.backend, self.binding = engine, backend, binding
        self.physical = list(physical)  # physical ids, in order
        self.port_base, self.workdir = port_base, workdir
        self.sampler = sampler
        self.stop = stop or threading.Event()
        self.model_seen = model_seen if model_seen is not None else set()
        self.sets_dir = sets_dir
        self.servers = []  # set by run(), so the runner can stop them from outside


def start_servers(p: dict, ctx: Context) -> list:
    servers = []
    rg = p["replica_gcds"]
    for r in range(p["replicas"]):
        group = ctx.physical[r * rg : (r + 1) * rg]
        env = dict(os.environ)
        env[VISIBLE_ENV[ctx.backend]] = ",".join(group)
        if ctx.backend == "rocm":
            # HIP applies these on top of ROCR_VISIBLE_DEVICES, as indices into
            # what ROCR left visible: a job-wide "0,...,7" from Slurm would
            # point a one-GCD server at devices it no longer has.
            env.pop("HIP_VISIBLE_DEVICES", None)
            env.pop("GPU_DEVICE_ORDINAL", None)
        log = os.path.join(ctx.workdir, f"{p['id']}.server{r}.log")
        server = Server(ctx.engine, ctx.port_base + r, server_args(p), env, log,
                        cores_for(ctx.binding, group))
        server.start()
        servers.append(server)
    return servers


def warm_up(p: dict, servers: list, ctx: Context, body: dict) -> dict:
    """One request per server, all at once: loads every copy of the model.
    Untimed for throughput; timed here, because loading is a metric."""
    cache = "warm" if p["model"] in ctx.model_seen else "cold"
    t0 = time.time()
    got = concurrently(
        len(servers), lambda i: request(servers[i].url, "/api/generate", body, WARMUP_TIMEOUT_S)
    )
    wall = time.time() - t0
    for i, g in enumerate(got):
        if isinstance(g, RequestError) and FIT_REFUSAL in str(g):
            raise DoesNotFit(str(g)[:300])
        if isinstance(g, RequestError) and is_missing_model(g):
            raise ModelMissing(f"model {p['model']!r} is not in the store: {g}")
        if isinstance(g, Exception):
            raise PointError(f"warm-up failed on server {i}: {g}\n" + servers[i].log_tail())
    ctx.model_seen.add(p["model"])
    return {
        "cache": cache,
        "wall_s": round(wall, 2),
        "load_ms": [round(g["load_duration_ns"] / 1e6, 1) for g in got],
    }


def run_throughput(p: dict, servers: list, ctx: Context) -> tuple:
    prompt = p["prompt"]
    if p["prompt_tokens"]:
        def body_for(i, rep):
            return generate_body(p, synthetic_prompt(p["prompt_tokens"], uuid.uuid4().hex[:12]))
    else:
        def body_for(i, rep):
            return generate_body(p, prompt)

    reps, first_t, last_t = [], None, None
    for rep in range(p["repeats"]):
        if ctx.stop.is_set():
            raise Interrupted("stopped between repeats")
        t0 = time.time()
        got = concurrently(
            p["concurrency"],
            lambda i: request(servers[i % len(servers)].url, "/api/generate", body_for(i, rep)),
        )
        t1 = time.time()
        first_t = first_t or t0
        last_t = t1
        ok = [g for g in got if isinstance(g, dict)]
        errors = [str(g)[:200] for g in got if not isinstance(g, dict)]
        wall = t1 - t0
        gen = sum(g["eval_count"] for g in ok)
        ttft = [g["ttft_ms"] for g in ok]
        decode = [rate(g["eval_count"], (g["total_ms"] - g["ttft_ms"]) / 1000) for g in ok]
        prefill = [rate(g["prompt_eval_count"], g["ttft_ms"] / 1000) for g in ok]
        srv_prefill = [rate(g["prompt_eval_count"], g["prompt_eval_duration_ns"] / 1e9) for g in ok]
        srv_decode = [rate(g["eval_count"], g["eval_duration_ns"] / 1e9) for g in ok]
        reps.append({
            "ok": len(ok),
            "failed": len(errors),
            "errors": errors[:3],
            "generated_tokens": gen,
            "prompt_tokens": sum(g["prompt_eval_count"] for g in ok),
            "wall_clock_s": round(wall, 3),
            "aggregate_tok_s": r1(rate(gen, wall)),
            "ttft_ms_p50": r1(percentile(ttft, 0.5)),
            "ttft_ms_p95": r1(percentile(ttft, 0.95)),
            "decode_tok_s_mean": mean(decode),
            "prefill_tok_s_mean": mean(prefill),
            # From the server's own durations; null until the engine reports
            # a measured prompt time (on main it is hard-coded to 0).
            "server_prefill_tok_s_mean": mean(srv_prefill),
            "server_decode_tok_s_mean": mean(srv_decode),
        })
        if not ok:
            raise PointError(f"every request failed in repeat {rep}: {errors[:3]}")

    agg = [r["aggregate_tok_s"] for r in reps if r["aggregate_tok_s"] is not None]
    summary = {
        "aggregate_tok_s_mean": r1(statistics.mean(agg)) if agg else None,
        "aggregate_tok_s_stdev": r1(statistics.stdev(agg)) if len(agg) > 1 else None,
        "aggregate_tok_s_cv_pct": (
            round(100 * statistics.stdev(agg) / statistics.mean(agg), 2)
            if len(agg) > 1 and statistics.mean(agg) else None
        ),
        "repeats": reps,
    }
    return summary, (first_t, last_t)


def load_items(p: dict, ctx: Context) -> list:
    """The point's sets, from the JSONL files `campaign.py prefetch` wrote on
    a login node: compute nodes have no network, so nothing is downloaded
    here."""
    if REFLEXBENCH not in sys.path:
        sys.path.insert(0, REFLEXBENCH)
    import ab_data

    items = []
    for name in p["sets"]:
        path = os.path.join(ctx.sets_dir or "", f"{name}.jsonl")
        if not os.path.exists(path):
            raise ModelMissing(f"set {name!r} not prefetched ({path}): run campaign.py prefetch")
        items += ab_data.from_jsonl(path, limit=p["limit"]).items
    return items


def run_workload(p: dict, servers: list, ctx: Context, answers_path=None) -> tuple:
    if REFLEXBENCH not in sys.path:
        sys.path.insert(0, REFLEXBENCH)
    import ab_grade
    import ab_methods

    items = load_items(p, ctx)
    n = len(items)
    duration = p.get("duration_s", 0)
    start = time.time()
    deadline = start + duration if duration else None
    lock = threading.Lock()
    counter = [0]
    records = []

    def worker():
        while not ctx.stop.is_set():
            with lock:
                i = counter[0]
                counter[0] += 1
            if deadline is None and i >= n:
                return
            if deadline is not None and time.time() >= deadline:
                return
            item = items[i % n]
            server = servers[i % len(servers)]
            path, body = ab_methods.chat_body(item, p["model"], p["think"], p["max_tokens"])
            rec = {"i": i, "pass": i // n, "item": item.id, "set": item.set}
            try:
                got = request(server.url, path, body)
                rec.update(ok=True, ttft_ms=got["ttft_ms"], total_ms=got["total_ms"],
                           eval=got["eval_count"], prompt=got["prompt_eval_count"],
                           sha=hashlib.sha1(got["text"].encode()).hexdigest()[:16])
                if rec["pass"] == 0:
                    rec["text"] = got["text"]
                    rec["correct"] = bool(ab_grade.correct(item, got["text"]))
            except Exception as e:
                rec.update(ok=False, error=str(e)[:200])
            rec["t"] = time.time()
            with lock:
                records.append(rec)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(p["concurrency"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end = time.time()
    if ctx.stop.is_set():
        raise Interrupted("stopped during the workload")

    ok = [r for r in records if r["ok"]]
    if records and len(ok) < len(records) / 2:
        raise PointError(f"{len(records) - len(ok)} of {len(records)} requests failed: "
                         f"{[r.get('error') for r in records if not r['ok']][:3]}")

    # Accuracy, from the first pass.
    per_set = {}
    for r in ok:
        if r["pass"] == 0:
            s = per_set.setdefault(r["set"], {"graded": 0, "correct": 0})
            s["graded"] += 1
            s["correct"] += r["correct"]
    for s in per_set.values():
        s["accuracy"] = round(s["correct"] / s["graded"], 4) if s["graded"] else None

    # Consistency: a later pass against the first, item by item.
    first = {r["item"]: r["sha"] for r in ok if r["pass"] == 0}
    later = [r for r in ok if r["pass"] > 0 and r["item"] in first]
    same = sum(r["sha"] == first[r["item"]] for r in later)

    # The time series.
    interval = max(1, int(p["interval_s"]))
    buckets = {}
    for r in records:
        buckets.setdefault(int((r["t"] - start) // interval), []).append(r)
    series = []
    for b in range(int((end - start) // interval) + 1):
        rows = buckets.get(b, [])
        good = [r for r in rows if r["ok"]]
        lat = [r["total_ms"] for r in good]
        b0, b1 = start + b * interval, min(start + (b + 1) * interval, end)
        row = {
            "t_s": b * interval,
            "completed": len(good),
            "errors": len(rows) - len(good),
            "tok_s": r1(rate(sum(r["eval"] for r in good), b1 - b0)),
            "latency_ms_p50": r1(percentile(lat, 0.5)),
            "latency_ms_p95": r1(percentile(lat, 0.95)),
            "ttft_ms_p50": r1(percentile([r["ttft_ms"] for r in good], 0.5)),
        }
        if ctx.sampler is not None:
            dev = ctx.sampler.window(b0, b1, ctx.physical)
            row["vram_peak_mib"] = {d: s["vram_peak_mib"] for d, s in dev.items()}
            row["use_mean"] = {d: s["use_mean"] for d, s in dev.items()}
        series.append(row)

    full = [s for s in series[:-1] if s["tok_s"] is not None] or series
    tenth = max(1, len(full) // 10)
    head = [s["tok_s"] for s in full[:tenth] if s["tok_s"]]
    tail = [s["tok_s"] for s in full[-tenth:] if s["tok_s"]]
    drift = (
        round(100 * (statistics.mean(tail) - statistics.mean(head)) / statistics.mean(head), 2)
        if head and tail else None
    )

    if answers_path:
        with open(answers_path, "w") as f:
            for r in sorted((r for r in ok if r["pass"] == 0), key=lambda r: r["i"]):
                f.write(json.dumps({k: r[k] for k in ("item", "set", "correct", "text",
                                                       "ttft_ms", "total_ms", "eval")}))
                f.write("\n")

    elapsed = end - start
    return {
        "items": n,
        "duration_s": round(elapsed, 1),
        "requests": len(records),
        "failed": len(records) - len(ok),
        "passes": round(len(records) / n, 2) if n else 0,
        "generated_tokens": sum(r["eval"] for r in ok),
        "aggregate_tok_s": r1(rate(sum(r["eval"] for r in ok), elapsed)),
        "latency_ms_p50": r1(percentile([r["total_ms"] for r in ok], 0.5)),
        "latency_ms_p95": r1(percentile([r["total_ms"] for r in ok], 0.95)),
        "latency_ms_p99": r1(percentile([r["total_ms"] for r in ok], 0.99)),
        "accuracy": per_set,
        "consistency": {
            "compared": len(later),
            "identical": same,
            "rate": round(same / len(later), 4) if later else None,
        },
        "throughput_drift_pct": drift,
        "series": series,
    }, (start, end)


WARMUP_BODY = {"prompt": "Ciao.", "stream": True, "think": False, "num_predict": 8,
               "options": {"num_predict": 8}}


def run(p: dict, ctx: Context, answers_path=None) -> dict:
    """Measure `p` on `ctx.physical`; the result, or an exception."""
    servers = start_servers(p, ctx)
    ctx.servers = servers
    try:
        for s in servers:
            s.wait_ready(stop=ctx.stop)
        if p["kind"] == "throughput":
            warm_body = generate_body(p, p["prompt"] if not p["prompt_tokens"]
                                      else synthetic_prompt(p["prompt_tokens"], "warmup"))
        else:
            warm_body = dict(WARMUP_BODY, model=p["model"])
        load = warm_up(p, servers, ctx, warm_body)
        if p["kind"] == "throughput":
            measured, window = run_throughput(p, servers, ctx)
        else:
            measured, window = run_workload(p, servers, ctx, answers_path)
        stats = ctx.sampler.window(*window, ctx.physical) if ctx.sampler else {}
        return {"load": load, p["kind"]: measured, "device_stats": stats}
    finally:
        for s in servers:
            s.stop()
