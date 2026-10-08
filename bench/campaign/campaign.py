#!/usr/bin/env python3
"""Keep every device of a node measuring something, for as long as the job runs.

    campaign.py plan     SPEC.json... --queue DIR      expand specs into the queue
    campaign.py pulls    SPEC.json... [--engine BIN]   models to pull for them (login node)
    campaign.py f32s     SPEC.json... --queue DIR      F32 models finetune points still need
    campaign.py prefetch --queue DIR [--sets ...]      fetch the sets and finetune text (login)
    campaign.py run      --queue DIR [--devices 0-7]   drain the queue on this node
    campaign.py status   --queue DIR                   what is waiting, running, done
    campaign.py unblock  --queue DIR                   retry points blocked on a model
    campaign.py retry    --queue DIR [--group PREFIX]  failed points back, once fixed
    campaign.py collect  --queue DIR [--out FILE.csv]  one row per measured point
    campaign.py budget   [--start --end --budget-node-hours]   spend against the calendar

`run` is the job's body. It packs points onto the node's devices — a pair of
GCDs on one MI250X module, half a node, the whole node, or one GCD each for
the narrow ones — and starts the next point as soon as devices free up. A wide
point that does not fit yet reserves the devices it is waiting for, and
narrower points only take them meanwhile if they will be finished by then
(EASY backfill). Workload points stretch between their minimum and maximum
duration to fill exactly the time that is free, which is what keeps the last
hours of a 48-hour job from being billed idle.

Several jobs can run against one queue at once, on different nodes: a claim
is an atomic rename (see fsqueue.py).

Every measured point leaves results/<campaign>/<point>.<job>.json and prints
the same object on one `BENCH_RESULT {...}` line, schema "eullm.bench/1",
with the provenance a number needs to be reproduced: engine version and
binary hash, this repository's revision, devices, core binding, and how many
other points shared the node at the time.

A `finetune` point runs `eullm finetune` on one device instead of a server:
an F32 GGUF from <queue>/f32/ trained on public text from <queue>/sets/, its
report (loss before and after, training tokens per second, memory) the
measurement. The trained model is deleted unless the point keeps it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import budget  # noqa: E402
import point  # noqa: E402
from devices import NodeSampler, aligned_group, cores_for, parse_devices, physical_map  # noqa: E402
from fsqueue import Queue  # noqa: E402
from spec import expand, order_key, width  # noqa: E402

SCHEMA = "eullm.bench/1"
REPO = os.path.dirname(os.path.dirname(HERE))
# MMLU is not here: its pinned source (people.eecs.berkeley.edu/~hendrycks/
# data.tar) answers 404 as of 05-10-2026.
DEFAULT_SETS = ("gsm8k", "arc-easy", "arc-challenge", "finetune-gsm8k-train")
# Text for `finetune` points to train on: public, pinned like the sets above,
# and never a split a workload point grades on (GSM8K's test split is one).
FINETUNE_SETS = {
    "finetune-gsm8k-train": (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl",
        "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    ),
}
# What a workload point needs on top of its duration: servers up, model read
# (a 400 GB GGUF off Lustre takes minutes), the requests in flight finished.
LOAD_ALLOWANCE_S = 1200
# A claim older than this, with no way to ask Slurm, is taken as abandoned:
# no job on LUMI lives longer than 48 hours.
STALE_CLAIM_S = 50 * 3600
# How often an otherwise idle scheduler looks at the queue again.
REPLAN_S = 120
PARAM_SKIP = ("id", "group", "campaign", "priority", "est_s", "attempts", "notes")


def now_iso(t=None) -> str:
    return dt.datetime.fromtimestamp(t or time.time(), dt.timezone.utc).isoformat(
        timespec="seconds")


def sh(cmd, timeout=60) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout if out.returncode == 0 else ""


def parse_time_left(text: str):
    """squeue's %L — '1-23:59:30', '23:59:30', '59:30' — in seconds, or None."""
    text = text.strip()
    if not text or not text[0].isdigit():
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(x) for x in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + s


def alive_jobs():
    """Job ids of this user still queued or running; None if Slurm cannot be
    asked (then claims are judged by age)."""
    user = os.environ.get("USER", "")
    out = sh(["squeue", "-h", "-u", user, "-o", "%i"]) if user else ""
    return set(out.split()) if out else None


def engine_provenance(engine: str) -> dict:
    info = {"path": engine, "version": None, "sha256": None}
    first = sh([engine, "--version"], timeout=30).strip().splitlines()
    info["version"] = first[0] if first else None
    try:
        h = hashlib.sha256()
        with open(engine, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        info["sha256"] = h.hexdigest()[:16]
    except OSError:
        pass
    return info


def bench_revision() -> str:
    rev = sh(["git", "-C", REPO, "rev-parse", "--short", "HEAD"]).strip()
    dirty = sh(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"]).strip()
    return (rev or "unknown") + ("-dirty" if dirty else "")


def runtime_info(backend: str) -> dict:
    if backend == "rocm":
        root = os.environ.get("ROCM_PATH", "/opt/rocm")
        try:
            with open(os.path.join(root, ".info", "version")) as f:
                return {"rocm": f.read().strip()}
        except OSError:
            return {"rocm": None}
    drv = sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return {"nvidia_driver": drv.strip().splitlines()[0] if drv.strip() else None}


def cores_set(text: str) -> set:
    return set(parse_devices(text)) if text else set()


class Running:
    def __init__(self, p, use, reserved, duration, ctx, expected_end):
        self.p, self.use, self.reserved = p, use, reserved
        self.duration, self.ctx, self.expected_end = duration, ctx, expected_end
        self.started = time.time()
        self.thread = None


class Runner:
    def __init__(self, args):
        self.args = args
        self.queue = Queue(args.queue)
        self.results = args.results or os.path.join(args.queue, "results")
        self.job = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"
        self.host = socket.gethostname()
        self.logs = os.path.join(self.results, "logs", self.job)
        os.makedirs(self.logs, exist_ok=True)
        self.devices = parse_devices(args.devices)
        self.physical = physical_map(args.backend, self.devices)
        self.free = set(self.devices)
        self.running = {}
        self.finished = []
        self.draining = []  # (servers that outlived their kill, their devices)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        # Set when a point finishes or a stop arrives, so the next point
        # starts at once rather than at the next poll.
        self.wake = threading.Event()
        self.model_seen = set()
        self.busy_s = {d: 0.0 for d in self.devices}
        self.counts = {"done": 0, "failed": 0, "blocked": 0, "released": 0, "retry": 0}
        self.started = time.time()
        walltime = args.walltime_s
        if walltime is None:
            jobid = os.environ.get("SLURM_JOB_ID")
            walltime = parse_time_left(sh(["squeue", "-h", "-j", jobid, "-o", "%L"])) \
                if jobid else None
        self.deadline = self.started + (walltime or 48 * 3600)
        try:
            self.allowed_cores = os.sched_getaffinity(0)
        except AttributeError:
            self.allowed_cores = set()
        self.sampler = NodeSampler(
            args.backend, interval_s=args.sample_s,
            log_path=os.path.join(self.results, f"{self.job}.node.jsonl"),
        )
        self.engine = engine_provenance(args.engine)
        self.revision = bench_revision()
        self.runtime = runtime_info(args.backend)
        self.sets_dir = os.path.join(args.queue, "sets")

    # ── scheduling ───────────────────────────────────────────────────────

    def width(self, p) -> int:
        """Devices `p` keeps to itself here: all of this node's when it must
        run alone, whatever the node's size."""
        return len(self.devices) if p.get("exclusive") else width(p)

    def devices_for(self, p):
        """(used, reserved) device lists, or None if they are not free."""
        w = self.width(p)
        block = aligned_group(self.free, self.devices, w)
        if block is None:
            return None
        return block[: p["gcds"]], block

    def duration_for(self, p, limit_s):
        """How long `p` will run if started now with `limit_s` seconds to
        spare: None if it cannot fit, 0 for a point with a fixed length."""
        if p["kind"] == "workload" and (p["min_duration_s"] or p["max_duration_s"]):
            avail = limit_s - LOAD_ALLOWANCE_S
            d = min(p["max_duration_s"], avail) if p["max_duration_s"] else avail
            return int(d) if d >= max(p["min_duration_s"], 60) else None
        return 0 if p.get("est_s", 900) <= limit_s else None

    def expected_end(self, p, duration, t):
        return t + (duration + LOAD_ALLOWANCE_S if duration else p.get("est_s", 900))

    def when_free(self, w, now):
        """When an aligned block of `w` devices will be free, freeing the
        running points' devices in the order they are expected to end. A
        point already past its estimate is given a quarter of it again, so
        one overrun does not freeze backfill for everything else."""
        ends = []
        for r in self.running.values():
            end = r.expected_end
            if end <= now:
                end = now + max(300, 0.25 * (r.expected_end - r.started))
            ends.append((end, r.reserved))
        free = set(self.free)
        for end, reserved in sorted(ends, key=lambda e: e[0]):
            free |= set(reserved)
            if aligned_group(free, self.devices, w) is not None:
                return end
        return max((e for e, _ in ends), default=now)

    def plan_next(self, now, todo):
        """The next point to start from `todo` (sorted), with its devices and
        duration, or None."""
        time_left = self.deadline - now - self.args.margin_s
        reserve_at = None
        for p in todo:
            if self.width(p) > len(self.devices):
                continue
            limit = time_left if reserve_at is None else min(time_left, reserve_at - now)
            duration = self.duration_for(p, limit)
            if duration is None:
                continue
            got = self.devices_for(p)
            if got is None:
                if reserve_at is None:
                    reserve_at = self.when_free(self.width(p), now)
                continue
            return p, got[0], got[1], duration
        return None

    # ── execution ────────────────────────────────────────────────────────

    def launch(self, p, use, reserved, duration):
        physical = [self.physical[d] for d in use]
        cores = cores_for(self.args.bind, physical)
        if cores and not cores_set(cores) <= self.allowed_cores:
            cores = None  # a partial allocation: those cores are not ours
        ctx = point.Context(
            self.args.engine, self.args.backend, self.args.bind if cores else "none",
            physical, self.args.port_base + 10 * min(reserved), self.logs,
            sampler=self.sampler, stop=self.stop, model_seen=self.model_seen,
            sets_dir=self.sets_dir, f32_dir=os.path.join(self.args.queue, "f32"),
        )
        run = Running(dict(p, duration_s=duration), use, reserved, duration, ctx,
                      self.expected_end(p, duration, time.time()))
        neighbours = len(self.running)
        busy = len(self.devices) - len(self.free)
        with self.lock:
            self.free -= set(reserved)
            self.running[p["id"]] = run
        run.thread = threading.Thread(
            target=self._execute, args=(run, cores, neighbours, busy), daemon=True)
        run.thread.start()
        print(f"[{now_iso()}] start {p['id']} on {physical}"
              f"{' (exclusive)' if p.get('exclusive') else ''}"
              f"{f' for {duration}s' if duration else ''}", flush=True)

    def _execute(self, run, cores, neighbours, busy):
        p = run.p
        result = {
            "schema": SCHEMA,
            "campaign": p["campaign"],
            "point": p["id"],
            "group": p["group"],
            "kind": p["kind"],
            "params": {k: v for k, v in p.items() if k not in PARAM_SKIP},
            "site": self.args.site,
            "host": self.host,
            "job": self.job,
            "devices": run.ctx.physical,
            "cores": cores,
            "neighbours_at_start": neighbours,
            "devices_busy_at_start": busy,
            "engine": self.engine,
            "bench_rev": self.revision,
            "runtime": self.runtime,
            "started": now_iso(run.started),
        }
        state, note = "done", ""
        answers = os.path.join(self.results, p["campaign"], f"{p['id']}.{self.job}.answers.jsonl")
        os.makedirs(os.path.dirname(answers), exist_ok=True)
        try:
            result.update(point.run(run.p, run.ctx,
                                    answers_path=answers if p["kind"] == "workload" else None))
            result["ok"], result["error"], result["outcome"] = True, None, "measured"
        except point.Interrupted as e:
            state, note = "released", str(e)
        except point.ModelMissing as e:
            state, note = "blocked", str(e)
        except point.DoesNotFit as e:
            result["ok"], result["error"], result["outcome"] = False, str(e), "does-not-fit"
        except Exception as e:
            if self.stop.is_set():
                # The job is ending and its servers are being killed: whatever
                # broke, it broke because of that. The point goes back.
                state, note = "released", f"stopped: {type(e).__name__}"
            else:
                state, note = "failed", f"{type(e).__name__}: {e}"
                result["ok"], result["error"], result["outcome"] = False, note, "failed"
                result["traceback"] = traceback.format_exc()[-4000:]
        ended = time.time()
        result["ended"], result["elapsed_s"] = now_iso(ended), round(ended - run.started, 1)
        if state in ("done", "failed"):
            sub = "" if state == "done" else "failed"
            out = os.path.join(self.results, p["campaign"], sub, f"{p['id']}.{self.job}.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                json.dump(result, f, indent=1)
            if state == "done":
                print("BENCH_RESULT " + json.dumps(result, separators=(",", ":")), flush=True)
        with self.lock:
            self.finished.append((run, state, note, ended))
        self.wake.set()

    def reap(self) -> int:
        with self.lock:
            finished, self.finished = self.finished, []
        # Devices whose server outlived its kill come back when it is gone.
        freed = 0
        for straggler in list(self.draining):
            if all(s.proc.poll() is not None for s in straggler[0]):
                self.draining.remove(straggler)
                with self.lock:
                    self.free |= set(straggler[1])
                print(f"[{now_iso()}] devices {sorted(straggler[1])} free again", flush=True)
                freed += 1
        for run, state, note, ended in finished:
            pid = run.p["id"]
            with self.lock:
                self.running.pop(pid, None)
                if run.ctx.stragglers:
                    self.draining.append((run.ctx.stragglers, set(run.reserved)))
                    print(f"[{now_iso()}] {pid}: a server outlived its kill; devices "
                          f"{sorted(run.reserved)} wait for it", flush=True)
                else:
                    self.free |= set(run.reserved)
            for d in run.reserved:
                self.busy_s[d] += ended - run.started
            try:
                if state == "released":
                    self.queue.release(pid)
                    final = "released"
                else:
                    final = self.queue.finish(pid, state, note)
            except FileNotFoundError:
                final = "released"  # already put back by the stop path
            key = "retry" if (state == "failed" and final == "todo") else final
            self.counts[key] = self.counts.get(key, 0) + 1
            print(f"[{now_iso()}] {final:8s} {pid} after {ended - run.started:.0f}s"
                  f"{': ' + note[:300] if note else ''}", flush=True)
        return len(finished) + freed

    def shutdown(self):
        """Stop: kill every server at once (Slurm allows seconds, not
        minutes), give the point threads a moment, put their claims back."""
        self.stop.set()
        for run in list(self.running.values()):
            for s in getattr(run.ctx, "servers", []):
                s.kill()
        deadline = time.time() + 20  # all of them together, not 20 s each
        for run in list(self.running.values()):
            run.thread.join(timeout=max(0.0, deadline - time.time()))
        self.reap()
        for pid in list(self.running):
            try:
                self.queue.release(pid)
                self.counts["released"] += 1
            except FileNotFoundError:
                pass

    def loop(self):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            try:
                signal.signal(sig, lambda *_: (self.stop.set(), self.wake.set()))
            except ValueError:
                pass  # not the main thread (the tests): stop is set directly
        released = self.queue.requeue_stale(alive_jobs(), time.time(), STALE_CLAIM_S)
        if released:
            print(f"[{now_iso()}] put back {len(released)} points left by ended jobs", flush=True)
        self.sampler.start()
        print(f"[{now_iso()}] job {self.job} on {self.host}: devices {self.devices} "
              f"(physical {list(self.physical.values())}), "
              f"{(self.deadline - time.time()) / 3600:.1f} h, engine {self.engine['version']}",
              flush=True)
        owner = {"job": self.job, "host": self.host}
        replan_at = 0.0
        while not self.stop.is_set():
            # Reading the queue is a directory listing and a file per point
            # on Lustre: do it when devices free up, not on every poll, and
            # every few minutes for points other jobs put back or added.
            if self.reap() or time.time() >= replan_at:
                todo = sorted((p for p in self.queue.todo() if point.can_run(p)),
                              key=order_key)
                while todo and not self.stop.is_set():
                    choice = self.plan_next(time.time(), todo)
                    if choice is None:
                        break
                    p, use, reserved, duration = choice
                    todo.remove(p)
                    claimed = self.queue.claim(p["id"], owner)
                    if claimed is not None:
                        self.launch(claimed, use, reserved, duration)
                replan_at = time.time() + REPLAN_S
            if not self.running and not self.draining:
                break
            self.wake.wait(self.args.poll_s)
            self.wake.clear()
        if self.stop.is_set():
            print(f"[{now_iso()}] stopping: releasing {len(self.running)} running points",
                  flush=True)
            self.shutdown()
        self.sampler.stop()
        self.summary()

    def summary(self):
        ended = time.time()
        span = max(ended - self.started, 1)
        stats = self.sampler.window(self.started, ended, list(self.physical.values()))
        doc = {
            "schema": SCHEMA + "/job",
            "job": self.job,
            "host": self.host,
            "started": now_iso(self.started),
            "ended": now_iso(ended),
            "elapsed_s": round(span, 1),
            "points": self.counts,
            "queue": self.queue.counts(),
            # The node-usage evidence: the share of the job each device had a
            # point assigned, and how busy it actually was.
            "assigned_fraction": {self.physical[d]: round(self.busy_s[d] / span, 3)
                                  for d in self.devices},
            "devices": stats,
            "engine": self.engine,
            "bench_rev": self.revision,
        }
        with open(os.path.join(self.results, f"{self.job}.summary.json"), "w") as f:
            json.dump(doc, f, indent=1)
        print("JOB_SUMMARY " + json.dumps(doc, separators=(",", ":")), flush=True)


# ── commands ─────────────────────────────────────────────────────────────


def hf_ref_to_id(ref: str) -> str:
    """The id `eullm pull hf.co/<owner>/<repo>[:<quant>]` stores a model
    under — the engine's own rule (models/pull.rs, hf_ref_to_model_id), so a
    spec can name a model before it is pulled."""
    body = ref.split("hf.co/", 1)[-1]
    repo, _, quant = body.partition(":")
    name = repo.rstrip("/").rsplit("/", 1)[-1]
    base = f"{name}-{quant}" if quant else name
    out = "".join(c if (c.isascii() and c.isalnum()) or c in "._-" else "-" for c in base.lower())
    return out.strip("-") or "model"


def load_specs(paths) -> list:
    specs = []
    for path in paths:
        with open(path) as f:
            specs.append((path, json.load(f)))
    return specs


def pull_refs(specs) -> list:
    """(model id, what to pull it as) for every model the specs serve: a
    `pull` entry of the spec when one maps to the id, else the id itself (a
    catalog model). Finetune points are not here: they train F32 files (see
    `f32_refs`), which no store holds."""
    refs, models = {}, set()
    for _, spec in specs:
        refs.update({hf_ref_to_id(r): r for r in spec.get("pull", [])})
        models |= {p["model"] for p in expand(spec) if p["kind"] != "finetune"}
    return [(m, refs.get(m, m)) for m in sorted(models)]


def f32_refs(specs) -> list:
    """(file, Hugging Face repo) for every model a finetune point trains: a
    file under <queue>/f32/, converted from the repo the spec's `f32` map
    names for it (None when it names none)."""
    files = {}
    for _, spec in specs:
        repos = spec.get("f32", {})
        for p in expand(spec):
            if p["kind"] == "finetune" and not os.path.isabs(p["model"]):
                files[p["model"]] = repos.get(p["model"])
    return sorted(files.items())


def has_finetune(engine) -> bool:
    return bool(sh([engine, "finetune", "--help"], timeout=60))


def missing_f32(queue, specs) -> list:
    return [(f, repo) for f, repo in f32_refs(specs)
            if not os.path.exists(os.path.join(queue, "f32", f))]


def node_hours(points, node_devices=8) -> float:
    """What the points cost at most: each holds its devices for its estimate,
    or for its maximum duration plus loading."""
    total = 0.0
    for p in points:
        seconds = (p["max_duration_s"] + LOAD_ALLOWANCE_S
                   if p["kind"] == "workload" and p["max_duration_s"] else p.get("est_s", 900))
        total += width(p) * seconds / 3600 / node_devices
    return total


def missing_models(engine, specs) -> list:
    listed = set(sh([engine, "list"], timeout=120).split())
    return [(m, ref) for m, ref in pull_refs(specs) if m not in listed]


def cmd_plan(args):
    q = Queue(args.queue)
    os.makedirs(os.path.join(args.queue, "specs"), exist_ok=True)
    specs = load_specs(args.specs)
    for path, spec in specs:
        points = expand(spec, args.round)
        added = [p for p in points if q.add(p)]
        name = os.path.basename(path)
        if args.round:
            name = name.replace(".json", f".{args.round}.json")
        with open(os.path.join(args.queue, "specs", name), "w") as f:
            json.dump(spec, f, indent=1)
        print(f"{path}{f' (round {args.round})' if args.round else ''}: {len(points)} points, "
              f"{len(added)} new, up to {node_hours(added):.0f} node-hours")
    print("queue:", q.counts())
    if args.engine:
        missing = missing_models(args.engine, specs)
        if missing:
            print("\nNot in the model store yet — pull them on a login node, or their "
                  "points will block:")
            for _, ref in missing:
                print(f"  {args.engine} pull {ref}")
    if args.engine and f32_refs(specs) and not has_finetune(args.engine):
        print(f"\n{args.engine} has no `finetune` command (a build from before it): the "
              "finetune points will block until EULLM_BIN is a build that has it "
              "(tools/lumi/build_engine.sh), then `campaign.py unblock`.")
    missing = missing_f32(args.queue, specs)
    if missing:
        print(f"\nF32 models for the finetune points, not in {args.queue}/f32 yet — "
              "tools/lumi/make_f32_models.sh converts them on a login node, or their points "
              "will block:")
        for f, repo in missing:
            print(f"  {f}  ← {repo or '(no repo in the spec f32 map)'}")
    return 0


def cmd_pulls(args):
    """What to pull for these specs, one reference per line: only what the
    store lacks when --engine is given."""
    specs = load_specs(args.specs)
    todo = missing_models(args.engine, specs) if args.engine else pull_refs(specs)
    for _, ref in todo:
        print(ref)
    return 0


def cmd_f32s(args):
    """The F32 models the specs' finetune points train, one `file repo` per
    line: only those not yet under <queue>/f32 (make_f32_models.sh reads it)."""
    for f, repo in missing_f32(args.queue, load_specs(args.specs)):
        print(f, repo or "-")
    return 0


def gsm8k_train_text(data: bytes) -> list:
    """GSM8K's training split as documents for `eullm finetune`: the question,
    the worked solution, the number. The calculator annotations
    (`<<48/2=24>>`) are dropped: the model is to write the arithmetic out."""
    docs = []
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        steps, _, final = row["answer"].rpartition("####")
        steps = re.sub(r"<<[^>]*>>", "", steps).strip()
        docs.append(f"Question: {row['question'].strip()}\n{steps}\nAnswer: {final.strip()}")
    return docs


def prefetch_finetune(name, out_dir) -> str:
    import ab_data
    from rb_data import fetch

    url, sha256 = FINETUNE_SETS[name]
    docs = gsm8k_train_text(ab_data.checked(fetch(url), sha256, url))
    path = os.path.join(out_dir, f"{name}.jsonl")
    with open(path + ".tmp", "w") as f:
        for text in docs:
            f.write(json.dumps({"text": text}) + "\n")
    os.replace(path + ".tmp", path)
    return f"{name}: {len(docs)} documents → {path}"


def cmd_prefetch(args):
    """Fetch the ReflexBench sets and the finetune text, and freeze them as
    JSONL beside the queue: compute nodes have no network."""
    sys.path.insert(0, point.REFLEXBENCH)
    import ab_data

    out_dir = os.path.join(args.queue, "sets")
    os.makedirs(out_dir, exist_ok=True)
    failed = []
    for name in args.sets.split(","):
        # One unreachable source must not cost the others: each set is
        # fetched on its own and what failed is reported at the end.
        try:
            if name in FINETUNE_SETS:
                print(prefetch_finetune(name, out_dir))
                continue
            data = ab_data.load(name)
        except Exception as e:
            failed.append(f"{name}: {type(e).__name__}: {e}")
            continue
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path + ".tmp", "w") as f:
            for item in data.items:
                row = {"id": item.id, "answer": item.answer, "grader": item.grader}
                row.update({"prompt": item.prompt} if item.prompt is not None
                           else {"messages": item.messages})
                f.write(json.dumps(row) + "\n")
        os.replace(path + ".tmp", path)
        print(f"{name}: {len(data.items)} items → {path}")
    for line in failed:
        print(f"[!!] {line[:300]}", file=sys.stderr)
    return 1 if failed else 0


def cmd_run(args):
    if not args.engine or not os.path.exists(args.engine):
        print(f"engine binary not found: {args.engine!r} (set EULLM_BIN)", file=sys.stderr)
        return 2
    Runner(args).loop()
    return 0


def cmd_status(args):
    q = Queue(args.queue)
    counts = q.counts()
    print("queue:", "  ".join(f"{k} {v}" for k, v in counts.items()))
    now = time.time()
    for pid, owner in q.running():
        if owner:
            age = (now - owner.get("claimed", now)) / 3600
            print(f"  running {pid}  job {owner.get('job')} on {owner.get('host')}, {age:.1f} h")
        else:
            print(f"  running {pid}  (no owner yet)")
    for state in ("failed", "blocked"):
        for pid in q.ids(state):
            notes = q.load(state, pid).get("notes") or [""]
            print(f"  {state} {pid}: {notes[-1][:200]}")
    todo = q.todo()
    if todo:
        by_group = {}
        for p in todo:
            by_group[p["group"]] = by_group.get(p["group"], 0) + 1
        print("  waiting by group:", ", ".join(f"{g} {n}" for g, n in sorted(by_group.items())))
    return 0


def cmd_retry(args):
    moved = Queue(args.queue).retry(args.group or "")
    print(f"{len(moved)} failed points back in todo")
    return 0


def cmd_unblock(args):
    moved = Queue(args.queue).unblock()
    print(f"{len(moved)} points back in todo")
    return 0


COLUMNS = (
    "campaign", "round", "group", "point", "kind", "outcome", "model", "gcds", "replicas",
    "mode", "batch", "ctx", "slot_ctx", "kv", "concurrency", "prompt_tokens", "job", "host",
    "devices", "neighbours_at_start", "load_cache", "load_storage", "load_wall_s",
    "agg_tok_s_mean", "agg_tok_s_cv_pct", "ttft_ms_p50", "decode_tok_s", "prefill_tok_s",
    "duration_s", "requests", "accuracy", "consistency", "drift_pct", "vram_peak_mib_max",
    "use_mean", "engine", "bench_rev", "ft_ctx", "optimizer", "train_tensors", "ft_loss_before",
    "ft_loss_after", "ft_tok_s", "ft_trainable_params", "lr", "epochs", "ft_mem_est_mib",
    "runtime", "decision_mode", "state_tokens", "questions", "dec_per_s", "dec_client_ms_p50",
    "dec_client_ms_p99", "dec_wait_ms_p50", "dec_decode_ms_p50", "dec_consistency",
    "extra_args", "trial",
)


def row_of(r: dict) -> dict:
    p = r.get("params", {})
    load = r.get("load", {})
    row = {k: p.get(k) for k in COLUMNS if k in p}
    row.update(campaign=r.get("campaign"), group=r.get("group"), point=r.get("point"),
               kind=r.get("kind"), outcome=r.get("outcome"), job=r.get("job"),
               host=r.get("host"),
               devices=" ".join(r.get("devices", [])),
               neighbours_at_start=r.get("neighbours_at_start"),
               load_cache=load.get("cache"), load_storage=load.get("storage"),
               load_wall_s=load.get("wall_s"),
               engine=(r.get("engine") or {}).get("version"), bench_rev=r.get("bench_rev"),
               extra_args=" ".join(p.get("extra_args", [])))
    t = r.get("throughput")
    if t:
        reps = t.get("repeats") or [{}]
        row.update(agg_tok_s_mean=t.get("aggregate_tok_s_mean"),
                   agg_tok_s_cv_pct=t.get("aggregate_tok_s_cv_pct"),
                   ttft_ms_p50=reps[-1].get("ttft_ms_p50"),
                   decode_tok_s=reps[-1].get("decode_tok_s_mean"),
                   prefill_tok_s=reps[-1].get("prefill_tok_s_mean"))
    w = r.get("workload")
    if w:
        row.update(agg_tok_s_mean=w.get("aggregate_tok_s"), duration_s=w.get("duration_s"),
                   requests=w.get("requests"), drift_pct=w.get("throughput_drift_pct"),
                   accuracy=" ".join(f"{k}={v.get('accuracy')}"
                                     for k, v in sorted(w.get("accuracy", {}).items())),
                   consistency=(w.get("consistency") or {}).get("rate"))
    ft = r.get("finetune")
    if ft:
        epochs = ft.get("per_epoch") or []
        last = (epochs[-1].get("validation") or {}) if epochs else {}
        speeds = [e.get("train_tok_s") for e in epochs if e.get("train_tok_s")]
        row.update(ft_loss_before=(ft.get("baseline") or {}).get("loss"),
                   ft_loss_after=last.get("loss"),
                   ft_tok_s=round(sum(speeds) / len(speeds), 1) if speeds else None,
                   ft_trainable_params=ft.get("trainable_params"),
                   ft_mem_est_mib=round(((ft.get("memory_estimate") or {}).get("total") or 0)
                                        / 2**20) or None,
                   train_tensors=",".join(p.get("train_tensors") or []))
    d = r.get("decision")
    if d:
        row.update(dec_per_s=d.get("decisions_per_s"),
                   dec_client_ms_p50=d.get("client_ms_p50"),
                   dec_client_ms_p99=d.get("client_ms_p99"),
                   dec_wait_ms_p50=d.get("wait_ms_p50"),
                   dec_decode_ms_p50=d.get("decode_ms_p50"),
                   dec_consistency=(d.get("consistency") or {}).get("rate"),
                   duration_s=d.get("duration_s"), requests=d.get("requests"))
    row["runtime"] = p.get("runtime", "eullm")
    devs = r.get("device_stats")
    if isinstance(devs, dict):
        peaks = [d.get("vram_peak_mib") for d in devs.values() if d.get("vram_peak_mib")]
        uses = [d.get("use_mean") for d in devs.values() if d.get("use_mean") is not None]
        row["vram_peak_mib_max"] = max(peaks) if peaks else None
        row["use_mean"] = round(sum(uses) / len(uses), 1) if uses else None
    return row


def result_rows(results: str) -> list:
    """A summary row for every result kept under `results` (failed ones,
    in `<campaign>/failed/`, are not results)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results, "*", "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if r.get("schema") == SCHEMA:
            rows.append(row_of(r))
    return rows


def cmd_collect(args):
    results = args.results or os.path.join(args.queue, "results")
    rows = result_rows(results)
    out = args.out or os.path.join(results, "summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} results → {out}")
    return 0


def cmd_report(args):
    """One table per group of what was measured (report.py)."""
    import report

    results = args.results or os.path.join(args.queue, "results")
    failed = {}
    # The points still failed in the queue, by group: a failed result file
    # stays behind after a retry that succeeds.
    for pid in Queue(args.queue).ids("failed"):
        p = Queue(args.queue).load("failed", pid)
        key = (p.get("campaign"), p.get("group"))
        failed[key] = failed.get(key, 0) + 1
    print(report.report(result_rows(results), failed, set(args.campaign or []) or None))
    return 0


def cmd_budget(args):
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    jobs = budget.read_sacct(args.start, args.account)
    gpu_h = sum(j["gpu_hours"] for j in jobs)
    by_part = {}
    for j in jobs:
        by_part[j["partition"]] = by_part.get(j["partition"], 0.0) + j["gpu_hours"]
    node_h = gpu_h / budget.GPU_HOURS_PER_NODE_HOUR
    pace = budget.pace(args.budget_node_hours, start, end, dt.datetime.now(), node_h)
    if args.json:
        print(json.dumps(dict(pace, gpu_hours=round(gpu_h, 1), by_partition=by_part)))
        return 0
    print(f"spent      {pace['spent_node_hours']:8.1f} node-h  ({pace['spent_pct']}% of "
          f"{args.budget_node_hours:.0f}; {gpu_h:.1f} GPU-h, from sacct)")
    print(f"calendar   {pace['calendar_pct']:8.1f} %       ({args.start} → {args.end}, "
          f"{pace['days_left']} days left)")
    print(f"target     {pace['target_to_date']:8.1f} node-h  to date on a straight line")
    print(f"behind     {pace['behind_node_hours']:8.1f} node-h")
    print(f"needed     {pace['needed_node_hours_per_day']:8.1f} node-h/day from now = "
          f"{pace['needed_nodes_continuous']} nodes busy around the clock")
    for part, h in sorted(by_part.items()):
        print(f"  {part:12s} {h / budget.GPU_HOURS_PER_NODE_HOUR:8.1f} node-h")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def with_queue(p):
        p.add_argument("--queue", required=True, help="campaign directory (on /scratch)")
        return p

    p = with_queue(sub.add_parser("plan"))
    p.add_argument("specs", nargs="+")
    p.add_argument("--round", help="label for this pass (engine revision, week): "
                   "a new round measures every point again")
    p.add_argument("--engine", default=os.environ.get("EULLM_BIN"),
                   help="check the models are in the store (eullm list)")

    p = sub.add_parser("pulls")
    p.add_argument("specs", nargs="+")
    p.add_argument("--engine", default=os.environ.get("EULLM_BIN"),
                   help="list only what `eullm list` does not have")

    p = with_queue(sub.add_parser("f32s"))
    p.add_argument("specs", nargs="+")

    p = with_queue(sub.add_parser("prefetch"))
    p.add_argument("--sets", default=",".join(DEFAULT_SETS))

    p = with_queue(sub.add_parser("run"))
    p.add_argument("--results")
    p.add_argument("--devices", default="0-7", help="the job's devices, e.g. 0-7")
    p.add_argument("--backend", choices=("rocm", "cuda"), default="rocm")
    p.add_argument("--bind", choices=("lumi", "none"), default="none")
    p.add_argument("--site", default="lumi-g")
    p.add_argument("--engine", default=os.environ.get("EULLM_BIN"))
    p.add_argument("--walltime-s", type=int, help="default: asked of squeue, else 48 h")
    p.add_argument("--margin-s", type=int, default=600,
                   help="stop starting points this long before the walltime")
    p.add_argument("--port-base", type=int, default=18000)
    p.add_argument("--poll-s", type=float, default=5.0)
    p.add_argument("--sample-s", type=float, default=2.0)

    with_queue(sub.add_parser("status"))
    with_queue(sub.add_parser("unblock"))
    p = with_queue(sub.add_parser("retry"))
    p.add_argument("--group", help="only points whose id starts with this, e.g. moe-671b")

    p = with_queue(sub.add_parser("report"))
    p.add_argument("--results")
    p.add_argument("--campaign", action="append",
                   help="only this campaign (repeatable), e.g. c06-mtp")
    p = with_queue(sub.add_parser("collect"))
    p.add_argument("--results")
    p.add_argument("--out")

    p = sub.add_parser("budget")
    p.add_argument("--start", default="2026-09-12")
    p.add_argument("--end", default="2027-03-12")
    p.add_argument("--budget-node-hours", type=float, default=4500)
    p.add_argument("--account", default=os.environ.get("SBATCH_ACCOUNT"))
    p.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    commands = {"plan": cmd_plan, "pulls": cmd_pulls, "f32s": cmd_f32s, "prefetch": cmd_prefetch,
                "run": cmd_run, "status": cmd_status, "unblock": cmd_unblock, "retry": cmd_retry,
                "collect": cmd_collect, "report": cmd_report, "budget": cmd_budget}
    if args.cmd not in commands:
        ap.print_help()
        return 2
    return commands[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
