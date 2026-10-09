"""A campaign spec, expanded into points.

A spec is JSON: a name, defaults, and groups. Each group fixes some fields
(`set`) and sweeps others (`axes`, a cartesian product). Every combination
becomes one point — one server configuration measured once, with its own
repeats inside it:

    {
      "campaign": "c01-baseline",
      "defaults": {"kind": "throughput", "repeats": 3},
      "groups": [
        {"name": "one-gcd", "priority": 50,
         "set": {"gcds": 1},
         "axes": {"model": ["qwen3-8b", "qwen3-14b"], "batch": [1, 16],
                  "kv": ["f16/f16", "q8_0/q8_0"]}}
      ]
    }

A point's id is a hash of what it measures, so expanding the same spec twice
adds nothing, and widening an axis later adds only the new combinations.
`priority` and `est_s` are scheduling hints and are left out of the hash:
changing them must not turn a finished point into a new one.
"""

from __future__ import annotations

import hashlib
import itertools
import json

KINDS = ("throughput", "workload", "finetune", "decision")
RUNTIMES = ("eullm", "llama-server", "ollama")
DECISION_MODES = ("shared_prefix", "batched", "separate")
# Points that need a runner newer than the first one carry this (see
# point.RUNNER_VERSION); the others keep the ids they always had.
NEW_RUNNER = 2
# `cold` points need the runner that drops their model from the page cache.
EVICT_RUNNER = 3
# Points planned for one engine build (`plan --engine-label`) need the runner
# that reads EULLM_ENGINE_LABEL: an older one would run them on any engine.
LABEL_RUNNER = 4
WIDTHS = (1, 2, 4, 8)

# Leonardo's prompt and length, verbatim (docs/cineca/leonardo.md): a
# different prompt tokenizes and generates differently, and the A100 rows it
# would be compared with stop meaning anything.
LEONARDO_PROMPT = "Scrivi una breve storia sul mare."
LEONARDO_NUM_PREDICT = 150

DEFAULTS = {
    # A label for one pass over a spec — an engine revision, a week. Part of
    # the id: the same spec planned under a new round is measured again, which
    # is how a campaign tracks the engine across the allocation.
    "round": None,
    "kind": "throughput",
    "gcds": 1,
    "replica_gcds": None,  # devices per server; None: all of `gcds` (one server)
    "exclusive": False,  # alone on the node, whatever its width
    "model": None,
    "batch": 1,
    "ctx": None,  # total KV pool, as --ctx-size
    "slot_ctx": None,  # or per slot: ctx = slot_ctx * batch
    "kv": "f16/f16",
    "concurrency": None,  # None: every slot of every server busy
    "extra_args": [],
    # throughput
    "repeats": 3,
    "num_predict": LEONARDO_NUM_PREDICT,
    "prompt": LEONARDO_PROMPT,
    "prompt_tokens": 0,  # >0: a synthetic prompt of about this many tokens
    "stream": True,
    # workload
    "sets": [],
    "limit": 0,
    "max_tokens": 768,
    "think": False,
    "min_duration_s": 0,  # 0 and 0: one pass over the sets
    "max_duration_s": 0,
    "interval_s": 60,
}

# Fields of a `finetune` point only (`eullm finetune`'s flags), added to the
# point when its kind is finetune so the other kinds' ids do not move.
FINETUNE_DEFAULTS = {
    "data": None,  # a file in <queue>/sets/, written by `campaign.py prefetch`
    "ft_ctx": 512,  # training window, a multiple of 256
    "epochs": 1,
    "lr": 1e-6,  # one window per step: see `eullm finetune --help`
    "optimizer": "adamw",
    "train_tensors": [],
    "limit_tokens": 0,
    "val_split": 0.05,
    "keep_output": False,  # the trained GGUF is a by-product of a measurement
}

# Scheduling hints: not part of what a point measures.
HINTS = ("priority", "est_s")

# Fields of a `decision` point only: /v1/systemone with a decision model
# (`model`, a store id: a Jev-Style release or a code-readout model).
DECISION_DEFAULTS = {
    "decision_ctx": 16384,  # --decision-ctx
    "state_tokens": 1024,  # synthetic ticket history, ~4 characters a token
    "questions": 8,  # per request, cycling noul / choice / score
    "decision_mode": "shared_prefix",  # eullm.mode (`mode` is the servers' layout)
    "requests": 400,  # in all, over `concurrency` clients
    "distinct_states": 97,  # states cycled; each asked several times
}

DEFAULT_EST_S = {"throughput": 900, "workload": 3600, "finetune": 3600, "decision": 600}


class SpecError(ValueError):
    pass


def normalize(raw: dict) -> dict:
    """A point with every field present and consistent, or SpecError."""
    p = dict(DEFAULTS)
    if raw.get("kind") == "finetune":
        p.update(FINETUNE_DEFAULTS)
    if raw.get("kind") == "decision":
        p.update(DECISION_DEFAULTS)
    p.update(raw)
    if p["kind"] not in KINDS:
        raise SpecError(f"kind must be one of {KINDS}, got {p['kind']!r}")
    if not p["model"]:
        raise SpecError("every point needs a model")
    if p["gcds"] not in WIDTHS:
        raise SpecError(f"gcds must be one of {WIDTHS}, got {p['gcds']!r}")
    if p["replica_gcds"] is None:
        p["replica_gcds"] = p["gcds"]
    if p["replica_gcds"] not in WIDTHS or p["gcds"] % p["replica_gcds"]:
        raise SpecError(
            f"replica_gcds {p['replica_gcds']!r} must be one of {WIDTHS} and divide "
            f"gcds {p['gcds']}"
        )
    p["replicas"] = p["gcds"] // p["replica_gcds"]
    p["mode"] = (
        "single" if p["replicas"] == 1
        else "replicas" if p["replica_gcds"] == 1
        else "split-replicas"
    )
    if p["batch"] < 1:
        raise SpecError("batch must be at least 1")
    if p["slot_ctx"] is not None:
        p["ctx"] = p["slot_ctx"] * p["batch"]
    if p["ctx"] is None:
        p["ctx"] = 4096 * p["batch"]
    p["slot_ctx"] = p["ctx"] // p["batch"]
    kv = str(p["kv"]).split("/")
    if len(kv) == 1:
        kv = kv * 2
    if len(kv) != 2 or not all(kv):
        raise SpecError(f"kv must be 'type' or 'k_type/v_type', got {p['kv']!r}")
    p["kv"] = f"{kv[0]}/{kv[1]}"
    if p["concurrency"] is None:
        p["concurrency"] = p["batch"] * p["replicas"]
    if p["kind"] == "finetune":
        if not p["data"]:
            raise SpecError("a finetune point needs data")
        if p["gcds"] != 1 or p["replicas"] != 1:
            raise SpecError("a finetune point trains on one device: gcds 1")
        if p["ft_ctx"] <= 0 or p["ft_ctx"] % 256:
            raise SpecError(f"ft_ctx must be a multiple of 256, got {p['ft_ctx']}")
        if p["optimizer"] not in ("adamw", "sgd"):
            raise SpecError(f"optimizer must be adamw or sgd, got {p['optimizer']!r}")
        if not p["lr"] > 0:
            raise SpecError("lr must be above zero")
        p["train_tensors"] = [str(t) for t in p["train_tensors"]]
    if p["kind"] == "decision":
        if p["decision_mode"] not in DECISION_MODES:
            raise SpecError(f"decision_mode must be one of {DECISION_MODES}, "
                            f"got {p['decision_mode']!r}")
        if not 1 <= p["questions"] <= 64:
            raise SpecError("questions must be 1 to 64, as /v1/systemone allows")
        if p["requests"] < 1 or p["distinct_states"] < 1:
            raise SpecError("requests and distinct_states must be at least 1")
        p["runner"] = NEW_RUNNER
    if "runtime" in raw:
        if p["runtime"] not in RUNTIMES:
            raise SpecError(f"runtime must be one of {RUNTIMES}, got {p['runtime']!r}")
        if p["kind"] not in ("throughput", "workload"):
            raise SpecError("only throughput and workload points compare runtimes")
        p["runtime_args"] = [str(a) for a in p.get("runtime_args", [])]
        p["runner"] = NEW_RUNNER
    if p["kind"] == "workload":
        if not p["sets"]:
            raise SpecError("a workload point needs sets")
        if p["max_duration_s"] < p["min_duration_s"]:
            raise SpecError("max_duration_s is below min_duration_s")
    if raw.get("engine_label"):
        # Only when given, as `cold`: every point planned before keeps its id.
        p["engine_label"] = str(p["engine_label"])
        p["runner"] = max(p.get("runner", 1), LABEL_RUNNER)
    elif "engine_label" in p:
        del p["engine_label"]
    if "cold" in raw:
        if p["kind"] == "finetune":
            raise SpecError("a finetune point loads no server: cold does not apply")
        p["cold"] = bool(p["cold"])
        p["runner"] = max(p.get("runner", 1), EVICT_RUNNER)
    p["extra_args"] = [str(a) for a in p["extra_args"]]
    for hint in HINTS:
        p.pop(hint, None)
    return p


def point_id(group: str, p: dict) -> str:
    canonical = json.dumps(p, sort_keys=True, separators=(",", ":"))
    return f"{group}-{hashlib.sha1(canonical.encode()).hexdigest()[:10]}"


def width(p: dict) -> int:
    """How many of the node's devices the point keeps to itself."""
    return 8 if p.get("exclusive") else p["gcds"]


def expand(spec: dict, round_=None, engine_label=None) -> list:
    """Every point of `spec`, each with its id, group, campaign and hints;
    `round_` overrides the spec's own round, and `engine_label` keeps every
    point for the jobs started with that label (point.can_run)."""
    campaign = spec.get("campaign")
    if not campaign:
        raise SpecError("the spec needs a campaign name")
    defaults = dict(spec.get("defaults", {}))
    if round_ is not None:
        defaults["round"] = round_
    if engine_label:
        defaults["engine_label"] = engine_label
    points, seen = [], set()
    for group in spec.get("groups", []):
        name = group.get("name")
        if not name:
            raise SpecError("every group needs a name")
        axes = group.get("axes", {})
        keys = sorted(axes)
        for combo in itertools.product(*(axes[k] for k in keys)):
            raw = dict(defaults)
            raw.update(group.get("set", {}))
            raw.update(zip(keys, combo))
            try:
                p = normalize(raw)
            except SpecError as e:
                raise SpecError(f"group {name!r}, {dict(zip(keys, combo))}: {e}") from None
            pid = point_id(name, p)
            if pid in seen:
                continue
            seen.add(pid)
            p.update(
                id=pid,
                group=name,
                campaign=campaign,
                priority=group.get("priority", 0),
                est_s=group.get("est_s", DEFAULT_EST_S[p["kind"]]),
            )
            points.append(p)
    return points


def order_key(p: dict):
    """Higher priority first; within it, wider first, so a wide point is
    not starved by a stream of narrow ones filling every device it frees."""
    return (-p.get("priority", 0), -width(p), p["id"])
