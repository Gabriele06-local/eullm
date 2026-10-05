"""Unit tests: spec expansion, the file queue, devices, billing and pace."""

import datetime as dt
import os

import pytest

import budget
from devices import (
    NodeSampler,
    aligned_group,
    cores_for,
    parse_devices,
    parse_nvidia_smi,
    parse_rocm_smi,
    physical_map,
)
from fsqueue import ORPHAN_GRACE_S, Queue
from spec import SpecError, expand, normalize, order_key, width

SPEC = {
    "campaign": "c-test",
    "defaults": {"repeats": 2},
    "groups": [
        {"name": "one", "priority": 5, "set": {"gcds": 1},
         "axes": {"model": ["a", "b"], "batch": [1, 16]}},
        {"name": "wide", "priority": 5, "set": {"gcds": 8, "replica_gcds": 2, "model": "a"}},
    ],
}


# ── spec ─────────────────────────────────────────────────────────────────


def test_expand_is_the_cartesian_product_and_stable():
    a, b = expand(SPEC), expand(SPEC)
    assert len(a) == 5
    assert [p["id"] for p in a] == [p["id"] for p in b]
    assert len({p["id"] for p in a}) == 5


def test_hints_do_not_change_the_id():
    other = dict(SPEC, groups=[dict(SPEC["groups"][0], priority=99, est_s=1)])
    ids = {p["id"] for p in expand(other)}
    assert ids <= {p["id"] for p in expand(SPEC)}


def test_normalize_fills_ctx_concurrency_and_mode():
    p = normalize({"model": "m", "batch": 16, "slot_ctx": 4096})
    assert p["ctx"] == 65536 and p["concurrency"] == 16 and p["mode"] == "single"
    r = normalize({"model": "m", "gcds": 8, "replica_gcds": 1, "batch": 4})
    assert r["replicas"] == 8 and r["mode"] == "replicas" and r["concurrency"] == 32
    s = normalize({"model": "m", "gcds": 8, "replica_gcds": 2})
    assert s["replicas"] == 4 and s["mode"] == "split-replicas"
    assert normalize({"model": "m", "kv": "q8_0"})["kv"] == "q8_0/q8_0"


@pytest.mark.parametrize("bad", [
    {"model": "m", "gcds": 3},
    {"model": "m", "gcds": 4, "replica_gcds": 8},
    {"kind": "workload", "model": "m"},
    {"model": "m", "kind": "nope"},
    {"gcds": 1},
])
def test_normalize_rejects(bad):
    with pytest.raises(SpecError):
        normalize(bad)


def test_order_is_priority_then_width():
    pts = [dict(id="x", priority=1, gcds=1), dict(id="y", priority=1, gcds=8),
           dict(id="z", priority=2, gcds=1), dict(id="w", priority=1, gcds=1, exclusive=True)]
    assert [p["id"] for p in sorted(pts, key=order_key)] == ["z", "w", "y", "x"]
    assert width(pts[3]) == 8


# ── queue ────────────────────────────────────────────────────────────────


def test_queue_add_is_idempotent_and_claim_exclusive(tmp_path):
    q = Queue(str(tmp_path))
    p = expand(SPEC)[0]
    assert q.add(p) and not q.add(p)
    assert q.claim(p["id"], {"job": "1"})["id"] == p["id"]
    assert q.claim(p["id"], {"job": "2"}) is None  # another node got there first
    assert q.where(p["id"]) == "running"
    assert not q.add(p)  # known while running too


def test_failure_is_retried_then_kept(tmp_path):
    q = Queue(str(tmp_path))
    p = expand(SPEC)[0]
    q.add(p)
    q.claim(p["id"], {"job": "1"})
    assert q.finish(p["id"], "failed", "boom") == "todo"
    q.claim(p["id"], {"job": "1"})
    assert q.finish(p["id"], "failed", "boom again") == "failed"
    assert q.load("failed", p["id"])["notes"] == ["boom", "boom again"]


def test_stale_claims_go_back(tmp_path):
    q = Queue(str(tmp_path))
    a, b, c = expand(SPEC)[:3]
    for p in (a, b, c):
        q.add(p)
    q.claim(a["id"], {"job": "dead"})
    q.claim(b["id"], {"job": "alive"})
    # c: claimed by a node that died before writing its owner file
    os.rename(os.path.join(q.root, "todo", c["id"] + ".json"),
              os.path.join(q.root, "running", c["id"] + ".json"))
    now = os.path.getmtime(os.path.join(q.root, "running", c["id"] + ".json"))
    assert q.requeue_stale({"alive"}, now, 3600) == [a["id"]]
    assert q.requeue_stale({"alive"}, now + ORPHAN_GRACE_S + 1, 3600) == [c["id"]]
    assert q.where(b["id"]) == "running"


def test_blocked_points_can_be_unblocked(tmp_path):
    q = Queue(str(tmp_path))
    p = expand(SPEC)[0]
    q.add(p)
    q.claim(p["id"], {"job": "1"})
    q.finish(p["id"], "blocked", "model not pulled")
    assert q.counts()["blocked"] == 1
    assert q.unblock() == [p["id"]] and q.where(p["id"]) == "todo"


# ── devices ──────────────────────────────────────────────────────────────


def test_parse_devices():
    assert parse_devices("0-3,6") == [0, 1, 2, 3, 6]
    assert parse_devices("7,1") == [1, 7]


def test_physical_map_follows_what_slurm_gave():
    assert physical_map("rocm", [0, 1], {}) == {0: "0", 1: "1"}
    assert physical_map("rocm", [0, 1], {"ROCR_VISIBLE_DEVICES": "4,5"}) == {0: "4", 1: "5"}
    with pytest.raises(ValueError):
        physical_map("cuda", [0, 1, 2], {"CUDA_VISIBLE_DEVICES": "0"})


def test_aligned_groups_keep_pairs_on_a_module():
    devs = list(range(8))
    assert aligned_group(set(devs), devs, 2) == [0, 1]
    assert aligned_group({1, 2, 3, 4, 5}, devs, 2) == [2, 3]
    assert aligned_group({1, 2, 3, 4, 5, 6, 7}, devs, 4) == [4, 5, 6, 7]
    assert aligned_group({0, 1, 2, 3, 4, 5, 6}, devs, 8) is None
    assert aligned_group({5}, devs, 1) == [5]


def test_lumi_cores_follow_the_documented_mask():
    assert cores_for("lumi", ["0"]) == "49-55"
    assert cores_for("lumi", ["4", "5"]) == "1-7,9-15"
    assert cores_for("none", ["0"]) is None
    assert cores_for("lumi", ["GPU-uuid"]) is None


ROCM_SMI = """
============================ ROCm System Management Interface ============================
GPU[0]          : GPU use (%): 97
GPU[1]          : GPU use (%): 0
GPU[0]          : VRAM Total Memory (B): 68702699520
GPU[0]          : VRAM Total Used Memory (B): 33554432000
GPU[1]          : VRAM Total Memory (B): 68702699520
GPU[1]          : VRAM Total Used Memory (B): 11288576
"""


def test_parse_smi_outputs():
    r = parse_rocm_smi(ROCM_SMI)
    assert r["0"] == {"use": 97, "total": 68702699520, "used": 33554432000}
    assert r["1"]["use"] == 0
    n = parse_nvidia_smi("0, 40000, 65536, 88\n1, 10, 65536, 0\n")
    assert n["0"] == {"used": 40000 * 2**20, "total": 65536 * 2**20, "use": 88}


def test_sampler_window_per_device(tmp_path):
    readings = iter([parse_rocm_smi(ROCM_SMI)] * 3)
    s = NodeSampler("rocm", log_path=str(tmp_path / "n.jsonl"), reader=lambda: next(readings))
    for _ in range(3):
        s.sample_once()
    w = s.window(0, 1e12, ["0", "1", "2"])
    assert w["0"]["vram_peak_mib"] == 32000 and w["0"]["use_mean"] == 97.0
    assert w["2"]["samples"] == 0
    assert len((tmp_path / "n.jsonl").read_text().splitlines()) == 3


# ── budget ───────────────────────────────────────────────────────────────


def test_lumi_billing():
    assert budget.billed_gpu_hours("standard-g", 3600, 1, {}) == 4
    one_gcd = {"gres/gpu": "1", "cpu": "7", "mem": "60G"}
    assert budget.billed_gpu_hours("small-g", 3600, 1, one_gcd) == 0.5
    # cores or memory beyond a GCD's share are billed as more GCDs
    assert budget.billed_gpu_hours("small-g", 3600, 1, dict(one_gcd, cpu="16")) == 1.0
    assert budget.billed_gpu_hours("small-g", 3600, 1, dict(one_gcd, mem="128G")) == 1.0
    assert budget.billed_gpu_hours("small", 3600, 1, one_gcd) == 0


def test_parse_sacct():
    text = ("123|eullm-campaign|standard-g|RUNNING|7200|1|cpu=56,gres/gpu=8,mem=480G,node=1\n"
            "124|eullm-bench|small-g|COMPLETED|3600|1|cpu=7,gres/gpu=1,mem=60G,node=1\n"
            "125|x|small-g|PENDING|0|1|\n")
    jobs = budget.parse_sacct(text)
    assert [j["gpu_hours"] for j in jobs] == [8.0, 0.5, 0.0]


def test_pace_against_the_calendar():
    start, end = dt.date(2026, 9, 12), dt.date(2027, 3, 12)
    p = budget.pace(4500, start, end, dt.datetime(2026, 10, 12), 0)
    assert 15 < p["calendar_pct"] < 18
    assert p["behind_node_hours"] == p["target_to_date"] > 700
    assert 1.1 < p["needed_nodes_continuous"] < 1.3


def test_shipped_campaigns_expand():
    """The specs in tools/lumi/campaigns are valid, and every model named in
    one is either a catalog id or covered by the spec's own `pull` list."""
    import glob
    import json

    import campaign

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    with open(os.path.join(root, "catalog", "v1", "catalog.json")) as f:
        catalog = {m["id"] for m in json.load(f)["models"]}
    paths = sorted(glob.glob(os.path.join(root, "tools", "lumi", "campaigns", "*.json")))
    assert paths
    # Pulled on LUMI on 12-09-2026 for the first measurements (docs/lumi/lumi-g.md).
    already_on_lumi = {"qwen3.8-27b-ud-q8_k_xl"}
    for path in paths:
        with open(path) as f:
            spec = json.load(f)
        points = expand(spec)
        assert points, path
        pulled = {campaign.hf_ref_to_id(r) for r in spec.get("pull", [])}
        unknown = {p["model"] for p in points} - catalog - pulled - already_on_lumi
        assert not unknown, f"{path}: models with no source: {sorted(unknown)}"
