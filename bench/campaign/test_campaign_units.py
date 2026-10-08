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
    {"kind": "finetune", "model": "m.gguf"},
    {"kind": "finetune", "model": "m.gguf", "data": "d", "gcds": 2},
    {"kind": "finetune", "model": "m.gguf", "data": "d", "ft_ctx": 300},
    {"kind": "finetune", "model": "m.gguf", "data": "d", "optimizer": "lion"},
    {"kind": "finetune", "model": "m.gguf", "data": "d", "lr": 0},
])
def test_normalize_rejects(bad):
    with pytest.raises(SpecError):
        normalize(bad)


def test_finetune_fields_belong_to_finetune_points_only():
    ft = normalize({"kind": "finetune", "model": "m.gguf", "data": "d.jsonl",
                    "train_tensors": ["blk.*.attn_*"]})
    assert ft["ft_ctx"] == 512 and ft["optimizer"] == "adamw" and ft["keep_output"] is False
    # The other kinds keep the fields, and so the ids, they had before.
    assert "ft_ctx" not in normalize({"model": "m"})


def test_finetune_command_resolves_the_queue_dirs():
    import point

    p = normalize({"kind": "finetune", "model": "m-f32.gguf", "data": "d.jsonl",
                   "train_tensors": ["blk.*.attn_*", "output.weight"], "lr": 1e-4})
    ctx = point.Context("/bin/eullm", "rocm", "none", ["3"], 0, "/w",
                        sets_dir="/q/sets", f32_dir="/q/f32")
    cmd = point.finetune_command(p, ctx, "/w/o.gguf", "/w/r.json")
    assert cmd[:3] == ["/bin/eullm", "finetune", "/q/f32/m-f32.gguf"]
    assert cmd[cmd.index("--data") + 1] == "/q/sets/d.jsonl"
    assert cmd[cmd.index("--lr") + 1] == "0.0001"
    assert cmd[cmd.index("--device") + 1] == "0"  # the one device left visible
    assert cmd[cmd.index("--train-tensors") + 1] == "blk.*.attn_*,output.weight"
    assert "--no-progress" in cmd


def test_gsm8k_train_text_drops_the_calculator_annotations():
    import json

    import campaign

    row = {"question": "Natalia sold 48 clips in April and half as many in May. How many? ",
           "answer": "In May: 48/2 = <<48/2=24>>24.\nIn all: 48+24 = <<48+24=72>>72.\n#### 72"}
    [doc] = campaign.gsm8k_train_text((json.dumps(row) + "\n\n").encode())
    assert doc == ("Question: Natalia sold 48 clips in April and half as many in May. How many?\n"
                   "In May: 48/2 = 24.\nIn all: 48+24 = 72.\nAnswer: 72")


def test_finetune_models_are_f32_files_not_pulls(tmp_path):
    import campaign

    spec = {"campaign": "c", "f32": {"a-f32.gguf": "Org/A"}, "groups": [
        {"name": "serve", "set": {"model": "qwen3-8b"}},
        {"name": "ft", "set": {"kind": "finetune", "data": "d.jsonl"},
         "axes": {"model": ["a-f32.gguf", "b-f32.gguf"]}},
    ]}
    specs = [("s.json", spec)]
    assert campaign.pull_refs(specs) == [("qwen3-8b", "qwen3-8b")]
    assert campaign.f32_refs(specs) == [("a-f32.gguf", "Org/A"), ("b-f32.gguf", None)]
    (tmp_path / "f32").mkdir()
    (tmp_path / "f32" / "a-f32.gguf").write_bytes(b"GGUF")
    assert campaign.missing_f32(str(tmp_path), specs) == [("b-f32.gguf", None)]


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
    for path in paths:
        with open(path) as f:
            spec = json.load(f)
        points = expand(spec)
        assert points, path
        pulled = {campaign.hf_ref_to_id(r) for r in spec.get("pull", [])}
        served = {p["model"] for p in points if p["kind"] != "finetune"}
        unknown = served - catalog - pulled
        assert not unknown, f"{path}: models with no source: {sorted(unknown)}"
        # A model a finetune point trains is converted from the repo the
        # spec's f32 map names, and its data is a set prefetch writes.
        for f, repo in campaign.f32_refs([(path, spec)]):
            assert repo and repo.count("/") == 1, f"{path}: {f} has no repo to convert"
        for p in points:
            if p["kind"] == "finetune":
                name = p["data"].rsplit(".jsonl", 1)[0]
                assert name in campaign.FINETUNE_SETS, f"{path}: {p['data']} is not prefetched"


def test_a_runner_leaves_points_it_cannot_run_in_the_queue():
    import point

    assert point.can_run({"kind": "throughput"})
    assert not point.can_run({"kind": "throughput", "runner": point.RUNNER_VERSION + 1})
    assert not point.can_run({"kind": "something-new"})
    assert not point.can_run({"kind": "throughput", "runtime": "unknown-server"})


def test_new_fields_leave_old_ids_alone_and_mark_their_points():
    old = normalize({"model": "m"})
    assert "runtime" not in old and "runner" not in old
    rt = normalize({"model": "m", "runtime": "llama-server"})
    assert rt["runner"] == 2 and rt["runtime_args"] == []
    dec = normalize({"kind": "decision", "model": "jev"})
    assert dec["runner"] == 2 and dec["decision_mode"] == "shared_prefix"
    assert dec["mode"] == "single" and dec["questions"] == 8
    cold = normalize({"model": "m", "cold": True})
    assert cold["runner"] == 3 and cold["cold"] is True
    assert normalize({"kind": "decision", "model": "jev", "cold": 1})["runner"] == 3
    for bad in ({"model": "m", "runtime": "vllm-maybe"},
                {"kind": "finetune", "model": "m", "data": "d", "cold": True},
                {"kind": "finetune", "model": "m", "data": "d", "runtime": "ollama"},
                {"kind": "decision", "model": "jev", "questions": 65},
                {"kind": "decision", "model": "jev", "decision_mode": "fast"}):
        with pytest.raises(SpecError):
            normalize(bad)


def test_answers_digest_ignores_noise_below_four_decimals():
    import point

    a = {"q": {"type": "choice", "choice": "x", "probabilities": {"x": 0.71234, "y": 0.28766}}}
    b = {"q": {"type": "choice", "choice": "x", "probabilities": {"x": 0.712341, "y": 0.287659}}}
    c = {"q": {"type": "choice", "choice": "y", "probabilities": {"x": 0.4, "y": 0.6}}}
    assert point.answers_digest(a) == point.answers_digest(b) != point.answers_digest(c)


def test_failed_points_can_be_retried_by_group(tmp_path):
    q = Queue(str(tmp_path))
    for pid in ("moe-1", "moe-2", "rt-1"):
        q.add({"id": pid, "kind": "throughput"})
        q.claim(pid, {"job": "j"})
        q.finish(pid, "failed", "first")
        q.claim(pid, {"job": "j"})
        q.finish(pid, "failed", "second")
    assert sorted(q.ids("failed")) == ["moe-1", "moe-2", "rt-1"]
    assert sorted(q.retry("moe-")) == ["moe-1", "moe-2"]
    p = q.load("todo", "moe-1")
    assert p["attempts"] == 0 and p["notes"] == ["first", "second"]
    assert q.ids("failed") == ["rt-1"]


def test_a_load_records_the_file_system_its_model_is_on_through_links(tmp_path, monkeypatch):
    import point

    store, flash = tmp_path / "store", tmp_path / "flash"
    (store / "m").mkdir(parents=True)
    flash.mkdir()
    (flash / "m-00001-of-00002.gguf").write_bytes(b"GGUF")
    (store / "m" / "m-00001-of-00002.gguf").symlink_to(flash / "m-00001-of-00002.gguf")
    (store / "m" / "manifest.json").write_text('{"gguf_file": "m-00001-of-00002.gguf"}')
    monkeypatch.setenv("EULLM_MODELS_DIR", str(store))
    root = "/" + os.path.realpath(tmp_path).split("/")[1]
    assert point.model_storage({"model": "m"}) == root
    assert point.model_storage({"model": "absent"}) is None
    monkeypatch.setenv("OLLAMA_MODELS", str(flash))
    assert point.model_storage({"model": "absent", "runtime": "ollama"}) == root
    assert point.mount_root("/scratch/project_1/someone/eullm-models/x.gguf") == "/scratch"


def test_a_cold_point_drops_every_part_of_its_model_from_the_page_cache(tmp_path, monkeypatch):
    import point

    d = tmp_path / "store" / "m"
    d.mkdir(parents=True)
    for i, size in ((1, 10), (2, 7)):
        (d / f"M-0000{i}-of-00002.gguf").write_bytes(b"x" * size)
    (d / "manifest.json").write_text('{"gguf_file": "M-00001-of-00002.gguf"}')
    monkeypatch.setenv("EULLM_MODELS_DIR", str(tmp_path / "store"))
    files = point.model_files({"model": "m"})
    assert [os.path.basename(f) for f in files] == ["M-00001-of-00002.gguf",
                                                    "M-00002-of-00002.gguf"]
    assert point.model_files({"model": "m", "runtime": "ollama"}) == []
    assert point.evict(files + [str(tmp_path / "gone.gguf")]) == 17

    ctx = point.Context("/bin/eullm", "rocm", "none", ["0"], 0, str(tmp_path))
    assert point.cache_state({"model": "m"}, ctx) == "cold"
    ctx.model_seen.add("m")
    assert point.cache_state({"model": "m"}, ctx) == "warm"
    assert point.cache_state({"model": "m", "cold": True}, ctx) == "evicted"


def test_the_report_puts_what_a_group_varies_beside_what_it_measured():
    import report

    rows = [
        {"campaign": "c06", "group": "mtp", "kind": "throughput", "outcome": "measured",
         "model": "m", "batch": "1", "extra_args": "--fit-strict", "agg_tok_s_mean": "100"},
        {"campaign": "c06", "group": "mtp", "kind": "throughput", "outcome": "measured",
         "model": "m", "batch": "1", "extra_args": "--fit-strict --mtp 2",
         "agg_tok_s_mean": "130"},
        {"campaign": "c06", "group": "mtp", "kind": "throughput", "outcome": "measured",
         "model": "m", "batch": "1", "extra_args": "--fit-strict --mtp 2",
         "agg_tok_s_mean": "134"},
        {"campaign": "c06", "group": "mtp", "kind": "throughput", "outcome": "does-not-fit",
         "model": "m", "batch": "1", "extra_args": "--fit-strict --mtp 3"},
        {"campaign": "c08", "group": "dec", "kind": "decision", "outcome": "measured",
         "model": "jev", "concurrency": "4", "dec_per_s": "12.5"},
    ]
    text = report.report(rows, {("c06", "mtp"): 2}, {"c06"})
    assert "=== c06 / mtp (throughput, 3 measured) ===" in text
    lines = text.splitlines()
    head = next(line for line in lines if "tok/s" in line)
    assert head.split()[0] == "extra_args"  # model and batch do not vary
    assert any("--mtp 2" in line and "132" in line and line.rstrip().endswith("2")
               for line in lines)  # the two results averaged, counted
    assert any("(none)" in line and "100" in line for line in lines)
    assert "not measured: does-not-fit 1, failed 2" in text
    assert "c08" not in text


def test_leftover_servers_are_those_of_this_job_outside_every_running_point():
    import campaign

    ps = ("  101   101 eullm\n  202   202 eullm\n  303   300 ollama\n"
          "  404   404 python3\n  505   505 llama-server\n  606   606 eullm\n")
    cgroups = {101: "0::/job_7/step_batch", 202: "0::/job_7/step_batch",
               303: "0::/job_7/step_batch", 404: "0::/job_7/step_batch",
               505: "0::/job_7/step_batch", 606: "0::/job_8/step_batch"}
    got = campaign.orphan_servers(ps, keep_pgids={101}, job="7", cgroup_of=cgroups.get)
    # 101 is a running point's server, 404 is no server, 606 is another job's.
    assert got == [(202, "eullm"), (303, "ollama"), (505, "llama-server")]
