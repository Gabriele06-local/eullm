"""The runner end to end, against fake_eullm.py instead of a GPU."""

import argparse
import csv
import json
import os
import sys
import time

import pytest

import campaign
from fsqueue import Queue
from spec import SpecError, expand

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_eullm.py")


@pytest.fixture
def engine(tmp_path):
    """fake_eullm.py behind an executable path, as EULLM_BIN would be."""
    path = tmp_path / "eullm"
    path.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE} \"$@\"\n")
    path.chmod(0o755)
    return str(path)


def run_args(queue, engine, **kw):
    a = dict(queue=queue, results=None, devices="0-3", backend="rocm", bind="none",
             site="test", engine=engine, walltime_s=3600, margin_s=0, port_base=0,
             poll_s=0.2, sample_s=0.5)
    a.update(kw)
    return argparse.Namespace(**a)


def free_port_base():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return 20000 + s.getsockname()[1] % 20000


SPEC = {
    "campaign": "c-e2e",
    "defaults": {"repeats": 2, "num_predict": 12, "est_s": 60},
    "groups": [
        {"name": "one", "priority": 1, "est_s": 60, "set": {"gcds": 1},
         "axes": {"model": ["qwen3-8b", "qwen3-4b"], "batch": [1, 4]}},
        {"name": "rep", "priority": 1, "est_s": 60,
         "set": {"model": "qwen3-8b", "gcds": 2, "replica_gcds": 1, "batch": 2}},
        {"name": "excl", "priority": 2, "est_s": 60,
         "set": {"model": "qwen3-8b", "gcds": 1, "exclusive": True}},
        {"name": "gone", "priority": 0, "est_s": 60, "set": {"model": "missing-model"}},
        {"name": "huge", "priority": 0, "est_s": 60, "set": {"model": "huge-model"}},
        {"name": "load", "priority": 0, "est_s": 60, "set": {
            "kind": "workload", "model": "qwen3-4b", "sets": ["tiny"], "concurrency": 3,
            "min_duration_s": 0, "max_duration_s": 0, "interval_s": 1}},
    ],
}


def test_runner_drains_a_campaign(tmp_path, engine, capsys):
    qdir = str(tmp_path / "q")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(SPEC))
    assert campaign.main(["plan", str(spec_path), "--queue", qdir]) == 0
    os.makedirs(os.path.join(qdir, "sets"), exist_ok=True)
    with open(os.path.join(qdir, "sets", "tiny.jsonl"), "w") as f:
        for i in range(5):
            f.write(json.dumps({"id": f"t{i}", "answer": "4" if i < 4 else "5",
                                "grader": "number",
                                "messages": [{"role": "user", "content": "2+2?"}]}) + "\n")

    runner = campaign.Runner(run_args(qdir, engine, port_base=free_port_base()))
    runner.loop()

    counts = Queue(qdir).counts()
    assert counts == {"todo": 0, "running": 0, "done": 8, "failed": 0, "blocked": 1}
    out = capsys.readouterr().out
    results = [json.loads(line.split(" ", 1)[1]) for line in out.splitlines()
               if line.startswith("BENCH_RESULT ")]
    assert len(results) == 8
    by_group = {}
    for r in results:
        assert r["schema"] == "eullm.bench/1"
        assert r["ok"] == (r["group"] != "huge")
        assert r["engine"]["version"].startswith("eullm 0.0.0-fake")
        by_group.setdefault(r["group"], []).append(r)

    one = by_group["one"][0]["throughput"]
    assert len(one["repeats"]) == 2 and one["aggregate_tok_s_mean"] > 0
    assert one["repeats"][0]["ttft_ms_p50"] is not None
    rep = by_group["rep"][0]
    assert rep["params"]["replicas"] == 2 and len(rep["load"]["load_ms"]) == 2
    excl = by_group["excl"][0]
    assert excl["neighbours_at_start"] == 0  # exclusive: alone on the node
    # The first load of a model on this node is cold, later ones warm. (Two
    # points loading the same model at the same moment are both cold.)
    assert excl["load"]["cache"] == "cold"
    assert "warm" in {r.get("load", {}).get("cache") for r in results}

    # Too large for the devices is a result, the memory boundary: recorded
    # and done, not retried.
    assert by_group["huge"][0]["outcome"] == "does-not-fit"

    work = by_group["load"][0]["workload"]
    assert work["accuracy"]["tiny"] == {"graded": 5, "correct": 4, "accuracy": 0.8}
    assert work["requests"] == 5 and work["failed"] == 0
    answers = os.path.join(qdir, "results", "c-e2e",
                           f"{by_group['load'][0]['point']}.{runner.job}.answers.jsonl")
    assert len(open(answers).read().splitlines()) == 5

    summary = json.load(open(os.path.join(qdir, "results", f"{runner.job}.summary.json")))
    assert summary["points"]["done"] == 8 and summary["points"]["blocked"] == 1
    assert set(summary["assigned_fraction"]) == {"0", "1", "2", "3"}

    assert campaign.main(["collect", "--queue", qdir]) == 0
    rows = list(csv.DictReader(open(os.path.join(qdir, "results", "summary.csv"))))
    assert len(rows) == 8 and {r["kind"] for r in rows} == {"throughput", "workload"}
    assert {r["outcome"] for r in rows} == {"measured", "does-not-fit"}


def test_workload_stretches_to_the_time_it_is_given(tmp_path, engine):
    qdir = str(tmp_path / "q")
    q = Queue(qdir)
    spec = {"campaign": "c-soak", "groups": [{"name": "soak", "set": {
        "kind": "workload", "model": "qwen3-4b", "sets": ["tiny"], "concurrency": 2,
        "min_duration_s": 60, "max_duration_s": 3}}]}
    with pytest.raises(SpecError):  # max below min
        expand(spec)
    spec["groups"][0]["set"].update(min_duration_s=60, max_duration_s=7200)
    p = expand(spec)[0]
    q.add(p)
    r = campaign.Runner(run_args(qdir, engine, walltime_s=4 * 3600, margin_s=600))
    got = r.duration_for(p, r.deadline - time.time() - r.args.margin_s)
    assert got == 7200  # capped by its maximum
    got = r.duration_for(p, campaign.LOAD_ALLOWANCE_S + 1800)
    assert got == 1800  # stretched to what is free
    assert r.duration_for(p, campaign.LOAD_ALLOWANCE_S + 30) is None  # below its minimum


def test_backfill_does_not_delay_a_waiting_wide_point(tmp_path, engine):
    qdir = str(tmp_path / "q")
    spec = {"campaign": "c-bf", "groups": [
        {"name": "wide", "priority": 1, "est_s": 600, "set": {"model": "m", "gcds": 4}},
        {"name": "long", "priority": 0, "est_s": 7200, "set": {"model": "m", "gcds": 1}},
        {"name": "short", "priority": 0, "est_s": 300, "set": {"model": "n", "gcds": 1}},
    ]}
    todo = sorted(expand(spec), key=campaign.order_key)
    r = campaign.Runner(run_args(qdir, engine, walltime_s=10 * 3600))
    now = time.time()
    # Two devices are busy with a point ending in 1000 s: the wide point
    # must wait for them, so only what ends before then may start.
    busy = campaign.Running({"id": "x"}, [0, 1], [0, 1], 0, None, now + 1000)
    r.running["x"] = busy
    r.free -= {0, 1}
    p, use, reserved, duration = r.plan_next(now, todo)
    assert p["group"] == "short" and use == [2]


def test_time_left_parsing():
    assert campaign.parse_time_left("1-23:59:30") == 86400 + 23 * 3600 + 59 * 60 + 30
    assert campaign.parse_time_left("47:00:00") == 47 * 3600
    assert campaign.parse_time_left("59:30") == 59 * 60 + 30
    assert campaign.parse_time_left("UNLIMITED") is None


def test_stop_releases_the_running_points(tmp_path, engine):
    """The walltime case: SIGTERM sets `stop`; servers die at once and the
    interrupted point goes back to todo for the next job, not to failed."""
    import threading

    qdir = str(tmp_path / "q")
    q = Queue(qdir)
    spec = {"campaign": "c-stop", "groups": [{"name": "soak", "set": {
        "kind": "workload", "model": "qwen3-4b", "sets": ["tiny"], "concurrency": 2,
        "min_duration_s": 60, "max_duration_s": 3600}}]}
    p = expand(spec)[0]
    q.add(p)
    os.makedirs(os.path.join(qdir, "sets"))
    with open(os.path.join(qdir, "sets", "tiny.jsonl"), "w") as f:
        f.write(json.dumps({"id": "t0", "answer": "4", "grader": "number",
                            "messages": [{"role": "user", "content": "2+2?"}]}) + "\n")
    r = campaign.Runner(run_args(qdir, engine, port_base=free_port_base(),
                                 walltime_s=4 * 3600))
    t = threading.Thread(target=r.loop)
    t.start()
    for _ in range(100):
        if q.counts()["running"]:
            break
        time.sleep(0.1)
    time.sleep(1.0)  # the workload is under way
    servers = [s for run in r.running.values() for s in run.ctx.servers]
    assert servers
    r.stop.set()
    t.join(timeout=60)
    assert not t.is_alive()
    assert q.counts()["todo"] == 1 and q.counts()["running"] == 0
    assert all(s.proc.poll() is not None for s in servers)
    assert r.counts["released"] == 1


def test_hf_ids_follow_the_engine_rule():
    assert campaign.hf_ref_to_id("hf.co/Qwen/Qwen3-235B-A22B-GGUF:Q4_K_M") == \
        "qwen3-235b-a22b-gguf-q4_k_m"
    assert campaign.hf_ref_to_id("hf.co/ggml-org/gpt-oss-120b-GGUF:MXFP4") == \
        "gpt-oss-120b-gguf-mxfp4"
    assert campaign.hf_ref_to_id("hf.co/owner/Some Repo") == "some-repo"


def test_pulls_name_the_hf_ref_or_the_catalog_id(tmp_path, engine, capsys):
    spec = {"campaign": "c", "pull": ["hf.co/Qwen/Qwen3-8B-GGUF:Q8_0"], "groups": [
        {"name": "g", "axes": {"model": ["qwen3-8b-gguf-q8_0", "qwen3-14b", "qwen3-8b"]}}]}
    path = tmp_path / "s.json"
    path.write_text(json.dumps(spec))
    assert campaign.main(["pulls", str(path), "--engine", engine]) == 0
    # fake_eullm lists qwen3-8b and qwen3-4b
    assert capsys.readouterr().out.split() == ["qwen3-14b", "hf.co/Qwen/Qwen3-8B-GGUF:Q8_0"]


def test_a_new_round_measures_everything_again(tmp_path, capsys):
    spec = {"campaign": "c", "groups": [{"name": "g", "axes": {"model": ["a", "b"]}}]}
    path = tmp_path / "s.json"
    path.write_text(json.dumps(spec))
    q = str(tmp_path / "q")
    campaign.main(["plan", str(path), "--queue", q])
    campaign.main(["plan", str(path), "--queue", q])
    campaign.main(["plan", str(path), "--queue", q, "--round", "v0.7.21"])
    out = capsys.readouterr().out
    assert "2 points, 2 new" in out and "2 points, 0 new" in out
    assert "(round v0.7.21): 2 points, 2 new" in out
    assert Queue(q).counts()["todo"] == 4


def test_when_free_follows_alignment_and_overruns(tmp_path, engine):
    r = campaign.Runner(run_args(str(tmp_path / "q"), engine, devices="0-7"))
    now = time.time()
    # Free: 1, 2, 5, 6 — four devices, but no aligned block of four.
    r.free = {1, 2, 5, 6}
    for name, devs, end in (("a", [0], now + 100), ("b", [3], now + 200),
                            ("c", [4], now + 300), ("d", [7], now + 400)):
        r.running[name] = campaign.Running({"id": name}, devs, devs, 0, None, end)
    assert r.when_free(4, now) == now + 200  # 0-3 complete once a and b end
    assert r.when_free(8, now) == now + 400
    # A point past its estimate is given a quarter of it again, at least 5 min.
    late = campaign.Running({"id": "late"}, [1], [1], 0, None, now - 10)
    late.started = now - 4010
    r.running = {"late": late}
    r.free = {0, 2, 3, 4, 5, 6, 7}
    assert abs(r.when_free(2, now) - (now + 1000)) < 1
