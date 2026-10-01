#!/usr/bin/env python3
"""ReflexBench, qualification: may this decision model replace that one?

MVP 4 of the Reflex roadmap (docs/reflex-roadmap.md). A decision model is
swapped in because it passed this test on the domain's own labelled
decisions, not because a configuration line names it. The candidate — a
model Forge trained on your decisions, say — and, optionally, the current
model it would replace answer the same labelled requests, each in the three
evaluation modes of `/v1/systemone` (`eullm.mode`: `separate`,
`shared_prefix`, `batched`), and the report says, per question type:

  * accuracy, in the mode the model will serve in, against always giving
    each question's commonest answer, with a 95% interval;
  * calibration: expected calibration error of the top answer, and NLL;
  * noise between the modes: how far the same request's probabilities move,
    and how many answers change;
  * latency p50 and p95, per mode;
  * coverage, for a code-readout model;

then PASS or FAIL against thresholds (qf_metrics.THRESHOLDS, each with its
reason; all configurable), and writes everything to a JSON report. The exit
status is 0 for PASS, 1 for FAIL.

    python3 bench/reflexbench/qualify.py \\
        --candidate http://localhost:11604 --current http://localhost:11434 \\
        --data decisions-data/test.labelled.jsonl --out qualify.json

The labelled requests (qf_data.py): a JSONL set — `eullm-forge decisions
build` writes the held-out states as one — or a traces directory, where
every decision with feedback is a request. Give each server an audit
directory of its own: a run is thousands of decisions.

A model is qualified as it will be served, at the temperature the server
applies. A model Forge exported carries the one fitted on its dev split in
its GGUF (`eullm.decision.temperature`), for the engine to apply by
default: with `--candidate-gguf` the test reads it there and checks that the
server applied it to every answer — an engine that does not read the key
serves, and would be qualified on, probabilities the fit did not calibrate.

Only the Python standard library is needed: the GGUF is read with Forge's
own reader (forge/eullm_forge/gguf_metadata.py), the code that writes it.
"""

import argparse
import datetime
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qf_data  # noqa: E402
import qf_metrics  # noqa: E402
from rb_methods import ServerError, post  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "forge"))
from eullm_forge.decisions.metrics import TEMPERATURE_KEY  # noqa: E402
from eullm_forge.gguf_metadata import GGUFError, read_metadata  # noqa: E402

MODES = ("separate", "shared_prefix", "batched")


def gguf_temperature(path):
    """The temperature the GGUF at `path` carries for the engine to apply
    by default, or None when it carries none."""
    return read_metadata(path, [TEMPERATURE_KEY]).get(TEMPERATURE_KEY)


class Server:
    """A `/v1/systemone` to qualify, as `candidate` or `current`.
    `temperature`, when given, goes with every request — the one the model
    will be served with; otherwise the server's default applies, which for
    a model whose GGUF carries one (`gguf_temperature`, read from `gguf`)
    should be that one."""

    def __init__(self, role, url, model=None, api_key=None, timeout=600.0, temperature=None,
                 gguf=None, gguf_temperature=None):
        self.role, self.base = role, url.rstrip("/")
        self.url = self.base + "/v1/systemone"
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.temperature = temperature
        self.gguf, self.gguf_temperature = gguf, gguf_temperature

    def ask(self, state, questions, mode):
        payload = {"state": state, "questions": questions, "eullm": {"mode": mode}}
        if self.temperature is not None:
            payload["eullm"]["temperature"] = self.temperature
        if self.model:
            payload["model"] = self.model
        started = time.perf_counter()
        body = post(self.url, payload, self.api_key, self.timeout)
        return body, (time.perf_counter() - started) * 1000


def refused_question(error):
    """The question a refusal names, when it was one question's."""
    try:
        detail = json.loads(error.detail)
    except ValueError:
        return None
    err = detail.get("error") if isinstance(detail, dict) else None
    return err.get("question") if isinstance(err, dict) else None


def class_probabilities(question, answer):
    """An answer's probabilities, in class order."""
    if question["type"] == "noul":
        p = float(answer["noul"])
        return [p, 1.0 - p]
    probabilities = answer["probabilities"]
    return [float(probabilities[label]) for label in qf_data.labels(question)]


def decide(server, item, mode):
    """One request in one mode: per question, `{"p", "coverage"}` or None
    when the server would not answer it, and the request's cost.

    A question the server refuses (a 422 naming it: 30 options for a model
    that answers 26, say) is dropped and the rest asked again; a refusal of
    the whole request (a state too long), or a server error on it, refuses
    every question. Anything else — no model loaded, a wrong key — stops
    the run: it says nothing about the model.
    """
    questions = dict(item.questions)
    out = {qid: None for qid in item.questions}
    refusals = {}
    while questions:
        try:
            body, ms = server.ask(item.state, questions, mode)
        except ServerError as e:
            qid = refused_question(e) if e.code == 422 else None
            if qid in questions:
                refusals[qid] = e.detail
                questions = {q: spec for q, spec in questions.items() if q != qid}
                continue
            if e.code == 422 or e.code >= 500:
                refusals.update({q: e.detail for q in questions})
                return out, None, {}, refusals
            raise
        for qid, question in questions.items():
            answer = body["answers"][qid]
            out[qid] = {"p": class_probabilities(question, answer),
                        "coverage": (answer.get("eullm") or {}).get("coverage")}
        info = body.get("eullm") or {}
        server_info = {"model": body.get("model"), "readout": info.get("readout"),
                       "request_ms": info.get("request_ms"),
                       "evaluated_tokens": info.get("evaluated_tokens"),
                       "temperature": info.get("temperature")}
        return out, ms, server_info, refusals
    return out, None, {}, refusals


def run(server, items, modes, serve_mode, details=None):
    """Every item in every mode — one mode at a time, so no request finds
    the previous mode's copy of its state still decoded — after one untimed
    request."""
    decide(server, items[0], serve_mode)
    records = [{"item": item.id, "qid": qid, "kind": item.kind(qid),
                "right": item.answers[qid], "modes": {}}
               for item in items for qid in item.questions]
    by_key = {(r["item"], r["qid"]): r for r in records}
    latency = {mode: [] for mode in modes}
    server_ms = {mode: [] for mode in modes}
    seen = {"models": set(), "readouts": set(), "temperatures": set()}
    reasons = {}
    for mode in modes:
        started = shown = time.perf_counter()
        for n, item in enumerate(items, 1):
            answers, ms, info, refusals = decide(server, item, mode)
            for qid, result in answers.items():
                by_key[(item.id, qid)]["modes"][mode] = result
            for detail in refusals.values():
                reasons[detail[:200]] = reasons.get(detail[:200], 0) + 1
            if ms is not None:
                latency[mode].append(ms)
                if info.get("request_ms") is not None:
                    server_ms[mode].append(info["request_ms"])
                seen["models"].add(info.get("model"))
                seen["readouts"].add(info.get("readout"))
                seen["temperatures"].add(info.get("temperature"))
            if details:
                for qid, result in answers.items():
                    details.write(json.dumps({
                        "server": server.role, "mode": mode, "item": item.id, "question": qid,
                        "right": item.answers[qid], "probabilities": result and result["p"],
                        "coverage": result and result["coverage"], "ms": ms,
                    }) + "\n")
            now = time.perf_counter()
            if n == len(items) or now - shown >= 30:
                shown = now
                print(f"  {server.role} {mode}: {n}/{len(items)} ({now - started:.0f} s)",
                      file=sys.stderr, flush=True)
    return {
        "url": server.base,
        "model_asked": server.model,
        "temperature_asked": server.temperature,
        "gguf": server.gguf,
        "temperature_gguf": server.gguf_temperature,
        # What the server reported applying (eullm.temperature), whatever
        # was asked: the temperature the numbers below were measured at.
        "temperatures_applied": sorted(t for t in seen["temperatures"] if t is not None),
        "models": sorted(m for m in seen["models"] if m),
        "readout": ", ".join(sorted(r for r in seen["readouts"] if r)) or None,
        "records": records,
        "refusals": reasons,
        "latency_ms": {m: {"p50": qf_metrics.percentile(v, 0.5),
                           "p95": qf_metrics.percentile(v, 0.95), "requests": len(v)}
                       for m, v in latency.items()},
        "server_ms": {m: {"p50": qf_metrics.percentile(v, 0.5),
                          "p95": qf_metrics.percentile(v, 0.95)}
                      for m, v in server_ms.items()},
    }


def measure(result, modes, serve_mode):
    reference = "separate" if "separate" in modes else modes[0]
    result["serve_mode"] = serve_mode
    result["reference_mode"] = reference
    result["by_type"] = qf_metrics.summarize(result["records"], serve_mode, reference, modes)
    return result


def compare(candidate, current, serve_mode):
    """Per type, which answers only one of the two got right (McNemar)."""
    mine = {(r["item"], r["qid"]): r for r in candidate["records"]}
    out = {}
    for kind in ("all",) + qf_metrics.KINDS:
        pairs = []
        for r in current["records"]:
            other = mine.get((r["item"], r["qid"]))
            if other is None or (kind != "all" and r["kind"] != kind):
                continue
            pairs.append(tuple(
                rec["modes"].get(serve_mode) is not None
                and qf_metrics.argmax(rec["modes"][serve_mode]["p"]) == rec["right"]
                for rec in (other, r)))
        if pairs:
            out[kind] = qf_metrics.mcnemar(pairs)
    return out


def table(results):
    head = ("| server | model | type | answers | refused | accuracy | 95% CI | commonest | ECE | "
            "NLL | coverage | max Δ between modes | changed | p50 ms | p95 ms |")
    lines = [head, "|---" * (head.count("|") - 1) + "|"]

    def num(x, digits=3):
        return "—" if x is None else f"{x:.{digits}f}"

    def pct(x):
        return "—" if x is None else f"{100 * x:.1f}%"

    for r in results:
        lat = r["latency_ms"].get(r["serve_mode"], {})
        for kind, m in r["by_type"].items():
            ci = m["accuracy_ci95"]
            lines.append(
                f"| {r['role']} | {', '.join(r['models']) or r['model_asked'] or '—'} | {kind} | "
                f"{m['answers']} | {m['refused']} | {pct(m['accuracy'])} | "
                f"{pct(ci[0]) if ci else '—'}–{pct(ci[1]) if ci else '—'} | "
                f"{pct(m['majority_baseline'])} | {num(m['ece'])} | {num(m['nll'])} | "
                f"{num(m['coverage'])} | {num(m['max_mode_delta'])} | "
                f"{pct(m['mode_flip_rate'])} | {num(lat.get('p50'), 1)} | "
                f"{num(lat.get('p95'), 1)} |"
            )
    return "\n".join(lines)


def temperature_line(result):
    """What temperature a server's numbers were measured at, and the one its
    GGUF carries; None when there is nothing to say."""
    applied = ", ".join(f"{t:.4g}" for t in result["temperatures_applied"])
    if not applied and not result["gguf"]:
        return None
    line = f"temperature applied {applied or 'not reported'}"
    if result["temperature_asked"] is not None:
        line += f" (asked for {result['temperature_asked']:.4g})"
    if result["gguf"]:
        carried = result["temperature_gguf"]
        line += (f"; its GGUF carries {carried:.4g}" if carried is not None
                 else "; its GGUF carries none")
    return line


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--candidate", required=True, help="EuLLM server with the candidate")
    parser.add_argument("--candidate-model", default=None,
                        help="decision model to ask for there (default: the one loaded)")
    parser.add_argument("--current", default=None,
                        help="EuLLM server with the model the candidate would replace")
    parser.add_argument("--current-model", default=None)
    for role in ("candidate", "current"):
        parser.add_argument(f"--{role}-temperature", type=float, default=None,
                            help=f"the temperature the {role} is served with (eullm."
                                 "temperature; default: the server's own)")
        parser.add_argument(f"--{role}-gguf", default=None,
                            help=f"the GGUF the {role}'s server serves: the temperature it "
                                 "carries (eullm.decision.temperature) is checked against the "
                                 "one the server applies")
    parser.add_argument("--data", action="append", default=[], help="a labelled JSONL set")
    parser.add_argument("--traces", action="append", default=[],
                        help="a traces directory: every decision with feedback is a request")
    parser.add_argument("--sources", default="",
                        help="only answers from these sources, comma-separated prefixes "
                             "(e.g. feedback); default: every labelled answer")
    parser.add_argument("--modes", default=",".join(MODES),
                        help="evaluation modes to compare, comma-separated")
    parser.add_argument("--serve-mode", default="shared_prefix",
                        help="the mode the model will serve in: accuracy, calibration and "
                             "latency are measured in it")
    parser.add_argument("--limit", type=int, default=0, help="requests per set, 0 for all")
    parser.add_argument("--api-key", default=os.environ.get("EULLM_API_KEY"),
                        help="API key, when the servers require one (default: $EULLM_API_KEY)")
    parser.add_argument("--timeout", type=float, default=600.0, help="per request, seconds")
    for name, (value, reason) in qf_metrics.THRESHOLDS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value,
                            help=f"default {value}: {reason}")
    parser.add_argument("--out", default=None, help="the report, as JSON")
    parser.add_argument("--details", default=None, help="every answer, one JSON line each")
    args = parser.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if any(m not in MODES for m in modes) or args.serve_mode not in MODES:
        parser.error(f"modes are {', '.join(MODES)}")
    if args.serve_mode not in modes:
        modes.append(args.serve_mode)
    if not args.data and not args.traces:
        parser.error("give --data or --traces")
    sets = [qf_data.from_jsonl(p) for p in args.data]
    sets += [qf_data.from_traces(d) for d in args.traces]
    prefixes = [p.strip() for p in args.sources.split(",") if p.strip()]
    items = []
    for labelled in sets:
        labelled = qf_data.only_sources(labelled, prefixes)
        for reason, n in labelled.skipped.items():
            print(f"  {labelled.name}: left out {n}: {reason}", file=sys.stderr)
        items += labelled.items[: args.limit] if args.limit else labelled.items
    if not items:
        parser.error("no labelled request to ask")
    answers = sum(len(i.questions) for i in items)
    print(f"{len(items)} requests, {answers} labelled answers; modes {', '.join(modes)}, "
          f"serving in {args.serve_mode}", file=sys.stderr, flush=True)

    thresholds = {name: getattr(args, name) for name in qf_metrics.THRESHOLDS}
    thresholds["min_answers"] = int(thresholds["min_answers"])
    carried = {}
    for role in ("candidate", "current"):
        path = getattr(args, f"{role}_gguf")
        if path:
            try:
                carried[role] = gguf_temperature(path)
            except (OSError, GGUFError) as e:
                parser.error(f"--{role}-gguf {path}: {e}")
    servers = [Server("candidate", args.candidate, args.candidate_model, args.api_key,
                      args.timeout, args.candidate_temperature, args.candidate_gguf,
                      carried.get("candidate"))]
    if args.current:
        servers.append(Server("current", args.current, args.current_model, args.api_key,
                              args.timeout, args.current_temperature, args.current_gguf,
                              carried.get("current")))
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(args.out or f"qualify-{stamp}.json")
    details = open(args.details, "w", encoding="utf-8") if args.details else None
    results = []
    try:
        for server in servers:
            result = measure(run(server, items, modes, args.serve_mode, details), modes,
                             args.serve_mode)
            result["role"] = server.role
            results.append(result)
    except (ServerError, OSError) as e:
        # No model loaded, a wrong key, nothing listening: nothing measured.
        print(f"error: the {server.role} at {server.base}: {e}", file=sys.stderr)
        return 2
    finally:
        if details:
            details.close()
    candidate = results[0]
    current = results[1] if len(results) > 1 else None
    found = qf_metrics.checks(candidate, thresholds, current)
    passed = all(c["passed"] for c in found)
    report = {
        "when": stamp,
        "verdict": "PASS" if passed else "FAIL",
        "thresholds": thresholds,
        "modes": modes,
        "serve_mode": args.serve_mode,
        "sets": [s.name for s in sets],
        "sources": prefixes,
        "requests": len(items),
        "checks": found,
        "comparison": compare(candidate, current, args.serve_mode) if current else None,
        "results": [{k: v for k, v in r.items() if k != "records"} for r in results],
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(table(results))
    print()
    for r in results:
        line = temperature_line(r)
        if line:
            print(f"  {r['role']}: {line}")
    for c in found:
        mark = "PASS" if c["passed"] else "FAIL"
        line = f"  [{mark}] {c['type']}: {c['text']}"
        print(line if c["passed"] else f"{line}\n         why it matters: {c['reason']}")
    if current:
        for kind, m in report["comparison"].items():
            print(f"  {kind}: right only for the candidate {m['candidate_only']}, only for the "
                  f"current model {m['current_only']} (McNemar p = {m['p_value']:.3f})")
    failed = sum(not c["passed"] for c in found)
    print(f"\n{report['verdict']}: " + ("every check passed" if passed else
                                       f"{failed} of {len(found)} checks failed"))
    print(f"report: {out}", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
