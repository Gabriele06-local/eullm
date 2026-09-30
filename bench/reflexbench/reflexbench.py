#!/usr/bin/env python3
"""ReflexBench: how well a decision picks the tools a request needs.

MVP 0 of the Reflex roadmap (docs/reflex-roadmap.md). For each request of a
set, every method ranks the tools on offer; the report says, per set and
method:

  * recall@k: how often every tool the request needs is among the first k;
  * k95 and k99: the k that keeps 95% and 99% of the requests whole, and
    the share of the tool specs a tool-calling model would still receive at
    that k;
  * for sets where no tool fits: how often Reflex said "no tool", and on
    the others how often it said so wrongly;
  * what the decision cost: latency p50 and p95 as the client sees it, and
    for Reflex the tokens the decision model evaluated per request.

The methods (see rb_methods.py): `bm25`, keyword matching; `embed`, EuLLM's
own embeddings (needs --embed-model); `reflex-a`, the request as the
decision state and the tools as options; `reflex-b`, the tool catalog as the
state, read once and reused while it stays the same, the request in the
question and the tool names as options; `two-stage`, the embeddings keep
--shortlist tools and `reflex-a` ranks those.

The sets (see rb_data.py) are downloaded on first use to
~/.cache/reflexbench ($REFLEXBENCH_CACHE); `--data` takes a set of your own.
Start the server with a Jev-Style decision model and room for a long
catalog, writing its audit trail somewhere of its own — a run is thousands
of decisions:

    EULLM_AUDIT_DIR=/tmp/reflexbench-audit \\
      eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m --decision-ctx 25600
    python bench/reflexbench/reflexbench.py --limit 200

Only the Python standard library is needed.
"""

import argparse
import datetime
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_data  # noqa: E402
import rb_methods  # noqa: E402
import rb_metrics  # noqa: E402

DEFAULT_SETS = "metatool-single,metatool-multi,bfcl-live-multiple,bfcl-live-irrelevance"


METHODS = ("bm25", "embed", "reflex-a", "reflex-b", "two-stage")


def build_methods(args, dataset):
    abstain = args.abstain == "on"

    def embeddings():
        return rb_methods.Embeddings(
            args.url,
            args.embed_model,
            args.api_key,
            args.timeout,
            args.embed_query_prefix.replace("\\n", "\n"),
        )

    def reflex(layout):
        return rb_methods.Reflex(args.url, layout, args.model, args.api_key, args.timeout, abstain)

    methods = []
    for name in args.methods.split(","):
        name = name.strip()
        if name in ("embed", "two-stage") and not args.embed_model:
            print(f"  {name} skipped: no --embed-model", file=sys.stderr)
        elif name == "bm25":
            methods.append(rb_methods.BM25(dataset))
        elif name == "embed":
            methods.append(embeddings())
        elif name in ("reflex-a", "reflex-b"):
            methods.append(reflex(name[-1].upper()))
        elif name == "two-stage":
            methods.append(rb_methods.TwoStage(embeddings(), reflex("A"), args.shortlist))
        elif name:
            raise SystemExit(f"unknown method {name!r}: {', '.join(METHODS)}")
    return methods


def run(method, dataset, details):
    """Rank every item, after ranking the last one once untimed: a cold
    start is not a decision, and warming up on the last item keeps the
    first from finding its own state already read. With one catalog for
    every item, the warm-up reads it, as a deployment would once. Progress
    every 25 items, or every 30 seconds when decisions are slow, as on a
    CPU."""
    rankings = []
    started = shown = time.perf_counter()
    if dataset.items:
        model = method.rank(dataset.items[-1]).server.get("model")
        if model:
            # Say which model decides before its decisions are counted.
            print(f"    decision model: {model}", file=sys.stderr, flush=True)
    for n, item in enumerate(dataset.items, 1):
        ranking = method.rank(item)
        rankings.append(ranking)
        if details:
            details.write(
                json.dumps(
                    {
                        "set": dataset.name,
                        "method": method.name,
                        "id": item.id,
                        "needed": item.needed,
                        "needed_ranks": [ranking.order.index(n) + 1 for n in item.needed],
                        "candidates": len(item.candidates),
                        "top": ranking.order[:20],
                        "refused": ranking.refused,
                        "none_score": ranking.none_score,
                        "best_score": ranking.best_score,
                        "ms": round(ranking.ms, 2),
                        "server": ranking.server,
                    }
                )
                + "\n"
            )
            details.flush()
        now = time.perf_counter()
        if n % 25 == 0 or n == len(dataset.items) or now - shown >= 30:
            shown = now
            print(
                f"    {n}/{len(dataset.items)} ({now - started:.0f} s)", file=sys.stderr, flush=True
            )
    return rankings


def table(results):
    """The report as Markdown."""
    head = (
        "| set | method | items | refused | R@1 | R@3 | R@5 | R@10 | k95 | k99 | "
        "specs kept @k95 | abstain ok | false abstain | p50 ms | p95 ms | tokens/decision |"
    )
    lines = [head, "|---" * (head.count("|") - 1) + "|"]

    def pct(x):
        return "—" if x is None else f"{100 * x:.1f}%"

    def ms(x):
        return "—" if x is None else f"{x:.1f}"

    for r in results:
        m = r["metrics"]
        recall = m.get("recall_at", {})
        lat = m["latency_ms"]
        lines.append(
            f"| {r['set']} | {r['method']} | {m['items']} | {m.get('refused', 0)} | "
            + " | ".join(pct(recall.get(k)) for k in ("1", "3", "5", "10"))
            + f" | {m.get('k95', '—')} | {m.get('k99', '—')} | "
            f"{pct(m.get('spec_kept_at_k95'))} | {pct(m.get('abstain_accuracy'))} | "
            f"{pct(m.get('false_abstain_rate'))} | "
            f"{ms(lat['p50'])} | {ms(lat['p95'])} | "
            + (f"{m['evaluated_tokens_mean']:.0f}" if "evaluated_tokens_mean" in m else "—")
            + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:11434", help="EuLLM server URL")
    parser.add_argument("--model", default=None, help="decision model (default: the one loaded)")
    parser.add_argument("--embed-model", default=None, help="embedding model for `embed`")
    parser.add_argument(
        "--embed-query-prefix",
        default="",
        help="text put before each request for `embed`, as some models want; \\n is a newline",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EULLM_API_KEY"),
        help="API key, when the server requires one (default: $EULLM_API_KEY)",
    )
    parser.add_argument(
        "--sets", default=DEFAULT_SETS, help=f"comma-separated: {', '.join(rb_data.SETS)}"
    )
    parser.add_argument(
        "--data", action="append", default=[], help="a JSONL set of your own (repeatable)"
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--shortlist",
        type=int,
        default=20,
        help="tools the embeddings keep for Reflex in `two-stage`",
    )
    parser.add_argument("--limit", type=int, default=100, help="items per set (0: all)")
    parser.add_argument(
        "--catalog-size",
        type=int,
        default=0,
        help="offer each request at most this many tools, its own among them",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--abstain",
        choices=["on", "off"],
        default="on",
        help='offer Reflex a "no tool" option on every set: it should win where no tool '
        "fits, and lose everywhere else",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request timeout, seconds")
    parser.add_argument("--out", default=None, help="the report, as JSON")
    parser.add_argument("--details", default=None, help="every ranking, one JSON line each")
    args = parser.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(args.out or f"reflexbench-{stamp}.json")
    details = open(args.details, "w", encoding="utf-8") if args.details else None
    datasets = [
        rb_data.load(s.strip(), args.limit, args.seed) for s in args.sets.split(",") if s.strip()
    ]
    datasets += [rb_data.from_jsonl(path) for path in args.data]
    if args.catalog_size:
        datasets = [rb_data.resized(d, args.catalog_size, args.seed) for d in datasets]

    results = []
    try:
        for dataset in datasets:
            print(f"{dataset.name}: {len(dataset.items)} requests", file=sys.stderr, flush=True)
            for method in build_methods(args, dataset):
                print(f"  {method.name}", file=sys.stderr, flush=True)
                rankings = run(method, dataset, details)
                served = sorted({r.server["model"] for r in rankings if r.server.get("model")})
                results.append(
                    {
                        "set": dataset.name,
                        "method": method.name,
                        "decision_model": ", ".join(served) or None,
                        "fixed_catalog": dataset.fixed_catalog,
                        "abstain_asked": getattr(method, "abstain", False),
                        "metrics": rb_metrics.summarize(dataset.items, rankings),
                    }
                )
    finally:
        if details:
            details.close()
        report = {
            "when": stamp,
            "url": args.url,
            "decision_model_asked": args.model,
            "decision_models": sorted({r["decision_model"] for r in results} - {None}),
            "embed_model": args.embed_model,
            "embed_query_prefix": args.embed_query_prefix,
            "limit": args.limit,
            "catalog_size": args.catalog_size,
            "shortlist": args.shortlist,
            "abstain": args.abstain,
            "seed": args.seed,
            "results": results,
        }
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if results:
            if report["decision_models"]:
                print(f"decision model: {', '.join(report['decision_models'])}\n")
            print(table(results))
        print(f"\nreport: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
