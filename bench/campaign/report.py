"""A readable summary of a campaign's results: one table per group.

`campaign.py report --queue Q [--campaign c06-mtp ...]` prints, for every
group, a row per configuration that was measured: the parameters that vary
within the group (model, batch, --mtp, runtime...) and the metrics that fit
the kind of point, averaged over the results that share a configuration
(retries, rounds). Points that failed or did not fit are counted under the
table, so a group with holes says so.
"""

from __future__ import annotations

from collections import defaultdict

# Parameters a group may vary; the ones that do become the table's first
# columns, in this order.
PARAMS = ("model", "runtime", "gcds", "replicas", "batch", "concurrency", "slot_ctx", "kv",
          "prompt_tokens", "decision_mode", "state_tokens", "questions", "extra_args",
          "trial", "round")

# What each kind is read by: (column in summary.csv, heading).
# Every kind that runs on the GPUs also shows the most VRAM a device held and
# how busy the devices were: a server that silently ran on the CPU, or shared
# its GCD with a stranger, shows there before anywhere else.
DEVICE = (("vram_start_mib_max", "vram0"), ("vram_peak_mib_max", "vram MiB"),
          ("use_mean", "use%"), ("neighbours_at_start", "neigh"))
METRICS = {
    "throughput": (("agg_tok_s_mean", "tok/s"), ("agg_tok_s_cv_pct", "cv%"),
                   ("decode_tok_s", "decode"), ("prefill_tok_s", "prefill"),
                   ("ttft_ms_p50", "ttft ms"), ("load_wall_s", "load s")) + DEVICE,
    "workload": (("accuracy", "accuracy"), ("agg_tok_s_mean", "tok/s"),
                 ("consistency", "consist"), ("requests", "requests"),
                 ("duration_s", "dur s")) + DEVICE,
    "decision": (("dec_per_s", "dec/s"), ("dec_client_ms_p50", "p50 ms"),
                 ("dec_client_ms_p99", "p99 ms"), ("dec_wait_ms_p50", "wait ms"),
                 ("dec_decode_ms_p50", "decode ms"), ("dec_consistency", "consist"),
                 ("dec_together", "together")) + DEVICE,
    "finetune": (("ft_loss_before", "loss0"), ("ft_loss_after", "loss1"),
                 ("ft_tok_s", "tok/s")),
}

SHORT = {"model": 34, "extra_args": 26}


def _number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(values):
    """The mean of the numbers among `values`; for text (accuracy is
    `set=0.8 ...`), the first value, since it is not averaged."""
    nums = [n for n in (_number(v) for v in values) if n is not None]
    if nums:
        m = sum(nums) / len(nums)
        return f"{m:.0f}" if abs(m) >= 100 else f"{m:.3g}"
    texts = [v for v in values if v not in (None, "")]
    return texts[0] if texts else "-"


def _cell(name, value):
    value = "-" if value in (None, "") else str(value)
    if name == "extra_args":
        value = value.replace("--fit-strict", "").strip() or "(none)"
    width = SHORT.get(name)
    return value if not width or len(value) <= width else value[: width - 1] + "…"


def table(rows, kind):
    """The lines of one group's table, `rows` its measured results."""
    varying = [p for p in PARAMS if len({r.get(p) for r in rows}) > 1]
    if not varying:
        varying = ["model"]
    metrics = METRICS.get(kind, ())
    by_config = defaultdict(list)
    for r in rows:
        by_config[tuple(r.get(p) for p in varying)].append(r)
    header = [p for p in varying] + [h for _, h in metrics] + ["n"]
    body = []
    for key in sorted(by_config, key=lambda k: tuple(_sort_key(v) for v in k)):
        group = by_config[key]
        body.append([_cell(p, v) for p, v in zip(varying, key)]
                    + [_mean([r.get(c) for r in group]) for c, _ in metrics]
                    + [str(len(group))])
    widths = [max(len(header[i]), *(len(b[i]) for b in body)) for i in range(len(header))]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    return [fmt.format(*header), fmt.format(*("-" * w for w in widths))] + [
        fmt.format(*b) for b in body]


def _sort_key(v):
    n = _number(v)
    return (0, n, "") if n is not None else (1, 0, str(v))


def report(rows, failed=None, campaigns=None) -> str:
    """The whole report: every campaign in `rows` (or those named), every
    group, measured configurations first and holes counted after.
    `failed` maps (campaign, group) to how many points failed."""
    failed = failed or {}
    out = []
    by_group = defaultdict(list)
    for r in rows:
        if campaigns and r.get("campaign") not in campaigns:
            continue
        by_group[(r.get("campaign"), r.get("group"))].append(r)
    for (campaign, group) in sorted(set(by_group) | {k for k in failed
                                                     if not campaigns or k[0] in campaigns}):
        rs = by_group.get((campaign, group), [])
        measured = [r for r in rs if r.get("outcome") == "measured"]
        other = defaultdict(int)
        for r in rs:
            if r.get("outcome") != "measured":
                other[r.get("outcome") or "?"] += 1
        if failed.get((campaign, group)):
            other["failed"] += failed[(campaign, group)]
        kind = (measured or rs or [{}])[0].get("kind", "?")
        out.append(f"=== {campaign} / {group} ({kind}, {len(measured)} measured) ===")
        if measured:
            out.extend("  " + line for line in table(measured, kind))
        if other:
            out.append("  not measured: " + ", ".join(f"{k} {v}" for k, v in sorted(other.items())))
        out.append("")
    return "\n".join(out)
