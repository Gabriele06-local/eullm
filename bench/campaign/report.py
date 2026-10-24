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
          "env", "trial", "round")

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
                 ("duration_s", "dur s"), ("died", "died")) + DEVICE,
    "decision": (("dec_per_s", "dec/s"), ("dec_client_ms_p50", "p50 ms"),
                 ("dec_client_ms_p99", "p99 ms"), ("dec_wait_ms_p50", "wait ms"),
                 ("dec_decode_ms_p50", "decode ms"), ("dec_consistency", "consist"),
                 ("dec_same_choices", "same choice"), ("dec_flipped", "flipped"),
                 ("dec_largest_shift", "max shift"),
                 ("dec_together", "together")) + DEVICE,
    "finetune": (("ft_loss_before", "loss0"), ("ft_loss_after", "loss1"),
                 ("ft_tok_s", "tok/s")),
}

SHORT = {"model": 34, "extra_args": 26, "env": 34}


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
    if "env" in varying:
        # the variables every row sets alike say nothing about the row
        common = set.intersection(*(set((r.get("env") or "").split()) for r in rows))
        rows = [r | {"env": " ".join(v for v in (r.get("env") or "").split()
                                     if v not in common)} for r in rows]
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


# ── a round against the ones before it ────────────────────────────────────

# What a configuration is read by when it is compared across engine builds.
HEADLINE = {"throughput": "agg_tok_s_mean", "workload": "agg_tok_s_mean",
            "decision": "dec_per_s"}
# What makes two results the same configuration: what the point was planned
# with, not the round, the job or the engine that measured it.
CONFIG = ("campaign", "group", "kind", "model", "runtime", "gcds", "replicas", "mode",
          "batch", "ctx", "slot_ctx", "kv", "concurrency", "prompt_tokens", "decision_mode",
          "state_tokens", "questions", "extra_args", "env", "trial")
# What a row is named by in the comparison, when it is not the group's only one.
LABEL = ("model", "runtime", "gcds", "replicas", "batch", "concurrency", "kv",
         "extra_args", "env")


def _label(row, params=LABEL) -> str:
    parts = []
    for p in params:
        v = row.get(p)
        if v in (None, "", "-") or (p == "runtime" and v == "eullm"):
            continue
        v = _cell(p, v)
        parts.append(v if p == "model" else f"{p}={v}")
    return " ".join(parts) or "-"


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2) if n else None


def compare(rows, is_new, threshold=0.95) -> str:
    """Every configuration measured by the rows `is_new` picks against its
    last earlier measurement: the mean of its headline metric (tokens/s, or
    decisions/s) in the new rows over the mean in the latest earlier round
    that measured it. One line a group: how many configurations, the median
    ratio and its range, and the slowest one by name; a group whose median
    is below `threshold` is marked. Configurations measured for the first
    time are counted, not compared."""
    new, old = defaultdict(list), defaultdict(list)
    for r in rows:
        if r.get("outcome") != "measured" or r.get("kind") not in HEADLINE:
            continue
        if _number(r.get(HEADLINE[r["kind"]])) is None:
            continue
        key = tuple(r.get(k) for k in CONFIG)
        (new if is_new(r) else old)[key].append(r)
    groups = defaultdict(list)
    first = defaultdict(int)
    for key, rs in new.items():
        metric = HEADLINE[rs[0]["kind"]]
        latest = max((r.get("ended") or "" for r in rs), default="")
        before = [r for r in old.get(key, []) if not latest or (r.get("ended") or "") < latest]
        if not before:
            first[(key[0], key[1])] += 1
            continue
        last_round = max(before, key=lambda r: r.get("ended") or "").get("round")
        ref = [r for r in before if r.get("round") == last_round]
        now = sum(_number(r.get(metric)) for r in rs) / len(rs)
        then = sum(_number(r.get(metric)) for r in ref) / len(ref)
        if then > 0:
            groups[(key[0], key[1])].append((now / then, now, then, last_round, rs[0]))
    out, ratios = [], []
    for g in sorted(set(groups) | set(first)):
        cs = groups.get(g, [])
        line = f"{g[0]} / {g[1]}: "
        if cs:
            rs = [c[0] for c in cs]
            ratios.extend(rs)
            med = _median(rs)
            slow = min(cs, key=lambda c: c[0])
            # named by what varies among the group's configurations
            varying = [p for p in LABEL if len({c[4].get(p) for c in cs}) > 1] or ["model"]
            line += (f"{len(cs)} compared, new/old {med:.2f} ({min(rs):.2f}-{max(rs):.2f})"
                     f"; slowest {_label(slow[4], varying)} {slow[0]:.2f}"
                     f" ({slow[2]:.4g} -> {slow[1]:.4g}"
                     f", {slow[3]})")
            if med < threshold:
                line = "[!!] " + line
        else:
            line += "nothing measured before"
        if first.get(g):
            line += f"; {first[g]} first measured"
        out.append(line)
    if ratios:
        out.append(f"all: {len(ratios)} configurations compared, new/old median "
                   f"{_median(ratios):.2f} ({min(ratios):.2f}-{max(ratios):.2f}), "
                   f"{sum(r < threshold for r in ratios)} below {threshold}")
    else:
        out.append("nothing to compare: no configuration measured before")
    return "\n".join(out)
