"""Tool-selection sets for ReflexBench, all brought to one shape.

Each set is downloaded from its publisher on first use, at a fixed
revision so that a report can be reproduced, and kept in a cache directory,
never in the repository:

  * MetaTool (MIT, https://github.com/HowieHwong/MetaTool): 20,614 requests
    that need one of 199 tools, and 497 that need two;
  * BFCL (Apache-2.0, the Berkeley Function Calling Leaderboard,
    https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard):
    requests offered a few functions of which one fits, and requests none of
    whose functions fits.

A tool has a name, the description a decision model reads, and the spec a
tool-calling model would receive in its prompt: for BFCL the function's JSON
schema, for MetaTool, which has no schemas, its name and description. The
spec is what selecting fewer tools keeps out of that prompt.
"""

import csv
import http.client
import io
import json
import os
import pathlib
import random
import time
import urllib.error
import urllib.request

# The revisions the reports are made on: MetaTool's master and the BFCL
# dataset as of September 2026.
METATOOL = (
    "https://raw.githubusercontent.com/HowieHwong/MetaTool/"
    "35e81bb7576826e980c80fed8f8c0a2b4a1e6fbb/"
)
BFCL = (
    "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
    "resolve/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda/"
)

SETS = {
    "metatool-single": "MetaTool: one tool of 199 needed",
    "metatool-multi": "MetaTool: two tools of 199 needed",
    "bfcl-multiple": "BFCL: one function of 2-4 fits",
    "bfcl-live-multiple": "BFCL live: one function of 2-37 fits",
    "bfcl-irrelevance": "BFCL: no function fits",
    "bfcl-live-irrelevance": "BFCL live: no function of 1-37 fits",
}


class Tool:
    """A tool on offer."""

    def __init__(self, name, description, spec):
        self.name, self.description, self.spec = name, description, spec


class Item:
    """A request, the tools offered with it, and the names of those it needs
    (none, one or several)."""

    def __init__(self, id, request, candidates, needed):
        self.id, self.request = id, request
        self.candidates, self.needed = candidates, needed


class Dataset:
    """Items of one set. `fixed_catalog` says every item offers the same
    tools in the same order, which is what lets the catalog be a decision
    state that is read once and reused."""

    def __init__(self, name, items, fixed_catalog):
        self.name, self.items, self.fixed_catalog = name, items, fixed_catalog


def cache_dir():
    root = os.environ.get("REFLEXBENCH_CACHE") or "~/.cache/reflexbench"
    path = pathlib.Path(root).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch(url, attempts=5):
    """The file at `url`, downloaded once. A transfer cut short is taken up
    where it stopped, a few times, and nothing incomplete is kept."""
    path = cache_dir() / url.split("//", 1)[1].replace("/", "_")
    if path.exists():
        return path.read_bytes()
    data = b""
    for attempt in range(attempts):
        headers = {"Range": f"bytes={len(data)}-"} if data else {}
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=120
            ) as r:
                if data and r.status != 206:
                    data = b""  # the whole file again, not the rest of it
                data += r.read()
            break
        except http.client.IncompleteRead as e:
            data += e.partial
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)
    else:
        raise OSError(f"{url}: the download kept being cut short")
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    part.replace(path)
    return data


def metatool_tools():
    """The 199 tools, by name. The merged descriptions of `big_tool_des`
    replace the plugin store's for the tools that have one."""
    tools = json.loads(fetch(METATOOL + "dataset/plugin_des.json"))
    tools.update(json.loads(fetch(METATOOL + "dataset/big_tool_des.json")))
    return {name: Tool(name, text, f"{name}: {text}") for name, text in sorted(tools.items())}


def metatool(kind, limit, seed):
    tools = metatool_tools()
    catalog = list(tools.values())
    if kind == "single":
        data = fetch(METATOOL + "dataset/data/all_clean_data.csv").decode("utf-8")
        rows = [(r["Query"], [r["Tool"]]) for r in csv.DictReader(io.StringIO(data))]
    else:
        rows = [
            (r["query"], list(r["tool"]))
            for r in json.loads(fetch(METATOOL + "dataset/data/multi_tool_query_golden.json"))
        ]
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    items = []
    for n in order[:limit] if limit else order:
        request, needed = rows[n]
        missing = [t for t in needed if t not in tools]
        if missing:
            raise ValueError(f"MetaTool row {n} needs unknown tools {missing}")
        items.append(Item(f"metatool-{kind}-{n}", request, catalog, needed))
    return Dataset(f"metatool-{kind}", items, fixed_catalog=True)


def bfcl(category, limit, seed):
    lines = fetch(BFCL + f"BFCL_v3_{category}.json").decode("utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    answers = {}
    if "irrelevance" not in category:
        for line in (
            fetch(BFCL + f"possible_answer/BFCL_v3_{category}.json").decode("utf-8").splitlines()
        ):
            if line.strip():
                row = json.loads(line)
                answers[row["id"]] = [next(iter(call)) for call in row["ground_truth"]]
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    items = []
    for n in order:
        row = rows[n]
        functions = row.get("function") or []
        if not functions:
            continue  # nothing offered: nothing to select from
        turn = row["question"][0]
        request = "\n".join(m["content"] for m in turn if m.get("role") == "user")
        if not request.strip():
            continue
        candidates = [Tool(f["name"], f.get("description") or "", json.dumps(f)) for f in functions]
        if len({t.name for t in candidates}) != len(candidates):
            continue  # a name offered twice cannot be scored apart
        needed = answers.get(row["id"], [])
        if "irrelevance" not in category and not needed:
            continue
        items.append(Item(row["id"], request, candidates, needed))
        if limit and len(items) >= limit:
            break
    return Dataset(f"bfcl-{category.replace('_', '-')}", items, fixed_catalog=False)


def load(name, limit=0, seed=1):
    """One of SETS, `limit` items at most (0: all), in a fixed shuffled order."""
    if name == "metatool-single":
        return metatool("single", limit, seed)
    if name == "metatool-multi":
        return metatool("multi", limit, seed)
    if name.startswith("bfcl-"):
        return bfcl(name[len("bfcl-") :].replace("-", "_"), limit, seed)
    raise ValueError(f"unknown set {name!r}: one of {', '.join(SETS)}")


def from_jsonl(path):
    """A set of your own, one JSON object per line:
    {"id": ..., "request": "...", "needed": ["tool", ...],
     "tools": [{"name": ..., "description": ..., "spec": optional}, ...]}.
    Items with the same tools in the same order share one catalog."""
    items, catalogs = [], {}
    for n, line in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        key = json.dumps(row["tools"], sort_keys=True)
        if key not in catalogs:
            catalogs[key] = [
                Tool(t["name"], t.get("description") or "", t.get("spec") or json.dumps(t))
                for t in row["tools"]
            ]
        item = Item(str(row.get("id", n)), row["request"], catalogs[key], row.get("needed", []))
        offered = {t.name for t in item.candidates}
        if any(name not in offered for name in item.needed):
            raise ValueError(f"{path}: {item.id} needs a tool it is not offered")
        items.append(item)
    return Dataset(pathlib.Path(path).stem, items, fixed_catalog=len(catalogs) == 1)


def resized(dataset, size, seed):
    """Every item offered `size` tools at most: the ones it needs, then
    others drawn from its own candidates at random. The scale experiment:
    how selection holds up as the catalog grows."""
    items = []
    for item in dataset.items:
        rng = random.Random(f"{seed}:{item.id}")
        needed = [t for t in item.candidates if t.name in item.needed]
        others = [t for t in item.candidates if t.name not in item.needed]
        rng.shuffle(others)
        chosen = needed + others[: max(0, size - len(needed))]
        chosen.sort(key=lambda t: t.name)
        items.append(Item(item.id, item.request, chosen, item.needed))
    return Dataset(f"{dataset.name}@{size}", items, fixed_catalog=False)
