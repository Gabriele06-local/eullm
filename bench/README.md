# bench/ — Stress Test & Parallelism Verification

## WP4 — CIX P1 ARM CPU baseline

See [`docs/arm-cix-p1-cpu-profile.md`](../docs/arm-cix-p1-cpu-profile.md) for
the full build profile, i8mm/repack runtime verification, and thread-pinning
recipe for the Radxa Orion O6 (CIX P1). `detect_arm_big_cores.sh` finds the
big Cortex-A720 cores on the actual board (core numbering isn't stable
across firmware, so this can't be a hardcoded range); `arm_cpu_bench.py` is
the T4.1 prefill/decode baseline harness.

Real stress test that **proves** whether an inference server processes requests in parallel or just queues them sequentially.

## `decision_bench.py` — shared-prefix benchmark for `/v1/systemone`

Measures what answering many questions about the same state in one request
buys over asking them one at a time, in each of the three evaluation modes:
`shared_prefix` (the default: the state decoded once, then each question on
its own right after it), `batched` (the state once, then every question in
one batch) and `separate` (each question from scratch, the baseline). For
every state size (`--states`, default 256/1k/4k tokens) and question count
(`--questions`, default 1/4/8/16/32/64): the tokens decoded with the state
shared and without, the decode time of each mode
(median of `--repeat` runs, from the server's own timings, which on a GPU
wait for the computation to finish), the speedup of `shared_prefix` over
`separate`, and how far each mode's answers are from the baseline's.
Standard library only.

```bash
eullm serve --decision-model qwen3-0.6b --decision-ctx 16384
python bench/decision_bench.py --url http://localhost:11434 --json decision-bench.json
```

`--decision-ctx 16384` covers the largest default case in `batched` mode (a
4k-token state with 64 questions needs about 9k tokens there; the other two
modes need only the state plus the longest question); a case over the limit
is reported as skipped. `--max-separate-tokens` (default 150k) skips the
one-at-a-time baseline where it would take too long — on a CPU, lower it.

The dP columns are not an error margin to shrink. The modes read the same
tokens but hand them to the kernels in batches of different shapes, and on
quantized weights that alone moves an answer by the model's own numerical
noise. The largest dP from `separate`, on a 4-core CPU
(`--states 256,1024 --questions 1,8,64`) and on an RTX 5070 Ti (the
defaults), where ggml-cuda's TF32 and half-precision arithmetic is coarser:

| Model | CPU `shared_prefix` | CPU `batched` | GPU `shared_prefix` | GPU `batched` |
|---|---|---|---|---|
| Qwen3-0.6B F16 | 0.017 | 0.017 | | |
| Qwen3-0.6B Q8_0 | 0.11 | 0.13 | | |
| Qwen3-0.6B Q4_K_M | 0.34 | 0.32 | 0.53 | 0.52 |
| Jev-Style-0.8B-Decision-v3 Q4_K_M | 0.024 | 0.028 | 0.039 | 0.033 |
| Jev-Style-2B-Decision-v3 Q4_K_M | | | 0 | 0 |

From Q4_K_M on, the noise of an instruction-tuned model is enough to change
the top answer of a question near a tie. Calibrate in the mode you serve in.
The Jev-Style 2B shows none: its input is cut into blocks where the state
ends, so every mode decodes the same blocks, and its `batched` requests are
answered as `shared_prefix` (see
[docs/engine.md](../docs/engine.md#jev-style-decision-models)).

What `shared_prefix` adds is that this noise no longer depends on the other
questions: a question is decoded alone, in the same cache cells and in
batches of the same shape whatever else the request asks, so its answer is
a function of the state and that question only. `--order-check` verifies
it: it asks the questions again in reverse order, the first and last alone,
and the whole request once more starting from the state the server kept
from the request before (`state kept`, timed), and in `shared_prefix` mode
every answer must come back bit for bit the same — `dP 0.0000 ...
identical`, measured on the CPU and on an RTX 5070 Ti — while `batched`
shows how much an answer moves only because of where its question sits in
the batch. The bench exits with an error if a `shared_prefix` answer moved.
The timed runs themselves never start from a kept state.

`--details` prints, per case and mode, the question that moved most from the
baseline with its coverage in both (a coverage that collapsed in one mode
would mean that mode read the wrong logits; a similar coverage with a
shifted distribution is arithmetic).

## `decision_calibration.py` — calibration comparison for `/v1/systemone`

Runs a labelled JSONL dataset through the decision model once and compares
no calibration, content-free calibration, temperature scaling (T fitted by
5-fold cross-fitting, never on the items it scores) and both together, on
accuracy, NLL, Brier score and ECE with 95% bootstrap intervals, plus the
share of items answered and their accuracy at several confidence
thresholds. The dataset format is in the script's docstring; the items are
yours to choose — a few hundred per question type is the least that makes
the intervals useful.

```bash
python bench/decision_calibration.py labelled.jsonl --url http://localhost:11434 \
    --json calibration.json
```

## `reflexbench/` — does a decision pick the tools a request needs?

MVP 0 of the [Reflex roadmap](../docs/reflex-roadmap.md): on MetaTool and
BFCL, how many tools a method can drop without losing the one needed, how
much of the tool-calling model's prompt that saves, and what the decision
costs — `/v1/systemone` in two layouts against BM25 and embeddings; and
`ragbench.py`, whether the passages a RAG system retrieved suffice to
answer. See [`reflexbench/README.md`](reflexbench/README.md).

## MTP: `mtp_sweep.sh`, `mtp_head_q8.sh`, `mtp_test_d.sh`

What decides where `--mtp` pays (roadmap 0.8-Z2), each printing one line per
setting — writing speed on a story and on a piece of code, and the share of
the drafts the model kept, which every answer reports:

- `mtp_sweep.sh EULLM MODEL.gguf [FLAGS]` — every `--mtp` setting of one
  model in EuLLM. `TEMPERATURE=0.8` measures the drafts kept at the usual
  sampling temperature: if that falls far below the greedy share, accepting
  drafts by their probabilities (Leviathan's rejection sampling) is worth
  writing; if not, it is not.
- `mtp_head_q8.sh EULLM SOURCE.gguf [FLAGS]` — the same model quantized twice
  to Q4_K_M from a Q8_0 or BF16 source, as unsloth's Q4_K_M (only the head's
  own tensors at Q8_0) and with the whole MTP layer at Q8_0, then
  `mtp_sweep.sh` on both: whether a Q8_0 MTP layer keeps more drafts for its
  few extra MiB. Needs `llama-quantize` (`LLAMA_QUANTIZE`).
- `mtp_test_d.sh LLAMA_SERVER MODEL.gguf [FLAGS]` — test D: llama.cpp's own
  MTP on an MoE with every expert in RAM, pinned (`--load-mode none`) and
  cached in VRAM (`--moe-cache-mib`), with 0, 1 and 2 drafts. Needs a
  `llama-server` built from the llama.cpp EuLLM pins, which carries the
  expert cache. If drafting gains there, it is worth measuring in EuLLM
  (`mtp_sweep.sh` with `--moe-cache auto`).

## `prefetch_check.sh` — phase 6 of `docs/moe-offload-plan.md`

`prefetch_check.sh EULLM MODEL.gguf [FLAGS]` starts `eullm serve` on an MoE
whose experts do not all fit in VRAM, with `LLAMA_MOE_PREFETCH=0` and then
`=1` (`--ctx-size 40960 --moe-cache auto --n-ubatch 4096` unless `CTX` and
`N_UBATCH` say otherwise). Each server answers the same long question twice,
greedy and with `cache_prompt: false`, and `speed_check.py` measures it over
a 33,200-token document (`PROMPT_TOKENS`). One line per setting: reading and
writing speeds, a checksum of each answer, and the server's `moe prefetch:`
line; then whether the four answers match, which they must, or why the
comparison says nothing (the prefetch stayed off, a server gave no answer).

## `interleave_check.py` — roadmap 0.7-D

`interleave_check.py --url URL --model NAME` streams an answer and, a few
tokens in, sends a long prompt of its own (8,000 tokens, `--prompt-tokens`)
with a one-token answer. It prints how long that prompt took to read, the
longest pause in the streamed answer meanwhile, and the answer's speed
before and during. With `--batch-size 2` or more EuLLM reads the prompt in
chunks between the answer's tokens and the answer keeps coming, a little
slower; a server that reads a prompt whole stops it for the whole reading.
Standard library only.

## `reuse_validation.py` — roadmap 0.7-A real-hardware checklist

Validates the KV-cache prefix reuse scheduler against the checklist in
[`docs/roadmap-engine-0.7-1.0.md`](../docs/roadmap-engine-0.7-1.0.md) § 0.7-A:
a 20-turn growing-history conversation, 8 concurrent conversations, abrupt
mid-stream disconnects, a slow-consuming client (the v0.6.20 `Full`-vs-`Closed`
channel regression test), and byte-identical output at a fixed seed.

Start the server headless first — note this is `eullm run <model>`, not
`eullm serve` (which starts the API with no model loaded and expects a
`model` field per-request instead). `--no-ui` skips the browser/chat-UI
auto-open, and `< /dev/null` keeps stdin a non-tty so it doesn't drop into
the interactive REPL when backgrounded. Bump `--ctx-size` for a 20-turn
growing conversation plus `--batch-size` concurrent slots — the default
4096 is split across all slots (`ctx_size / batch_size`) and fills up fast:

```bash
bin/eullm run <model-id-or-path> --no-ui --batch-size 8 --ctx-size 16384 \
    < /dev/null > server.log 2>&1 &
```

Then:

```bash
pip install aiohttp

python bench/reuse_validation.py \
    --url http://localhost:11434 \
    --model <same-model-id-or-path> \
    --server-log server.log
```

Run a subset with `--tests multiturn,slow-consumer` (see `--help` for every
flag: turn/concurrency counts, token budgets, timeouts). Exit code is nonzero
if any test fails. Pass `--baseline-url` to diff the determinism test's output
against a second server (e.g. an old binary on another port) for a true A/B
instead of a self-comparison.

This exercises the same scheduler code path the `--cli` REPL uses (both
resend the full growing history and share the scheduler), so it doubles as
an automated stand-in for the 20-turn CLI conversation check — driving the
REPL by hand and grepping its log for `reused N from cache` remains a useful
manual cross-check but isn't required to run this suite.

## What it measures

| Metric | What it proves |
|--------|---------------|
| **TTFT** (Time To First Token) | Do all requests start generating immediately, or do later ones wait? |
| **Timeline overlap** | Are generation periods overlapping in time? |
| **Token interleaving** | Do tokens from different requests arrive interleaved (true batching) or in sequence? |
| **Throughput** | Total tok/s across all concurrent requests |
| **Latency distribution** | P50, P95, P99 for TTFT and total latency |

## Quick start

```bash
# Install dependency
pip install aiohttp

# Test EULLM
python bench/stress_test.py \
    --url http://localhost:11434 \
    --model Qwen3.5-9B-Q4_K_M \
    --concurrency 1,2,4,8 \
    --tokens 100 \
    --warmup

# Test Ollama
python bench/stress_test.py \
    --url http://localhost:11435 \
    --model qwen3.5:9b \
    --concurrency 1,2,4,8 \
    --tokens 100 \
    --warmup

# Compare both (requires both servers running)
./bench/compare.sh Qwen3.5-9B-Q4_K_M qwen3.5:9b
```

## How to read the output

### Timeline

```
  req 1 |....############################################  | TTFT   120ms  94.2 tok/s  100 tokens
  req 2 |....############################################  | TTFT   125ms  93.8 tok/s  100 tokens
  req 3 |....#############################################| TTFT   130ms  93.1 tok/s  100 tokens
  req 4 |.....############################################| TTFT   135ms  92.5 tok/s  100 tokens
```

- `.` = prefill/waiting (submit → first token)
- `#` = generating tokens (first token → last token)
- If `#` bars overlap vertically → **real parallel processing**
- If `#` bars are sequential (no vertical overlap) → **queued processing**

### Parallelism verdict

```
  PARALLELISM ANALYSIS (EULLM):
    Overlap:      YES — 6/6 pairs overlap (100%)
    Interleaving: YES — 285 context switches (72% transition rate)
    VERDICT:      REAL PARALLEL PROCESSING
```

vs

```
  PARALLELISM ANALYSIS (Ollama):
    Overlap:      NO  — requests appear sequential
    Interleaving: NO  — tokens arrive in sequence, not interleaved
    VERDICT:      SEQUENTIAL PROCESSING (no real parallelism)
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | (required) | Server base URL |
| `--model` | (required) | Model name |
| `--label` | auto | Label for output |
| `--concurrency` | `1,2,4,8` | Comma-separated concurrency levels |
| `--tokens` | `100` | Tokens to generate per request |
| `--rounds` | `1` | Rounds per concurrency level (for averaging) |
| `--warmup` | off | Send a warmup request first |
| `--json` | none | Write JSON results to file |

## JSON output

Use `--json results.json` to get machine-readable results for further analysis:

```json
{
  "label": "EULLM",
  "model": "Qwen3.5-9B-Q4_K_M",
  "tokens_per_request": 100,
  "results": [
    {
      "concurrency": 4,
      "wall_ms": 3300,
      "throughput": 121.2,
      "overlap": {"is_parallel": true, "overlap_ratio": 1.0},
      "interleaving": {"is_interleaved": true, "transition_rate": 0.72},
      "requests": [...]
    }
  ]
}
```

## Why `bench.sh` was not enough

The old `bench.sh` fired N curl requests with `"stream": false` and measured only wall time. This doesn't prove parallelism — a fast sequential processor could achieve similar wall times. The new stress test uses **streaming** to track individual token arrival timestamps, enabling definitive proof of parallel vs sequential processing.
