# Benchmarks — EULLM Engine continuous batching

EULLM Engine uses **continuous batching** to decode multiple requests in parallel within a single GPU pass, enabled by default across all slots. The scheduler runs a dedicated decode loop on an OS thread; each iteration calls `llama_decode` with a batch containing tokens from every active sequence, so requests start receiving tokens immediately instead of queueing for a slot.

This page documents how that scaling behaves on a consumer GPU.

## October 2026: Qwen3-8B on the RTX 5070 Ti

The same GPU and the same [`bench.sh`](../bench.sh) (150 tokens a request,
throughput over the wall time of the slowest), with the model the data-centre
runs below use, on 17 October 2026:

| Concurrent requests | Throughput | Speedup vs N=1 | Each request |
|:---:|:---:|:---:|:---:|
| 1 | 132 tok/s | 1.0× | 144 tok/s |
| 2 | 248 tok/s | 1.9× | 135 tok/s |
| 4 | 433 tok/s | 3.3× | 117 tok/s |
| 8 | 720 tok/s | 5.5× | 98 tok/s |
| 16 | **1,065 tok/s** | **8.1×** | 72 tok/s |

Qwen3-8B Q4_K_M from the catalog (`qwen3-8b`), a local CUDA build of
0.7.70,
`eullm serve --batch-size 16 --ctx-size 65536 --no-fit`: 16 slots of 4,096
tokens, all 36 layers on the GPU, 14.1 of 16.3 GB in use.

`--no-fit` matters here. The sizing keeps 12% of the card free once the
model and its context are in, a margin set after a load that ran out of
memory on the same 16 GB card. With all 36 layers llama.cpp's own breakdown
was 4,455 MiB of weights, 9,216 of f16 KV cache and 108 of compute buffers,
leaving 1,812 of 15,841 MiB free: 11.4%, under the margin, so without the
flag 2 layers stayed in RAM and the same run gave 52 tok/s for one request
and 300 for 16. The load with 11.4% free ran the whole test without a
failure; it is a thinner margin, not a broken one. `--ctx-size 32768` (2,048
tokens a slot, plenty here) or `--cache-type-k q8_0 --cache-type-v q8_0`
halves the KV cache and fits all 36 layers without the flag.

Since 0.7.80 the margin is a fixed 1 GiB (`--vram-reserve`), and this run loads
all 36 layers without `--no-fit`: 144.6 tok/s for one request and 1,054.9 for
16 (the run above with `--no-fit`, repeated the same day, gave 145.1 and 977.2),
the card peaking at 14,069 MiB in both.

The March run below, Qwen3.5-9B on an earlier engine, is kept as it was.

## Test setup (March 2026)

| | |
|---|---|
| **GPU** | NVIDIA RTX 5070 Ti (16 GB VRAM) |
| **Model** | Qwen3.5-9B GGUF |
| **Tokens per request** | 150 (`num_predict=150`) |
| **Benchmark script** | [`bench.sh`](../bench.sh) |
| **Method** | Fire N concurrent requests, measure wall time and total tokens |
| **Engine port** | `localhost:11434` |
| **Engine parallelism** | Continuous batching, 16 slots, shared KV cache pool |
| **Date** | March 2026 |

## Results — Engine scaling

### Total throughput

How many tokens per second the system produces across all concurrent requests.

<p align="center">
  <img src="assets/bench-throughput.svg" alt="Engine throughput scaling 1→16 concurrent" width="720" />
</p>

| Concurrent requests | Engine throughput | Speedup vs N=1 |
|:---:|:---:|:---:|
| 1 | 94 tok/s | 1.0× |
| 2 | 143 tok/s | **1.5×** |
| 4 | 183 tok/s | **1.9×** |
| 8 | 206 tok/s | **2.2×** |
| 16 | **259 tok/s** | **2.75×** |

Continuous batching extracts more throughput per GPU step as concurrency rises because the same forward pass amortises across multiple sequences.

### Time to complete all requests (wall time)

How long until every concurrent request has finished — the metric that matters for user experience.

<p align="center">
  <img src="assets/bench-latency.svg" alt="Engine wall time vs concurrency" width="720" />
</p>

| Concurrent requests | Engine wall time (16×150 tok) |
|:---:|:---:|
| 1 | 1.6 s |
| 2 | 2.1 s |
| 4 | 3.3 s |
| 8 | 5.8 s |
| 16 | **9.3 s** |

With 16 concurrent users, the slowest of the 16 responses still finishes within ~9 seconds, and every user starts receiving tokens via SSE streaming immediately — no slot queueing.

### Per-request latency

Individual request performance under load. Each user gets fewer tok/s as concurrency increases (the GPU time is shared between active sequences), but tokens arrive steadily via SSE streaming.

| Concurrent requests | Engine per-request |
|:---:|:---:|
| 1 | 97.6 tok/s |
| 2 | ~71 tok/s |
| 4 | ~46 tok/s |
| 8 | ~26 tok/s |
| 16 | ~16.5 tok/s |

At ~16 tok/s text still arrives faster than humans can read; the trade-off chosen by continuous batching is "everyone gets tokens in real-time" rather than "one user at full speed, the rest queued."

## Data-centre GPUs (October 2026)

The same scaling, measured on EuroHPC machines with Q4_K_M models and
the October engines, tokens/s across all requests:

| GPU | model | 1 request | 16 | 32 |
|---|---|---:|---:|---:|
| one LUMI-G MI250X GCD | Qwen3-4B | 149 | 921 | 1,242 |
| | Qwen3-8B | 114 | 705 | 958 |
| | Qwen3-14B | 75.3 | 470 | 667 |
| | Qwen3-32B | 38 | 267 | 378 |
| one JUPITER GH200 | Qwen3-8B | 229 | 1,434 | |
| | Qwen3-14B | 142 | 1,066 | |
| | Qwen3-32B | 70 | 589 | 875 |
| | Qwen3.6-27B (hybrid) | 75 | 513 | 700 |
| | Qwen3.6-35B-A3B (MoE) | 218 | 888 | |

Against llama-server at the same llama.cpp commit:

- **LUMI-G, Qwen3-8B, 16 requests:** 695 against 646 tok/s, and the median
  first token after 33 ms against 274 ms, since a new prompt is read inside
  the running answers' steps.
- **JUPITER:** equal on dense models one request at a time, ahead with many
  requests on the hybrid and MoE models: 700 against 392 tok/s with 32
  requests on Qwen3.6-27B, 888 against 723 with 16 on Qwen3.6-35B-A3B.
- **One JUPITER node** (four GH200), Qwen3-30B-A3B-Instruct Q8_0, 32 requests
  per GPU: 2,279 tok/s.

Sources, with the jobs and the engine builds:
[LUMI-G](lumi/allocation-plan.md) (rounds c01, 10 to 13 October) and
[JUPITER](jupiter/dossier.md#51-eullm-engine-on-gh200-deliverable-5)
(14 October).

## Reproduce

### Stress test with parallelism verification (recommended)

The [`bench/stress_test.py`](../bench/stress_test.py) script uses streaming to track individual token timestamps, proving whether requests are truly processed in parallel or just queued. See [`bench/README.md`](../bench/README.md) for full details.

```bash
pip install aiohttp

python bench/stress_test.py --url http://localhost:11434 --model Qwen3.5-9B-Q4_K_M \
    --concurrency 1,2,4,8,16 --tokens 150 --warmup --json results_eullm.json
```

### Quick throughput benchmark

The [`bench.sh`](../bench.sh) script fires N concurrent non-streaming requests and measures wall time. Simpler but does not verify parallelism.

```bash
# Build Engine with CUDA
cargo build --release --features cuda

# Run Engine
./target/release/eullm run ./Qwen3.5-9B-Q4_K_M.gguf --batch-size 16

# In another terminal, run the benchmark
./bench.sh http://localhost:11434 Qwen3.5-9B-GGUF
```

## Hardware notes

These results are on a consumer RTX 5070 Ti (16 GB). Results will vary by GPU, model size, and quantization level. The relative gain of continuous batching grows with:

- **More concurrent requests** — the speedup vs N=1 widens
- **Longer generations** — more time spent in decode = more batching opportunity
- **Faster GPUs** — shared-pass decoding extracts more parallelism per GPU step

On server GPUs (A100, H100) with higher memory bandwidth, the throughput scaling with concurrency is more dramatic still.
