# EULLM Engine

The EULLM Engine is a CLI + API server for running GGUF models locally, with real llama.cpp inference, built-in EU model catalog, local-only AI Act audit trail, and no network telemetry of any kind. Single Rust binary — no Python, no Docker.

## Installation

### From source

```bash
cd engine

# CPU only
cargo build --release

# With GPU acceleration
cargo build --release --features cuda     # NVIDIA (CUDA)
cargo build --release --features rocm     # AMD (ROCm)
cargo build --release --features vulkan   # Cross-platform (NVIDIA + AMD + Intel)
cargo build --release --features metal    # macOS Apple Silicon

# Binary will be at target/release/eullm
```

#### Build requirements

- Rust 1.75+
- C/C++ compiler (gcc/clang) — needed by llama.cpp
- CMake 3.14+
- libclang (`libclang-dev` on Debian/Ubuntu, `clang-devel` on Fedora) — needed by `bindgen` for FFI bindings
- (Optional) CUDA toolkit, ROCm, Vulkan SDK, or Xcode for GPU support

**Ubuntu/Debian one-liner:** `sudo apt install build-essential cmake libclang-dev`

### Docker

```bash
# CPU only
docker build -t eullm-engine engine/
docker run -p 11434:11434 -v eullm-models:/models eullm-engine

# With NVIDIA GPU
docker build -t eullm-engine --build-arg FEATURES=cuda engine/
docker run --gpus all -p 11434:11434 -v eullm-models:/models eullm-engine

# Or via docker compose (from repo root)
docker compose up engine              # CPU
docker compose --profile gpu up engine-gpu   # GPU
```

## CLI Commands

### `eullm run <model> [--port PORT]`

Load a model and start the API server. Supports local GGUF files and catalog models.

```bash
# Run a local GGUF file (inference works immediately)
eullm run ./qwen3-7b-q4_k_m.gguf

# Run a catalog model (auto-downloads from HuggingFace)
eullm run legal-it-4b

# With options
eullm run ./model.gguf --port 8080
eullm run ./model.gguf --gpu-layers 0      # CPU only
eullm run ./model.gguf --gpu-layers 20     # At most 20 layers on the GPU
eullm run ./model.gguf --ctx-size 8192     # Larger context window
eullm run ./model.gguf --threads 8         # Limit CPU threads
```

**Options** (`eullm run --help` is authoritative; this table is the short form):

| Option | Default | Description |
|---|---|---|
| `model` | (optional) | GGUF path, catalog id, URL, or `hf.co/<owner>/<repo>[:<quant>]`. Omitted on a terminal, opens the model picker |
| `--port, -p` | `11434` | API server port |
| `--ui-port` | `11435` | Embedded chat UI port (separate from the API) |
| `--gpu-layers` | automatic | **Upper bound** on GPU layers (-1 = all, 0 = CPU only). Sizing may offload fewer — see below |
| `--no-fit` | false | Disable automatic sizing; use `--gpu-layers` as given |
| `--fit` | (implied) | Same sizing, plus an interactive confirmation before a partial split |
| `--fit-strict` | false | Refuse to load rather than offload a partial split |
| `--cpu-moe` | false | MoE models: all expert tensors on CPU RAM (sized automatically when unset) |
| `--n-cpu-moe` | `0` | MoE models: expert tensors on CPU for the first N layers only |
| `--ctx-size, -c` | `4096` | Total context window (split across batch slots) |
| `--threads, -t` | all CPUs | Number of CPU threads |
| `--batch-size` | `1` | Concurrent requests served by the batching scheduler (raise `--ctx-size` with it) |
| `--n-batch` | `2048` | Prefill batch size (tokens per eval) |
| `--cache-type-k` | `f16` | KV cache type for keys (f16, q8_0, q4_0). Quantizing frees VRAM for more layers |
| `--cache-type-v` | `f16` | KV cache type for values (f16, q8_0, q4_0) |
| `--no-flash-attn` | false | Disable flash attention (on by default), for the generation model and the decision model alike |
| `--web` | false | Fetch URLs found in user messages and inject their content |
| `--mmproj` | (auto) | Multimodal projector path, when it is not beside the weights |
| `--ctx-checkpoints` | `0` | Prompt-prefix state snapshots for hybrid/recurrent models |
| `--checkpoint-min-step` | `8192` | Minimum new tokens between checkpoints |
| `--rs-seq` | `0` | Recurrent-state rollback window — leave off unless you know why |
| `--rust-debug` | false | Per-token NaN/Inf scan of the logits (diagnostics) |
| `--replace` | false | Replace an existing service on the port |
| `--daemon` | false | Run as a background daemon |
| `--pidfile` | `/tmp/eullm.pid` | PID file path (with `--daemon`) |
| `--logfile` | `~/.eullm/logs/eullm.log` | Daemon log file (with `--daemon`). Set `--pidfile` alone and the log stays beside it |
| `--keep-alive` | (unset) | Idle-unload a model this many seconds/minutes/hours after its last use (e.g. `5m`). Unset = never automatic; a request's own `keep_alive` field overrides it for that load. Applies to the generation, embedding and decision models independently |
| `--embedding-model` | (unset) | Load a text-embedding model (GGUF path or store name) at startup as a **reserved companion**: its VRAM is subtracted from free VRAM before `--fit` sizes the generation model, so both stay resident together instead of depending on load order. See [Text Embeddings and the Embedding Slot](#text-embeddings-and-the-embedding-slot) |
| `--decision-model` | (unset) | Load a decision model for `POST /v1/systemone` at startup, as a reserved companion like `--embedding-model`. See [Decisions: `/v1/systemone`](#decisions-v1systemone-and-the-decision-slot) |
| `--decision-ctx` | `8192` | Most tokens of context one `/v1/systemone` request may use: the state plus its longest question (plus every other question in `batched` mode). The context is sized per request; this ceiling is what the decision slot keeps free in VRAM |

#### Automatic GPU sizing

Since 0.6.80 the engine decides how much of the model goes on the GPU, on
every load — at startup and on every model swap, on `run` and on `serve`
alike. It reads free VRAM and the model's own metadata, then charges each
layer its share of the weights plus its KV-cache slice for the context and
cache type in use, and offloads as many layers as that budget allows. MoE
models get their expert tensors moved to CPU RAM first, since only a few
experts fire per token: that buys far more headroom than whole-layer
offload can.

You do not need a flag for this. It exists because the alternative default
is worse: a model larger than free VRAM used to die with an out-of-memory
error at load, while sized, the worst case is a slower partial split.

| You want | Use |
|---|---|
| The engine to decide | nothing — this is the default |
| To cap how much of the card is used | `--gpu-layers N` (an upper bound; sizing may go lower) |
| CPU only | `--gpu-layers 0` |
| To force a count past the estimate | `--no-fit --gpu-layers N` |
| To be asked before a partial split | `--fit` (interactive terminals only) |
| To refuse a load that doesn't fully fit | `--fit-strict` |

Three properties worth knowing.

`--gpu-layers` is a ceiling rather than a fixed count on purpose: a number
chosen against one model is not a fact about the next one the process loads,
and applying it blindly to a swapped-in model is exactly how an
out-of-memory error happens.

Automatic sizing never asks questions — it applies the split and logs one
line naming the flags that override it — because a default that interrupts
every launch is its own kind of failure. Where free VRAM cannot be probed
(any non-CUDA build) it stays silent and `--gpu-layers` is used as-is.

The budget leaves headroom on purpose, and it is the same headroom the
loader requires: enough of the card's total memory must stay free for the
context and its compute buffers, or the weights load and then no context can
be allocated at all. Sizing that aims past that floor produces a split the
loader refuses, which is a worse failure than a conservative one — so the
sizer takes the stricter of its own fragmentation margin and the loader's
minimum. Expect a loaded model to leave roughly 12% of the card free; that
is not waste, it is what the next token's compute buffer is allocated from.

### `eullm pull <model>`

Download a model from HuggingFace (or the EU registry when available).

```bash
eullm pull legal-it-4b
eullm pull eullm/legal-it-4b     # Full name works too
```

The model is stored in `~/.eullm/models/<model>/` with a GGUF file and manifest.

Gated and private Hugging Face repositories need an access token: set
`HF_TOKEN` in the environment (for a gated one, after accepting its terms with
the same account). It is sent only to `https://huggingface.co`, for the API
calls and every download request alike. See the README section "Gated and
private Hugging Face models".

### `eullm list`

Show locally downloaded models. If none are available, displays the EU catalog.

```bash
eullm list
```

### `eullm show <model>`

Display detailed information about a model (local or from catalog).

```bash
eullm show legal-it-4b
```

### `eullm serve [--port PORT]`

Start the API server without loading any model. The first API request with a `"model"` field will load that model dynamically.

```bash
eullm serve
eullm serve --port 8080
```

`serve` takes the same runtime flags as `run` (they are one shared set), and
they apply to every model it loads or swaps to: automatic GPU sizing runs on
each load, `--gpu-layers` caps it, `--no-fit` disables it. The one difference
is that `serve` never prompts — a daemon has nobody at the keyboard — so
`--fit` adds nothing there and `--fit-strict` surfaces a refused load as an
error to the API caller.

### `eullm import-ollama <model> [--ollama-dir PATH]`

Import a model from a local Ollama installation into EULLM's model store. Copies the GGUF blob so you can benchmark both engines with the exact same model file.

```bash
# Import from Ollama (tag "latest")
eullm import-ollama llama3.2

# Import a specific tag
eullm import-ollama qwen3:14b

# Custom Ollama directory
eullm import-ollama gemma3 --ollama-dir /custom/path
```

**How it works:**

1. Reads the Ollama manifest at `~/.ollama/models/manifests/registry.ollama.ai/library/{name}/{tag}`
2. Locates the model layer (`application/vnd.ollama.image.model`) — the GGUF blob
3. Copies the blob to `~/.eullm/models/{name}/{name}.gguf`
4. Applies GGUF metadata patches if needed (e.g. fixes array lengths for llama.cpp compatibility)
5. Writes a EULLM manifest so the model appears in `eullm list`

After import:

```bash
eullm run llama3.2        # Runs on EULLM Engine
ollama run llama3.2       # Same model on Ollama — identical comparison
```

**Licensing note:** Ollama does not add any license on top of the original model weights. The GGUF blob is the same file distributed by the upstream model author. The license of the model itself applies (e.g. Apache 2.0 for Qwen3, MIT for DeepSeek).

**GGUF compatibility:** Some Ollama GGUF files contain metadata arrays with fewer elements than upstream llama.cpp expects (e.g. `qwen35.rope.dimension_sections` with 3 elements instead of 4). The import command automatically patches these during copy. Models with hybrid architectures (e.g. Qwen3.5 with SSM/Mamba2 layers) may have incompatible tensor layouts — use the HuggingFace GGUF instead.

### `eullm forge`

Delegate to the EULLM Forge Python pipeline for model verticalizzazione.

```bash
eullm forge Qwen/Qwen3-14B --profile legal-it --identity "LegalAI"
```

## Dynamic Model Swap

EULLM Engine can swap models at runtime, like Ollama. When an API request specifies a `"model"` that differs from the currently loaded one, the server automatically unloads the current model and loads the new one.

```bash
# Start with one model
eullm run qwen3-14b

# Any API request with a different model triggers a swap
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen3-7b", "prompt": "Ciao"}'
# → Unloads qwen3-14b, loads qwen3-7b, then responds

# Or start with no model and load on first request
eullm serve
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen3-14b", "prompt": "Ciao"}'
# → Loads qwen3-14b on the fly
```

**Behavior:**

- In-flight requests on the old model complete normally (they hold cloned handles)
- The new model loads on a blocking thread, then atomically replaces the slot
- The model name must be an imported model (`eullm import-ollama`) or a local GGUF path

**What carries across a swap, and what does not.** The flags you launched
with are settings, and they apply to every model the process loads: context
size, cache types, batch size, thread count, and `--gpu-layers` as an upper
bound. What a *model* needs is decided per model, freshly, on each load:

| Per model, re-decided on every load | Why |
|---|---|
| How many layers go on the GPU | Sized against the VRAM actually free at that moment, for that model's own weights and KV cache |
| MoE expert offload | Only MoE models have experts to move |
| The multimodal projector | A projector belongs to the weights it was trained with; pairing it with another model fails the load |
| Sequential vs batched execution | Vision models need the sequential engine; text models get the scheduler back |

That split is not a detail: treating a per-model property as a process-wide
setting is how a launch model's projector ended up on its successors, and how
a layer count chosen for one model produced an out-of-memory error on the
next one.

## KV Cache Quantization

By default, EULLM uses F16 KV cache for maximum GPU compatibility. Quantized types save VRAM but may cause GPU compute fallback to CPU on some architectures — verify GPU utilisation with `nvtop` before deploying in production.

| Setting | VRAM for 14B @ 16K context |
|---------|---------------------------|
| **`--cache-type-k f16 --cache-type-v f16`** | **~10 GB (default, best GPU compat)** |
| `--cache-type-k q8_0 --cache-type-v q8_0` | ~5 GB |
| `--cache-type-k q8_0 --cache-type-v q4_0` | ~2.5 GB (⚠️ verify GPU usage) |

```bash
# Default (F16) — maximum GPU compatibility
eullm run qwen3-14b --ctx-size 8192

# Save VRAM with quantized KV cache (check GPU usage with nvtop!)
eullm run qwen3-14b --ctx-size 16384 --cache-type-k q8_0 --cache-type-v q4_0

# Aggressive quantization (minimum VRAM, may fall back to CPU)
eullm run qwen3-14b --ctx-size 32768 --cache-type-k q4_0 --cache-type-v q4_0
```

Available types: `f16`, `f32`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`.

> **Note:** Quantized V cache types (Q4_0, Q8_0) require Flash Attention. On GPUs where Flash Attention doesn't support these types, the engine automatically falls back to F16 KV cache and logs a warning. You can also set F16 explicitly by omitting the `--cache-type-v` flag.
>
> **TurboQuant note:** v0.5.x of the engine integrated an experimental TurboQuant (Walsh-Hadamard + Lloyd-Max) KV compression via the AmesianX/llama.cpp fork. It is **not in v0.5.8 onwards** — see [Research and experiments](research.md) and the archived numbers in [`turboquant-quality-report.md`](turboquant-quality-report.md) / [`turboquant-kv-stress-report.md`](turboquant-kv-stress-report.md).

## Constrained JSON Decoding (`format: "json"`)

When `format: "json"` is set in a request, EULLM uses GBNF grammar-based constrained decoding to guarantee valid JSON output. This matches Ollama's behavior and prevents malformed JSON in extraction pipelines.

```bash
curl http://localhost:11434/api/generate \
  -d '{
    "model": "qwen3-14b",
    "prompt": "Extract the name and age from: John is 30 years old",
    "format": "json"
  }'
# → Always returns valid JSON: {"name": "John", "age": 30}
```

Works on all endpoints: `/api/generate`, `/api/chat`, `/v1/chat/completions`. Both sequential and continuous batching modes.

## Continuous Batching

EULLM's continuous batching scheduler decodes multiple requests in parallel on a single GPU pass. This is a key performance differentiator over Ollama, which processes requests one at a time.

```bash
# Enable continuous batching with 8 parallel slots (default)
eullm run ./model.gguf --batch-size 8

# More slots for high-throughput RAG workloads
eullm run ./model.gguf --batch-size 16

# Sequential mode (one request at a time, like Ollama)
eullm run ./model.gguf --batch-size 0
```

With 16 concurrent requests on a consumer GPU, EULLM achieves ~2.5x throughput vs Ollama. See [benchmarks](benchmarks.md) for details.

### Context window and batch slots

The `--ctx-size` flag sets the **total** KV cache budget, shared across all batch slots (matching Ollama/llama.cpp server behaviour). Each slot gets `ctx_size / batch_size` tokens of context:

```bash
# 16K total, 4 slots → 4096 tokens/slot
eullm run ./model.gguf --ctx-size 16384 --batch-size 4

# 32K total, 8 slots → 4096 tokens/slot
eullm run ./model.gguf --ctx-size 32768 --batch-size 8
```

### Choosing batch-size

More slots increase parallelism but reduce per-request throughput (shared GPU time). General guideline:

| Parallel slots | Per-request throughput | Aggregate throughput | Use case |
|:-:|:-:|:-:|---|
| 4 | High | High | Chat, general inference |
| 8 | Medium | Higher | Batch extraction, RAG pipelines |
| 16+ | Lower | Highest | High-concurrency APIs, multi-GPU |

Start with `--batch-size 4` for the best per-request latency. Increase when your workload requires more concurrent slots and can tolerate slower individual responses.

## Dynamic Model Swap

The Engine supports hot-swapping models at runtime. When a request specifies a different `model`, the server automatically:

1. **Shuts down** the old scheduler thread (waits for it to fully exit)
2. **Frees VRAM** — the old model, KV cache, and LlamaBackend are destroyed
3. **Loads** the new model with the requested configuration
4. **Resumes** serving requests on the new model

### Basic swap (via model field)

Any generation request with a different `model` triggers the swap:

```bash
# Currently running qwen3-14b — this switches to qwen3-8b automatically
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
  "prompt": "Hello"
}'
```

### Dynamic batch_size and ctx_size

The `batch_size` and `ctx_size` can be overridden per model swap. This is useful when switching between a large model (fewer slots) and a small model (more slots):

```bash
# Switch to 8B with 8 parallel slots and 32K context
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
  "batch_size": 8,
  "ctx_size": 32768,
  "prompt": "Hello"
}'

# Switch back to 14B with 4 slots and 16K context
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:14b",
  "batch_size": 4,
  "ctx_size": 16384,
  "prompt": "Hello"
}'
```

When `batch_size` or `ctx_size` are not specified, the values from `--batch-size` and `--ctx-size` at startup are used.

### Model name resolution

The `model` field accepts:

| Format | Example | Resolution |
|--------|---------|------------|
| Full GGUF path | `/models/qwen3-8b.gguf` | Direct file |
| Ollama-style name | `qwen3:8b` | Normalized to `qwen3-8b`, searched in `/models/` and model store |
| Path without extension | `/models/qwen3-8b` | Tries appending `.gguf` |
| Directory | `/models/mymodel/` | Picks the first `.gguf` file inside |
| Registered name | `legal-it-4b` | Looked up in `~/.eullm/models/` |

### Concurrent swap safety

Multiple requests arriving simultaneously for a different model are handled safely:
- Only one swap runs at a time (serialized via Mutex)
- Other requests wait for the swap to complete, then use the new model
- In-flight requests on the old model continue normally via reference counting

### VRAM budget reference

Approximate VRAM usage with F16 KV cache (default). Actual values depend on model architecture and GPU.

| Model size | batch_size | ctx_size | tok/slot | VRAM est. |
|:----------:|:----------:|:--------:|:--------:|:---------:|
| 14B Q4 | 4 | 16384 | 4096 | ~12.5 GB |
| 8B Q4 | 4 | 16384 | 4096 | ~7 GB |
| 8B Q4 | 8 | 32768 | 4096 | ~9.5 GB |
| 8B Q4 | 8 | 16384 | 2048 | ~7 GB |
| 70B Q4 | 16 | 65536 | 4096 | ~45 GB |

## Text Embeddings and the Embedding Slot

`POST /api/embed` (Ollama) and `POST /v1/embeddings` (OpenAI) embed one or
more texts with any GGUF text-embedding model — BGE, E5, and similar.
Naming a model in the request loads it the first time it is requested,
exactly like the generation model:

```bash
curl -s http://localhost:11434/api/embed -d '{
  "model": "bge-m3",
  "input": ["first chunk", "second chunk"]
}'
```

The embedding model lives in a **second, independent slot** — it does not
replace whatever generation model is loaded. Both are kept resident when
they fit; this is a decision the server makes for itself on every embedding
request, not something the caller has to reason about:

- If the embedding model fits in currently free VRAM alongside the loaded
  generation model, it is loaded next to it. Both stay resident.
- If it does not fit, the generation model is evicted first (it reloads
  automatically on the next generation request), and the embedder gets the
  whole card.
- The reverse also holds: a generation request with `--fit` enabled evicts
  a resident embedder first if the sizing needs the VRAM back.

On a build with no VRAM probe (non-CUDA), the server always tries to keep
both resident and lets a genuine out-of-memory surface as a normal load
error — the same posture `--fit` itself takes there.

How many times either direction of eviction has happened is in
`/api/version`'s `model_swaps` field — a rate of roughly one per request
means a card too small for both is being asked to alternate rather than
batch (do all ingestion, then all generation, rather than interleaving).

The eviction dance above assumes the embedding model was loaded on demand,
by naming it in a request. `--embedding-model <path-or-name>` skips that:
the embedder loads at startup and becomes a **reserved companion** — it
loads first, so its weights already count as used VRAM by the time `--fit`
reads free VRAM to size the generation model, on both `eullm run` and
`eullm serve`; `--fit` additionally keeps a small compute-buffer margin free
on top, for the `LlamaContext` an embedding call opens and closes per
request, both at launch and again on every later generation-model swap. A reserved
companion is never evicted to make room for a generation load; it keeps its
place for the life of the process. If reserving its space would leave the
generation model no room at all, the launch proceeds anyway with a warning:
the reservation is dropped and the embedder falls back to the normal
evict-on-demand behavior described above, exactly as if the flag had not
been given.

```bash
eullm run legal-it-4b --fit --embedding-model bge-m3
```

Use this when a companion (e.g. bge-m3 for a RAG pipeline) should always be
resident alongside the generation model whenever the card has room for both,
rather than depending on which model happens to be requested first.

Pooling is read from the model's own GGUF metadata (CLS for BGE, mean for
E5, and so on) rather than guessed; a model that declares no pooling type
falls back to mean-pooling its per-token embeddings. Output vectors are
L2-normalized. Rerankers (RANK-pooling models) are out of scope for this
endpoint — normalizing a single relevance score would collapse it.

## Decisions: `/v1/systemone` and the Decision Slot

`POST /v1/systemone` answers typed questions about a *state* — a ticket, a
document, a conversation, a JSON object — without generating any text. It
takes the request and response shape of the System One API (TypeSafe's
Jev), so a client written for it can point its base URL at EuLLM. Three
question types:

| Type | Question | Answer |
|---|---|---|
| `noul` | Is this statement true of the state? (`criteria`, optional: `{"true": "…", "false": "…"}`, what each answer means) | `noul`: P(yes) |
| `choice` | Which of these options? (`criteria`: an object of name → description, 2–26 options; up to 255 with a [Jev-Style model](#jev-style-decision-models)) | `choice`, `probabilities`, `confidence` |
| `score` | Which level of this scale? (`criteria`: an array of 2–10 level descriptions, lowest first; a level may also be `{"label": "…", "description": "…"}`) | `score` (Σ level × p), `legend`, `probabilities`, `confidence` |

```bash
curl -s http://localhost:11434/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?" },
    "team": { "type": "choice", "instructions": "Which team should handle it?",
              "criteria": { "billing": "Payments and payouts", "tech": "Bugs", "other": "Anything else" } },
    "severity": { "type": "score", "instructions": "How severe is it?",
                  "criteria": ["Cosmetic", "Degraded, with a workaround", "Blocking"] }
  }
}'
```

```json
{
  "model": "qwen3-4b",
  "answers": {
    "is_urgent": { "type": "noul", "noul": 0.95, "eullm": { ... } },
    "team": { "type": "choice", "choice": "billing",
              "probabilities": { "billing": 0.91, "tech": 0.07, "other": 0.02 },
              "confidence": 0.865, "eullm": { ... } },
    "severity": { "type": "score", "score": 1.43,
                  "legend": { "0": "Cosmetic", "1": "Degraded, with a workaround", "2": "Blocking" },
                  "probabilities": { "0": 0.0, "1": 0.57, "2": 0.43 },
                  "confidence": 0.355, "eullm": { ... } }
  },
  "usage": { "input_tokens": 312, "output_tokens": 0 },
  "timing": { "total_ms": 187.4 },
  "eullm": { "mode": "shared_prefix", "prompt_tokens": 520, "shared_prefix_tokens": 104,
             "evaluated_tokens": 312, "timings_ms": { ... }, "request_ms": 187.43, ... }
}
```

`timing.total_ms` is the request's wall time, model resolution included, as
jev-style's server reports it and its MCP tools and guard show it:
`eullm.request_ms` to 0.1 ms. `eullm.timings_ms` splits the decode into its
phases.

Answers and options come back in the order the request listed them; options
are shown to the model lettered in that order.

`instructions` may be a string, or, as the System One API allows, an object
or an array — the question in one field and the data it refers to in
others. The model then reads it as one line of compact JSON, in the order
it was written, which is what jev-style's server gives its models:

```json
{"record":{"name":"John Smith","city":"Oakland"},"question":"Is this resume the same person as the record?"}
```

A score level written as `{"label": "high", "description": "loses data"}` is
shown to the model as `high: loses data` — the label alone when there is no
description — and named `high` in the `legend`, as jev-style's own server
does. Any other object or array is shown as one line of JSON and named in the
legend by its label, when it has one, or by its compact JSON.

**How an answer is computed.** With an instruction-tuned model — the *code
readout* — each question becomes one chat prompt (the model's own template,
reasoning switched off) ending where the answer would begin, with the
options coded `A`…`Z`, the levels `0`…`9`, or `Yes`/`No`. The logits at
that position are read once — nothing is generated — and restricted to the
codes. When the model loads, the server checks that every code is a single
token for its tokenizer right after its prompt, and refuses a question whose
codes are not; the load log lists what it found. Everything a request
sends — the state, the instructions, the options — is tokenized as text: a
chat template's own turn markers inside it (`<|im_end|>`,
`<|im_start|>assistant`) stay text instead of closing the user turn and
writing the rest of the prompt. Only the template's text is read for control
tokens; for the rare template that cannot be tokenized in those pieces
exactly as it is whole, a request containing such a marker is refused with
a 422. A model trained for this endpoint reads its answers differently — see
[Jev-Style decision models](#jev-style-decision-models) — and the response
says which readout was used in `eullm.readout` (`codes` or `verdict`).

Next to the System One fields, every answer carries an `eullm` object with
what it was derived from, so a stored response can be re-examined or
re-calibrated later:

| Field | Meaning |
|---|---|
| `logprobs` | Code readout: full-vocabulary log-probability of each answer's code |
| `scores` | Verdict readout: each option's score, `logit(" yes") − logit(" no")` at its slot |
| `raw_probabilities` | The same, renormalized over the answers (a softmax of the scores), before calibration and temperature |
| `coverage` | Code readout: share of the model's probability on a valid code. Near 1: it answered in the format asked for. Low: most of its probability went elsewhere (a thinking tag, a sentence) and the answer describes a minority of what it would have said — check this before trusting an answer |
| `prior_logprobs` | The content-free prior that was divided out (with `content_free` calibration only) |
| `confidence_entropy` | `choice` and `score`: `1 − H(p) / ln K` of the same probabilities, `confidence` as it was defined up to 0.7.20 |

`confidence` (for `choice` and `score`) is `(K · p_max − 1) / (K − 1)` over
the `K` answers, clipped to [0, 1]: how far the top answer's probability is
above an even split, as a share of the most it could be — 1 when all
probability is on one answer, 0 when it is spread evenly; `2 · p_max − 1`
for two options. It is jev-style's definition, so a threshold on it means
the same against jev-style's server and against EuLLM. The response names
it in `eullm.confidence_method` (`normalized_max_probability`), and each
answer's `eullm.confidence_entropy` keeps `1 − H(p) / ln K`, the
entropy-based value `confidence` was up to 0.7.20, which reads the
runner-up answers too.

**Calibration is not a solved problem here, and nothing is claimed about
it yet.** The mechanism reproduces with any model; calibrated probabilities
do not come with it. The request's `eullm` object picks what is applied, so
the options can be compared on labelled data before one is trusted:

| `eullm` option | Values | Default |
|---|---|---|
| `calibration` | `none`; `content_free` (code readout): divide out the answer the model gives the same question about the state `N/A` (Zhao et al., 2021), cached per question | `none` |
| `temperature` | Temperature scaling after calibration: `> 1` flattens, `< 1` sharpens | `1`; a Jev-Style model's own calibrated temperature |
| `mode` | `shared_prefix`; `batched`; `separate` (see below) | `shared_prefix` |

The content-free prior is not always noise to remove: when the options
themselves imply a base rate, dividing it out moves probability towards
options that are rarely right. Measure before choosing.

**Many questions, one pass.** Every question's prompt starts with the same
tokens — system prompt, template, the state — and differs only at the end.
`shared_prefix` decodes that common part once, then each question on its
own right after it, in a sequence that starts as a copy of the state's (no
copy of data: on a unified cache the cells are only tagged with the extra
sequence) and is dropped once the question's answer is read: about
`S + Q·q` tokens for `Q` questions of `q` tokens on a state of `S`, instead
of the `Q·(S + q)` of asking one at a time. `batched` decodes the same
tokens, but every question's in one batch: fewer, larger decode calls.
`separate` asks one question at a time from an empty cache; it exists as
the baseline. The response reports the token counts (`prompt_tokens`
against `evaluated_tokens`), the timings of each phase and `flash_attn` —
`auto`, or `off` under `--no-flash-attn` — since both the timings and the
last digits of every probability depend on it. `bench/decision_bench.py`
measures all three for 1–64 questions on states of several sizes.

**Each answer depends on its own question only.** The three modes read the
same tokens but hand them to the kernels in batches of different shapes,
and on quantized weights that alone moves an answer — a quantized model's
matrix products round their inputs to 8 bits, which turns a last-digit
difference in a sum into a different rounding one layer later. How far,
measured with `bench/decision_bench.py` (the largest difference in any
probability from `separate`; on the CPU over states of 256 and 1,024 tokens
with 1–64 questions, on the GPU with the benchmark's defaults):

| Model | 4-core CPU: `shared_prefix` | `batched` | RTX 5070 Ti: `shared_prefix` | `batched` |
|---|---|---|---|---|
| Qwen3-0.6B F16 | 0.017 | 0.017 | | |
| Qwen3-0.6B Q8_0 | 0.11 | 0.13 | | |
| Qwen3-0.6B Q4_K_M | 0.34 | 0.32 | 0.53 | 0.52 |
| Jev-Style-0.8B-Decision-v3 Q4_K_M | 0.024 | 0.028 | 0.039 | 0.033 |
| Jev-Style-2B-Decision-v3 Q4_K_M | | | 0 | 0 |

With Qwen3-0.6B Q4_K_M, the catalog's `qwen3-0.6b`, that is enough to
change the top answer of a question near a tie, and more so on CUDA, where
the arithmetic is coarser: ggml-cuda runs cuBLAS in TF32 mode and
accumulates some F16 products in half precision. None of the modes is the
exact one; they are the same model with different rounding. The Jev-Style
models, trained to answer this way, are an order of magnitude steadier than
an instruction-tuned model at the same quantization, and the 2B's modes read
the same blocks, so they agree exactly (see
[below](#jev-style-decision-models)).

What differs is what the rounding depends on. In `batched` mode a question
sits somewhere in a batch with the others, so its answer moves with the
other questions asked and with their order: asking the same questions in
reverse moved answers by up to 0.31 on the CPU (Q4_K_M). In `shared_prefix`
mode a question is decoded alone, in the same cache cells, in batches of the
same shape and over the same attention window whatever else the request
asks, so its answer is a function of the state and that question: asked
alone, among 63 others or in reverse order, it comes back bit for bit the
same (measured on the CPU and on an RTX 5070 Ti;
`bench/decision_bench.py --order-check` checks it on your
hardware and fails if it does not hold). That is the default because a
decision should not change with the questions asked next to it, and
because a calibration measured on labelled data then holds however the
questions are grouped into requests when serving. The price is one decode
call per question instead of one per batch. On a 4-core CPU that is 2–14%
slower than `batched`. On a GPU each call has a fixed cost of about 3 ms
(Qwen3-0.6B on an RTX 5070 Ti) — the GPU running a few hundred small
kernels, not launching them: padding every question to a common length so
the calls replay as CUDA graphs was tried and measured 25% slower — so with
many questions `shared_prefix` is 2–3× slower than `batched` there, and
still 2–5× faster than `separate`. Measured with 64 questions: 316 / 347 /
473 ms for states of 256 / 1k / 4k tokens, against 111 / 130 / 235 ms
`batched` and 630 / 1645 ms / — `separate`. Calibrate in the mode that
will serve.

**Limits:** 64 questions per request, 26 options per `choice` (255 with a
Jev-Style model), 2–10 levels per `score`, and `--decision-ctx` tokens of
context per request (default 8192). A request over the context limit is
refused with a 422 `input_budget_exceeded` that says how many tokens it
needed; nothing is truncated.

**Errors** come back in the shape System One clients and
[jev-style](https://github.com/lawrence3699/jev-style)'s parse — a
status and `{"error": {"code": "…", "message": "…", "question": "…"}}`, with
`question` present when one question is at fault — so its MCP server, CLI
and guard show EuLLM's message instead of failing on the body:

| Status | `code` | When |
|---|---|---|
| 422 | `invalid_json` | The body is not JSON |
| 422 | `invalid_request` | The request as a whole fails validation: no `state`, no questions, more than 64, an unknown `eullm` option |
| 422 | `invalid_question` | One question fails validation — an unknown `type`, one option, 11 levels; `question` names it |
| 422 | `input_budget_exceeded` | Longer than `--decision-ctx`, or than a Jev-Style model's budgets; `question` names the question when it was one question's. Nothing was truncated |
| 400 | `model_not_loaded` | No `model`, or a System One name such as `jev-latest`, and no decision model loaded |
| 404 | `not_found` | `model` names a model the server does not have |
| 401 / 403 / 429 | `unauthorized` / `forbidden` / `too_many_requests` | Refused by the API key, IP allowlist or origin checks, or over the key's quota |
| 405 / 413 / 415 | `method_not_allowed` / `payload_too_large` / `unsupported_media_type` | Not a `POST`, a body over the limit, not `Content-Type: application/json` |
| 500 | `internal_error` | The model failed to load, or llama.cpp failed |

The other endpoints keep the error bodies Ollama and OpenAI clients read.

**Which model.** Preferably one trained for this endpoint, a
[Jev-Style model](#jev-style-decision-models): calibrated by its authors, up
to 255 options, and an order of magnitude steadier than an instruction-tuned
model at the same quantization (see the table above). A small
instruction-tuned model also works: Qwen3 0.6B–4B from the catalog are the
intended size. It must be able to answer without reasoning first; a model
that always opens a reasoning block (the DeepSeek-R1 family) spends its
first token on the tag, and every answer's `coverage` shows it.

**The decision slot.** The model lives in a third slot, next to the
generation and embedding models, with the same residency rules as the
embedding slot: loaded next to the generation model when it fits, the
generation model evicted first when it does not. What counts is the weights
plus a request's context — its KV cache at `--decision-ctx` and a compute
buffer. The model keeps that context between requests, sized by the largest
request so far, and releases it before a generation model is sized, so the
sizing sees it as reserved, not as used. `--decision-model <path-or-name>`
loads it at startup as a reserved companion, exactly like
`--embedding-model`: `--fit` keeps its context's VRAM free and a chat-model
swap never evicts it.

**The same state again costs only the questions.** The context keeps the
state the last request decoded. A request whose shared part is token for
token the same — the next round of questions an agent asks about the same
document — starts from it instead of decoding the state again, and reports
`prefix_reused: true`, with `evaluated_tokens` counting only what was
decoded. It changes no answer: the questions land in the same cells, in
batches of the same shape, as they would from a freshly decoded state (and
`--order-check` in the benchmark checks exactly that). A request about
another state replaces the one kept.

```bash
eullm serve --decision-model qwen3-1.7b --decision-ctx 16384
```

The `model` field may name any model the server can load (it then loads into
the decision slot). Left out, or set to a System One model name such as
`jev-latest`, it means the decision model already loaded — so a Jev client
works unchanged. With no decision model loaded the request is refused with a
400 `model_not_loaded` saying so.

Every decision is written to the audit trail with `request_type:
"systemone"` and a `decision` record: each answer with its log-probabilities
and its probabilities before and after calibration. The state itself is not
stored, only its SHA-256.

**A client that disconnects.** The record is written by the thread that
computed the decision, not by the connection that asked for it, so every
decision the model computes is recorded — one whose client disconnected
before its answers were ready with `client_disconnected: true`, since they
were never sent. A request whose client disconnects before that stops at its
next question, or before it starts if it was still waiting for the model, so
the next request does not wait behind work nobody will read. It decided
nothing and, like a request refused as invalid or one llama.cpp failed on, is
not recorded; the server log says it was abandoned. A single question is not
interrupted once it is being decoded.

### Jev-Style decision models

[Jev-Style](https://github.com/lawrence3699/jev-style) (Apache-2.0)
publishes Qwen3.5 fine-tunes trained for the three question types of this
endpoint, and they read an answer differently — the *verdict readout*. Every
option is followed by a ` ->` slot, and the option's score is
`logit(" yes") − logit(" no")` at its slot. The options are all read in the
same pass, not as competing codes, so a `choice` can have up to 255 of them.
The probabilities are `softmax(scores / T)`, with the temperature `T` fitted
on held-out data and released with the model. It is applied by default; a
request's `temperature` overrides it, and `1` gives the probabilities of the
raw scores.

| Model | Input | Get it |
|---|---|---|
| Jev-Style-0.8B-Decision-v3 (0.53 GB in Q4_K_M) | Causal attention; question, options and slots within 2,048 tokens | `eullm pull hf.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF:Q4_K_M` |
| Jev-Style-2B-Decision-v3 (1.3 GB in Q4_K_M) | Block-causal attention over 2,048-token blocks; options that do not fit one block become a numbered catalogue | `eullm pull hf.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF:Q4_K_M` |

```bash
eullm pull hf.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF:Q4_K_M
eullm serve --decision-model jev-style-0.8b-decision-v3-gguf-q4_k_m --decision-ctx 25600
```

The server recognizes such a model when it loads, in one of two ways:

- from the `readout_config.json` released with the model, when that file
  sits next to the GGUF;
- otherwise from the model's name (`general.name`) — the case of a GGUF
  pulled on its own.

The load log says which and prints the temperature.

**The input is the one the model was trained on.** It is not a chat
prompt. The input is rebuilt token for token: each segment tokenized on
its own, with text never parsed for special tokens, and a structured
state serialized the way the release's runtime writes JSON. It is
computed the way that runtime computes it: the same micro-batches, flash
attention off on the CPU for the 2B, and one block per decode call for
the block-causal attention.

This was checked against that runtime, Q4_K_M on the CPU, on 24 cases for
the 0.8B and 28 for the 2B:

- short, JSON, long, and special-token-laden states;
- every question type, up to 30 options — and 70 in the 2B's numbered
  catalogue form.

The token ids were identical and, in `separate` mode, the scores agreed to
4e-16, asked one at a time and all together.

**Differences from the code readout:**

- `content_free` calibration is refused, since the model brings its own.
- `eullm.scores` replaces `eullm.logprobs`, and there is no `coverage`:
  every option is read at a slot of its own.
- Input over 25,600 tokens, or options that do not fit the model's
  budget, is refused rather than truncated. Raise `--decision-ctx` to
  25600 to allow the longest input the models accept.
- Only the release's global temperature is applied, as its own runtime
  does when it is not given a category.

`shared_prefix` decodes the state once, as with any model, and each
question on its own after it:

- **2B:** its input is cut into 2,048-token blocks at the end of the state
  anyway, so sharing the state changes no score: `shared_prefix` gives the
  runtime's scores exactly, on the CPU and on an RTX 5070 Ti. It decodes one
  block per call, so its `batched` requests are answered, and reported, as
  `shared_prefix`.
- **0.8B:** its arithmetic depends on where a 1,024-token micro-batch
  starts, because its recurrent layers are computed in chunks from there,
  and sharing the state moves where its questions' micro-batches start. Its
  `shared_prefix` scores are therefore within the model's own noise of the
  runtime's: up to 0.025 in probability on the CPU, 0.039 on an RTX 5070 Ti.
  `separate` gives them exactly, at the cost of decoding the state again for
  every question: 64 questions about a 1,024-token state take 0.62 s on an
  RTX 5070 Ti in `shared_prefix` mode, 3.2 s in `separate`.

## API Reference

The Engine exposes two sets of endpoints: the native EULLM API (Ollama-compatible) and an OpenAI-compatible API. CORS is enabled for browser-based tools.

### EULLM API (Ollama-compatible)

#### `GET /api/version`

Returns the Engine version.

```bash
curl http://localhost:11434/api/version
```

```json
{
  "version": "0.1.0"
}
```

#### `GET /api/tags`

List available models. Returns the currently loaded model first (what admin dashboards check for health), followed by catalog entries.

```bash
curl http://localhost:11434/api/tags
```

```json
{
  "models": [
    {
      "name": "eullm/legal-it-4b",
      "size": 4500000000,
      "digest": "sha256:le7a1it0...",
      "details": {
        "format": "gguf",
        "family": "qwen3",
        "parameter_size": "12B",
        "quantization_level": "Q4_K_M",
        "domain": "legal",
        "source_model": "Qwen/Qwen3-14B"
      }
    }
  ]
}
```

#### `POST /api/generate`

Generate text from a prompt. Uses real llama.cpp inference.

```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "eullm/legal-it-4b", "prompt": "Cosa dice l'\''art. 2043 del Codice Civile?"}'
```

```json
{
  "model": "eullm/legal-it-4b",
  "created_at": "2026-03-21T10:00:00Z",
  "response": "L'articolo 2043 del Codice Civile...",
  "done": true,
  "done_reason": "stop",
  "total_duration": 1500000000,
  "load_duration": 0,
  "prompt_eval_count": 15,
  "prompt_eval_duration": 0,
  "eval_count": 128,
  "eval_duration": 1200000000
}
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model` | loaded model | Model name |
| `prompt` | (required) | Input prompt |
| `max_tokens` / `num_predict` | 512 | Maximum tokens to generate (see note below) |
| `temperature` | 0.7 | Sampling temperature |
| `stream` | true | Stream response token-by-token (NDJSON) |
| `num_ctx` | server per-slot ctx | Per-request context window budget (clamped to per-slot max) |
| `format` | — | Set to `"json"` for constrained JSON decoding (GBNF grammar) |
| `options` | — | Ollama-style nested object for `num_predict`, `temperature`, `num_ctx` |

**Ollama `options` support:** Parameters can be passed at the top level (OpenAI style) or nested inside an `options` object (Ollama style). Top-level values take precedence.

```json
{
  "prompt": "Ciao!",
  "options": {
    "num_predict": 1024,
    "temperature": 0.5,
    "num_ctx": 8192
  }
}
```

**`num_predict` capping:** If `num_predict` (or `max_tokens`) would exceed the remaining context budget (`effective_ctx - prompt_tokens`), it is automatically capped. The Engine logs a `WARN` when this happens — see the [Logging & Troubleshooting](#logging--troubleshooting) section.

**Streaming:** When `"stream": true` (the default), the response is sent as **NDJSON** (newline-delimited JSON). Each line is a complete JSON object with `"response"` (the token) and `"done": false`. The final line has `"done": true` with timing stats. Content-Type is `application/x-ndjson`.

```bash
# Streaming example (NDJSON — same format as Ollama)
curl -N http://localhost:11434/api/generate \
  -d '{"model": "local", "prompt": "Hello", "stream": true}'
# Each line: {"model":"...","response":"token","done":false}
# Final line: {"model":"...","response":"","done":true,"done_reason":"stop",...}
```

#### `POST /api/chat`

Chat completion with message history. Messages are formatted as ChatML internally. Supports `"stream": true` for token-by-token NDJSON streaming (same format as Ollama).

```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-4b",
    "messages": [
      {"role": "user", "content": "Spiegami il GDPR in breve."}
    ]
  }'

# Streaming
curl -N http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-4b",
    "messages": [{"role": "user", "content": "Ciao!"}],
    "stream": true
  }'
```

#### `POST /api/show`

Get model metadata.

```bash
curl -X POST http://localhost:11434/api/show \
  -H "Content-Type: application/json" \
  -d '{"name": "eullm/legal-it-4b"}'
```

#### `POST /api/pull`

Trigger a model download.

```bash
curl -X POST http://localhost:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "eullm/legal-it-4b"}'
```

#### `POST /api/embed`

Embed one or more texts. `input` accepts a single string or an array. See
[Text Embeddings and the Embedding Slot](#text-embeddings-and-the-embedding-slot)
for how this coexists with the generation model.

```bash
curl -X POST http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": ["first chunk", "second chunk"]}'
```

```json
{
  "model": "bge-m3",
  "embeddings": [[0.013, -0.021, ...], [0.008, 0.044, ...]]
}
```

### OpenAI-Compatible API

These endpoints allow using EULLM as a drop-in backend for any tool that supports the OpenAI API: Open WebUI, LangChain, LlamaIndex, n8n, Flowise, etc.

#### `GET /v1/models`

List models in OpenAI format.

```bash
curl http://localhost:11434/v1/models
```

When a decision model is loaded, it is listed for System One clients too —
jev-style's `model_info` tool and the System One SDKs' `models.list()`.
Its entry in `data` carries `context_tokens`, the most tokens one
`/v1/systemone` request may hold (`--decision-ctx`, or less when the model
itself reads fewer), for a Jev-Style model `head_max_tokens`, what one
question with its options may take on its own, and `"eullm": {"slot":
"decision"}`; and the top-level `models`, the list the System One SDKs read,
names it. With no decision model loaded, `models` is empty.

```json
{
  "object": "list",
  "data": [
    { "id": "qwen3-4b", "object": "model", "created": 1700000000, "owned_by": "eullm" },
    { "id": "Jev-Style-0.8B-Decision-v3-Q4_K_M", "object": "model", "created": 1700000000,
      "owned_by": "eullm", "context_tokens": 8192, "head_max_tokens": 2048,
      "eullm": { "slot": "decision", "readout": "verdict" } },
    ...
  ],
  "models": [
    { "name": "Jev-Style-0.8B-Decision-v3-Q4_K_M", "release_date": "2026-09-24",
      "description": "Jev-Style-0.8B-Decision-v3 on EuLLM: typed decisions (noul / choice / score)" }
  ]
}
```

#### `POST /v1/chat/completions`

Chat completion in OpenAI format. Real inference with token counts. Supports `"stream": true` for SSE streaming (OpenAI `chat.completion.chunk` format with `[DONE]` terminator).

```bash
# Non-streaming
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-4b",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'

# Streaming (same format as OpenAI API)
curl -N http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-4b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "eullm/legal-it-4b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 25,
    "total_tokens": 35
  }
}
```

#### `POST /v1/embeddings`

Embed one or more texts in OpenAI format. Same underlying embedding slot as
`/api/embed`.

```bash
curl -X POST http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": "first chunk"}'
```

```json
{
  "object": "list",
  "data": [{"object": "embedding", "embedding": [0.013, -0.021, ...], "index": 0}],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 0, "total_tokens": 0}
}
```

`usage` is honestly reported as zero rather than a fabricated token count —
the embedding path does not run a text tokenizer count today.

#### `POST /v1/systemone`

Typed decisions (`noul`, `choice`, `score`) about a state, in the System One
API shape. Not an OpenAI endpoint; it sits under `/v1` because that is where
System One clients look for it. See
[Decisions: `/v1/systemone`](#decisions-v1systemone-and-the-decision-slot).

## Model Catalog

The Engine ships with a built-in catalog of EU models:

| Model | Domain | Base | VRAM | Size | Languages |
|---|---|---|---|---|---|
| `eullm/legal-it-4b` | Legal | Qwen3 | 6 GB | 4.5 GB | IT, EN |
| `eullm/medical-de-7b` | Medical | Qwen3 | 6 GB | 4.5 GB | DE, EN |
| `eullm/finance-fr-7b` | Finance | Qwen3 | 6 GB | 4.5 GB | FR, EN |
| `eullm/general-eu-7b` | General | Qwen3 | 6 GB | 4.5 GB | EN, IT, DE, FR, ES, PT, NL |
| `eullm/general-eu-14b` | General | Qwen3 | 10 GB | 8.5 GB | EN, IT, DE, FR, ES, PT, NL |
| `eullm/code-eu-14b` | Code | DeepSeek | 10 GB | 8.5 GB | EN, IT, DE, FR, ES |
| `eullm/legal-it-14b` | Legal | Qwen3 | 10 GB | 8.2 GB | IT, EN |

All models are Apache 2.0 or MIT licensed.

## Audit Trail

Every inference request is logged to a persistent JSONL file at `~/.eullm/audit/audit.jsonl`. Each line is a self-contained JSON object.

| Field | Type | Description |
|---|---|---|
| `id` | UUID v4 | Unique inference ID |
| `timestamp` | DateTime (UTC) | Request time |
| `model` | String | Model name |
| `request_type` | String | `generate`, `chat`, `chat.completions`, `systemone` |
| `input_tokens` | u32 | Input token count |
| `output_tokens` | u32 | Output token count |
| `duration_ms` | u64 | Inference duration |
| `user_id` | Option\<String\> | Optional user identifier |
| `decision` | Object, `systemone` only | `state_sha256`, `readout`, `mode`, `calibration`, `temperature`, `confidence_method` (`normalized_max_probability`; absent, and `normalized_entropy`, on lines written up to 0.7.20), `client_disconnected` (only when true: the answers were computed after the client had gone, and never sent), and per answer: `id`, `type`, `labels`, `logprobs` or `scores`, `raw_probabilities`, `probabilities`, `coverage`, `answer`, `confidence` |

**Example audit entry:**

```json
{"id":"a1b2c3d4-...","timestamp":"2026-03-21T14:30:00Z","model":"eullm/legal-it-4b","request_type":"chat","input_tokens":15,"output_tokens":128,"duration_ms":1200,"user_id":null}
```

The JSONL format allows:
- Append-only writes (crash-safe)
- Easy to grep, tail, stream
- Each line independently parseable
- Compatible with log analysis tools (Loki, ELK, etc.)

This provides the traceability required by the EU AI Act (Regulation 2024/1689).

## GPU Acceleration

| Feature flag | GPU backend | Build command |
|---|---|---|
| `cuda` | NVIDIA CUDA | `cargo build --release --features cuda` |
| `rocm` | AMD ROCm | `cargo build --release --features rocm` |
| `vulkan` | Cross-platform | `cargo build --release --features vulkan` |
| `metal` | Apple Silicon | `cargo build --release --features metal` |
| *(none)* | CPU only | `cargo build --release` |

How much of the model goes on the GPU is decided automatically on every
load — see [Automatic GPU sizing](#automatic-gpu-sizing) above. `--gpu-layers
0` forces CPU-only inference; `--gpu-layers N` caps the offload at N layers.
Sizing needs a CUDA build to read free VRAM; on the other backends
`--gpu-layers` is used as given.

## Integration Examples

### With Open WebUI

```bash
# Start Engine
eullm run ./model.gguf

# In Open WebUI settings, set API URL to:
# http://localhost:11434
```

### With LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    model="eullm/legal-it-4b",
    api_key="not-needed"
)

response = llm.invoke("Spiegami l'art. 2043 del Codice Civile.")
```

### With curl

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local", "messages": [{"role": "user", "content": "Ciao!"}]}'
```

## Logging & Troubleshooting

The Engine uses [`tracing`](https://docs.rs/tracing) for structured logging. Control verbosity with the `RUST_LOG` environment variable.

### Log levels

```bash
# Minimal (errors only)
RUST_LOG=error eullm run ./model.gguf

# Normal operation (recommended) — shows request params, context budget, cap warnings
RUST_LOG=eullm_engine=info eullm run ./model.gguf

# Verbose — adds prefill chunk details, decode loop, batch scheduling
RUST_LOG=eullm_engine=debug eullm run ./model.gguf

# Everything (very noisy, includes llama.cpp internals)
RUST_LOG=trace eullm run ./model.gguf
```

### What gets logged

| Level | Message | When |
|---|---|---|
| `INFO` | `Request params: max_tokens=N, temperature=T, num_ctx=X` | Every API request (routes layer) |
| `INFO` | `Seq N: prompt=P tokens, max_output=M, effective_ctx=C` | After tokenization (scheduler) |
| `INFO` | `Generate: prompt=P tokens, max_output=M, effective_ctx=C` | Sequential generate path |
| `INFO` | `Stream: prompt=P tokens, max_output=M, effective_ctx=C` | Sequential streaming path |
| `WARN` | `num_predict capped: requested=R, effective=E (context=C, prompt_tokens=P)` | When `num_predict` exceeds remaining context budget |
| `DEBUG` | `Prefilling seq N with P tokens in chunks of B` | Prefill chunking details (scheduler) |
| `DEBUG` | `Sequence N prefilled (P prompt tokens)` | After successful prefill |
| `ERROR` | `Prompt (P tokens) does not fit in context window (C)` | Prompt too long for context |

### Common issues

#### Truncated output / fewer tokens than expected

The model generates fewer tokens than `num_predict` requested.

**Diagnose:** Run with `RUST_LOG=eullm_engine=info` and look for the `WARN num_predict capped` message.

**Cause:** The prompt consumed most of the context window, leaving less room than `num_predict`.

**Fix:** Either:
- Increase `--ctx-size` on the server (requires more RAM/VRAM)
- Send `num_ctx` in the request to override per-request: `"num_ctx": 8192`
- Reduce prompt length
- Lower `num_predict` to match your actual needs

#### Prompt does not fit in context window

**Error:** `Prompt (N tokens) does not fit in context window (C)`

**Cause:** The tokenized prompt is longer than the effective context size.

**Fix:** Increase `--ctx-size` or send a shorter prompt. You can also pass `"num_ctx": 16384` per-request (clamped to server max).

#### Long generation latency on first request

**Cause:** The first prefill is slow because the KV cache is being allocated. Subsequent requests reuse allocated memory.

**Fix:** This is normal. For benchmarking, discard the first request as warmup.

## Implementation Status

| Component | Status |
|---|---|
| CLI (pull, run, list, show, serve, forge, import-ollama) | Implemented |
| Real inference (llama.cpp via llama-cpp-2 0.1.141) | Implemented |
| EULLM API routes (Ollama-compatible) | Implemented |
| OpenAI-compatible API | Implemented |
| GPU acceleration (CUDA, ROCm, Vulkan, Metal) | Implemented (feature flags) |
| CORS (Open WebUI compatibility) | Implemented |
| Model catalog (7 models) | Implemented |
| Local model store (~/.eullm/models/) | Implemented |
| Model download (HuggingFace, streaming with progress) | Implemented |
| Import from Ollama (import-ollama with GGUF patching) | Implemented |
| Dynamic model swap (load/unload via API, dynamic batch_size/ctx_size) | Implemented |
| KV cache (F16 default, automatic fallback from quantized types) | Implemented |
| Constrained JSON decoding (format: "json" via GBNF) | Implemented |
| Continuous batching scheduler | Implemented |
| Audit trail (persistent JSONL) | Implemented |
| Streaming (NDJSON for Ollama, SSE for OpenAI) | Implemented |
| ChatML prompt formatting | Implemented |
| Interactive chat REPL | Implemented |
| Daemon mode (--daemon) | Implemented |
| EU registry download | Implemented (client ready, registry server coming soon) |
