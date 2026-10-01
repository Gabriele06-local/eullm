# MVP 3 — several resident chat models, then `model: "auto"`: implementation plan

**Status:** plan, not started · 1 October 2026. Written against `main` at 6f886c7: line numbers refer to that commit and will drift as the code moves. Ollama's behaviour was checked against docs.ollama.com/faq and docs.ollama.com/api/ps. Part of the [Reflex roadmap](reflex-roadmap.md), MVP 3.

---

## 0. Where the code is today, and seven defects found while reading it

**Current design**
- **Slots.** There is one generation slot, `AppState.slot: RwLock<ModelSlot>` (`api/mod.rs`:45-52, 101), next to the `embedding` slot (214) and the `decision` slot (218).
- **`swap_lock`** (104) serializes every load and unload across all three slots.
- **`swap_model`** (292-615) runs in this order:
  1. Re-checks whether the model is already loaded (307-320) and resolves its path, so an unknown name is a 404 before anything is unloaded.
  2. Unloads the current model with `unload_current` (652-666), which joins the scheduler thread.
  3. Under fit, evicts ad-hoc embedding and decision models (369-370, 794-818, 928-948) and releases the decision context (410).
  4. Sizes the load against free VRAM minus the reserved companions' reserve (407-482).
  5. Loads on a blocking thread (526-566) and installs the result (569-574).
- **`routes::ensure_model`** (303-358) is the only way a request reaches a model. It compares the requested name with the slot, swaps if they differ, and snapshots the handles into `SlotSnapshot` (290-294).
- **keep_alive.** There is one deadline per slot (238-242). `touch_main_slot` (986-991) sets it at request start, and a loop checks it every 30 s (1019-1064).

**Defects found.** Each one is either a constraint on the design or a fix commit of its own.

- **F1.** `keep_alive: 0` with a real prompt breaks scheduler-backed models.
  - `generate`, `chat` and `chat_completions` call `touch_main_slot` right after `ensure_model` (`routes.rs`:1522, 1689, 2252).
  - For `Immediate`, that calls `unload()` **before** the request is submitted, and the scheduler thread is joined.
  - `SchedulerHandle::submit` then fails `try_send` on a disconnected channel and reports "Scheduler queue full — try again later" (`scheduler.rs`:312-323).
  - The `KeepAlive` doc (`mod.rs`:1259-1260) promises the unload happens "after this request completes".
- **F2.** The idle deadline is set at request start (`touch_deadline`, 1358-1381), and the loop never checks whether the slot is in use. A generation longer than keep_alive is therefore unloaded mid-stream. This contradicts `KeepAlive::For` ("counted from the end of this request", 1262-1264).
- **F3.** A swap aborts in-flight requests on a scheduler-backed model.
  - Shutdown sends "Server shutting down" to every active sequence and returns (`scheduler.rs`:976-988). Queued requests lose their channel.
  - `docs/engine.md` ("Concurrent swap safety") and `api/mod.rs`:11-12 claim these requests complete normally.
  - With one model (N=1) the plan keeps this, because it is today's behaviour. With N>1, eviction must never do it to another client's model.
- **F4.** `unload_current` waits for a scheduler thread to exit, but not for a sequential engine.
  - Requests hold `Arc<InferenceEngine>` clones (`sequential_to_channel`, `routes.rs`:2714-2723), so the weights stay in VRAM after the slot is emptied.
  - The next `--fit` measurement then undercounts free VRAM.
- **F5.** A sequential engine (every multimodal model, and `--batch-size 0`) creates its context on every request (`inference/mod.rs`:2004-2022). While it is idle, its KV cache is not part of the free-VRAM figure.
  - `fits_in_free_vram` (`mod.rs`:1234-1241) lets an embedder or decision model take that space.
  - The next image request then fails to allocate its context.
  - With several resident models this becomes the main sizing hazard.
- **F6.** There is no `/api/ps` (`api_routes`, `routes.rs`:124-136), so Ollama clients that call it get a 404.
- **F7.** The `RuntimeOpts` doc comment (`main.rs`:140-151) still says `batch_size` and `--fit` stay out of the shared struct; both are in it now. Fix it while adding the new flags.

---

## 1. Several resident generation models (flag, default 1)

### 1.1 Flags and plumbing (`main.rs` `RuntimeOpts`, 153-467)

**`--max-loaded-models <N>`**
- Type and limits: `usize`, default **1**, `value_parser` range `1..=16`.
- What it counts: generation models only. The embedding and decision slots are separate and never count toward N.
- How it differs from `OLLAMA_MAX_LOADED_MODELS`: Ollama counts every model and defaults to 3 × the number of GPUs (3 on CPU).
- The doc comment must state why the default is 1, in the CLAUDE.md wording: a second model only ever gets what the first one left. A default of 2 would silently halve the card for whichever model loads second.
- It must also state the co-residency rule (section 1.4), and that a request needing a place waits up to 120 s for a model to go idle.

**Other new flags**
- `--default-model` (section 2.10).
- `--auto-model` and `--auto-timeout-ms` (section 2.1).

**Plumbing**
- The `let RuntimeOpts {..} = opts` destructurings in `main()` (744-776 and 904-937) must name every field; the compiler enforces it.
- Build one `api::ResidencyConfig { max_loaded_models, default_model, router: Option<route::RouteTable>, auto_timeout }` there, with a pure constructor.
- Pass it as a single parameter to `cmd_run` (2044) and `cmd_serve` (2811), into a new `ServeConfig.residency` field (`mod.rs`:1656-1724), and from there into `AppState`.
- These are the user's original values, as the CLAUDE.md rule requires.

**Startup log**
- One line: `Generation models kept resident: up to N (--max-loaded-models)`.
- Warnings:
  - N>1 with `--no-fit`, or on a build that cannot probe VRAM: "kept up to the count only; whether they fit together is not checked".
  - A pure `ollama_env_hint(env: Option<&str>, flag_default: bool) -> Option<String>`: if `OLLAMA_MAX_LOADED_MODELS` is set, say that EuLLM takes this as a flag, not an environment variable.

### 1.2 Data model (new `engine/src/api/resident.rs`)

```rust
pub(crate) struct LoadedModel {
    pub name: String,              // normalized name it was loaded under
    pub path: PathBuf,             // resolved GGUF; identity also by same_file()
    pub engine: Option<Arc<InferenceEngine>>,
    pub scheduler: Option<SchedulerHandle>,
    pub launch: bool,              // `eullm run`'s model: evicted last
    pub ctx_size: u32, pub batch_size: usize, pub gpu_layers: i32,
    pub size_bytes: u64,           // file + projector + KV estimate (/api/ps `size`)
    pub size_vram: Option<u64>,    // drop in free VRAM measured across the load
    pub unallocated_reserve: u64,  // sequential engine: KV at its ctx + COMPUTE_BUFFER_RESERVE_BYTES (F5); 0 for a scheduler
    pub usage: Arc<Usage>,
}
pub(crate) struct Usage {          // no await while any of these is held
    in_flight: AtomicUsize, last_used: parking_lot::Mutex<Instant>,
    deadline: parking_lot::Mutex<Option<Instant>>, forever: AtomicBool, unload_when_idle: AtomicBool,
}
pub(crate) struct ResidentModels { models: Vec<LoadedModel> }   // N ≤ 16: a Vec, not a map
pub(crate) struct Lease { usage: Arc<Usage>, keep_alive: KeepAlive, default: Option<Duration>, idle: Arc<tokio::sync::Notify> }
// Drop: in_flight -= 1; last_used = now; deadline = now + resolved keep_alive (None for Forever);
//       Immediate → unload_when_idle = true; if in_flight hit 0 → idle.notify_waiters().
```

**`AppState` changes**
- `slot` (101) becomes `models: RwLock<ResidentModels>`.
- `main_deadline` (238) is removed.
- New fields: `max_loaded_models`, `idle: Arc<Notify>`, `generation_evictions: AtomicU64`, `default_model`, `router`, `auto_timeout`, and `idle_tick: Duration` (30 s in production; tests shorten it).

**Launch model**
- The launch model's handles in `ServeConfig.model_name/engine/scheduler` become the first resident, with `launch = true`.
- `launch_model: Option<(String, PathBuf)>` already supplies its path.

**Tests and `SlotSnapshot`**
- Add `#[cfg(test)] AppState::for_tests(store, api_keys)` so `http_tests::spawn_with_keys` (2359-2404) stops breaking every time a field is added.
- `SlotSnapshot` gains `lease: Lease`. The lease is taken while the `models.read()` guard is held.

### 1.3 Request path (`routes.rs` `ensure_model`, 303-358)

1. **No model named** (`model` absent or `""`): use `state.default_target()`, in this order:
   - `--default-model`, if set;
   - otherwise the most recently used resident. At N=1 that is the only resident, which is exactly today's behaviour;
   - otherwise today's 503 "No model loaded" (348).
2. **Named and resident** (`model_names_match`, 2827): take a lease and a snapshot. No `swap_lock` is involved.
3. **Named and not resident:** call `state.load_generation_model(name, overrides)`, then look it up again.

**Lease lifetime**
- Every handler moves the lease into the response it returns: `ndjson_stream_response` (2654), `stream_from_channel_sse` (2598), `buffered_message_sse` (2554), `multimodal_to_channel` (656).
- On the collect paths, the handler holds the lease until it returns.
- `in_flight` therefore covers the whole response, including streaming.
- The `touch_main_slot` calls (1522, 1689, 2252) go away; keep_alive is recorded in the lease instead.

### 1.4 Loading beside other models: `AppState::load_generation_model` (replaces `swap_model`, `mod.rs`:292-615)

**a. Before anything is unloaded**
- Take `swap_lock`.
- Re-check residency by name.
- Resolve the path, so an unknown name is still a 404 before any unload.
- Re-check residency by `same_file` (1178). This is new: with N>1, the same file asked for by its path and by its store name would otherwise load twice.

**b. Unchanged:** mmproj resolution and the Gemma KV correction (333-391).

**c. Make room.** Loop on `next_step(views, max, busy_policy, fits_now, now)`:
- **`Evict(i)`**
  1. Under `models.write()`, remove model i only if `in_flight == 0` still holds. Checking under the write guard closes the race with a new lease.
  2. Drop the guard.
  3. Shut the scheduler down on a blocking thread, as `unload_current` does. For a sequential engine, instead wait (at most 30 s) until `Arc::strong_count == 1` (F4).
  4. Increment `generation_evictions`, log the model's name, the reason and how long it has been idle, and continue the loop.
- **`WaitFor`**
  1. Release `swap_lock`.
  2. Wait with `tokio::time::timeout(remaining of BUSY_EVICTION_WAIT = 120 s, idle.notified())`.
  3. Re-acquire `swap_lock` and re-plan.
  4. When the time runs out, return the new `ModelError::Busy`, which becomes a 503 with `Retry-After: 5`. Map it in `ensure_model` (323-338), `embedding_model_error` (1304) and systemone (1154).
- **`Load`**: go on to sizing.
- **`busy_policy`** is `Abort` when `max == 1`: the resident model is replaced even mid-request, as today (F3). It is `Wait` otherwise.
- Once the count is satisfied, the existing ad-hoc companion eviction (369-370) and `release_decision_context` (410) run unchanged.

**d. Sizing as one pure plan**
- Extract the core of 399-482 into `fit::plan_offload(vram, info, layout, file_size, ctx, kv_k, kv_v, reserve, mmproj_bytes, flags) -> OffloadPlan { gpu_layers, cpu_moe, n_cpu_moe, mmproj: MmprojPlacement, full: bool }`. It reuses `compute_moe_fit`, `compute_fit`, `place_mmproj` and `apply_gpu_layers_ceiling`, prints nothing, and only the plan finally chosen is logged.
- `reserve` = `reserved_embedding_bytes + reserved_decision_bytes + mmproj placement` (as today) **+ the sum of the other residents' `unallocated_reserve`** (F5).
- `full` means sizing cut nothing below what the flags ask for:
  - dense model: `FitsFully`;
  - MoE: `Proceed { n_cpu_moe: 0 }`, or, when the user forced `--cpu-moe`/`--n-cpu-moe`, the rest of the model entirely on the GPU;
  - projector on the GPU, unless `--no-mmproj-offload` forced it to RAM;
  - `gpu_layers` equal to what the user's `--gpu-layers` ceiling allows.
- If `!plan.full` and another generation model is resident, set `fits_now = Some(false)` and go back to step c to evict the next model.
- If no other model is resident, load with the plan as today, partial split included.
- If VRAM cannot be probed, or `--no-fit` is set, `fits_now = None`: only the count is enforced.
- Ollama has the same rule: "new models must be able to completely fit in VRAM to allow concurrent model loads". This is what keeps the card from being silently divided.

**e. Load**
- Load on a blocking thread as now (526-566).
- Measure `fit::vram_bytes()` before and after the load to get `size_vram`.
- If a sequential engine shrank its context (`eng.context_size() < requested`, 558) while other models are resident, that is a silent cut: unload it, evict one more model, and retry once.

**f. Install** under `models.write()` and print the banner as today (587-612).

**Pure planner (`resident.rs`)**

```rust
pub(crate) enum BusyPolicy { Abort, Wait }
pub(crate) enum Step { Load, Evict(usize), WaitFor }
pub(crate) fn next_step(rs: &[ResidentView], max: usize, busy: BusyPolicy, fits_now: Option<bool>, now: Instant) -> Step
pub(crate) fn eviction_order(rs: &[ResidentView], now: Instant) -> Vec<usize>
```

- **Eviction order:** idle models before busy ones. Among idle models: an expired deadline first, then a finite keep_alive before "forever", then non-launch before the launch model, then least recently used.
- At N=1 this reproduces today's sequence exactly: unload the resident, evict ad-hoc companions, size alone, load.
- **Warm-up mode** (section 2.1) passes `EvictPolicy::Never`: the planner refuses instead of evicting.

### 1.5 What fits on a 16 GiB card (arithmetic to validate on the 5070 Ti)

**Inputs**
- Catalog Q4_K_M sizes: 1.7B 1.12 GiB, 4B 2.33 GiB, 8B 4.66 GiB, 14B 8.38 GiB, 32B 18.63 GiB.
- F16 KV cache per token: 112 KiB (0.6B/1.7B), 144 KiB (4B/8B), 160 KiB (14B), 256 KiB (32B).
- Each model also costs a 0.31 GiB compute reserve. The floor is 12% of total VRAM, about 1.9 GiB.

| Combination | Estimate (incl. the ≈1.9 GiB floor) | Expected outcome |
|---|---|---|
| **A**: 4B + 8B at 8k context, with a reserved Jev-Style 0.8B decision model | 3.8 + 6.1 + ~2 + floor ≈ 13.8 GiB | Fits |
| **B**: 1.7B + 14B at 4k, decision model at `--decision-ctx 4096` | 1.9 + 9.3 + ~1.5 + floor ≈ 14.6 GiB | Tight: a test of the floor |
| **C**: 4B + 14B at 8k | ≈ 17.6 GiB | Does not fit: the planner must **evict**, never split |

**LUMI GCD (64 GiB):** 4B + 14B + 32B at 8k plus Jev-Style 2B is about 45 GiB including the floor. Adding qwen3.6-27b then forces an LRU eviction at N=4.

### 1.6 Eviction, summarized

**Count limit**
- The least recently used idle model goes first.
- If every model is busy:
  - N>1: wait up to 120 s for one to go idle, then answer 503.
  - N=1: the resident model is replaced even if busy, as today.

**VRAM**
- The new model must fit fully beside the others; otherwise evict until it does, or until it is alone.
- Alone, today's partial-split rules apply.

**keep_alive**
- Expired models are the first victims. "Forever" models (keep_alive -1) go after finite ones.
- A model with requests in flight is never idle-unloaded.

**Reserved companions** (`--embedding-model`, `--decision-model`)
- Never evicted for a generation load. Their reserves are subtracted from free VRAM, as today.
- Ad-hoc companions are evicted under fit before sizing, as today.

**Counters**
- `generation_evictions` is new.
- `cross_slot_evictions` keeps its current meaning; it is reported as `model_swaps`.

### 1.7 Lock granularity

**Requests to resident models never take `swap_lock`.** They only take `models.read()` long enough to look up the model and take a lease.

**Loads stay serialized by `swap_lock`.** Free-VRAM measurements are only attributable one load at a time.

**What `models.write()` covers.** It is taken only to install a model, or to remove one. The scheduler `shutdown()` join happens after the guard is dropped, as `unload_current` does today.

**Waiting releases the lock.** A load that is waiting for a busy model releases `swap_lock`. Embedding and decision loads are therefore not held up behind it.

**Lock order**

```
swap_lock → models → embedding/decision → per-model Usage mutexes
```

- Never acquire `swap_lock` while holding any of the guards after it.
- Never hold `models`, `embedding` or `decision` across an await of a load or shutdown.
- `Lease::drop` takes no lock: atomics, a parking_lot mutex, and `Notify`.

**Determinism in tests.** A `#[cfg(test)] load_gate: Option<Arc<Notify>>` in `AppState`, awaited before the blocking load, lets a test prove that a request to model A finishes while model B is still loading.

### 1.8 keep_alive and expiry per model

**Per-request value.** Each request's `keep_alive` applies to the model that served it, through its lease. The deadline is set when the last lease is released, which fixes F1 and F2.

**`run_idle_unload_loop` becomes a janitor.** It wakes every `idle_tick` (30 s), or on `idle.notified()`. Each pass:
1. Collect, under `models.read()`, the ids that are idle and either expired or `unload_when_idle`.
2. Drop the guard.
3. Unload each one through `unload_generation(id)`, which takes `swap_lock`.

The decision of whether a model is due is a pure function, `due(view, now) -> bool`.

**Empty request with `keep_alive: 0`**
- An empty prompt (or empty `messages`) with `keep_alive: 0` unloads **only that model**.
- If the model is not resident, it is not loaded first.
- The reply is `done_reason: "unload"`, Ollama's value.
- Today it loads the model (evicting whatever was resident) and then unloads it.

**`--keep-alive`:** unchanged; it is still the server default.

### 1.9 Embedding and decision slots

**`ensure_embedding_model` and `ensure_decision_model`**
- The checks at 729-742 and 879-891 call `unload_current()` when the companion does not fit.
- They now evict generation models LRU-first, idle ones first, until the companion fits or none is left.
- Each eviction counts in `cross_slot_evictions`.
- `fits_in_free_vram` also subtracts the residents' `unallocated_reserve` (F5).

**Unchanged:** reserved companions keep their place, and their reserve is subtracted when sizing.

### 1.10 Ollama-compatible surfaces

| Ollama | EuLLM after MVP 3 |
|---|---|
| `OLLAMA_MAX_LOADED_MODELS` (environment variable, default 3×GPUs, every model counted) | `--max-loaded-models` (flag, default 1, generation models only; documented divergence, with a startup hint if the environment variable is set) |
| New models must completely fit in VRAM to load concurrently | Same rule (section 1.4) |
| Requests queue until idle models are unloaded (`OLLAMA_MAX_QUEUE` 512) | Wait up to 120 s for an idle model, then 503 + `Retry-After` |
| `OLLAMA_NUM_PARALLEL` (default 1) | `--batch-size` (default 1), per model |
| `GET /api/ps` | New: see the field list below |
| `keep_alive: 0` + empty request unloads the model | Same, for that model only |
| `ollama stop` | `POST /api/unload {"model": "..."}`, `eullm unload --model` |

**`GET /api/ps` fields**
- Ollama's: `name`, `model`, `size`, `digest`, `details{parent_model, format, family, families, parameter_size, quantization_level}`, `expires_at` (RFC 3339), `size_vram`, `context_length`.
- `expires_at` must be a far-future instant, never `null`, for a model with no deadline; Ollama reports one for models kept forever.
- Field sources:
  - `size_vram`: the measured drop in free VRAM at load; otherwise an estimate from `gpu_layers`.
  - `family`: `GgufInfo.architecture`, or the catalog.
  - `digest`: from the manifest or catalog.
- It lists the embedding and decision models too, because Ollama lists embedding models.
- EuLLM extension: `eullm: {slot, in_flight, last_used, batch_size, gpu_layers, reserved_companion, launch, size_vram_measured}`.

**`/api/tags`** (`list_models`, 1391-1507)
- Mark **every** resident `loaded: true`, most recently used first.
- The `models.first_mut()` replacement (1434) becomes a keyed replacement.

**`/v1/models`** (2150-2234)
- Add every resident that is not in the store (2167-2178 already does this for one model).
- Add `auto` when routing is configured.

**`/api/unload`** (1098)
- An optional `{"model": "..."}` body unloads that model.
- With no body, it unloads every generation model.
- `unloaded` stays a string (or null), because `cmd_unload` reads it as a string (`main.rs`:3023). A new `unloaded_all: [..]` list is added.

**`/api/version`:** add `max_loaded_models`, `loaded_models` and `generation_evictions`.

**`load_duration`:** measure it when a request triggered a load. It is hard-coded to 0 today.

### 1.11 Chat UI and `eullm run`

**Chat UI: no change.**
- `loadModels` (`app.js`:684-737) already groups every `loaded: true` entry under "Loaded" and preselects `reallyLoaded[0]`. That is the most recently used model, which is also what the server uses by default.
- `send` (823-986) always sends `model` explicitly.
- At N=1 nothing the UI sees changes.

**`eullm run`: unchanged at N=1.**
- The terminal REPL holds its own `ChatBackend` handle (`main.rs`:3525, 2697).
- At N>1 the launch model is evicted last, so the REPL keeps working unless memory pressure forces the eviction, exactly like an API swap today.
- `SchedulerHandle::submit` should say "the model was unloaded" on a disconnected channel instead of "queue full". This also gives F1 an honest message.

---

## 2. `model: "auto"`

### 2.1 Configuration (CLI flags in `RuntimeOpts`: model configuration, not perimeter)

**`--auto-model <NAME[=DESCRIPTION]>`** (repeatable, `ArgAction::Append`)
- Split each value at the first `=`.
- Allow 2 to 8 candidates (`MAX_AUTO_MODELS = 8`). Reflex is good at few options, per the MVP 0 and MVP 1 results.
- List candidates from smallest to largest.
- Refuse `auto` as a candidate, duplicates (by identity or same file), names that cannot be resolved (exit, like `--decision-model` at 2967-2970), and descriptions longer than 400 characters or containing NUL. Long descriptions are refused so the question fits the Jev-Style 0.8B's 2,048-token budget for one question with its options.

**`--auto-timeout-ms <MS>`** (default 1000, range 10..=60000): the most the router may add to a request before it gives up and uses the fallback.

**Fallback:** `--default-model` if it is a candidate, otherwise the **last** candidate.

**Resolution at startup**
- In `main.rs`: `route::RouteTable::resolve(specs, default_model, |name| lookup(store, catalog))`. Pure, with an injected lookup.
- Result: `RouteTable { candidates: Vec<RouteCandidate{name, description, source, has_projector}>, fallback, timeout }`.
- `has_projector` comes from `store.mmproj_path` or `mmproj_beside`.

**Startup warnings**
- `--auto-model` without `--decision-model`: auto will always answer with the fallback.
- `--max-loaded-models` lower than the number of candidates: every switch reloads a model.
- `--keep-alive` together with auto: the decision model can be unloaded when idle, and routing then falls back.

**Warm-up**
- After the listener binds, a task loads the candidates, fallback first, with `EvictPolicy::Never`.
- It stops at the first candidate that would need an eviction, and logs the numbers.

**Visibility.** When routing is configured, `auto` is listed in `/v1/models` and `/api/tags` with `details.family: "eullm-router"`. When it is not configured, `auto` is an ordinary name, which gives a 404 as today.

### 2.2 Candidates and descriptions

**Which models are candidates.** Only the configured ones, resident or not; one that is not resident gets loaded. Resident models that are not configured are never candidates: otherwise any client loading a model would change routing for everyone.

**Where each description comes from**, in this order:
1. The flag's text after `=`. Preferred: it should say which requests the model should get.
2. The store manifest's `description` (`store.rs`:20).
3. The catalog's `description`, plus `(about NB parameters, domain …)`. This is a warning case, because catalog descriptions are product blurbs such as "Runs on any laptop CPU".
4. The bare name. Also a warning case.

The startup log prints each option's final text and its source.

### 2.3 Code filters first ("code filters, the model judges")

These are deterministic, applied by a pure `eligible(table, facts) -> (Vec<&RouteCandidate>, Vec<Excluded{model, why}>)`:
- **Attachments.** If the request carries images or audio (`collect_chat_media`, 473, or OpenAI `image_url` parts), only candidates with a projector are eligible.
- **Context length.** A candidate whose context per slot (ctx / batch) is smaller than the estimated prompt (characters / 3.5) is excluded.

Then:
- **None eligible:** use the fallback (`no_eligible_candidate`), and the existing error path (for example `cannot_read_media`) does the rest.
- **One eligible:** use it without asking (`only_candidate`).
- **Two or more:** ask Reflex.

### 2.4 The Reflex question and its state

The decision is one `Question::Choice` (`decision.rs`:168-186). The options are `(model name, description)` in the configured order.

**Initial question text** (the bench tunes it before commit 12):

> Which model should answer the latest message? Choose the smallest model that will answer it correctly and completely; choose a larger one only when the message needs more reasoning, knowledge, code or length than a smaller one can give.

**The state is a deterministic digest built by code, not a model-written summary.** A summary would be a generation, with its own latency, and "a decision and its arguments are separate things". The digest comes from a pure `routing_state(&RouteInput) -> String`:

```
Request to answer, with its context.
Conversation: 6 messages, 3 from the user. Attachments: none. Tools offered: none.
System instructions (start): <first 300 chars, whitespace collapsed>
Earlier turns (most recent last):
user (start): <first 300 chars of the previous user turn>
assistant (end): <last 300 chars of the previous assistant turn>
Latest message:
<the latest user message whole if ≤ 4,000 chars; else first 3,000 + "[… N characters left out …]" + last 1,000>
```

How the digest is built:
- Follow-ups ("and in Python?") cannot be judged without the previous turn, so at most the two turns before the latest message are included.
- For `/api/generate`, the digest is `Prompt:` plus the head and tail of the prompt.
- OpenAI array content is reduced to its text parts, and image parts are counted as attachments.
- NUL characters are removed.
- The digest is capped at about 6,000 characters (about 1,500 tokens), well inside `--decision-ctx 8192`.

**Reading the answer**
- Use `DecisionModel::decide(state, &[q], DecideOptions{ mode: SharedPrefix, content_free: false }, &cancel)` (1349).
- Probabilities: `calibrated_probabilities(lp, None, model.default_temperature())` (1851).
- The answer is the **argmax**; `confidence` is `max_probability_confidence` (1882).
- **No threshold in the MVP.** The roadmap rule is: no thresholds without calibration on the domain's own data. Probabilities are kept everywhere, so a calibrated threshold (`--auto-min-p`) can be added later, once the bench has fitted one.

### 2.5 Running the decision: latency budget and concurrency

**How it runs**
- The decision runs on a blocking thread under `tokio::time::timeout(auto_timeout, ..)`.
- On timeout, `Cancel` is set (a `CancelOnDrop`-style guard). The decision stops at its next check, and the request uses the fallback.

**Budget on GPU**
- Measured on the 5070 Ti: 26–28 ms (BFCL, few options) and 49–50 ms (RAG gate, about 770 tokens).
- Target: router p50 ≤ 60 ms and p95 ≤ 150 ms with states of about 1,000 tokens. That is under 10% of a typical request on the larger model.

**CPU**
- The 2B took 1.5–1.8 s on few options in MVP 0, so on CPU auto mostly times out to the fallback: safe, but useless.
- The 0.8B on CPU has not been measured; the bench decides whether CPU routing is ever worth it.

**Concurrency**
- One decision worker per model (`decision/engine.rs`: a single thread, and `eval_lock` is held per request). Routing decisions and `/v1/systemone` traffic therefore queue behind each other.
- At 16 concurrent routed requests, the last one waits about 0.8 s. The timeout bounds this, and the bench measures it at concurrency 1, 4 and 16.
- Routing requests touch the decision slot's deadline (`KeepAlive::Default`), like any decision traffic.

### 2.6 Fallbacks

Every fallback answers with the fallback model and records one of these reasons:

| Reason | When |
|---|---|
| `no_decision_model` | The decision slot is empty: no `--decision-model`, or it was evicted or expired |
| `timeout` | The decision did not finish within `--auto-timeout-ms` |
| `decision_error` | `DecisionError`: over budget, control text refused, runtime failure (the message is logged) |
| `no_eligible_candidate` / `only_candidate` | Section 2.3 |
| `load_failed` | The chosen model could not be loaded: retry once with the fallback, which is usually resident |

**`raw: true` with auto on `/api/generate` → 400.** A raw prompt is written for one model's template, so it cannot be routed.

### 2.7 Response (which model answered, and why)

**`model` field.** It carries the **chosen** model on all three endpoints and in every streamed chunk (`format_token_event` and `format_done_event` already take the name). OpenAI also returns the concrete model rather than the alias that was requested.

**Extension object.** On the non-streaming response, and on the final done line or final chunk when streaming (`format_done_event` 2763, `buffered_message_sse` 2554):

`"eullm": {"route": {"requested": "auto", "model": "qwen3-4b", "reason": "decided", "confidence": 0.71, "probabilities": {"qwen3-4b": 0.85, "qwen3-8b": 0.15}, "decision_model": "...", "decision_ms": 38.2, "id": "<uuid>"}}`

Ollama and OpenAI clients ignore unknown fields.

**Headers** on all three endpoints, sent before the body starts streaming:
- `X-EuLLM-Model`, `X-EuLLM-Route` (the reason) and `X-EuLLM-Route-Id`.
- Values that are not valid header values are skipped.
- Add them to `expose_headers` in `cors_layer` (`mod.rs`:2197-2208).

### 2.8 Audit trail

**One `request_type: "route"` line per routed request**
- `decision`: the existing `DecisionRecord` (`audit/mod.rs`:71-94): `state_sha256` (never the text), readout, mode, temperature, and the choice answer. Extract `pub(crate) fn choice_record(...)` from `systemone::build_answers` (989-1100) so both paths write the same record.
- `routing: Option<RoutingRecord>`, a new field with `#[serde(default, skip_serializing_if)]`:

```rust
pub struct RoutingRecord {
  requested: String, model: String, reason: String, fallback: String,
  candidates: Vec<String>, excluded: Vec<ExcludedCandidate>,
  decision_model: Option<String>, decision_ms: f64,
  dry_run: bool /* skip if false */, error: Option<String>,
}
```

**Who writes the line**
- The route id is minted by the handler.
- A decision that finishes writes the line from its blocking thread, like `DecisionJob::run` (1301-1363), so the record survives a client disconnect.
- A shared `AtomicU8` decides which side writes: the job moves it Pending→Decided, the handler moves it Pending→TimedOut. Whoever wins the compare-exchange writes. A decision that finishes after its timeout is not recorded; its answer was never used.

**Generation lines of routed requests**
- They gain `route: Option<RouteRef{ id, requested, fallback: Option<String> /* "load_failed: …" */ }>`.
- Old lines still parse.
- The benchmark writes to its own `EULLM_AUDIT_DIR`.

### 2.9 `POST /api/route` (EuLLM extension, dry run)

**Request.** The same body as the three endpoints.

**Response**

```json
{"model", "reason", "fallback", "candidates": [{"model", "description", "probability", "resident"}],
 "excluded", "confidence", "decision_model", "decision_ms",
 "state": "<rendered>", "question": {"type": "choice", ...systemone shape}, "route_id"}
```

**Behaviour**
- It never loads a generation model.
- It is audited with `dry_run: true`.
- It lets the bench (a) measure the router alone, cheaply, and (b) replay the exact state the server built against alternative question wordings through `/v1/systemone`. The engine remains the single source of truth for the state.

### 2.10 Server-side default model

**`--default-model <NAME>`** (in `RuntimeOpts`)
- Used when a request names no model, and as auto's fallback when it is a candidate.
- Resolved at startup; an unknown name makes the process exit.
- `--default-model auto` is allowed only together with `--auto-model`. Clients that omit `model` are then routed.
- It is loaded lazily, unless it is a warm-up candidate.

**Without the flag:** N=1 behaves as today; N>1 uses the most recently used resident.

### 2.11 Endpoint specifics

**Where routing runs.** In `generate` (1509), `chat` (1676) and `chat_completions` (2236):
1. If `router.is_some()` and `model` equals `auto` (case-insensitive), call `route::decide_route(...)`.
2. Pass the chosen model to `ensure_model`.
3. Attach the route to the response and to the audit context.

Routing runs before `inject_web_content` and reads the user's own messages.

**Warm-load with auto.** Empty `messages`, or an empty prompt, warms every candidate within `max_loaded_models`, using `NoEvict`.

**Tools.** Tool names are included in the state. There is no tool-capability filter in the MVP; this is a documented limitation.

**`/api/show {"name": "auto"}`** returns a short router modelfile (optional; some UIs call it).

---

## 3. Evaluation harness: `bench/reflexbench/autobench.py`

Conventions are ReflexBench's: standard library only, one command, sets downloaded at run time to `~/.cache/reflexbench` at pinned revisions (`rb_data.fetch`), `--out` JSON and `--details` JSONL, a Markdown table, and the server run with its own `EULLM_AUDIT_DIR`.

**New files**
- `ab_data.py`, the sets:
  - `gsm8k`: openai/grade-school-math `test.jsonl`, MIT, graded by the number after "Answer:".
  - `arc-easy` and `arc-challenge`: AllenAI's ARC-V1-Feb2018 zip, CC BY-SA 4.0, graded by letter.
  - `mmlu`: Hendrycks' `data.tar`, MIT, graded by letter.
  - Together they give a natural easy/hard mix. Confirm each licence when pinning the URLs.
  - `--data` for a set of one's own: JSONL `{id, messages|prompt, answer?, grader}`, which leaves room for an Italian set built from consented traffic.
- `ab_grade.py`: number and letter extraction, exact match and F1, and judge verdict parsing.
- `ab_methods.py`: the routers and the generation runner.
- `ab_metrics.py`: paired bootstrap confidence intervals, AUROC, threshold fitting, latency statistics.
- `test_autobench.py`: offline, with `post()` stubbed the way `test_ragbench.py` (166-215) does it.

**Server for a run**

```bash
EULLM_AUDIT_DIR=/tmp/autobench-audit eullm serve --max-loaded-models 2 \
  --auto-model 'qwen3-4b=Short everyday requests: ...' --auto-model 'qwen3-8b=Multi-step reasoning, maths, code, ...' \
  --decision-model <jev-style-0.8b|2b> --embedding-model qwen3-embedding-0.6b-gguf-q8_0
python3 bench/reflexbench/autobench.py --small qwen3-4b --large qwen3-8b \
  --sets gsm8k,arc-easy,arc-challenge,mmlu --limit 200 --out auto.json --details auto.jsonl
```

**Stage 1: generate** (`--answers` caches it)
- Both models answer every item, each named explicitly, deterministically: `temperature 0, top_k 1, seed 1`, `think` off by default (recorded; `--think` switches it).
- Record correctness, TTFT, total time and generated tokens.
- Label each item `small_ok`: the small model is right, or both are wrong.

**Stage 2: route** (cheap, no generation)
- `/api/route` gives Reflex's P(small) on every item.
- Baselines:
  - always-large, always-small;
  - random routing at the same small share;
  - a length threshold fitted on the dev half;
  - embedding kNN: k=10 over dev items labelled `small_ok`, through `/v1/embeddings`;
  - the oracle.
- Quality(auto) = Σ correct[chosen model], valid because decoding is deterministic.
- `--questions` replays alternative wordings and option orders (position bias) against `/v1/systemone`, using the server's own `state`.

**Stage 3: end-to-end** (after commit 12)
- `model: "auto"` on a subset, at concurrency 1, 4 and 16.
- It validates the stage-2 combination: the same answer text on at least 99% of items at temperature 0.
- It measures the real TTFT, including routing and any model loads.

**Metrics per set and router**
- Large-model calls avoided (the share routed small), and large-model GPU-seconds avoided.
- Accuracy, and Δ against always-large with a paired-bootstrap 95% CI.
- Answers lost (large right, auto wrong) and answers gained.
- AUROC of P(small) against `small_ok`.
- At Reflex's own argmax **and** at a threshold fitted on the dev half for at most 1 point of accuracy loss: calls avoided on the test half (ragbench's own/fitted columns).
- Router p50/p95 (server `decision_ms` and client wall time); TTFT and total time of auto against always-large; loads triggered (`generation_evictions` delta).
- The header records the server version, `/api/ps` and the decision model, as ReflexBench names its decision model before counting.

**Judging quality locally**
- Verifiable sets are graded by exact match, as above. This is the primary evidence.
- For open-ended custom sets, `--judge-model`: pairwise comparison of the small model's answer against the large one's at temperature 0, asked **in both orders** (a tie unless both orders agree). Use a judge from a different family than the large model, against self-preference bias, and report agreement with a hand-labelled sample of 50 items (Cohen's κ) before trusting it.

**Kill criterion** (roadmap style): if the length threshold or the embedding kNN router avoids as many calls at equal accuracy (within the CI), Reflex is not the answer for routing. Several resident models stay, because they are useful on their own.

**Done when** one command reproduces a report on the 5070 Ti and on a CPU (and one LUMI run with a large pair) that answers: calls avoided at ≤1 point of accuracy loss, router p50/p95, and TTFT overhead.

---

## 4. Commits, in order

Each commit passes `cargo test` and `clippy -D warnings` (no dead code: pure functions land with their first caller) and changes nothing at the default except the defects it names.

Real-model tests are `#[ignore]`, CPU, and run single-threaded:
`EULLM_GENERATION_TEST_MODEL=stories260K.gguf EULLM_DECISION_TEST_MODEL=Qwen3-0.6B-Q8_0.gguf cargo test -p eullm-engine -- --ignored real_model_ --test-threads=1`.
Tests copy the tiny GGUF under several names into a temporary store, so they get distinct models without `same_file` deduplicating them.

**1. `fix(engine): keep_alive counts from the end of a request, and 0 unloads after it`** (F1, F2)
- **Change**
  - `Usage` and `Lease` are created in their final home, `api/resident.rs`, and used for the single slot.
  - The janitor is woken by `Notify`; `idle_tick` becomes injectable.
  - `submit` reports a disconnected channel honestly.
- **Files:** `api/mod.rs` (45-52, 986-1064, 1358-1381), `routes.rs` (290-358, 1522/1689/2252, the stream helpers), `scheduler.rs` (308-331).
- **Unit tests:** `lease_drop_sets_the_deadline_from_the_end_of_the_request`, `immediate_unloads_only_after_the_last_lease`, `a_busy_slot_is_never_due`.
- **Real-model tests:** `real_model_keep_alive_zero_with_a_prompt_answers_then_unloads`; `real_model_a_long_generation_outlives_a_short_keep_alive` (100 ms tick).

**2. `refactor(engine): the generation slot becomes a list of resident models, still one`**
- **Change**
  - `ResidentModels`, and `next_step`/`eviction_order` with max 1 and `Abort`.
  - `load_generation_model` and `unload_generation`.
  - `same_file` deduplication.
  - `/api/tags` and `/v1/models` iterate over the residents.
  - `AppState::for_tests`.
- **Unit tests:** `a_resident_is_found_by_path_store_name_and_ollama_tag`, `the_same_file_under_two_names_is_one_model`, `at_one_model_the_resident_is_replaced_even_when_busy`, `every_resident_is_marked_loaded_and_keeps_catalog_metadata`.
- **Real-model test:** `real_model_one_model_swaps_exactly_as_before` (A→B→A).

**3. `refactor(fit): one pure offload plan for loads; sequential engines keep their context reserved`** (F4, F5)
- **Change:** `fit::plan_offload` and `OffloadPlan.full`; `unallocated_reserve` is subtracted in `fits_in_free_vram`; drain wait for engine `Arc`s.
- **Unit tests**
  - `plan_offload` reproduces the old `run_fit_headless`/`run_moe_fit`/`place_mmproj` outcomes on the existing fixtures.
  - A truth table for `full`.
  - `a_sequential_resident_reserves_its_context_for_a_companion`.
- **GPU:** at N=1, the fit logs match main for qwen3-14b, qwen3-32b, qwen3.6-35b-a3b and gemma-4-e4b.

**4. `feat(engine): --max-loaded-models keeps several generation models resident`**
- **Change**
  - The flag and its plumbing; the `Wait` policy and the co-residency rule.
  - `ModelError::Busy` → 503.
  - `generation_evictions` and the `/api/version` fields; startup logs; the F7 doc fix.
- **Tests**
  - clap: default 1 on both commands; 0 and 17 refused.
  - `next_step` tables: count → LRU idle; all busy → `WaitFor`; not full with residents → `Evict`; `None` → count only; expired first; forever after finite; launch model last.
- **Real-model tests:** `real_model_two_models_answer_side_by_side`, `real_model_a_third_model_evicts_the_least_recently_used`, `real_model_eviction_waits_for_a_busy_model_instead_of_aborting_it`, `real_model_a_request_to_a_resident_model_is_not_held_by_a_load` (with `load_gate`).
- **Merge gate:** V1–V6 on the 5070 Ti (section 5).

**5. `feat(engine): keep_alive and unload per model`**
- **Change:** per-model janitor; an empty request with `keep_alive: 0` unloads only that model, without loading it (`done_reason: "unload"`); `/api/unload {"model"}`; `eullm unload --model`.
- **Tests**
  - Pure `due()` tests.
  - HTTP: `api_unload_of_a_model_not_loaded_is_a_200_null`.
  - Real-model: `real_model_each_model_expires_on_its_own`.

**6. `feat(api): GET /api/ps, and every resident in /api/tags and /v1/models`**
- **Change:** `/api/ps` including companions; measured `load_duration`.
- **Tests**
  - HTTP: `api_ps_answers_with_no_model` → `{"models": []}`.
  - Unit: `a_ps_entry_has_ollamas_fields` (including a far-future `expires_at`).
  - Real-model: two models listed, with distinct `expires_at`.

**7. `feat(engine): companions make room by evicting generation models, least recently used first`**
- **Unit test:** pure `companion_evictions(...)`.
- **GPU:** V5 (an ad-hoc embedder beside two models).

**8. `feat(engine): --default-model`**
- **Tests**
  - clap.
  - HTTP: no model field, with a default pointing at the non-GGUF fixture → 500 naming it.
  - The existing 503 test (2753) still passes without the flag.
  - Real-model: the default is used at N=2.

**9. `refactor(api): one builder for a generation's audit entry`**
- **Change:** `AuditCtx { user_id, route }` replaces the `user_id` parameter. It collapses the ten copies at 1598, 1651, 1809, 1880, 1935, 2373, 2462, 2516, 2615 and 2672.
- **Tests:** unit equality with the old fields.

**10. `feat(api): route table (--auto-model) and POST /api/route`**
- **Change:** `route.rs`: resolve, filters, `routing_state`, the question, `decide_route` with the timeout and the audit compare-exchange; the `RoutingRecord` audit field.
- **Unit tests**
  - Resolve errors; description sources and warnings; fallback choice.
  - Golden states for chat and generate; head/tail truncation.
  - Proptest: `routing_state` never panics, and its size stays bounded, for arbitrary messages.
  - Filters.
  - Old audit lines without `routing` still parse.
- **HTTP tests (CI):** routing not configured → 404 "auto routing is not configured"; configured without a decision model → 200 `no_decision_model`, with `state` and `question` echoed.
- **Real-model tests (Qwen3-0.6B decision):** `decided` with probabilities summing to 1; a 10 ms timeout → `timeout`.

**11. `feat(bench): autobench — generation and router stages`**
- **Tests (offline):** graders, bootstrap with a fixed seed, AUROC, threshold fit, loaders on made-up files, routers against a stubbed `post()`, and the table has every column.
- **Use:** pick the question wording and option order on the 5070 Ti before commit 12.

**12. `feat(api): model "auto" on /api/chat, /api/generate and /v1/chat/completions`**
- **Change:** wiring; headers and `eullm.route`; the linked audit line; `load_failed` → fallback; `raw` → 400; warm-up; `auto` listed in the model lists.
- **HTTP tests (CI):** routing not configured → 404 naming `auto`, as before; configured, with an unloadable fallback fixture → 500 naming the fallback; `raw` → 400.
- **Unit tests:** the extension JSON and the header values.
- **Real-model test:** two tiny copies plus Qwen3-0.6B, on every endpoint, streamed and not: the chosen `model`, the headers, and the final chunk carrying the route.
- **Merge gate:** V7–V9.

**13. `feat(bench): autobench end-to-end stage`**, plus `tools/lumi/sbatch_autobench.slurm`. It emits one `BENCH_RESULT {...}` line, as `sbatch_replicas.slurm` does.

**14. `docs: several resident models and model "auto"`**
- `docs/engine.md`: rewrite "Dynamic Model Swap" (242/369), correct "Concurrent swap safety" per F3, add `/api/ps`, `/api/route` and the auto sections.
- `reflex-roadmap.md`: MVP 3 tags and results.
- `CHANGELOG`: per the release-and-ci skill; the user cuts releases.


---

## 5. Hardware validation

**RTX 5070 Ti, 16 GB** (merge gate for commits 3, 4 and 12)

- **V1. N=1 regression.** Run the sequence qwen3-4b → qwen3-14b → gemma-4-e4b with an image → qwen3-4b through the chat UI and `eullm run --cli`. The fit decisions and banners should match main.
- **V2. Pair A, 4B + 8B at 8k with a reserved Jev-Style 0.8B.**
  - Both models load fully on the GPU.
  - `/api/ps` `size_vram` matches the `nvidia-smi` delta within 5%.
  - Read `memory_breakdown_print` for each model.
- **V3. Pairs B and C.**
  - B (tight) fits, or the planner evicts, and the log says which. There must never be a partial split while two models are resident.
  - C must evict.
  - Then qwen3-32b: everything is evicted and it loads partially alone, as today.
- **V4. Multimodal beside text.** gemma-4-e4b plus qwen3-4b, with 20 image requests interleaved with text requests: no context allocation failure (F5).
- **V5. Ad-hoc embedder and evictions.**
  - An ad-hoc embedder beside two models: watch the churn counters.
  - Two long streams while a third model is requested: neither stream is aborted; the third model loads after one ends, or the request gets a 503 after 120 s.
- **V6. A load does not stall other requests.** Requests to model A while model B loads: A's TTFT changes by no more than 10%.
- **V7. Auto bench.**
  - Pairs A and B, with the Jev-Style 0.8B and the 2B.
  - Calls avoided, Δ accuracy with its CI, router p50/p95.
  - Concurrency 1, 4 and 16.
- **V8. One-hour soak.**
  - Mixed `/api/chat`, `/v1/chat/completions`, `/api/embed`, `/v1/systemone` and `auto`, at concurrency 8.
  - VRAM sampled every 5 s stays stable, with no OOM and no `GGML_ASSERT`.
  - Every audit line parses, and the counts match.
- **V9. CPU-only machine.** N=2 tiny models, and auto with a Qwen3-0.6B decision model: router latency, stated honestly.

**LUMI-G, MI250X** (`eullm-linux-x64-rocm-gfx90a`, ROCm 6.3.4; or `tools/lumi/build_engine.sh`)

Run on `small-g`/`dev-g`, one GCD (`ROCR_VISIBLE_DEVICES=0`; 0.5 GPU-h per GCD-hour), under `sbatch`, with the job sampling device utilisation itself.

- **L1. VRAM probe on ROCm.** `vram_bytes()` through ggml on HIP, and `size_vram` against `rocm-smi --showmeminfo vram`.
- **L2. N=3 then N=4.** 4B + 14B + 32B at 8k with the Jev-Style 2B, then qwen3.6-27b forces an LRU eviction. Measure load time from Lustre, cold and warm: this is what auto pays when a candidate is not resident, and there is no local disk.
- **L3. Concurrency.** 16 streams spread over 3 resident models for 30 minutes.
- **L4. Auto bench with realistic cost ratios.** The pairs (qwen3-4b, qwen3-32b) and (qwen3-8b, qwen3.6-27b Q8, 29.3 GiB).
- **L5. Two visible GCDs.**
  - Document that every model is layer-split across both devices; there is no per-model placement yet (gap 2 in `lumi-g.md`).
  - `vram_bytes()` sums devices, so per-device placement plus `--main-gpu`/`--split-mode none` per model is the follow-up.
  - Not a merge blocker.

**Leonardo, A100 64 GB** (only if the legal-it-4b budget leaves room; the allocation ends 2 Nov 2026)
- Build from source: the login nodes' glibc 2.28 blocks the Ubuntu artifacts (see `docs/cineca/leonardo.md`).
- One job of at most 1 node-hour: L2 on CUDA, for the CUDA↔ROCm comparison (proposal objective 4).
- If a `legal-it-4b` GGUF exists by then: a domain-routing pair (legal-it-4b for Italian law questions, qwen3-14b for everything else) on the Italian set built with `rg_openbook.py`.

---

## 6. Risks, and how each step limits them

1. **Co-residency mis-sizing, leading to OOM or a crash mid-request.**
   - The "fits fully" decision comes from the same plan the load then uses.
   - The 12% floor; the sequential-engine reserves (commit 3); retry alone after a context shrink.
   - Default 1; the GPU gate before merging commit 4; the V8 soak.
2. **Killing other clients' streams.**
   - Leases; idle-first eviction; a bounded wait, then 503.
   - At N=1 today's behaviour is kept and documented (F3); unifying it is a separate decision.
3. **Deadlock or starvation.**
   - One documented lock order; no guard held across a load or join.
   - A waiting load releases `swap_lock`.
   - The deterministic `load_gate` test.
4. **Unnoticed behaviour changes at N=1.** Each one is confined to fix commits 1, 3 and 5, with its own test, plus the V1 regression run.
5. **Ollama compatibility.**
   - `/api/ps` field test; `unloaded` stays a string.
   - The flag-versus-environment divergence is documented, with a startup hint.
6. **Routing quality.**
   - Auto is opt-in; argmax only; the fallback is the large model.
   - The response and the audit always say which model answered and why.
   - The bench, with its kill criterion, gates any claim.
7. **Router latency** (CPU, concurrency, a decision worker shared with `/v1/systemone`): the timeout falls back to the large model; a capped state; measured at concurrency 1, 4 and 16; GPU recommended.
8. **Router model missing** (keep_alive expiry, or an ad-hoc decision model evicted by a generation load): `no_decision_model` fallback, startup warnings, and `--decision-model` (a reserved companion) recommended.
9. **Models alternating within one conversation.**
   - The other model reads earlier answers as its own.
   - The conversation's prompt cache is lost: every switch re-prefills the whole conversation.
   - Documented; the headers let a UI tag each turn with its model; a multi-turn set in the bench later.
10. **Multi-GPU:** summed free VRAM against per-device placement. Validate on one device; per-model pinning is a follow-up.
11. **Audit volume and privacy.** The state is stored as SHA-256 only; route lines are small; the bench uses its own `EULLM_AUDIT_DIR`.
12. **Flag sprawl.** Everything is in `RuntimeOpts` and one `ResidencyConfig`, and the parity tests cover it.
13. **Benchmark validity.**
    - The deterministic-decoding assumption is checked by the end-to-end stage.
    - Judge bias is controlled with both orders, a different family, and agreement with human labels.
    - Licences are pinned, and seeds are reported.

---
