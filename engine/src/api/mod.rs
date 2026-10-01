//! eullm REST API.
//!
//! Exposes a standard LLM API (both `/api` and `/v1` OpenAI-compatible)
//! so that existing tools (Open WebUI, LangChain, n8n) work out of the box.
//!
//! Supports two inference backends:
//! - **Sequential** (`InferenceEngine`): one request at a time.
//! - **Continuous batching** (`SchedulerHandle`): multiple concurrent requests.
//!
//! Supports **dynamic model swapping**: when a request specifies a model that
//! is not loaded, the server unloads the current model and loads the new one.
//! Requests the old model was still answering are cut off with an error, as
//! they always were; the residents are kept in `resident::ResidentModels`.

mod auth;
mod ip_allowlist;
mod origin;
#[cfg(test)]
mod real_model_tests;
mod resident;
// `routes` is not part of the public API, but the terminal REPL in `main.rs`
// reuses `routes::sequential_to_channel` so that a model without a scheduler
// (multimodal forces `batch_size = 0`) streams through exactly the same code
// path as an HTTP request instead of a second, divergent one.
pub(crate) mod routes;
mod systemone;

pub use auth::Identity;

use axum::Router;
use axum::extract::{ConnectInfo, DefaultBodyLimit, State};
use axum::response::IntoResponse;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use llama_cpp_2::llama_backend::LlamaBackend;

use crate::inference::decision::DecisionModel;
use crate::inference::embedding::EmbeddingModel;
use crate::inference::{
    BatchScheduler, InferenceConfig, InferenceEngine, SchedulerConfig, SchedulerHandle,
};
use crate::models::ModelStore;

/// The embedding slot — independent of the generation models on purpose. See
/// `AppState::ensure_embedding_model` for why this coexists with the
/// generation model rather than sharing its slot.
pub struct EmbeddingSlot {
    pub model_name: String,
    pub model: Arc<EmbeddingModel>,
    /// Whether this slot was populated by `--embedding-model` at launch
    /// (`true`) or loaded ad hoc by a runtime `/api/embed`/`/v1/embeddings`
    /// request (`false`). The distinction is what `--embedding-model` is
    /// *for*: a reserved companion is never evicted just because the
    /// generation model is being swapped (`load_generation_model`'s own `--fit`
    /// reserves its footprint instead, so it keeps its place), where an ad
    /// hoc embedder has no such guarantee and is evicted on any generation
    /// swap that needs the room back (`evict_embedding_if_present_for_generation_load`).
    pub is_reserved_companion: bool,
}

/// The decision slot behind `/v1/systemone` — a third slot, handled like the
/// embedding one and for the same reason: a small decision model can stay
/// resident next to the chat model instead of taking turns with it. See
/// `AppState::ensure_decision_model`.
pub struct DecisionSlot {
    pub model_name: String,
    pub model: Arc<DecisionModel>,
    /// Same meaning as `EmbeddingSlot::is_reserved_companion`, for
    /// `--decision-model`.
    pub is_reserved_companion: bool,
    /// VRAM a request can take on top of the weights: its KV cache at the
    /// per-request ceiling plus a compute buffer
    /// (`fit::decision_reserve_bytes`). The weights show up as used VRAM
    /// once loaded; the context the model keeps between requests is
    /// released before a generation model is sized, so it does not.
    pub reserve_bytes: u64,
}

/// Shared state for API handlers.
pub struct AppState {
    /// The one `LlamaBackend` this process created (see
    /// `inference::init_shared_backend`), shared by every model load —
    /// the launch model, every later `load_generation_model`, and every embedding
    /// model. `LlamaBackend::init()` is a process-wide one-time marker;
    /// a second independent instance fails with `BackendAlreadyInitialized`
    /// while the first is still alive, so this must be the same `Arc` the
    /// launch model itself loaded with, not a fresh one.
    pub backend: Arc<LlamaBackend>,
    /// The generation models in memory. A request takes the read guard just
    /// long enough to find its model and take a lease on it; the write guard
    /// is taken only to install a model or take one out, never across a load
    /// or a scheduler's shutdown.
    pub(crate) models: tokio::sync::RwLock<resident::ResidentModels>,
    /// Serializes loads and unloads of every slot — generation, embedding
    /// and decision — so that two loads never size themselves against the
    /// same free VRAM. Taken before any other lock here, never after one:
    /// `swap_lock` → `models` → `embedding`/`decision`.
    swap_lock: tokio::sync::Mutex<()>,

    // ── Immutable inference settings (from CLI flags) ────────────────
    pub gpu_layers: i32,
    /// `--fit`: size the GPU offload against measured free VRAM before
    /// every load this server performs — the initial lazy load and every
    /// API-triggered swap. Never prompts (a daemon cannot ask questions):
    /// the MoE path always resolves to a loadable configuration, and the
    /// dense path proceeds with the computed split, or refuses the load
    /// when `fit_strict` is set. Without this, a swap reused the *launch*
    /// model's layer split for whatever model came next — observed live:
    /// `run --fit` sized a dense 27B at 43/64 layers, then a web-UI switch
    /// to a 22 GB MoE loaded with those same settings and OOM'd.
    pub fit: bool,
    /// `--fit-strict`: with `fit`, a model that does not fully fit is not
    /// loaded; the API caller gets the error instead of a partial split.
    pub fit_strict: bool,
    pub ctx_size: u32,
    pub threads: u32,
    pub flash_attn: bool,
    pub n_batch: u32,
    /// KV cache quantization type for keys (e.g. Q8_0 — reduces VRAM).
    pub cache_type_k: crate::inference::KvCacheType,
    /// KV cache quantization type for values (e.g. Q4_0 — reduces VRAM).
    pub cache_type_v: crate::inference::KvCacheType,
    /// 0 = sequential, >0 = continuous batching with this many slots.
    pub batch_size: usize,
    /// Keep MoE expert tensors on CPU RAM (see `InferenceConfig::cpu_moe`).
    /// Applied to every model this server loads or swaps to.
    pub cpu_moe: bool,
    /// Keep MoE expert tensors on CPU RAM for only the first N layers (see
    /// `InferenceConfig::n_cpu_moe`). Applied to every model this server
    /// loads or swaps to.
    pub n_cpu_moe: u32,
    /// Recurrent-state rollback window for hybrid/recurrent architectures
    /// (see `InferenceConfig::rs_seq`). Applied to every model this server
    /// loads or swaps to.
    pub rs_seq: u32,
    /// Max full-sequence-state checkpoints kept for prompt-prefix restore
    /// (see `SchedulerConfig::ctx_checkpoints`). 0 disables checkpointing.
    /// Applied to every model this server loads or swaps to.
    pub ctx_checkpoints: usize,
    /// Min new tokens since the closest checkpoint before taking another
    /// one (see `SchedulerConfig::checkpoint_min_step`).
    pub checkpoint_min_step: u32,
    /// Enable extra internal diagnostics for the Rust engine layer (see
    /// `ServeConfig::rust_debug`). Applied to every model this server
    /// loads or swaps to.
    pub rust_debug: bool,

    /// Enable transparent web fetching: URLs in user messages are fetched
    /// and their content is injected into the prompt before inference.
    pub web_enabled: bool,

    /// Port the canonical API listener runs on (Ollama-compatible, default
    /// 11434). Exposed via `/api/version` so the chat UI (served on its own
    /// port) can display the endpoint external clients should point at.
    pub api_port: u16,

    /// Model store for resolving names → GGUF paths.
    pub store: ModelStore,

    /// Which source IPs may reach the API/UI — see `ip_allowlist`. Loaded
    /// once at startup; not affected by later edits to `.env` without a
    /// restart.
    pub ip_allowlist: ip_allowlist::IpAllowlist,

    /// Optional bearer-token authentication with per-key quotas — see `auth`.
    /// When enabled it runs *outside* the IP allowlist and a valid key admits
    /// the request regardless of source address, which is the only ordering
    /// that works behind Docker's address translation.
    pub api_keys: Arc<auth::ApiKeys>,

    /// Which browser origins may call the API — see `origin`.
    pub allowed_origins: origin::AllowedOrigins,

    /// What the web tool is allowed to fetch — see `tools::guard`. Resolved
    /// once at startup so a request cannot pay for re-reading the environment,
    /// and so the posture is logged before the first fetch rather than after.
    pub web_policy: crate::tools::guard::WebPolicy,

    /// Projector to fall back on when the model being loaded declares none
    /// of its own. Only ever an explicit `--mmproj`, never a projector that
    /// was discovered for some other model: pairing a projector with weights
    /// it was not trained on fails the load outright (`mismatch between text
    /// model and mmproj`), which is what happened when `run`'s auto-detected
    /// projector was passed here and then applied to every later swap.
    /// Normally `None`.
    pub fallback_mmproj: Option<PathBuf>,

    /// `--mmproj-offload` / `--no-mmproj-offload` as the user gave them, or
    /// `None` to let sizing place each model's projector. The flag and not a
    /// decision, for the same reason `gpu_layers` is the flag: a placement
    /// worked out for the launch model says nothing about the next one.
    pub mmproj_offload: Option<bool>,

    /// Whether a request's `model` field may name an arbitrary filesystem
    /// path. Off by default — see `resolve_model`.
    pub allow_model_paths: bool,

    /// The `(name, path)` this process was launched with, if any. Always
    /// resolvable even when `allow_model_paths` is off: `/api/tags` advertises
    /// this name, clients echo back what they were told, and refusing our own
    /// answer would break `eullm run ./model.gguf` on the first model swap
    /// back to it.
    pub launch_model: Option<(String, PathBuf)>,

    /// Second, independent model slot for text embeddings — see
    /// `ensure_embedding_model`. `None` until the first `/v1/embeddings` or
    /// `/api/embed` request names a model.
    pub embedding: tokio::sync::RwLock<Option<EmbeddingSlot>>,

    /// Third model slot, for `/v1/systemone` — see `ensure_decision_model`.
    /// `None` until `--decision-model` or the first request naming a model.
    pub decision: tokio::sync::RwLock<Option<DecisionSlot>>,
    /// Most tokens of context one decision request may use
    /// (`--decision-ctx`). Every model loaded into the decision slot gets it.
    pub decision_ctx: u32,

    /// How many times a model was evicted to make VRAM room for another
    /// slot (generation displacing the embedder or the decision model, or
    /// either of those displacing generation). Not itself a problem — an
    /// ingestion run that evicts the LLM once and restores it once is
    /// exactly the intended use — but a steady-state rate of one eviction
    /// per request means a caller on a card too small for both is
    /// alternating instead of batching, paying a full model load on every
    /// call. Surfaced in `/api/version` (`model_swaps`) so that pattern is
    /// visible over time rather than only as an unexplained slowdown.
    pub cross_slot_evictions: std::sync::atomic::AtomicU64,

    /// Woken when the last request on a generation model finishes, so the
    /// idle-unload loop acts on `keep_alive: 0` as soon as the request is
    /// over instead of at its next tick. Each generation model's own
    /// deadline lives in its `resident::Usage`, set when its requests end.
    idle: Arc<tokio::sync::Notify>,
    /// How often the idle-unload loop looks for an expired keep_alive: 30 s,
    /// shorter in tests.
    idle_tick: std::time::Duration,
    /// Idle-unload deadline for the embedding slot — `None` means no timer
    /// is running (slot empty, or the model that loaded it asked to be kept
    /// forever). Reset on every request that touches the slot; checked by
    /// the background task spawned in `serve()`. See `KeepAlive`.
    embedding_deadline: tokio::sync::Mutex<Option<tokio::time::Instant>>,
    /// Same as `embedding_deadline`, for the decision slot.
    decision_deadline: tokio::sync::Mutex<Option<tokio::time::Instant>>,
    /// Applied when a request does not set its own `keep_alive` field —
    /// see `RuntimeOpts`/`ServeConfig::keep_alive`. `None` disables the
    /// idle-unload timer by default (a request can still opt in with an
    /// explicit `keep_alive`).
    pub default_keep_alive: Option<std::time::Duration>,
}

/// Why a model could not be made ready.
///
/// The distinction exists because it is the difference between a 4xx and a
/// 5xx, and getting it wrong is not cosmetic: a client with automatic retry
/// treats a 500 as "try again" and will hammer a request that can never
/// succeed. Asking for a model that does not exist is a client mistake;
/// failing to load one that does is ours.
#[derive(Debug)]
pub enum ModelError {
    /// No such model, by any accepted spelling. The caller should get a 404.
    NotFound(String),
    /// The model exists but could not be loaded: out of VRAM, corrupt GGUF,
    /// a context that will not allocate. The caller should get a 500.
    LoadFailed(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(m) | Self::LoadFailed(m) => f.write_str(m),
        }
    }
}

impl From<String> for ModelError {
    /// Everything that is not explicitly a lookup miss is a load failure.
    fn from(m: String) -> Self {
        Self::LoadFailed(m)
    }
}

impl AppState {
    /// Load a generation model that is not resident, making room for it
    /// first: the resident gives way, and in-flight requests on it are cut off
    /// as they always were. The new model loads with the same inference
    /// settings as every other.
    ///
    /// `override_batch_size` allows the caller to change the number of
    /// concurrent batch slots for the new model (e.g. more slots for a
    /// smaller model that uses less VRAM).  Pass `None` to keep the
    /// batch size from the CLI launch.
    ///
    /// Returns the model with a lease on it for the request that asked,
    /// taken under the same guard that found or installed it, so the request
    /// is answered by the model it named even when another load follows
    /// straight after. `keep_alive` is that request's.
    ///
    /// This is the **write** path — only one load runs at a time.
    pub(crate) async fn load_generation_model(
        &self,
        name: &str,
        override_batch_size: Option<usize>,
        override_ctx_size: Option<u32>,
        keep_alive: KeepAlive,
    ) -> Result<resident::SlotSnapshot, ModelError> {
        // Serialize loads — if another request is already loading, wait for
        // it to finish instead of starting a parallel load.
        let _swap_guard = self.swap_lock.lock().await;

        // Normalize Ollama-style names: "qwen3:14b" → "qwen3-14b"
        let normalized = normalize_model_name(name);

        // Re-check after acquiring the lock — another request may have
        // loaded it while this one waited.
        if let Some(snapshot) = self.lease_resident(&normalized, keep_alive).await {
            tracing::info!(
                "Model {} already loaded (by another request)",
                crate::audit::sanitize_for_log(&normalized)
            );
            return Ok(snapshot);
        }

        // Resolved before anything is unloaded, so that a name that does not
        // exist is a 404 that costs nobody their model.
        let gguf_path = self.resolve_model(&normalized)?;
        // The same file under another of its names — a store name for a model
        // launched by its path, or another name `eullm pull` linked to the
        // same weights: loading it again would hold it twice.
        {
            let models = self.models.read().await;
            if let Some(model) = models.find_file(&gguf_path) {
                tracing::info!(
                    "{} is {}, already loaded",
                    crate::audit::sanitize_for_log(&normalized),
                    crate::audit::sanitize_for_log(&model.name)
                );
                return Ok(self.lease(model, keep_alive));
            }
        }
        // Resolve an mmproj sibling (vision projector) if the model store
        // declares one. Presence of a projector is the signal that this is
        // a multimodal model — we then force sequential loading (next step)
        // because the continuous-batching scheduler is text-only.
        // A projector beside the weights counts too: that is how every
        // HuggingFace vision repo is laid out, and a model resolved from a
        // path has no store entry to declare one. `--mmproj` is the last
        // resort, and it is logged loudly because pairing a projector with
        // weights it was not trained on produces confident nonsense rather
        // than an error.
        let mmproj_path = self
            .store
            .mmproj_path(&normalized)
            .or_else(|| crate::models::store::mmproj_beside(&gguf_path))
            .or_else(|| {
                self.fallback_mmproj.clone().inspect(|p| {
                    tracing::warn!(
                        "no projector of its own for {}; using --mmproj {}",
                        crate::audit::sanitize_for_log(&normalized),
                        p.display()
                    );
                })
            });
        if let Some(ref p) = mmproj_path {
            tracing::info!("Multimodal model detected — mmproj: {}", p.display());
        }
        tracing::info!(
            "Swapping model → {} ({})",
            crate::audit::sanitize_for_log(&normalized),
            gguf_path.display()
        );

        // ── 1. Make room: unload the resident that gives way and WAIT for
        //       its scheduler thread to fully exit before loading the new
        //       model.
        //
        // Without this, both models would be in VRAM simultaneously —
        // causing OOM or a C-level crash in llama.cpp.
        self.make_room_for(&normalized).await;

        // An embedder left resident from an earlier ingestion run would
        // otherwise shrink the free VRAM `--fit` measures below, sizing this
        // load as if the card were smaller than it actually is once the
        // embedder itself is later evicted by `ensure_embedding_model`. See
        // `evict_embedding_if_present_for_generation_load`.
        self.evict_embedding_if_present_for_generation_load().await;
        self.evict_decision_if_present_for_generation_load().await;

        // ── 2. Load the new model ───────────────────────────────────
        // Gemma 4 requires f16 KV cache regardless of the server's configured
        // baseline (mixed SWA architecture) — see
        // `inference::correct_kv_cache_for_model` for the rationale. This
        // must run here, not just in the CLI `run` startup path, so a swap
        // triggered by a request (on `run` after startup, or on any `serve`)
        // hits the same correction instead of silently loading incompatible
        // KV cache types.
        let (cache_type_k, cache_type_v, kv_corrected) =
            crate::inference::correct_kv_cache_for_model(
                &normalized,
                self.cache_type_k,
                self.cache_type_v,
            );
        if kv_corrected {
            tracing::warn!(
                "Gemma 4 detected ({}) with non-f16 KV cache — auto-correcting to f16/f16 (mixed SWA architecture requires it)",
                crate::audit::sanitize_for_log(&normalized)
            );
        }
        // ── 2a. Size the offload for THIS model (--fit) ─────────────
        // The launch flags describe the launch model; whatever is being
        // swapped in has its own size, layer count, and (possibly) expert
        // layout. Runs after the unload above so the measured free VRAM is
        // real. Never prompts — same decision order as the `run` startup
        // flow: projector placement, MoE auto-sizing (always resolves), then
        // the dense split, headless — all in one `fit::plan_offload`.
        let effective_ctx = override_ctx_size.unwrap_or(self.ctx_size);
        let info = crate::fit::read_gguf_info(&gguf_path);
        let file_size = std::fs::metadata(&gguf_path).map(|m| m.len()).unwrap_or(0);
        let kv_bpe_k = crate::inference::cache_type_bytes_per_elem(&cache_type_k);
        let kv_bpe_v = crate::inference::cache_type_bytes_per_elem(&cache_type_v);
        // The projector is loaded with the model, always, so sizing has to
        // count it — see `fit::place_mmproj` for where it goes and why.
        let mmproj_bytes = crate::fit::mmproj_footprint_bytes(mmproj_path.as_deref());
        let flags = crate::fit::OffloadFlags {
            gpu_layers: self.gpu_layers,
            cpu_moe: self.cpu_moe,
            n_cpu_moe: self.n_cpu_moe,
            mmproj_offload: self.mmproj_offload,
        };
        let plan = if self.fit {
            // Counted below as reserved; the context the decision model
            // keeps between requests must not show up as used as well.
            self.release_decision_context().await;
            // Memory the free-VRAM figure does not show yet: the reserved
            // companions' requests, and the context every sequential
            // resident creates per request.
            let reserve_bytes = self
                .reserved_embedding_bytes()
                .await
                .saturating_add(self.reserved_decision_bytes().await)
                .saturating_add(self.models.read().await.unallocated_reserve());
            let layout = match (&info, file_size) {
                (Some(i), size) if size > 0 => {
                    crate::fit::read_gguf_moe_layout(&gguf_path, size, i.n_layers)
                }
                _ => None,
            };
            let plan = crate::fit::plan_offload(
                crate::fit::vram_bytes(),
                info.as_ref(),
                layout.as_ref(),
                file_size,
                effective_ctx,
                kv_bpe_k,
                kv_bpe_v,
                reserve_bytes,
                mmproj_bytes,
                flags,
            );
            plan.print_decision(file_size, self.fit_strict);
            if self.fit_strict && plan.refused_by_strict() {
                return Err(ModelError::LoadFailed(format!(
                    "--fit-strict: model '{normalized}' does not fully fit in the \
                     currently free VRAM; not loading. Retry without --fit-strict \
                     to allow a partial CPU/GPU split."
                )));
            }
            // A `--gpu-layers` given at startup is an upper bound for every
            // model this server loads, not a count to apply blindly to a
            // model it was never chosen for.
            if plan.capped_from.is_some() {
                tracing::info!(
                    "--gpu-layers {}: offloading {} layers for {}",
                    self.gpu_layers,
                    plan.gpu_layers,
                    crate::audit::sanitize_for_log(&normalized)
                );
            }
            Some(plan)
        } else {
            None
        };
        let (gpu_layers, cpu_moe, n_cpu_moe, mmproj_placement) = match &plan {
            Some(plan) => (plan.gpu_layers, plan.cpu_moe, plan.n_cpu_moe, plan.mmproj),
            None => (
                self.gpu_layers,
                self.cpu_moe,
                self.n_cpu_moe,
                crate::fit::MmprojPlacement::from_flag(self.mmproj_offload),
            ),
        };

        let config = InferenceConfig {
            model_path: gguf_path.clone(),
            gpu_layers,
            context_size: effective_ctx,
            threads: self.threads,
            flash_attn: self.flash_attn,
            n_batch: self.n_batch,
            cache_type_k,
            cache_type_v,
            // Multimodal: when the model store declares an mmproj sibling we
            // load it here so HTTP requests with `images` can route through
            // `engine.generate_multimodal()`. Models without an mmproj keep
            // the text-only fast path (None → no extra VRAM, no init cost).
            mmproj_path: mmproj_path.clone(),
            mmproj_on_gpu: mmproj_placement.on_gpu(),
            cpu_moe,
            n_cpu_moe,
            rs_seq: self.rs_seq,
        };
        if mmproj_path.is_some() {
            tracing::info!("{}", mmproj_placement.describe());
        }

        // The continuous-batching scheduler is text-only — it does not route
        // mtmd chunks. For multimodal models we therefore force the sequential
        // `InferenceEngine` (batch_size=0). Vision is interactive single-user
        // anyway, so losing batching here is not a practical regression.
        let batch_size = if mmproj_path.is_some() {
            0
        } else {
            override_batch_size.unwrap_or(self.batch_size)
        };
        let model_name = normalized.clone();
        let ctx_checkpoints_for_swap = self.ctx_checkpoints;
        let checkpoint_min_step_for_swap = self.checkpoint_min_step;
        let rust_debug_for_swap = self.rust_debug;
        // What the banner shows unless the sequential engine has to shrink it.
        // The scheduler never shrinks: if its context does not fit, the whole
        // swap fails, so requested and actual are always the same there.
        let requested_ctx_size = override_ctx_size.unwrap_or(self.ctx_size);
        let backend_for_swap = self.backend.clone();

        let (new_engine, new_scheduler, ready_info, effective_ctx_size) =
            tokio::task::spawn_blocking(move || {
                if batch_size > 0 {
                    let sched_config = SchedulerConfig {
                        max_batch_size: batch_size,
                        queue_capacity: batch_size * 8,
                        ctx_checkpoints: ctx_checkpoints_for_swap,
                        checkpoint_min_step: checkpoint_min_step_for_swap,
                        debug_logit_check: rust_debug_for_swap,
                    };
                    let sched = BatchScheduler::new(config, sched_config);
                    match sched.start(backend_for_swap) {
                        Ok((handle, model_info)) => {
                            Ok((None, Some(handle), Some(model_info), requested_ctx_size))
                        }
                        Err(e) => Err(format!("Failed to start scheduler: {e}")),
                    }
                } else {
                    match InferenceEngine::load(config, backend_for_swap) {
                        Ok(eng) => {
                            // Read it here, on the blocking thread that already
                            // owns the model, rather than after the move into the
                            // slot: the estimate needs the model's own metadata.
                            let info = eng.ready_info();
                            // May be smaller than `requested_ctx_size`: `load()`
                            // shrinks it automatically when the requested size
                            // does not fit, and the banner has to say what
                            // actually loaded — `info`'s KV estimate already
                            // reflects the shrunk size, so showing the
                            // requested one here would state a KV cost that
                            // belongs to a different context than the one
                            // printed next to it.
                            let actual_ctx_size = eng.context_size();
                            Ok((Some(Arc::new(eng)), None, Some(info), actual_ctx_size))
                        }
                        Err(e) => Err(format!("Failed to load model: {e}")),
                    }
                }
            })
            .await
            .map_err(|e| format!("Task join error: {e}"))??;

        // ── 3. Install the new model in the slot ─────────────────────
        // A sequential engine creates its context per request, so the
        // memory that context takes is free while it is idle; it is held
        // back from everything sized next to it instead (F5).
        let unallocated_reserve = match &new_engine {
            Some(engine) if gpu_layers != 0 => crate::fit::context_reserve_bytes(
                info.as_ref(),
                engine.context_size(),
                kv_bpe_k,
                kv_bpe_v,
            ),
            _ => 0,
        };
        let snapshot = {
            let mut models = self.models.write().await;
            let mut model = resident::LoadedModel::new(
                model_name.clone(),
                gguf_path,
                new_engine,
                new_scheduler,
            );
            model.unallocated_reserve = unallocated_reserve;
            let model = models.insert(model);
            self.lease(model, keep_alive)
        };

        tracing::info!(
            "Model swap complete → {} (batch_size={batch_size})",
            crate::audit::sanitize_for_log(&model_name)
        );

        // The diagnostic banner `run` prints at startup. `serve` starts with no
        // model, so this is the only place it can be emitted — and until it was
        // here, anyone driving the engine as a daemon never saw which backend
        // actually initialised, how many layers were offloaded, or what the KV
        // cache costs. That is the audience least able to guess and most likely
        // to be filing a report. See `crate::banner`.
        let info = ready_info.unwrap_or_default();
        crate::banner::ModelBanner {
            model_name: model_name
                .strip_prefix("eullm/")
                .unwrap_or(&model_name)
                .to_string(),
            gpu_layers: self.gpu_layers,
            cpu_moe: self.cpu_moe,
            n_cpu_moe: self.n_cpu_moe,
            rs_seq: self.rs_seq,
            ctx_checkpoints: self.ctx_checkpoints,
            checkpoint_min_step: self.checkpoint_min_step,
            batch_size,
            ctx_size: effective_ctx_size,
            n_ctx_train: info.n_ctx_train,
            flash_attn: self.flash_attn,
            cache_type_k,
            cache_type_v,
            kv_k_mib: info.kv_k_mib,
            kv_v_mib: info.kv_v_mib,
            web: self.web_enabled,
            threads: self.threads,
            n_batch: self.n_batch,
            rust_debug: self.rust_debug,
        }
        .print();

        Ok(snapshot)
    }

    /// A resident's handles, with a lease on it for one request. Only with a
    /// guard on the residents held — see `resident::Usage::lease`.
    pub(crate) fn lease(
        &self,
        model: &resident::LoadedModel,
        keep_alive: KeepAlive,
    ) -> resident::SlotSnapshot {
        resident::SlotSnapshot {
            model_name: model.name.clone(),
            engine: model.engine.clone(),
            scheduler: model.scheduler.clone(),
            lease: model
                .usage
                .lease(keep_alive, self.default_keep_alive, &self.idle),
        }
    }

    /// The resident `requested` names, with a lease on it, if it is loaded.
    async fn lease_resident(
        &self,
        requested: &str,
        keep_alive: KeepAlive,
    ) -> Option<resident::SlotSnapshot> {
        let models = self.models.read().await;
        models
            .find(requested)
            .map(|model| self.lease(model, keep_alive))
    }

    /// Unload residents until there is room for one more model, in the order
    /// `resident::next_step` gives. One model is resident at a time, so this
    /// unloads the one there is, busy or not, exactly as a swap always did.
    async fn make_room_for(&self, incoming: &str) {
        loop {
            let views = self.models.read().await.views();
            match resident::next_step(&views, 1, std::time::Instant::now()) {
                resident::Step::Load => return,
                resident::Step::Evict(i) => {
                    if let Some(name) = self.remove_generation(views[i].id, Removal::Always).await {
                        tracing::info!(
                            "Unloaded {} to make room for {}",
                            crate::audit::sanitize_for_log(&name),
                            crate::audit::sanitize_for_log(incoming)
                        );
                    }
                }
            }
        }
    }

    /// Take the generation model `id` out of the residents — if `when` still
    /// holds for it under their write guard, where no lease can be taken, so
    /// that a request which took one in the meantime keeps its model — and
    /// free its memory. Returns its name. Call with `swap_lock` held.
    async fn remove_generation(&self, id: u64, when: Removal) -> Option<String> {
        let model = {
            let mut models = self.models.write().await;
            let usage = models.get(id)?.usage.view();
            let allowed = match when {
                Removal::Always => true,
                Removal::IfDue => resident::due(&usage, std::time::Instant::now()),
            };
            if !allowed {
                return None;
            }
            models.remove(id)?
        };
        let name = model.name.clone();
        retire(model).await;
        Some(name)
    }

    /// Unload every generation model, freeing its VRAM, without loading a
    /// replacement — a later request with a `model` field (or another
    /// `eullm run`) loads one again. Requests still running on one are cut
    /// off: an explicit unload means now.
    ///
    /// The primary use case is freeing VRAM for a co-resident process (e.g.
    /// an embedding model used during RAG document ingestion) without
    /// restarting the eullm server. Serialized against loads via the same
    /// lock, so an unload can't race a concurrent load.
    ///
    /// Returns the names of the models unloaded, none if nothing was loaded
    /// (a no-op, not an error).
    pub(crate) async fn unload_all(&self) -> Vec<String> {
        let _swap_guard = self.swap_lock.lock().await;
        let unloaded = self.unload_generation_models().await;
        if !unloaded.is_empty() {
            tracing::info!("Generation models unloaded — none resident");
        }
        unloaded
    }

    /// Shared by `unload_all` and the companions that need the whole card:
    /// take every generation model out and wait for each one's scheduler
    /// thread to fully exit, so their VRAM is guaranteed freed by the time
    /// this resolves — the caller needs the VRAM actually free before handing
    /// it to another model or process. Call with `swap_lock` held.
    async fn unload_generation_models(&self) -> Vec<String> {
        let ids: Vec<u64> = self
            .models
            .read()
            .await
            .views()
            .iter()
            .map(|r| r.id)
            .collect();
        let mut unloaded = Vec::new();
        for id in ids {
            unloaded.extend(self.remove_generation(id, Removal::Always).await);
        }
        unloaded
    }

    /// Unload every generation model that is due (`resident::due`): no
    /// request is using it, and the last one to finish asked for
    /// `keep_alive: 0` or left a deadline that has passed. Each is checked
    /// again under the residents' write guard, where no lease can be taken,
    /// so a request that arrived in the meantime keeps its model.
    async fn unload_due_generation_models(&self) {
        let now = std::time::Instant::now();
        let due: Vec<resident::ResidentView> = self
            .models
            .read()
            .await
            .views()
            .into_iter()
            .filter(|r| resident::due(&r.usage, now))
            .collect();
        for model in due {
            let _swap_guard = self.swap_lock.lock().await;
            let Some(name) = self.remove_generation(model.id, Removal::IfDue).await else {
                continue;
            };
            let name = crate::audit::sanitize_for_log(&name);
            if model.usage.unload_when_idle {
                tracing::info!("keep_alive 0 — unloading {name} now that its request is over");
            } else {
                tracing::info!("keep_alive expired — unloading idle generation model {name}");
            }
        }
    }

    /// Ensure the named embedding model is loaded, loading or swapping it in
    /// if needed, and return a handle to it.
    ///
    /// The residency decision — coexist with the generation model, or evict
    /// it — is made here rather than left to the caller, because the
    /// caller (a RAG pipeline) does not know how much VRAM the card has:
    /// the same request works unchanged on a 12 GB card (where the two
    /// cannot fit together) and a 16 GB one (where they can), and only this
    /// process can tell which situation it is in right now.
    ///
    /// 1. Already loaded under this name → return it, no eviction, no load.
    /// 2. Not loaded, and it fits in free VRAM alongside whatever is in the
    ///    main slot → load it into the embedding slot; the main slot is
    ///    untouched.
    /// 3. Not loaded, and it does not fit → evict the main slot first (a
    ///    generation request will reload it later; `resolve_model` and the
    ///    embedded chat UI both work unchanged against an empty main slot),
    ///    then load the embedder, which now has the whole card.
    ///
    /// On a non-CUDA build `fit::vram_bytes()` cannot answer "does it fit",
    /// so this always takes the coexist path (case 2) and lets a real
    /// allocation failure surface as a normal load error — the same
    /// posture `--fit` itself takes on those builds.
    pub async fn ensure_embedding_model(
        &self,
        name: &str,
        n_ctx: u32,
    ) -> Result<Arc<EmbeddingModel>, ModelError> {
        let _swap_guard = self.swap_lock.lock().await;

        let normalized = normalize_model_name(name);
        let wanted = model_identity_key(&normalized);
        {
            // By the name it was loaded under, or by its file's own: the
            // name an `--embedding-model` given as a store name was loaded
            // under until it kept the one it was given.
            let slot = self.embedding.read().await;
            if let Some(ref loaded) = *slot
                && (model_identity_key(&loaded.model_name) == wanted
                    || model_identity_key(&loaded.model.path().to_string_lossy()) == wanted)
            {
                return Ok(loaded.model.clone());
            }
        }

        let gguf_path = self.resolve_model(&normalized)?;
        // The loaded model asked for under another of its names — a store
        // name for one launched by its path, or another name `eullm pull`
        // linked to the same file: loading it again would hold it twice
        // while both are in use, and put a model no longer reserved in
        // place of a reserved companion.
        {
            let slot = self.embedding.read().await;
            if let Some(ref loaded) = *slot
                && same_file(loaded.model.path(), &gguf_path)
            {
                return Ok(loaded.model.clone());
            }
        }
        let weights_bytes = std::fs::metadata(&gguf_path).map(|m| m.len()).unwrap_or(0);

        let (main_loaded, unallocated) = {
            let models = self.models.read().await;
            (!models.is_empty(), models.unallocated_reserve())
        };
        let fits_alongside = fits_in_free_vram(
            weights_bytes,
            crate::fit::EMBEDDING_COMPUTE_RESERVE_BYTES,
            unallocated,
        )
        .unwrap_or(true);
        if main_loaded && !fits_alongside {
            tracing::info!(
                "Embedding model {} does not fit alongside the loaded generation model — \
                 evicting it to make room (will reload on the next generation request)",
                crate::audit::sanitize_for_log(&normalized)
            );
            let evicted = self.unload_generation_models().await;
            self.cross_slot_evictions
                .fetch_add(evicted.len() as u64, std::sync::atomic::Ordering::Relaxed);
        }

        tracing::info!(
            "Loading embedding model {} ({})",
            crate::audit::sanitize_for_log(&normalized),
            gguf_path.display()
        );
        let threads = self.threads;
        let backend_for_load = self.backend.clone();
        let model = tokio::task::spawn_blocking(move || {
            EmbeddingModel::load(&gguf_path, threads, n_ctx, backend_for_load)
        })
        .await
            .map_err(|e| ModelError::LoadFailed(format!("Task join error: {e}")))?
            .map_err(|e| ModelError::LoadFailed(format!("Failed to load embedding model: {e}")))?;
        let model = Arc::new(model);

        {
            let mut slot = self.embedding.write().await;
            *slot = Some(EmbeddingSlot {
                model_name: normalized.clone(),
                model: model.clone(),
                // Never reserved: a model loaded ad hoc by a runtime request
                // has no launch-time guarantee behind it. Only
                // `--embedding-model` produces a reserved companion — see
                // `EmbeddingSlot::is_reserved_companion`.
                is_reserved_companion: false,
            });
        }
        tracing::info!(
            "Embedding model ready → {}",
            crate::audit::sanitize_for_log(&normalized)
        );
        Ok(model)
    }

    /// The mirror of the eviction inside `ensure_embedding_model`: called
    /// from `load_generation_model` before sizing a generation load, so an embedder
    /// left resident from a prior ingestion run does not silently shrink
    /// the VRAM budget `--fit` sizes against. Cheap when nothing is loaded
    /// (`RwLock::read` + an `Option` check) and a no-op unless `--fit` is
    /// on, since only `--fit` reads free VRAM to make a sizing decision in
    /// the first place — without it, evicting the embedder would trade a
    /// real, working configuration for a guess.
    ///
    /// Skips a **reserved companion** (`EmbeddingSlot::is_reserved_companion`
    /// — populated via `--embedding-model` at launch): that is exactly the
    /// case `--embedding-model` exists to guarantee against — the whole
    /// point of reserving its footprint up front is that it survives a
    /// later chat-model swap instead of being kicked out for one. Its
    /// footprint is subtracted from free VRAM by the caller
    /// (`load_generation_model`'s `reserve_bytes`) instead of it being evicted.
    async fn evict_embedding_if_present_for_generation_load(&self) {
        if !self.fit {
            return;
        }
        let is_reserved = self
            .embedding
            .read()
            .await
            .as_ref()
            .is_some_and(|s| s.is_reserved_companion);
        if is_reserved {
            return;
        }
        let was_loaded = self.embedding.read().await.is_some();
        if !was_loaded {
            return;
        }
        tracing::info!(
            "Generation request — evicting the resident embedding model to free VRAM for sizing \
             (reload it with a later /v1/embeddings or /api/embed request)"
        );
        *self.embedding.write().await = None;
        self.cross_slot_evictions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// The VRAM footprint to protect for a **reserved companion** (see
    /// `EmbeddingSlot::is_reserved_companion`) before sizing a generation
    /// load — `0` when no reserved companion is resident, which is every
    /// server that never used `--embedding-model` and every ad hoc
    /// `/api/embed`-loaded embedder (those are evicted outright instead,
    /// see `evict_embedding_if_present_for_generation_load`).
    ///
    /// Deliberately **not** `weights_bytes + EMBEDDING_COMPUTE_RESERVE_BYTES`:
    /// a reserved companion is always already loaded by the time this runs
    /// (eagerly, at launch — see `main.rs`), so its weights already show up
    /// as used memory in the free-VRAM figure `--fit` reads. Adding
    /// `weights_bytes` again would subtract its footprint twice and
    /// under-offload the generation model for no reason. Its context too:
    /// built at launch for its longest input and kept for every request
    /// (`EmbeddingModel::reserve_context`), it is used memory already. Only
    /// the margin is genuinely not yet reflected — what a decode allocates
    /// beside the context, in the GPU backend's scratch pool — so that is
    /// the only part worth protecting ahead of time.
    async fn reserved_embedding_bytes(&self) -> u64 {
        self.embedding
            .read()
            .await
            .as_ref()
            .filter(|s| s.is_reserved_companion)
            .map(|_| crate::fit::EMBEDDING_COMPUTE_RESERVE_BYTES)
            .unwrap_or(0)
    }

    /// Ensure the named decision model is loaded into the decision slot and
    /// return it. The same residency rules as `ensure_embedding_model` —
    /// already loaded: reuse it; fits next to the generation model: load it
    /// alongside; does not: evict the generation model first — with the VRAM
    /// a request's context needs (`fit::decision_reserve_bytes`) counted in,
    /// not only the weights.
    ///
    /// A different decision model already in the slot is dropped before the
    /// new one loads, so the free-VRAM check sees the room it leaves. A
    /// request still running on it keeps it alive until it finishes.
    pub async fn ensure_decision_model(
        &self,
        name: &str,
    ) -> Result<Arc<DecisionModel>, ModelError> {
        let _swap_guard = self.swap_lock.lock().await;

        let normalized = normalize_model_name(name);
        {
            let slot = self.decision.read().await;
            if let Some(ref loaded) = *slot
                && model_identity_key(&loaded.model_name) == model_identity_key(&normalized)
            {
                return Ok(loaded.model.clone());
            }
        }

        let gguf_path = self.resolve_model(&normalized)?;
        *self.decision.write().await = None;
        let weights_bytes = std::fs::metadata(&gguf_path).map(|m| m.len()).unwrap_or(0);
        let reserve_bytes = crate::fit::decision_reserve_bytes(&gguf_path, self.decision_ctx);

        let (main_loaded, unallocated) = {
            let models = self.models.read().await;
            (!models.is_empty(), models.unallocated_reserve())
        };
        let fits_alongside =
            fits_in_free_vram(weights_bytes.saturating_add(reserve_bytes), 0, unallocated)
                .unwrap_or(true);
        if main_loaded && !fits_alongside {
            tracing::info!(
                "Decision model {} does not fit alongside the loaded generation model — \
                 evicting it to make room (will reload on the next generation request)",
                crate::audit::sanitize_for_log(&normalized)
            );
            let evicted = self.unload_generation_models().await;
            self.cross_slot_evictions
                .fetch_add(evicted.len() as u64, std::sync::atomic::Ordering::Relaxed);
        }

        tracing::info!(
            "Loading decision model {} ({})",
            crate::audit::sanitize_for_log(&normalized),
            gguf_path.display()
        );
        let threads = self.threads;
        let max_ctx = self.decision_ctx;
        let flash_attn = self.flash_attn;
        let backend_for_load = self.backend.clone();
        let model = tokio::task::spawn_blocking(move || {
            DecisionModel::load(&gguf_path, threads, max_ctx, flash_attn, backend_for_load)
        })
        .await
        .map_err(|e| ModelError::LoadFailed(format!("Task join error: {e}")))?
        .map_err(|e| ModelError::LoadFailed(format!("Failed to load decision model: {e}")))?;
        let model = Arc::new(model);

        *self.decision.write().await = Some(DecisionSlot {
            model_name: normalized.clone(),
            model: model.clone(),
            // Loaded by a request, so no launch-time guarantee behind it —
            // see `EmbeddingSlot::is_reserved_companion`.
            is_reserved_companion: false,
            reserve_bytes,
        });
        tracing::info!(
            "Decision model ready → {}",
            crate::audit::sanitize_for_log(&normalized)
        );
        Ok(model)
    }

    /// The decision slot's counterpart of
    /// `evict_embedding_if_present_for_generation_load`, with the same
    /// conditions: only under `--fit`, never a reserved companion.
    async fn evict_decision_if_present_for_generation_load(&self) {
        if !self.fit {
            return;
        }
        let evictable = self
            .decision
            .read()
            .await
            .as_ref()
            .is_some_and(|s| !s.is_reserved_companion);
        if !evictable {
            return;
        }
        tracing::info!(
            "Generation request — evicting the resident decision model to free VRAM for sizing \
             (reload it with a later /v1/systemone request)"
        );
        *self.decision.write().await = None;
        self.cross_slot_evictions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Free the context the loaded decision model keeps between requests
    /// (`DecisionModel::release_context`), so a free-VRAM figure measured
    /// next shows its weights as used and its context as free — the context
    /// being what `reserved_decision_bytes` counts. The next decision
    /// request creates it again. Waits for a request in progress to finish.
    async fn release_decision_context(&self) {
        let model = self
            .decision
            .read()
            .await
            .as_ref()
            .map(|slot| Arc::clone(&slot.model));
        if let Some(model) = model {
            let _ = tokio::task::spawn_blocking(move || model.release_context()).await;
        }
    }

    /// The decision slot's counterpart of `reserved_embedding_bytes`: a
    /// reserved companion's per-request context, which the free-VRAM figure
    /// does not show once `release_decision_context` has run. Its weights
    /// do, so they are not counted again.
    async fn reserved_decision_bytes(&self) -> u64 {
        self.decision
            .read()
            .await
            .as_ref()
            .filter(|s| s.is_reserved_companion)
            .map(|s| s.reserve_bytes)
            .unwrap_or(0)
    }

    /// Reset the embedding slot's idle-unload deadline. Called on every
    /// request that uses the slot, so an active ingestion run is never
    /// unloaded out from under it. `Immediate` empties the slot right away:
    /// the request holds its own `Arc` to the model and finishes on it. See
    /// `KeepAlive`.
    pub async fn touch_embedding_slot(&self, keep_alive: KeepAlive) {
        touch_deadline(
            &self.embedding_deadline,
            keep_alive,
            self.default_keep_alive,
        );
        if keep_alive == KeepAlive::Immediate {
            *self.embedding.write().await = None;
        }
    }

    /// Same as `touch_embedding_slot`, for the decision slot.
    pub async fn touch_decision_slot(&self, keep_alive: KeepAlive) {
        touch_deadline(&self.decision_deadline, keep_alive, self.default_keep_alive);
        if keep_alive == KeepAlive::Immediate {
            *self.decision.write().await = None;
        }
    }

    /// Background loop spawned once from `serve()`: unload whatever is due.
    /// It looks every `idle_tick` (30 s), which is coarse on purpose — this
    /// is a power-saving idle timer, not a latency-sensitive path, and
    /// checking on every request would mean taking the slot locks on every
    /// single call for a comparison that is false almost all the time — and
    /// also whenever the generation model's last request finishes, so that
    /// `keep_alive: 0` takes effect when the request is over rather than up
    /// to 30 s later.
    async fn run_idle_unload_loop(self: Arc<Self>) {
        loop {
            // Registered before anything is checked: a request that ends
            // while this pass runs wakes the next pass instead of being
            // missed until the tick after.
            let mut went_idle = std::pin::pin!(self.idle.notified());
            went_idle.as_mut().enable();

            self.unload_due_generation_models().await;

            let now = tokio::time::Instant::now();
            let embedding_expired = {
                let mut deadline = self.embedding_deadline.lock().await;
                let expired = deadline.is_some_and(|d| now >= d);
                if expired {
                    *deadline = None;
                }
                expired
            };
            if embedding_expired {
                tracing::info!("keep_alive expired — unloading idle embedding model");
                *self.embedding.write().await = None;
            }

            let decision_expired = {
                let mut deadline = self.decision_deadline.lock().await;
                let expired = deadline.is_some_and(|d| now >= d);
                if expired {
                    *deadline = None;
                }
                expired
            };
            if decision_expired {
                tracing::info!("keep_alive expired — unloading idle decision model");
                *self.decision.write().await = None;
            }

            tokio::select! {
                () = went_idle.as_mut() => {}
                () = tokio::time::sleep(self.idle_tick) => {}
            }
        }
    }

    /// Resolve a model name to a GGUF file path.
    ///
    /// Search order:
    /// 1. Direct GGUF file path — **only** when `allow_model_paths` is set, or
    ///    when it is the path this process was launched with
    /// 2. Directory containing a single .gguf file — same condition
    /// 3. Path without extension — try appending `.gguf`, same condition
    /// 4. Well-known container mount points (`/models`, `/data/models`)
    /// 5. Exact name in model store (`~/.eullm/models/{name}/*.gguf`)
    /// 6. Normalized name (Ollama tags: `qwen3:14b` → `qwen3-14b`)
    ///
    /// # Why steps 1–3 are gated
    ///
    /// The `model` field of an API request used to be handed straight to
    /// `PathBuf::from` and accepted if `is_file()`. Combined with an
    /// unauthenticated API that made every readable file on the host a valid
    /// model name. A caller could not read the file's contents back, but the
    /// error messages distinguished "not found" from "found but failed to
    /// load", which is a working oracle for probing the filesystem — and
    /// pointing the loader at, say, a 40 GB file is a denial of service on its
    /// own. Names that resolve inside the model store or a deliberate mount
    /// point cover every documented workflow; arbitrary paths are opt-in via
    /// `EULLM_ALLOW_MODEL_PATHS=1`.
    ///
    /// The launch path is always accepted regardless: `/api/tags` reports it as
    /// the loaded model's name, clients echo back what they were told, and
    /// refusing our own answer would break `eullm run ./model.gguf`.
    fn resolve_model(&self, name: &str) -> Result<PathBuf, ModelError> {
        let path = PathBuf::from(name);

        // 0. The model this process was launched with, by the name the API
        //    advertises for it or by its literal path. Exact match on either —
        //    never a stem or prefix comparison, which would turn this
        //    allowance into a way to reach any similarly named file.
        if let Some((launch_name, launch_path)) = &self.launch_model
            && (name == launch_name.as_str() || &path == launch_path)
            && launch_path.is_file()
        {
            return Ok(launch_path.clone());
        }

        if self.allow_model_paths {
            // 1. Direct GGUF file path?
            if path.is_file() {
                return Ok(path);
            }

            // 2. Directory containing .gguf files? Pick the first one.
            if path.is_dir()
                && let Some(gguf) = find_gguf_in_dir(&path)
            {
                return Ok(gguf);
            }

            // 3. Try appending .gguf extension.
            let with_ext = path.with_extension("gguf");
            if with_ext.is_file() {
                return Ok(with_ext);
            }
        }

        // 4. Try common model directories (Docker volumes, etc.). These are
        //    deliberate mount points, not arbitrary paths, so they stay
        //    available without the opt-in — but only a plain file name may be
        //    joined onto them, or `../` would walk straight back out.
        if crate::models::store::is_safe_filename(&format!("{name}.gguf")) {
            for dir in &["/models", "/data/models"] {
                let candidate = PathBuf::from(dir).join(format!("{name}.gguf"));
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }

        // 5. Exact name in model store.
        if let Some(p) = self.store.gguf_path(name) {
            return Ok(p);
        }

        // 5. Try normalized name (Ollama tag format).
        let normalized = normalize_model_name(name);
        if normalized != name
            && let Some(p) = self.store.gguf_path(&normalized)
        {
            return Ok(p);
        }

        Err(ModelError::NotFound(format!(
            "Model '{name}' not found. Accepted formats:\n  \
             - GGUF file path: /models/model.gguf\n  \
             - Directory with GGUF: /models/mymodel/\n  \
             - Registered name: eullm import-ollama {name}"
        )))
    }
}

/// When `AppState::remove_generation` may take a model out.
#[derive(Debug, Clone, Copy)]
enum Removal {
    /// Whatever it is doing: requests still running on it are cut off.
    Always,
    /// Only if it is still due — idle, and its keep_alive over.
    IfDue,
}

/// Free a generation model taken out of the residents, so that its memory
/// is free when this returns — the next load is sized against what is free.
///
/// A scheduler is stopped, and its thread joined on a blocking thread: the
/// join waits for the decode loop to notice. A sequential engine has no
/// thread to join; it is freed when the last `Arc` to it goes, and every
/// request running on it holds one. Those are waited for (F4): taking it out
/// of the residents used to be all, so the next load measured free VRAM with
/// the old weights still in it.
async fn retire(model: resident::LoadedModel) {
    if let Some(handle) = model.scheduler
        && let Err(e) = tokio::task::spawn_blocking(move || handle.shutdown()).await
    {
        tracing::warn!("Failed to join scheduler thread: {e}");
    }
    if let Some(engine) = model.engine {
        let released = resident::wait_for_release(
            &engine,
            &model.usage,
            ENGINE_RELEASE_LIMIT,
            ENGINE_RELEASE_GRACE,
        )
        .await;
        if !released {
            tracing::warn!(
                "{} is still held elsewhere — the terminal chat, or a request \
                 whose client left before it finished; its memory is freed when \
                 that ends",
                crate::audit::sanitize_for_log(&model.name)
            );
        }
    }
}

/// The longest a sequential engine's in-flight requests are waited for
/// before its memory is given up on (see `retire`): a load waits behind it.
const ENGINE_RELEASE_LIMIT: std::time::Duration = std::time::Duration::from_secs(30);

/// How long a sequential engine still held after its last request ended is
/// waited for: the moment the blocking thread that ran the request takes to
/// let go. A holder that outlasts it is not a request — `eullm run`'s
/// terminal chat holds the launch model for as long as it runs.
const ENGINE_RELEASE_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// Find the first `.gguf` file in a directory.
fn find_gguf_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() && p.extension().is_some_and(|e| e == "gguf") {
            return Some(p);
        }
    }
    None
}

/// Whether two paths name the same file: through the symlinks a model store
/// is often reached by (`~/.eullm/models` linked to a data disk), and, where
/// the filesystem says so, through the hard links `eullm pull` gives a model
/// pulled under a second name.
fn same_file(a: &std::path::Path, b: &std::path::Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(a), Ok(b)) = (std::fs::metadata(a), std::fs::metadata(b)) {
            return a.dev() == b.dev() && a.ino() == b.ino();
        }
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

#[cfg(test)]
mod fits_tests {
    use super::fits_in;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    /// A sequential generation model creates its context per request, so
    /// while it is idle that memory looks free. An embedder sized into it
    /// left the next image request without room for its context (F5).
    #[test]
    fn a_sequential_resident_reserves_its_context_for_a_companion() {
        // 16 GiB card, 8 GiB free; the floor keeps 12% of it back.
        let card = (8 * GIB, 16 * GIB);
        let embedder = 2 * GIB;
        assert_eq!(fits_in(card, embedder, 256 * MIB, 0), Some(true));
        let vision_model_context = 4 * GIB;
        assert_eq!(
            fits_in(card, embedder, 256 * MIB, vision_model_context),
            Some(false)
        );
        // Saturating, not wrapping, when the reservations exceed what is free.
        assert_eq!(fits_in(card, 0, 256 * MIB, 64 * GIB), Some(true));
        assert_eq!(fits_in(card, 1, 256 * MIB, 64 * GIB), Some(false));
    }
}

#[cfg(all(test, unix))]
mod same_file_tests {
    use super::same_file;

    #[test]
    fn a_file_is_the_same_through_symlinks_and_hard_links() {
        let root = std::env::temp_dir().join(format!("eullm-same-file-{}", uuid::Uuid::new_v4()));
        let disk = root.join("disk");
        std::fs::create_dir_all(disk.join("emb")).unwrap();
        std::fs::write(disk.join("emb/Model-Q8_0.gguf"), b"GGUF").unwrap();
        std::fs::write(disk.join("emb/Other-Q8_0.gguf"), b"GGUF").unwrap();
        std::os::unix::fs::symlink(&disk, root.join("models")).unwrap();

        let direct = disk.join("emb/Model-Q8_0.gguf");
        std::fs::create_dir_all(disk.join("alias")).unwrap();
        std::fs::hard_link(&direct, disk.join("alias/Model-Q8_0.gguf")).unwrap();
        assert!(same_file(&direct, &root.join("models/emb/Model-Q8_0.gguf")));
        assert!(same_file(&direct, &disk.join("emb/../emb/Model-Q8_0.gguf")));
        assert!(same_file(&direct, &root.join("models/alias/Model-Q8_0.gguf")));
        assert!(!same_file(&direct, &disk.join("emb/Other-Q8_0.gguf")));
        assert!(!same_file(&direct, &root.join("missing.gguf")));
        std::fs::remove_dir_all(&root).unwrap();
    }
}

/// Whether `additional_bytes` fits in currently free VRAM, applying the same
/// floor `fit.rs` reserves for a normal model load
/// (`fit::MIN_FREE_TOTAL_RATIO`) plus `compute_reserve_bytes` for the
/// model's own compute buffer — `fit::EMBEDDING_COMPUTE_RESERVE_BYTES` for an
/// embedder, 256 MiB rather than `fit.rs`'s 640 MiB, since an embedding
/// model's context and micro-batch are both a fraction of an LLM's. A
/// decision model passes 0 and counts its whole per-request context in
/// `additional_bytes` instead (`fit::decision_reserve_bytes`).
///
/// Deliberately not the layer-by-layer machinery in `fit.rs`: an embedding
/// model loads fully onto the GPU or not at all (see `EmbeddingModel::load`),
/// so this only ever needs a yes/no answer, never a partial split.
///
/// `None` when VRAM cannot be probed at all (non-CUDA build) — the caller
/// decides what "unknown" means for it; `ensure_embedding_model` treats it as
/// "assume yes" so a build that cannot measure VRAM behaves as it always has,
/// letting a real allocation failure surface as a normal load error.
///
/// `unallocated_bytes` is memory the free figure shows but is already
/// spoken for: the contexts sequential residents create per request (F5).
fn fits_in_free_vram(
    additional_bytes: u64,
    compute_reserve_bytes: u64,
    unallocated_bytes: u64,
) -> Option<bool> {
    fits_in(
        crate::fit::vram_bytes()?,
        additional_bytes,
        compute_reserve_bytes,
        unallocated_bytes,
    )
}

/// [`fits_in_free_vram`] against a given `(free, total)`.
fn fits_in(
    (free, total): (u64, u64),
    additional_bytes: u64,
    compute_reserve_bytes: u64,
    unallocated_bytes: u64,
) -> Option<bool> {
    let floor = (total as f64 * crate::fit::MIN_FREE_TOTAL_RATIO) as u64;
    let usable = free
        .saturating_sub(floor)
        .saturating_sub(compute_reserve_bytes)
        .saturating_sub(unallocated_bytes);
    Some(additional_bytes <= usable)
}

/// How long a loaded model should be kept resident after a request, decoded
/// from a request's `keep_alive` field (Ollama's field of the same name and
/// meaning) — a pure function over the parsed JSON value, testable without a
/// running server, per the perimeter-config convention in
/// `engine/CLAUDE.md` (even though this is not a perimeter setting, the same
/// reason applies: precedence logic belongs in a function that can be
/// tested directly).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepAlive {
    /// No `keep_alive` in the request — use the server's `--keep-alive`
    /// default (`AppState::default_keep_alive`), which may itself be "never
    /// idle-unload" if the server was started without the flag.
    Default,
    /// `keep_alive: -1` (or any negative number/duration) — never
    /// idle-unload this load; the deadline is cleared rather than set.
    Forever,
    /// `keep_alive: 0` — unload right after this request completes, instead
    /// of waiting for the idle timer.
    Immediate,
    /// A positive duration to keep the model resident, counted from the end
    /// of this request.
    For(std::time::Duration),
}

impl KeepAlive {
    /// `Default` replaced by what it stands for: the server's `--keep-alive`
    /// (`default`), or `Forever` when none was given. Every other value is
    /// the request's own and is returned as it is.
    pub(crate) fn resolve(self, default: Option<std::time::Duration>) -> KeepAlive {
        match self {
            KeepAlive::Default => default.map_or(KeepAlive::Forever, KeepAlive::For),
            other => other,
        }
    }
}

/// Parse a request body's `keep_alive` field. Accepts what Ollama accepts:
/// a bare number of seconds (`300`), a duration string with a unit
/// (`"5m"`, `"30s"`, `"2h"`), or a plain numeric string (`"300"`). An
/// absent field, `null`, or anything unparseable falls back to `Default`
/// rather than erroring — a malformed `keep_alive` should not fail the
/// request it rides along with.
pub fn parse_keep_alive(value: Option<&serde_json::Value>) -> KeepAlive {
    let Some(value) = value else {
        return KeepAlive::Default;
    };
    let seconds = if let Some(n) = value.as_f64() {
        Some(n)
    } else if let Some(s) = value.as_str() {
        parse_duration_string(s)
    } else {
        None
    };
    match seconds {
        None => KeepAlive::Default,
        Some(s) if s < 0.0 => KeepAlive::Forever,
        Some(0.0) => KeepAlive::Immediate,
        // `from_secs_f64` panics on NaN, infinity, and magnitudes past what
        // a Duration holds — all reachable from a request body so go
        // through the standard library's checked constructor and take the
        // malformed-value fallback for whatever it refuses.
        Some(s) => match std::time::Duration::try_from_secs_f64(s) {
            Ok(d) => KeepAlive::For(d),
            Err(_) => KeepAlive::Default,
        },
    }
}

/// Parse `--keep-alive`'s CLI value into a duration, for `main.rs` to build
/// `ServeConfig::keep_alive` — the same duration grammar `parse_keep_alive`
/// accepts on a request's `keep_alive` field ("5m", "30s", "300"), minus the
/// 0/negative special cases: a *default* keep-alive of "unload immediately"
/// or "never" is nonsensical (the former means every load evicts itself, the
/// latter is just the flag being absent), so those are rejected here rather
/// than silently accepted and misread as `Duration::ZERO`.
pub fn parse_keep_alive_flag(s: &str) -> Result<std::time::Duration, String> {
    match parse_duration_string(s) {
        Some(secs) if secs > 0.0 => match std::time::Duration::try_from_secs_f64(secs) {
            Ok(d) => Ok(d),
            Err(_) => Err(format!(
                "--keep-alive must be a positive duration, got '{s}' \
                 (the value is too large to represent as a duration)"
            )),
        },
        Some(_) => Err(format!(
            "--keep-alive must be a positive duration, got '{s}' \
             (0 or negative only make sense as a per-request keep_alive override)"
        )),
        None => Err(format!(
            "--keep-alive: cannot parse '{s}' as a duration — expected e.g. '5m', '30s', '2h', or a bare number of seconds"
        )),
    }
}

/// `"300"` → 300.0, `"5m"` → 300.0, `"1.5h"` → 5400.0. Returns `None` for
/// anything that is not a bare number optionally followed by one of
/// `s`/`m`/`h`.
fn parse_duration_string(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        // An empty (or whitespace-only) string has no number to split off:
        // without this, the split below underflows.
        return None;
    }
    if let Ok(n) = s.parse::<f64>() {
        return Some(n);
    }
    // Split off the last CHARACTER, not the last byte. `s.len() - 1` lands
    // inside a multi-byte one and panics the handler task: `"¡"` is the
    // smallest input that does it, and a `keep_alive` of `"5à"` or `"30s€"`
    // arrives from a request body like any other string. The empty case is
    // already out above, so there is always a last character to measure.
    let split = s.len() - s.chars().next_back().map_or(0, char::len_utf8);
    let (number, unit) = s.split_at(split);
    let n: f64 = number.parse().ok()?;
    match unit {
        "s" => Some(n),
        "m" => Some(n * 60.0),
        "h" => Some(n * 3600.0),
        _ => None,
    }
}

/// Shared implementation behind `touch_embedding_slot`/`touch_decision_slot`:
/// resolve `keep_alive` (falling back to `default` when it is `Default`)
/// into a new deadline, or clear it for `Forever`/`Immediate` (the caller
/// handles the actual unload for `Immediate`).
fn touch_deadline(
    deadline: &tokio::sync::Mutex<Option<tokio::time::Instant>>,
    keep_alive: KeepAlive,
    default: Option<std::time::Duration>,
) {
    let new_deadline = match keep_alive.resolve(default) {
        // `checked_add`: a duration a request may carry (up to ~584 billion
        // years) overflows an `Instant`, and `+` panics on that. A deadline
        // that far away is no deadline.
        KeepAlive::For(d) => tokio::time::Instant::now().checked_add(d),
        KeepAlive::Forever | KeepAlive::Immediate | KeepAlive::Default => None,
    };
    // `try_lock`: this runs on the hot request path (once per request, to
    // reset the idle timer) and must never block a response on the idle
    // loop's own lock acquisition, which happens at most once every 30s and
    // holds the lock only briefly. Losing a single deadline reset to a rare
    // collision is harmless — the next request resets it again, and the
    // idle loop only unloads a slot that has had no reset for the entire
    // keep_alive window.
    if let Ok(mut guard) = deadline.try_lock() {
        *guard = new_deadline;
    }
}

#[cfg(test)]
mod keep_alive_tests {
    use super::*;
    use std::time::Duration;

    fn v(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("valid json literal")
    }

    #[test]
    fn an_absent_field_is_the_server_default() {
        assert_eq!(parse_keep_alive(None), KeepAlive::Default);
    }

    #[test]
    fn a_bare_number_is_seconds() {
        assert_eq!(
            parse_keep_alive(Some(&v("300"))),
            KeepAlive::For(Duration::from_secs(300))
        );
    }

    #[test]
    fn duration_strings_match_ollamas_grammar() {
        assert_eq!(
            parse_keep_alive(Some(&v("\"30s\""))),
            KeepAlive::For(Duration::from_secs(30))
        );
        assert_eq!(
            parse_keep_alive(Some(&v("\"5m\""))),
            KeepAlive::For(Duration::from_secs(300))
        );
        assert_eq!(
            parse_keep_alive(Some(&v("\"2h\""))),
            KeepAlive::For(Duration::from_secs(7200))
        );
        // A numeric string with no unit is still seconds.
        assert_eq!(
            parse_keep_alive(Some(&v("\"300\""))),
            KeepAlive::For(Duration::from_secs(300))
        );
    }

    /// `0` means "unload right after this request" — distinct from an
    /// absent field, which means "use the server default" and may well
    /// keep the model resident.
    #[test]
    fn zero_means_immediate() {
        assert_eq!(parse_keep_alive(Some(&v("0"))), KeepAlive::Immediate);
        assert_eq!(parse_keep_alive(Some(&v("\"0\""))), KeepAlive::Immediate);
    }

    /// Negative, in either form, means "never idle-unload this load" —
    /// Ollama's `-1`.
    #[test]
    fn negative_means_forever() {
        assert_eq!(parse_keep_alive(Some(&v("-1"))), KeepAlive::Forever);
        assert_eq!(parse_keep_alive(Some(&v("\"-1\""))), KeepAlive::Forever);
    }

    /// A malformed value must not fail the request it rides along with —
    /// it falls back to the server default rather than erroring.
    #[test]
    fn garbage_falls_back_to_default_rather_than_erroring() {
        assert_eq!(parse_keep_alive(Some(&v("\"banana\""))), KeepAlive::Default);
        assert_eq!(parse_keep_alive(Some(&v("null"))), KeepAlive::Default);
        assert_eq!(parse_keep_alive(Some(&v("true"))), KeepAlive::Default);
    }

    /// An empty (or whitespace-only) string is malformed like any other
    /// garbage — the documented fallback above — except it used to never get
    /// there: `split_at(s.len() - 1)` underflows on a zero-length string and
    /// panics the request task instead.
    #[test]
    fn an_empty_string_falls_back_to_default_rather_than_panicking() {
        assert_eq!(parse_keep_alive(Some(&v("\"\""))), KeepAlive::Default);
        assert_eq!(parse_keep_alive(Some(&v("\"   \""))), KeepAlive::Default);
        assert!(parse_keep_alive_flag("").is_err());
        assert!(parse_keep_alive_flag("   ").is_err());
    }

    /// Absurd magnitudes must behave like any other malformed value rather
    /// than panicking inside `from_secs_f64`: NaN, infinity, and anything
    /// past what a `Duration` holds are all reachable from a request body
    /// (`"nan"`/`"inf"` parse as floats; JSON numbers have no range check).
    #[test]
    fn absurd_durations_fall_back_to_default_rather_than_panicking() {
        for raw in ["1e20", "1e30", "\"1e30\"", "\"nan\"", "\"inf\""] {
            assert_eq!(parse_keep_alive(Some(&v(raw))), KeepAlive::Default);
        }
        assert_eq!(
            parse_keep_alive(Some(&v("300"))),
            KeepAlive::For(Duration::from_secs(300))
        );
        for s in ["1e30", "inf", "nan"] {
            assert!(parse_keep_alive_flag(s).is_err());
        }
        assert_eq!(
            parse_keep_alive_flag("5m").unwrap(),
            Duration::from_secs(300)
        );
    }

    #[test]
    fn the_cli_flag_parser_accepts_only_a_positive_duration() {
        assert_eq!(
            parse_keep_alive_flag("5m").unwrap(),
            Duration::from_secs(300)
        );
        assert!(
            parse_keep_alive_flag("0").is_err(),
            "0 as a *default* keep-alive is nonsensical — every load would evict itself"
        );
        assert!(parse_keep_alive_flag("-1").is_err());
        assert!(parse_keep_alive_flag("banana").is_err());
    }

    #[test]
    fn touch_deadline_resolves_default_against_the_servers_own_default() {
        let deadline = tokio::sync::Mutex::new(None);
        // No server default configured (`None`) → Default resolves to
        // Forever, i.e. no timer at all — matches every release before
        // --keep-alive existed, where nothing unloaded a model on its own.
        touch_deadline(&deadline, KeepAlive::Default, None);
        assert!(
            deadline.try_lock().unwrap().is_none(),
            "no --keep-alive configured means Default must not start a timer"
        );

        // A server default of 5 minutes turns Default into an actual deadline.
        touch_deadline(&deadline, KeepAlive::Default, Some(Duration::from_secs(300)));
        assert!(deadline.try_lock().unwrap().is_some());
    }

    /// 1e19 seconds is a `Duration`, so it parses, and is not an `Instant`
    /// away from now: adding it panicked the embedding and decision handlers.
    #[test]
    fn a_keep_alive_past_the_end_of_time_is_no_deadline_not_a_panic() {
        let huge = parse_keep_alive(Some(&v("1e19")));
        assert!(matches!(huge, KeepAlive::For(_)), "{huge:?}");
        let deadline = tokio::sync::Mutex::new(None);
        touch_deadline(&deadline, huge, None);
        assert!(deadline.try_lock().unwrap().is_none());
    }

    #[test]
    fn touch_deadline_forever_and_immediate_both_clear_any_running_timer() {
        let deadline = tokio::sync::Mutex::new(Some(tokio::time::Instant::now()));
        touch_deadline(&deadline, KeepAlive::Forever, Some(Duration::from_secs(300)));
        assert!(deadline.try_lock().unwrap().is_none());

        let deadline = tokio::sync::Mutex::new(Some(tokio::time::Instant::now()));
        touch_deadline(&deadline, KeepAlive::Immediate, Some(Duration::from_secs(300)));
        assert!(deadline.try_lock().unwrap().is_none());
    }

    /// A duration whose last character is multi-byte used to panic the
    /// handler task: the unit was split off by byte index, and `s.len() - 1`
    /// lands inside such a character. The empty-string guard above covered
    /// only the length-zero case. Found by the property below, which shrank
    /// it to a single `"¡"`; kept here by name because a failing list reads
    /// better than a seed hash.
    #[test]
    fn a_multibyte_tail_is_malformed_not_a_panic() {
        for raw in ["¡", "5à", "30s€", "1h☃", "¡¡¡"] {
            assert_eq!(
                parse_keep_alive(Some(&serde_json::json!(raw))),
                KeepAlive::Default
            );
            assert!(parse_keep_alive_flag(raw).is_err());
        }
        // The ASCII grammar is untouched.
        assert_eq!(
            parse_keep_alive_flag("5m").unwrap(),
            Duration::from_secs(300)
        );
        assert_eq!(
            parse_keep_alive_flag("30s").unwrap(),
            Duration::from_secs(30)
        );
        assert_eq!(
            parse_keep_alive_flag("2h").unwrap(),
            Duration::from_secs(7200)
        );
    }

    // ── Properties ───────────────────────────────────────────────────────
    //
    // The tests above name values somebody thought of. These name the rules
    // instead, and let proptest hunt for the value that breaks one. Both
    // panics fixed in this function during September were a value nobody had
    // thought to write down: an empty string (#454) and `1e20` (#483). A
    // property would have produced each of them on the first run.
    //
    // `keep_alive` arrives inside a request body, so the strategy generates
    // what a body can actually carry: any JSON scalar, and strings both
    // arbitrary and duration-shaped. Note that NaN and infinity cannot be
    // JSON *numbers* — `serde_json` has no representation for them — which is
    // exactly why `"nan"` and `"inf"` have to be reachable as strings.
    use proptest::prelude::*;

    /// Any JSON scalar a `keep_alive` field can hold.
    fn any_keep_alive_value() -> impl Strategy<Value = serde_json::Value> {
        prop_oneof![
            proptest::num::f64::ANY.prop_map(|f| serde_json::json!(f)),
            any::<i64>().prop_map(|i| serde_json::json!(i)),
            ".*".prop_map(|s: String| serde_json::json!(s)),
            // Duration-shaped strings, the grammar the parser documents.
            (
                any::<f64>(),
                prop_oneof![Just(""), Just("s"), Just("m"), Just("h")]
            )
                .prop_map(|(n, u)| serde_json::json!(format!("{n}{u}"))),
            any::<bool>().prop_map(|b| serde_json::json!(b)),
            Just(serde_json::Value::Null),
        ]
    }

    proptest! {
        /// Whatever a request body carries, this returns — it never takes the
        /// handler task down with it. The documented contract is that a
        /// malformed value falls back to the server default; a panic is not a
        /// fallback.
        #[test]
        fn parse_keep_alive_never_panics(v in any_keep_alive_value()) {
            let _ = parse_keep_alive(Some(&v));
        }

        /// Sign decides the variant, and nothing else does. Negative means
        /// "never unload", zero means "unload now" — for every negative and
        /// every zero, not just the ones in the examples above.
        #[test]
        fn sign_alone_decides_forever_and_immediate(f in proptest::num::f64::NEGATIVE) {
            prop_assert_eq!(parse_keep_alive(Some(&serde_json::json!(f))), KeepAlive::Forever);
            prop_assert_eq!(parse_keep_alive(Some(&serde_json::json!(0.0))), KeepAlive::Immediate);
        }

        /// The CLI flag answers or refuses, never panics.
        #[test]
        fn parse_keep_alive_flag_never_panics(s in ".*") {
            let _ = parse_keep_alive_flag(&s);
        }

        /// And when it answers, the answer is a positive duration. Accepting
        /// zero here is the bug the flag parser exists to prevent: it would be
        /// read as `Duration::ZERO` and evict every model the moment it loads.
        #[test]
        fn the_flag_never_accepts_a_non_positive_duration(s in ".*") {
            if let Ok(d) = parse_keep_alive_flag(&s) {
                prop_assert!(d > Duration::ZERO, "accepted {s:?} as {d:?}");
            }
        }
    }
}

/// Normalize an Ollama-style model name for EULLM's store.
///
/// Ollama uses `name:tag` (e.g. `qwen3:14b`), but EULLM stores models
/// with dashes (e.g. `qwen3-14b`).  This converts `:` → `-` so that
/// API requests using Ollama naming conventions find the right model.
fn normalize_model_name(name: &str) -> String {
    name.replace(':', "-")
}

/// Canonical comparison key for "is this model already loaded?" checks: the
/// last path component (a loaded model may be a full `.gguf` path), with a
/// `.gguf` extension stripped, compared case-insensitively.
///
/// Deliberately NOT `Path::file_stem`. Model names legitimately contain dots
/// (`qwen3.6-27b`, `ornith-1.0-35b-gguf-ud-q5_k_xl`), and `file_stem` cuts
/// at the LAST dot, which collapsed every `ornith-1.*` quant into the same
/// `ornith-1` identity — reported as #345: switching between two quants of
/// the same repo was a silent no-op because the swap believed the requested
/// model was already loaded.
pub(crate) fn model_identity_key(name: &str) -> String {
    let last = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let base = match last.char_indices().rev().nth(4) {
        Some((i, _)) if last[i..].eq_ignore_ascii_case(".gguf") => &last[..i],
        _ => last,
    };
    base.to_ascii_lowercase()
}

/// Check if a loaded model name matches a requested name.
///
/// Handles the common case where the loaded model is a full path
/// (e.g. `/models/qwen3-8b.gguf`) but the request uses a short name
/// (e.g. `qwen3-8b` or `qwen3:8b`).
pub(crate) fn model_names_match(loaded: &str, normalized_request: &str) -> bool {
    // Exact match.
    if loaded == normalized_request {
        return true;
    }
    // Otherwise compare identity keys: last path component, `.gguf` stripped,
    // case-insensitive. See `model_identity_key` for why this is not
    // `file_stem` — model names contain dots, and cutting at the last one
    // made every quant of a repo look like the same model (#345).
    model_identity_key(loaded) == model_identity_key(normalized_request)
}

/// Configuration for starting the API server.
pub struct ServeConfig {
    /// The one `LlamaBackend` the process created at startup — see
    /// `AppState::backend`. Every model this server ever loads, launch
    /// model included, must share this same instance.
    pub backend: Arc<LlamaBackend>,
    pub port: u16,
    /// See `AppState::fallback_mmproj`.
    pub mmproj: Option<PathBuf>,
    /// See `AppState::mmproj_offload`.
    pub mmproj_offload: Option<bool>,
    pub model_name: Option<String>,
    pub engine: Option<Arc<InferenceEngine>>,
    pub scheduler: Option<SchedulerHandle>,
    pub gpu_layers: i32,
    /// Auto-size the GPU offload before every model load (see
    /// `AppState::fit`). Pass the user's `--fit` flag, never a value a
    /// previous fit computed.
    pub fit: bool,
    /// With `fit`, refuse a load that does not fully fit instead of
    /// offloading a partial split (see `AppState::fit_strict`).
    pub fit_strict: bool,
    pub ctx_size: u32,
    pub threads: u32,
    pub flash_attn: bool,
    pub n_batch: u32,
    pub cache_type_k: crate::inference::KvCacheType,
    pub cache_type_v: crate::inference::KvCacheType,
    pub batch_size: usize,
    pub cpu_moe: bool,
    pub n_cpu_moe: u32,
    pub rs_seq: u32,
    pub ctx_checkpoints: usize,
    pub checkpoint_min_step: u32,
    /// Enable extra internal diagnostics for the Rust engine layer (NaN/Inf
    /// logit scan per token — see `SchedulerConfig::debug_logit_check`).
    /// Applied to every model this server loads or swaps to. Off by
    /// default: zero added per-token cost, matches upstream llama.cpp.
    pub rust_debug: bool,
    pub web_enabled: bool,
    pub store: ModelStore,
    /// Optional embedded chat UI. When `Some(port)`, a second listener is
    /// spawned on that port serving the chat at `/` (plus the API on the
    /// same port for same-origin fetches). When `None`, only the API
    /// listener on `cfg.port` is started — pure API surface, nothing on `/`.
    pub ui_port: Option<u16>,
    /// The `(advertised name, GGUF path)` this process was launched with, when
    /// it was launched with a model (`eullm run`). `None` for headless `serve`,
    /// which starts with an empty slot. See `AppState::launch_model`.
    pub launch_model: Option<(String, PathBuf)>,
    /// Default idle-unload duration for a load whose request did not set its
    /// own `keep_alive` — see `AppState::default_keep_alive`. `None` (the
    /// default) means never idle-unload automatically, matching every
    /// release before this flag existed.
    pub keep_alive: Option<std::time::Duration>,
    /// An embedding model already loaded via `--embedding-model`, ready to
    /// seed the embedding slot at startup instead of waiting for the first
    /// `/api/embed`/`/v1/embeddings` request. Its
    /// `EmbeddingSlot::is_reserved_companion` must be `true` — the caller
    /// (`cmd_run`/`cmd_serve`) already reserved its footprint against the
    /// generation model's own `--fit` sizing before loading it, and that
    /// guarantee only means anything if it is then treated as reserved here
    /// too. `None` when `--embedding-model` was not given — the ordinary
    /// case, unaffected by any of this.
    pub launch_embedding: Option<EmbeddingSlot>,
    /// The `--decision-model` counterpart of `launch_embedding`.
    pub launch_decision: Option<DecisionSlot>,
    /// `--decision-ctx`: see `AppState::decision_ctx`.
    pub decision_ctx: u32,
}

/// Start the API server on the given port with graceful shutdown support.
///
/// The server shuts down cleanly on SIGTERM or SIGINT (Ctrl+C), finishing
/// in-flight requests before exiting. This is critical for Docker containers
/// (which send SIGTERM on `docker stop`) and systemd services.
pub async fn serve(cfg: ServeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let env_file = std::path::Path::new(".env");
    let ip_allowlist = ip_allowlist::IpAllowlist::load(env_file);

    // Authentication first: configuration that is present but unusable is fatal
    // here. An operator who set EULLM_API_KEYS asked for authentication, and
    // starting an open API because of a typo in it is the one outcome that must
    // not be possible.
    let api_keys = Arc::new(auth::ApiKeys::load(env_file).map_err(|e| {
        format!(
            "{e}\n  Expected id:secret[:rpm=N] entries, comma-separated. \
             Refusing to start: serving without the authentication you configured \
             would be worse than not starting."
        )
    })?);
    let allowed_origins = origin::AllowedOrigins::load(env_file);
    let web_policy = crate::tools::guard::WebPolicy::from_env();
    let allow_model_paths = matches!(
        std::env::var("EULLM_ALLOW_MODEL_PATHS")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    );

    if api_keys.is_enabled() {
        tracing::info!(
            "API authentication: enabled — keys: {}  [source: {}]",
            api_keys.describe(),
            api_keys.source(),
        );
        // State this explicitly. It is the one place where enabling a control
        // relaxes another, and an operator discovering it from behaviour rather
        // than from a log line is how a deployment ends up unintentionally open.
        tracing::info!(
            "A valid API key admits a request from any source address; the IP allowlist \
             below then applies only to requests without a key (which are refused with 401)."
        );
    } else {
        tracing::info!(
            "API authentication: disabled ({}). Set EULLM_API_KEYS=id:secret to require \
             a bearer token — necessary behind Docker's published ports, where every \
             external client arrives as the bridge gateway address.",
            api_keys.source()
        );
    }
    tracing::info!(
        "Allowed source IPs/subnets: {}  [source: {}]",
        ip_allowlist.describe(),
        ip_allowlist.source(),
    );
    tracing::info!(
        "Allowed browser origins: {}  [source: {}]",
        allowed_origins.describe(),
        allowed_origins.source(),
    );
    if cfg.web_enabled {
        tracing::info!("Web tool: enabled — fetchable: {}", web_policy.describe());
    }
    if allow_model_paths {
        tracing::warn!(
            "EULLM_ALLOW_MODEL_PATHS is set: a request's `model` field may name any \
             GGUF path readable by this process. Intended for local use; do not \
             combine it with an API reachable by untrusted callers."
        );
    }

    // Fail loudly at startup if the audit destination is unusable, rather than
    // warning once per request after the fact. The trail exists to produce a
    // defensible record; degrading silently to "no record" is the one failure
    // mode it must not have.
    let audit = crate::audit::AuditLogger::new();
    {
        // Which store the server resolves model names against. Omitting this
        // is how `eullm list` and the API came to disagree about whether a
        // model existed, with no way to tell that they were reading different
        // directories.
        let (root, source) = cfg.store.root_with_source();
        tracing::info!("Model store: {}  [source: {source}]", root.display());
    }
    match audit.check_writable() {
        Ok(()) => tracing::info!("Audit trail: {}", audit.log_path().display()),
        // Explicitly configured destination that doesn't work → refuse to
        // start. Someone who set EULLM_AUDIT_DIR (or mounted a volume at it)
        // asked for the trail; serving without one silently is the failure
        // this check exists to prevent.
        Err(e) if crate::audit::AuditLogger::is_explicitly_configured() => {
            return Err(format!(
                "EULLM_AUDIT_DIR is set but the audit trail is not writable: {e}\n  \
                 Point it at a writable, persistent path (in Docker, one backed by a \
                 mounted volume), or unset it to fall back to ~/.eullm/audit."
            )
            .into());
        }
        // Nobody asked for a specific destination — warn loudly and serve.
        // Refusing to start an inference server over a log file the operator
        // never configured would turn a read-only home directory into an outage.
        Err(e) => tracing::warn!(
            "Audit trail disabled: {} is not writable ({e}). Inference will work, but \
             no audit records will be kept. Set EULLM_AUDIT_DIR to a writable path to \
             enable it.",
            audit.log_path().display(),
        ),
    }

    // `eullm run`'s model, loaded before the server started, is the first
    // resident; `serve` starts with none.
    let mut models = resident::ResidentModels::default();
    if let Some(name) = cfg.model_name
        && (cfg.engine.is_some() || cfg.scheduler.is_some())
    {
        let path = cfg
            .launch_model
            .as_ref()
            .map_or_else(|| PathBuf::from(&name), |(_, path)| path.clone());
        // A sequential launch model reserves its per-request context like
        // any other (see `LoadedModel::unallocated_reserve`).
        let unallocated_reserve = match &cfg.engine {
            Some(engine) if cfg.gpu_layers != 0 => crate::fit::context_reserve_bytes(
                crate::fit::read_gguf_info(&path).as_ref(),
                engine.context_size(),
                crate::inference::cache_type_bytes_per_elem(&cfg.cache_type_k),
                crate::inference::cache_type_bytes_per_elem(&cfg.cache_type_v),
            ),
            _ => 0,
        };
        let mut launch = resident::LoadedModel::new(name, path, cfg.engine, cfg.scheduler);
        launch.launch = true;
        launch.unallocated_reserve = unallocated_reserve;
        models.insert(launch);
    }

    let state = Arc::new(AppState {
        backend: cfg.backend,
        fallback_mmproj: cfg.mmproj.clone(),
        mmproj_offload: cfg.mmproj_offload,
        models: tokio::sync::RwLock::new(models),
        swap_lock: tokio::sync::Mutex::new(()),
        gpu_layers: cfg.gpu_layers,
        fit: cfg.fit,
        fit_strict: cfg.fit_strict,
        ctx_size: cfg.ctx_size,
        threads: cfg.threads,
        flash_attn: cfg.flash_attn,
        n_batch: cfg.n_batch,
        cache_type_k: cfg.cache_type_k,
        cache_type_v: cfg.cache_type_v,
        batch_size: cfg.batch_size,
        cpu_moe: cfg.cpu_moe,
        n_cpu_moe: cfg.n_cpu_moe,
        rs_seq: cfg.rs_seq,
        ctx_checkpoints: cfg.ctx_checkpoints,
        checkpoint_min_step: cfg.checkpoint_min_step,
        rust_debug: cfg.rust_debug,
        web_enabled: cfg.web_enabled,
        api_port: cfg.port,
        store: cfg.store,
        ip_allowlist,
        api_keys,
        allowed_origins,
        web_policy,
        allow_model_paths,
        launch_model: cfg.launch_model,
        embedding: tokio::sync::RwLock::new(cfg.launch_embedding),
        decision: tokio::sync::RwLock::new(cfg.launch_decision),
        decision_ctx: cfg.decision_ctx,
        cross_slot_evictions: std::sync::atomic::AtomicU64::new(0),
        idle: Arc::new(tokio::sync::Notify::new()),
        idle_tick: IDLE_TICK,
        embedding_deadline: tokio::sync::Mutex::new(None),
        decision_deadline: tokio::sync::Mutex::new(None),
        default_keep_alive: cfg.keep_alive,
    });
    let idle_unload_state = state.clone();
    tokio::spawn(async move {
        idle_unload_state.run_idle_unload_loop().await;
    });
    let api_port = cfg.port;
    let ui_port_opt = cfg.ui_port;

    let api_app = api_router(state.clone());
    let api_addr = format!("0.0.0.0:{api_port}");
    let api_listener = TcpListener::bind(&api_addr).await?;
    tracing::info!("eullm API listening on {api_addr}");

    // Spawn the optional chat-UI listener on a separate port. It exposes the
    // same API surface (so the embedded JS can call same-origin) plus the
    // HTML/CSS/JS for the chat at `/`. Disabled by default for `eullm serve`
    // (headless) and enabled by default for `eullm run` (interactive).
    let ui_handle = if let Some(ui_port) = ui_port_opt {
        if ui_port == api_port {
            tracing::warn!(
                "ui_port == api_port ({ui_port}); refusing to bind UI to avoid collision. \
                 Pick a different --ui-port or pass --no-ui."
            );
            None
        } else {
            let ui_app = ui_router(state.clone());
            let ui_addr = format!("0.0.0.0:{ui_port}");
            match TcpListener::bind(&ui_addr).await {
                Ok(ui_listener) => {
                    tracing::info!(
                        "eullm chat UI listening on {ui_addr}  (open http://localhost:{ui_port}/)"
                    );
                    Some(tokio::spawn(async move {
                        if let Err(e) = axum::serve(
                            ui_listener,
                            ui_app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
                        )
                        .with_graceful_shutdown(shutdown_signal())
                        .await
                        {
                            tracing::error!("UI listener failed: {e}");
                        }
                    }))
                }
                Err(e) => {
                    tracing::warn!(
                        "Could not bind chat UI on {ui_addr}: {e}. \
                         API still served on {api_addr}; pass --ui-port to override."
                    );
                    None
                }
            }
        }
    } else {
        None
    };

    axum::serve(
        api_listener,
        api_app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    if let Some(h) = ui_handle {
        // The UI server listens for the same shutdown signal, but the signal
        // is observed by whichever task wakes first. Abort the leftover task
        // on the way out to avoid lingering listeners during repeated runs
        // (notably in tests).
        h.abort();
    }

    tracing::info!("Server shut down gracefully.");
    Ok(())
}

/// How often the idle-unload loop looks for an expired keep_alive — see
/// `AppState::run_idle_unload_loop`.
const IDLE_TICK: std::time::Duration = std::time::Duration::from_secs(30);

#[cfg(test)]
impl AppState {
    /// A server with nothing loaded, and every other field at a test's
    /// starting value — named once, here, so that adding a field does not
    /// break every test that builds a state. Perimeter settings take their
    /// defaults: `.env` is read from a path that does not exist.
    pub(crate) fn for_tests(store: ModelStore, api_keys: auth::ApiKeys) -> Self {
        let absent = std::path::Path::new("/nonexistent/eullm-test/.env");
        Self {
            backend: crate::inference::test_backend(),
            fallback_mmproj: None,
            mmproj_offload: None,
            models: tokio::sync::RwLock::new(resident::ResidentModels::default()),
            swap_lock: tokio::sync::Mutex::new(()),
            gpu_layers: 0,
            fit: false,
            fit_strict: false,
            ctx_size: 4096,
            threads: 1,
            flash_attn: false,
            n_batch: 512,
            cache_type_k: crate::inference::KvCacheType::F16,
            cache_type_v: crate::inference::KvCacheType::F16,
            batch_size: 1,
            cpu_moe: false,
            n_cpu_moe: 0,
            rs_seq: 0,
            ctx_checkpoints: 0,
            checkpoint_min_step: 8192,
            rust_debug: false,
            web_enabled: false,
            api_port: 0,
            store,
            ip_allowlist: ip_allowlist::IpAllowlist::load(absent),
            api_keys: Arc::new(api_keys),
            allowed_origins: origin::AllowedOrigins::load(absent),
            web_policy: crate::tools::guard::WebPolicy::from_env(),
            allow_model_paths: false,
            launch_model: None,
            embedding: tokio::sync::RwLock::new(None),
            decision: tokio::sync::RwLock::new(None),
            decision_ctx: crate::inference::decision::DEFAULT_DECISION_CTX,
            cross_slot_evictions: std::sync::atomic::AtomicU64::new(0),
            idle: Arc::new(tokio::sync::Notify::new()),
            idle_tick: IDLE_TICK,
            embedding_deadline: tokio::sync::Mutex::new(None),
            decision_deadline: tokio::sync::Mutex::new(None),
            default_keep_alive: None,
        }
    }
}

/// Wait for a shutdown signal (SIGTERM, SIGINT, or Ctrl+C).
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();

    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to register SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => { tracing::info!("Received SIGINT, shutting down..."); }
            _ = sigterm.recv() => { tracing::info!("Received SIGTERM, shutting down..."); }
        }
    }

    #[cfg(not(unix))]
    {
        ctrl_c.await.ok();
        tracing::info!("Received Ctrl+C, shutting down...");
    }
}

/// Maximum request body size. Axum defaults to 2 MB, which is fine for text
/// but far too small for multimodal `/api/chat` requests: a base64-encoded
/// image or audio clip easily exceeds it (base64 inflates bytes by ~33%), so
/// a stock photo returns `413 length limit exceeded`. 64 MB comfortably fits
/// images and reasonable audio clips while still bounding abuse.
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Reject any request whose source IP isn't in `state.ip_allowlist`, before
/// it reaches CORS, body parsing, or any handler. See `ip_allowlist` for why
/// the socket always binds `0.0.0.0` regardless and this check is the real
/// boundary.
///
/// Runs *inside* [`enforce_auth`], so by the time it executes an [`Identity`]
/// is always present. A request that presented a valid key is admitted whatever
/// its source address: behind Docker's published ports every external client
/// arrives as the bridge gateway, so refusing an authenticated caller on
/// address grounds would leave the operator with no working configuration at
/// all. See the `auth` module docs for the full reasoning.
async fn enforce_ip_allowlist(
    State(state): State<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let authenticated = req
        .extensions()
        .get::<Identity>()
        .is_some_and(Identity::is_authenticated);
    if authenticated || state.ip_allowlist.is_allowed(addr.ip()) {
        next.run(req).await
    } else {
        tracing::warn!("Rejected request from disallowed IP {}", addr.ip());
        let message = "source IP not in the configured allowlist";
        if req.uri().path() == systemone::PATH {
            return systemone::ApiError::new(
                axum::http::StatusCode::FORBIDDEN,
                "forbidden",
                message,
            )
            .into_response();
        }
        (axum::http::StatusCode::FORBIDDEN, message).into_response()
    }
}

/// Verify the bearer token, attach an [`Identity`] to the request, and charge
/// the key's quota. Outermost layer on both routers.
///
/// When no keys are configured this attaches an anonymous identity and gets out
/// of the way — the IP allowlist is then the only control, which is the right
/// default for the single-user local case the engine is most often used for.
///
/// `allow_query_token` is true only on the UI listener. The embedded chat runs
/// in a browser, which cannot be handed a header before its first navigation,
/// so `?api_key=…` bootstraps it (the page then stores the key and sends it as
/// a header on every subsequent fetch). It is deliberately **not** accepted on
/// the API listener: a token in a URL ends up in proxy logs, browser history
/// and `Referer` headers, and no programmatic client needs it.
async fn enforce_auth(
    State((state, allow_query_token)): State<(Arc<AppState>, bool)>,
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let presented = extract_token(&req, allow_query_token);
    // `/v1/systemone` refusals carry the body its clients parse (see
    // `systemone::ApiError`); every other endpoint keeps the one it had.
    let structured = req.uri().path() == systemone::PATH;
    match state.api_keys.authenticate(presented.as_deref()) {
        Ok(identity) => {
            req.extensions_mut().insert(identity);
            next.run(req).await
        }
        Err(auth::AuthError::Missing) => unauthorized(
            "missing API key — send it as `Authorization: Bearer <key>` or `X-Api-Key: <key>`",
            structured,
        ),
        Err(auth::AuthError::Invalid) => {
            // No key id to name: logging the presented token would write a
            // credential into the log file, and it may be a valid key for a
            // *different* deployment.
            tracing::warn!("Rejected request with an invalid API key");
            unauthorized("invalid API key", structured)
        }
        Err(auth::AuthError::RateLimited {
            key_id,
            retry_after_s,
        }) => {
            tracing::warn!("Key '{key_id}' is over its per-minute quota");
            let message =
                format!("rate limit exceeded for key '{key_id}' — retry in {retry_after_s}s");
            let retry_after = [(axum::http::header::RETRY_AFTER, retry_after_s.to_string())];
            if structured {
                let status = axum::http::StatusCode::TOO_MANY_REQUESTS;
                let error = systemone::ApiError::new(status, "too_many_requests", message);
                return (retry_after, error).into_response();
            }
            (
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                retry_after,
                axum::Json(serde_json::json!({ "error": message })),
            )
                .into_response()
        }
    }
}

/// 401 with the `WWW-Authenticate` challenge, so a client library can tell an
/// authentication failure from a generic refusal. `structured`: the body of
/// a `/v1/systemone` error.
fn unauthorized(message: &str, structured: bool) -> axum::response::Response {
    let challenge = [(
        axum::http::header::WWW_AUTHENTICATE,
        "Bearer realm=\"eullm\"",
    )];
    if structured {
        let status = axum::http::StatusCode::UNAUTHORIZED;
        return (
            challenge,
            systemone::ApiError::new(status, "unauthorized", message),
        )
            .into_response();
    }
    (
        axum::http::StatusCode::UNAUTHORIZED,
        challenge,
        axum::Json(serde_json::json!({ "error": message })),
    )
        .into_response()
}

/// Pull the token out of `Authorization: Bearer`, `X-Api-Key`, or — on the UI
/// listener only — an `api_key` query parameter.
fn extract_token(req: &axum::extract::Request, allow_query_token: bool) -> Option<String> {
    let headers = req.headers();
    if let Some(v) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        // Case-insensitive scheme, per RFC 7235.
        let v = v.trim();
        if let Some(rest) = v
            .split_once(' ')
            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
            .map(|(_, rest)| rest)
        {
            return Some(rest.trim().to_string());
        }
    }
    if let Some(v) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(v.trim().to_string());
    }
    if allow_query_token {
        return req.uri().query().and_then(|q| {
            q.split('&')
                .filter_map(|kv| kv.split_once('='))
                .find(|(k, _)| *k == "api_key")
                .map(|(_, v)| v.to_string())
        });
    }
    None
}

/// Refuse a cross-origin request that has side effects, before it reaches a
/// handler.
///
/// CORS is not this check. CORS decides whether a browser hands the *response*
/// back to the calling page; the request itself is still executed. For `GET` on
/// a read-only endpoint that distinction is academic, but `POST /api/unload` or
/// a model swap take effect regardless of whether the attacker can read the
/// reply — and a simple `POST` with `Content-Type: text/plain` needs no
/// preflight, so the CORS layer never gets a chance to object.
///
/// Requests with no `Origin` header are left alone: that is every non-browser
/// client, and an origin policy has never applied to them.
async fn enforce_origin(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let method = req.method().clone();
    let is_safe = matches!(
        method,
        axum::http::Method::GET | axum::http::Method::HEAD | axum::http::Method::OPTIONS
    );
    if !is_safe
        && let Some(origin) = req
            .headers()
            .get(axum::http::header::ORIGIN)
            .and_then(|v| v.to_str().ok())
        && !state.allowed_origins.is_allowed(origin)
    {
        tracing::warn!(
            "Rejected {} from disallowed origin {}",
            method,
            crate::audit::sanitize_for_log(origin)
        );
        let message = "request origin is not allowed — set EULLM_ALLOWED_ORIGINS \
                       if this frontend should be permitted";
        if req.uri().path() == systemone::PATH {
            return systemone::ApiError::new(
                axum::http::StatusCode::FORBIDDEN,
                "forbidden",
                message,
            )
            .into_response();
        }
        return (
            axum::http::StatusCode::FORBIDDEN,
            axum::Json(serde_json::json!({ "error": message })),
        )
            .into_response();
    }
    next.run(req).await
}

/// CORS layer honouring `state.allowed_origins`.
///
/// `allow_headers(Any)` and `allow_methods(Any)` stay permissive on purpose:
/// once the *origin* is constrained, restricting which headers that trusted
/// origin may send buys nothing and breaks frontends that send their own
/// (Open WebUI sends several).
fn cors_layer(state: &Arc<AppState>) -> CorsLayer {
    let origins = state.allowed_origins.clone();
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin, _req| {
            origin
                .to_str()
                .map(|o| origins.is_allowed(o))
                .unwrap_or(false)
        }))
        .allow_methods(Any)
        .allow_headers(Any)
}

/// Build the EULLM API router (Ollama + OpenAI compat) with CORS enabled
/// for Open WebUI and other frontends.
///
/// This router never serves the chat UI — clients hitting the API port get
/// only `/api/*` and `/v1/*`, so RAG systems and OpenAI-compatible tooling
/// see a pure API surface with no HTML on `/`.
fn api_router(state: Arc<AppState>) -> Router {
    let cors = cors_layer(&state);

    // Layers run outermost-last: `enforce_auth` is added last, so it sees the
    // request first. The order is load-bearing — see `enforce_ip_allowlist` for
    // why authentication must precede the address check rather than follow it.
    Router::new()
        .nest("/api", routes::api_routes())
        .nest("/v1", routes::openai_routes())
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(cors)
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            enforce_origin,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            enforce_ip_allowlist,
        ))
        .layer(axum::middleware::from_fn_with_state(
            (state.clone(), false),
            enforce_auth,
        ))
        .with_state(state)
}

/// Build the chat-UI router. Includes the same API routes (so the embedded
/// JS can fetch same-origin), plus `/` and `/eullm-ui/*` for HTML/CSS/JS.
///
/// Always served on a separate port from the API so the two surfaces are
/// independently togglable and never collide.
fn ui_router(state: Arc<AppState>) -> Router {
    let cors = cors_layer(&state);

    // The UI listener nests the same API routes, so it must enforce the same
    // controls — exempting it would simply move the open door to another port.
    // It differs in one respect: a token may arrive as `?api_key=…`, because a
    // browser cannot set a header on its first navigation. See `enforce_auth`.
    Router::new()
        .nest("/api", routes::api_routes())
        .nest("/v1", routes::openai_routes())
        .merge(crate::ui::router())
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(cors)
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            enforce_origin,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            enforce_ip_allowlist,
        ))
        .layer(axum::middleware::from_fn_with_state(
            (state.clone(), true),
            enforce_auth,
        ))
        .with_state(state)
}

#[cfg(test)]
mod http_tests {
    //! End-to-end tests over a real listener.
    //!
    //! Every unit test in this crate exercises a function; none of them ever
    //! sent an HTTP request. That gap is not theoretical. Three defects found
    //! by hand in two days lived entirely on the `serve` path: the model lists
    //! ignored the store so a pulled model could not be selected from an
    //! editor, the diagnostic banner was never printed, and a whole family of
    //! models answered nothing at all. A suite of 217 green tests had nothing
    //! to say about any of them.
    //!
    //! These bind an ephemeral port and speak real HTTP through the real
    //! middleware stack, because that is where the behaviour lives: the
    //! allowlist reads a peer address, and a handler tested in isolation never
    //! has one.
    //!
    //! Deliberately no model is loaded. Inference needs a GGUF that CI cannot
    //! download on every push, and the endpoints that answer without one are
    //! exactly the ones that broke.

    use super::*;
    use std::net::SocketAddr;

    /// A store directory with one model in it, laid out the way a pull leaves
    /// it: a directory named after the id, a manifest, and the weights.
    fn store_with_one_model(dir: &std::path::Path, id: &str) -> ModelStore {
        let model_dir = dir.join(id);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::write(model_dir.join("model.gguf"), b"not a real gguf").expect("weights");
        let manifest = serde_json::json!({
            "id": id,
            "name": id,
            "description": "test fixture",
            "languages": ["en"],
            "base": "test",
            "vram_gb": 1,
            "size_bytes": 15,
            "license": "Apache-2.0",
            "digest": "sha256:0",
            "pulled_at": "2026-07-30T00:00:00Z",
            "status": "ready",
            "gguf_file": "model.gguf",
        });
        std::fs::write(
            model_dir.join("manifest.json"),
            serde_json::to_string(&manifest).expect("manifest json"),
        )
        .expect("manifest");
        ModelStore::at(dir.to_path_buf())
    }

    /// Start the API on 127.0.0.1:0 and return its base URL.
    ///
    /// Port 0 rather than a fixed one: these run in parallel with every other
    /// test in the binary, and a hardcoded port makes the suite fail depending
    /// on what else is listening on the machine.
    async fn spawn(store: ModelStore) -> String {
        // A path that does not exist, so the perimeter types fall back to
        // their defaults instead of reading a developer's real `.env`.
        let absent = std::path::Path::new("/nonexistent/eullm-test/.env");
        spawn_with_keys(
            store,
            auth::ApiKeys::load(absent).expect("no keys configured"),
        )
        .await
    }

    /// `spawn`, with API keys configured.
    async fn spawn_with_keys(store: ModelStore, api_keys: auth::ApiKeys) -> String {
        spawn_state(AppState::for_tests(store, api_keys)).await
    }

    /// `spawn`, for a state the test has prepared.
    async fn spawn_state(state: AppState) -> String {
        let state = Arc::new(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let app = api_router(state);
        tokio::spawn(async move {
            let _ = axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await;
        });
        format!("http://{addr}")
    }

    async fn get_json(url: &str) -> (reqwest::StatusCode, serde_json::Value) {
        let r = reqwest::get(url).await.expect("request");
        let status = r.status();
        let body = r.json().await.unwrap_or(serde_json::Value::Null);
        (status, body)
    }

    async fn post_json(url: &str, body: serde_json::Value) -> (reqwest::StatusCode, String) {
        let r = reqwest::Client::new()
            .post(url)
            .json(&body)
            .send()
            .await
            .expect("request");
        let status = r.status();
        (status, r.text().await.unwrap_or_default())
    }

    #[tokio::test]
    async fn api_tags_lists_a_model_that_is_on_disk_but_not_loaded() {
        // The shape of the bug reported in #294: both model lists were built
        // from the built-in catalog plus whatever happened to be loaded, so a
        // model pulled from a URL or a HuggingFace repo was invisible to them
        // even though it was sitting in the store, ready to run.
        let tmp = std::env::temp_dir().join(format!("eullm-tags-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        let (status, body) = get_json(&format!("{base}/api/tags")).await;
        assert_eq!(status, 200);
        let names: Vec<String> = body["models"]
            .as_array()
            .expect("models array")
            .iter()
            .filter_map(|m| m["name"].as_str().map(str::to_string))
            .collect();
        assert!(
            names.iter().any(|n| n.contains("a-pulled-model")),
            "a model in the store must appear in /api/tags, got {names:?}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Every resident is listed as loaded, the most recently used first —
    /// the one the chat UI preselects — and a catalog model keeps its
    /// catalog metadata while marked so.
    #[tokio::test]
    async fn every_resident_is_marked_loaded_and_keeps_catalog_metadata() {
        let tmp = std::env::temp_dir().join(format!("eullm-tags-loaded-{}", uuid::Uuid::new_v4()));
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let absent = std::path::Path::new("/nonexistent/eullm-test/.env");
        let state = AppState::for_tests(store, auth::ApiKeys::load(absent).expect("no keys"));
        let catalog = &crate::models::EU_CATALOG[0];
        {
            let mut models = state.models.write().await;
            for name in [catalog.id.as_str(), "/elsewhere/Custom-Model-Q4_K_M.gguf"] {
                models.insert(resident::LoadedModel::new(
                    name.into(),
                    name.into(),
                    None,
                    None,
                ));
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
        let base = spawn_state(state).await;

        let (status, body) = get_json(&format!("{base}/api/tags")).await;
        assert_eq!(status, 200);
        let models = body["models"].as_array().expect("models array");
        let loaded: Vec<&serde_json::Value> =
            models.iter().filter(|m| m["loaded"] == true).collect();
        let names: Vec<&str> = loaded.iter().filter_map(|m| m["name"].as_str()).collect();
        assert_eq!(
            names,
            ["/elsewhere/Custom-Model-Q4_K_M.gguf", catalog.id.as_str()],
            "every resident, the most recently used first"
        );
        assert_eq!(loaded[1]["digest"], catalog.digest.as_str());
        assert_eq!(loaded[1]["details"]["family"], catalog.base());
        assert_eq!(
            models
                .iter()
                .filter(|m| m["name"] == catalog.id.as_str())
                .count(),
            1,
            "listed once, as loaded"
        );

        let (_, body) = get_json(&format!("{base}/v1/models")).await;
        let ids: Vec<&str> = body["data"]
            .as_array()
            .expect("data array")
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        for name in names {
            assert_eq!(
                ids.iter().filter(|id| **id == name).count(),
                1,
                "{name} in {ids:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn openai_models_lists_a_model_that_is_on_disk_but_not_loaded() {
        // This endpoint is the one that decides whether a model can be picked
        // at all: a coding editor offers what `/v1/models` names, so a model
        // it never names cannot be selected, however well it runs elsewhere.
        let tmp = std::env::temp_dir().join(format!("eullm-v1models-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        let (status, body) = get_json(&format!("{base}/v1/models")).await;
        assert_eq!(status, 200);
        let ids: Vec<String> = body["data"]
            .as_array()
            .expect("data array")
            .iter()
            .filter_map(|m| m["id"].as_str().map(str::to_string))
            .collect();
        assert!(
            ids.iter().any(|i| i.contains("a-pulled-model")),
            "a model in the store must appear in /v1/models, got {ids:?}"
        );
        // Still an OpenAI list; `models`, the System One SDKs' list of
        // decision models, is there and empty with none loaded.
        assert_eq!(body["object"], "list");
        assert_eq!(body["models"], serde_json::json!([]));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn an_unknown_model_is_refused_by_name_and_not_with_a_500() {
        // A wrong model name is a client mistake and must read like one. The
        // failure mode worth pinning is a panic or a bare 500, which tells the
        // caller nothing and looks like the server is broken.
        let tmp = std::env::temp_dir().join(format!("eullm-unknown-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        let (status, body) = post_json(
            &format!("{base}/api/chat"),
            serde_json::json!({
                "model": "this-model-does-not-exist",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false,
            }),
        )
        .await;
        assert!(
            status.is_client_error() || status == 503,
            "expected a client error or 503, got {status}"
        );
        assert!(
            body.contains("this-model-does-not-exist"),
            "the refusal must name the model the caller asked for, got: {body}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // The catalog endpoints answer before any model is loaded, and both
    // refuse malformed input without reaching the network -- so these run in
    // CI, where there is none.
    #[tokio::test]
    async fn the_catalog_search_answers_an_empty_query_without_calling_out() {
        let tmp = std::env::temp_dir().join(format!("eullm-hfsearch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        let (status, body) = get_json(&format!("{base}/api/hf/search?q=")).await;
        assert_eq!(status, 200);
        assert_eq!(
            body["models"].as_array().map(Vec::len),
            Some(0),
            "an empty query is an empty result, not a search for everything"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // This id is interpolated into a huggingface.co URL, so it is validated
    // before the request is built rather than after it comes back.
    #[tokio::test]
    async fn the_catalog_refuses_a_repo_id_that_is_not_owner_slash_repo() {
        let tmp = std::env::temp_dir().join(format!("eullm-hfrepo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        for bad in ["", "owner", "a/b/c", "..%2Fetc"] {
            let (status, _) = get_json(&format!("{base}/api/hf/repo?id={bad}")).await;
            assert_eq!(status, 400, "`{bad}` must be refused, not requested");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn the_version_endpoint_answers_without_a_model() {
        // `serve` starts with an empty slot, and a client probing whether the
        // server is up must get an answer before any model exists.
        let tmp = std::env::temp_dir().join(format!("eullm-version-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        let (status, body) = get_json(&format!("{base}/api/version")).await;
        assert_eq!(status, 200);
        assert_eq!(
            body["version"].as_str(),
            Some(env!("CARGO_PKG_VERSION")),
            "/api/version must report the crate version"
        );
        assert_eq!(
            body["model_swaps"].as_u64(),
            Some(0),
            "a fresh server has evicted nothing yet"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn embedding_endpoints_refuse_an_unknown_model_by_name() {
        // Same shape as `an_unknown_model_is_refused_by_name_and_not_with_a_500`
        // for generation: a model name that resolves to nothing is a client
        // mistake and must read like one on both embedding endpoints, not as
        // a generic 500.
        let tmp = std::env::temp_dir().join(format!("eullm-embed-404-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        for path in ["/api/embed", "/v1/embeddings"] {
            let (status, body) = post_json(
                &format!("{base}{path}"),
                serde_json::json!({ "model": "this-model-does-not-exist", "input": "hi" }),
            )
            .await;
            assert!(
                status.is_client_error(),
                "{path}: expected a client error, got {status}"
            );
            assert!(
                body.contains("this-model-does-not-exist"),
                "{path}: the refusal must name the model the caller asked for, got: {body}"
            );
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn embedding_endpoints_require_input() {
        let tmp = std::env::temp_dir().join(format!("eullm-embed-noinput-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        // Refused before any model resolution is attempted — a malformed
        // request should not trigger a load/swap on its way to being
        // rejected.
        let (status, _) = post_json(
            &format!("{base}/api/embed"),
            serde_json::json!({ "model": "a-pulled-model" }),
        )
        .await;
        assert_eq!(status, 400);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The `error` object of a `/v1/systemone` error body: the shape its
    /// clients parse, `{"error": {"code", "message", "question"?}}`.
    fn systemone_error(body: &str) -> serde_json::Value {
        let body: serde_json::Value = serde_json::from_str(body).expect("a JSON body");
        let error = body["error"].clone();
        assert!(error["code"].is_string(), "{body}");
        assert!(error["message"].is_string(), "{body}");
        error
    }

    /// `/v1/systemone` on a server with no decision model: the ways a
    /// request can fail before any model runs must each read as the
    /// client's mistake it is, with the body System One clients parse.
    #[tokio::test]
    async fn systemone_refuses_what_it_cannot_answer_as_client_errors() {
        let tmp = std::env::temp_dir().join(format!("eullm-systemone-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;
        let url = format!("{base}/v1/systemone");
        let question = serde_json::json!({ "q": { "type": "noul", "instructions": "Urgent?" } });

        // `jev-latest` means "the loaded decision model", and there is none.
        let (status, body) = post_json(
            &url,
            serde_json::json!({ "model": "jev-latest", "state": "x", "questions": question }),
        )
        .await;
        assert_eq!(status, 400, "{body}");
        let error = systemone_error(&body);
        assert_eq!(error["code"], "model_not_loaded");
        assert!(body.contains("--decision-model"), "{body}");

        let (status, body) = post_json(
            &url,
            serde_json::json!({
                "model": "this-model-does-not-exist", "state": "x", "questions": question
            }),
        )
        .await;
        assert_eq!(status, 404, "{body}");
        assert_eq!(systemone_error(&body)["code"], "not_found");
        assert!(body.contains("this-model-does-not-exist"), "{body}");

        // Malformed: refused before any model resolution, not with a 500.
        let (status, body) = post_json(
            &url,
            serde_json::json!({ "model": "a-pulled-model", "state": "x", "questions": {} }),
        )
        .await;
        assert_eq!(status, 422, "{body}");
        assert_eq!(systemone_error(&body)["code"], "invalid_request");
        assert!(body.contains("at least one question"), "{body}");

        // One question wrong: named in `question`.
        let (status, body) = post_json(
            &url,
            serde_json::json!({ "state": "x", "questions": {
                "fine": { "type": "noul", "instructions": "Urgent?" },
                "team": { "type": "choice", "instructions": "Which?", "criteria": { "a": null } }
            } }),
        )
        .await;
        assert_eq!(status, 422, "{body}");
        let error = systemone_error(&body);
        assert_eq!(error["code"], "invalid_question");
        assert_eq!(error["question"], "team");

        let client = reqwest::Client::new();
        let r = client
            .post(&url)
            .header("content-type", "application/json")
            .body("{\"state\": ")
            .send()
            .await
            .expect("request");
        assert_eq!(r.status(), 422);
        assert_eq!(
            systemone_error(&r.text().await.unwrap())["code"],
            "invalid_json"
        );

        let r = client.get(&url).send().await.expect("request");
        assert_eq!(r.status(), 405);
        assert_eq!(
            systemone_error(&r.text().await.unwrap())["code"],
            "method_not_allowed"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The middleware's refusals: in the structured shape on
    /// `/v1/systemone`, unchanged everywhere else, where Ollama and OpenAI
    /// clients read `{"error": "<message>"}`.
    #[tokio::test]
    async fn systemone_refusals_from_the_middleware_are_structured_there_only() {
        let tmp = std::env::temp_dir().join(format!("eullm-systemone-auth-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let keys = auth::ApiKeys::from_spec("ci:0123456789abcdef01").expect("keys");
        let base = spawn_with_keys(ModelStore::at(tmp.clone()), keys).await;

        let (status, body) = post_json(
            &format!("{base}/v1/systemone"),
            serde_json::json!({ "state": "x", "questions": {} }),
        )
        .await;
        assert_eq!(status, 401, "{body}");
        let error = systemone_error(&body);
        assert_eq!(error["code"], "unauthorized");
        assert!(
            error["message"]
                .as_str()
                .unwrap()
                .contains("missing API key")
        );

        let (status, body) = get_json(&format!("{base}/api/tags")).await;
        assert_eq!(status, 401);
        assert!(body["error"].is_string(), "{body}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn generate_with_no_model_field_and_nothing_loaded_still_503s_before_the_warm_load_check() {
        // The empty-prompt warm-load short-circuit sits after `ensure_model`
        // on purpose (see `generate`'s comment): it must not bypass "no
        // model available" and answer a fabricated "loaded" response for a
        // server with nothing loaded and no model named.
        let tmp = std::env::temp_dir().join(format!("eullm-warmload-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = ModelStore::at(tmp.clone());
        let base = spawn(store).await;

        let (status, body) = post_json(
            &format!("{base}/api/generate"),
            serde_json::json!({ "prompt": "" }),
        )
        .await;
        assert_eq!(status, 503);
        assert!(body.contains("No model loaded"));
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
