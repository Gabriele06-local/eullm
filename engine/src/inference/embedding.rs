//! A second, independent model slot for text embeddings.
//!
//! Deliberately not built on `InferenceEngine`: that type exists to generate
//! text token by token, with a KV cache sized for a whole conversation and a
//! sampler chain behind it. An embedding request is one forward pass per
//! input, clears its KV cache before the next one, and produces a fixed-size
//! vector instead of a token stream. Reusing `InferenceEngine` would mean
//! carrying all of that machinery to switch it back off.
//!
//! Runs alongside the generation model on purpose — see
//! `AppState::ensure_embedding_model` for why keeping both resident (when
//! they fit) beats swapping a 20 GB LLM out for a 500 MB embedder and back on
//! every request.
//!
//! The model's llama.cpp context is kept from one request to the next, on a
//! thread of its own: a `LlamaContext` borrows its model and is not `Send`,
//! so inputs reach it over a channel and are embedded one at a time, in the
//! order they came — as `inference::decision` keeps its own.

use std::num::NonZeroU32;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::pin::pin;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread::JoinHandle;

use llama_cpp_2::EmbeddingsError;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;

/// Default embedding context. A request to `/api/embed`/`/v1/embeddings` may
/// override it with `options.num_ctx`; a launch-time `--embedding-model`
/// (see `main.rs`) has no per-request body to read one from, so it uses this
/// directly. 2048 comfortably covers a single RAG chunk — the usual unit
/// these are called on — without paying for a window sized to the
/// generation model's much longer conversations.
pub const DEFAULT_EMBEDDING_CTX: u32 = 2048;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

/// A loaded embedding model: a `LlamaModel` bound to the process-wide shared
/// `LlamaBackend` (see `load`), kept alive independently of whatever
/// generation model is loaded in the main slot, and the context its inputs
/// are embedded in, kept between requests by its worker thread (see
/// `embed` for why kept, and `reserve_context`).
pub struct EmbeddingModel {
    model: Arc<LlamaModel>,
    /// The GGUF it was loaded from: the same file asked for under another
    /// name is this model (`AppState::ensure_embedding_model`).
    path: PathBuf,
    /// Most tokens one input may have. Inputs longer than this are
    /// truncated (see `embed`) rather than rejected — the alternative is a
    /// hard error on the first oversized chunk an ingestion pipeline sends
    /// it, which is a worse failure mode than a documented truncation.
    n_ctx: u32,
    worker: Worker,
}

/// What `EmbeddingModel::embed` produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedded {
    /// One L2-normalized vector per input, in input order.
    pub vectors: Vec<Vec<f32>>,
    /// Tokens the model read, over all inputs and after truncation: what
    /// `usage.prompt_tokens` on `/v1/embeddings` and `prompt_eval_count` on
    /// `/api/embed` report.
    pub prompt_tokens: usize,
}

/// KV cells are allotted in steps of this many: llama.cpp rounds a context
/// up to a multiple of 256 whatever it is asked for, and attends over the
/// used cells rounded up to 256 too. A context sized in these steps is
/// therefore the one llama.cpp would build anyway, and an input sees the
/// same attention window whatever the size of the context it is decoded in.
const KV_CELL_STEP: u32 = 256;

/// The cells an input of `tokens` needs: a batch that holds it whole in one
/// decode — what keeps every vector the same as it always was — and the KV
/// cells that batch fills. `largest` is the largest context the model
/// builds ([`largest_context`]), which holds any input it takes.
fn cells_for(tokens: usize, largest: u32) -> u32 {
    let tokens = u32::try_from(tokens).unwrap_or(largest).clamp(1, largest);
    tokens
        .checked_next_multiple_of(KV_CELL_STEP)
        .unwrap_or(largest)
        .min(largest)
}

/// The largest context an embedder that takes inputs of up to `max` tokens
/// builds: one that holds the longest of them.
fn largest_context(max: u32) -> u32 {
    max.max(1)
        .checked_next_multiple_of(KV_CELL_STEP)
        .unwrap_or(max)
}

/// The size to build the kept context at for an input needing `needed`
/// cells, or `None` when the one kept (`kept` cells) already holds it.
/// Grown, never shrunk — an input the old context held fits again — and at
/// least doubled, so a run of slightly longer inputs does not rebuild it for
/// each one.
fn grown_context(kept: Option<u32>, needed: u32, largest: u32) -> Option<u32> {
    match kept {
        Some(kept) if kept >= needed => None,
        Some(kept) => Some(needed.max(kept.saturating_mul(2).min(largest))),
        None => Some(needed),
    }
}

impl EmbeddingModel {
    /// Load an embedding model fully onto the GPU (`gpu_layers = -1`) if any
    /// GPU backend is compiled in, CPU otherwise — mirrors
    /// `InferenceEngine::load`'s use of `check_gpu_support`. Embedding models
    /// are small enough (typically 100 MB-1 GB) that a partial CPU/GPU split
    /// is not worth the complexity `--fit` applies to a multi-gigabyte LLM;
    /// full offload or full CPU are the only two shapes this needs.
    ///
    /// `backend` is shared with whatever generation model is loaded in this
    /// process (or loaded later) rather than created fresh here:
    /// `LlamaBackend::init()` marks a process-wide `AtomicBool` and fails
    /// with `BackendAlreadyInitialized` on a second call while the first
    /// instance is still alive — two independently-initialized backends can
    /// never coexist in one process, which used to make the whole point of
    /// this second model slot (staying resident alongside a loaded
    /// generation model) fail with exactly that error on the first request
    /// that tried it. See `main.rs`/`AppState`, which own the one instance
    /// the whole process shares.
    ///
    /// No context is built yet: the first input builds one, or
    /// `reserve_context` does.
    pub fn load(
        path: &Path,
        threads: u32,
        n_ctx: u32,
        backend: Arc<LlamaBackend>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if !path.exists() {
            return Err(format!("Embedding model file not found: {}", path.display()).into());
        }
        // A context of no tokens embeds nothing, and llama.cpp refuses to
        // create one. Refused here, where the value arrives
        // (`options.num_ctx: 0`), rather than as a failure on every request.
        if n_ctx == 0 {
            return Err("an embedding context needs at least one token (num_ctx is 0)".into());
        }

        let gpu_layers = crate::inference::check_gpu_support(-1);
        let model_params = if gpu_layers >= 0 {
            LlamaModelParams::default().with_n_gpu_layers(gpu_layers as u32)
        } else {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        };
        let model_params = pin!(model_params);

        tracing::info!("Loading embedding model: {}", path.display());
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .map_err(|e| format!("Failed to load embedding model: {e}"))?;
        tracing::info!("Embedding model loaded — dimension {}", model.n_embd());

        let model = Arc::new(model);
        let worker = Worker::start(model.clone(), backend, threads, largest_context(n_ctx))?;
        Ok(Self {
            model,
            path: path.to_path_buf(),
            n_ctx,
            worker,
        })
    }

    pub fn n_embd(&self) -> usize {
        usize::try_from(self.model.n_embd()).unwrap_or(0)
    }

    /// The GGUF this model was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The bytes of its weights as llama.cpp holds them: what `/api/ps`
    /// reports as its size.
    pub fn weights_bytes(&self) -> u64 {
        self.model.size()
    }

    /// Tokens of context the model keeps once its longest input has come:
    /// [`n_ctx`](Self::embed) rounded up to whole steps of 256 cells.
    pub fn largest_context(&self) -> u32 {
        largest_context(self.n_ctx)
    }

    /// Build the kept context now, at the size of the longest input, instead
    /// of when an input first needs it. `--embedding-model` does this at
    /// launch, before the generation model is sized: the context is then
    /// memory already in use when `--fit` reads what is free, and a long
    /// input never needs more than the model holds — where a context built
    /// by the first long request, with a generation model sized to fill the
    /// card, found no room and failed.
    pub fn reserve_context(&self) -> Result<(), String> {
        let cells = self.largest_context();
        self.worker.ask(|reply| Job::Reserve(cells, reply))
    }

    /// Embed each input text independently, returning one vector per input in
    /// the same order.
    ///
    /// Pooling is deliberately left as `Unspecified`: llama.cpp then reads
    /// the model's own declared pooling type from the GGUF (`hparams`) —
    /// CLS for BGE, mean for E5, and so on — falling back to `None` only for
    /// a model that declares nothing. Overriding it here would silently mask
    /// a real mismatch instead of using the pooling the model was trained
    /// with. When the resolved type genuinely is `None`, this falls back to
    /// mean-pooling the per-token embeddings itself, which is the standard
    /// substitute and better than refusing to answer.
    ///
    /// One text at a time, KV cache cleared between them, each decoded whole
    /// in one batch. Not the fastest possible shape — batching independent
    /// texts into one multi-sequence decode call would amortize the fixed
    /// per-decode cost — but it is the simple, obviously-correct one.
    ///
    /// The inputs are embedded in the context the model keeps between
    /// requests, grown when one needs more cells than it has, never shrunk.
    /// A context per request cost far more than the input: llama.cpp
    /// allocates and clears the KV cache for every cell, and reserves
    /// compute buffers for the largest batch the context could take — for a
    /// decoder-based embedder a logits row per token, ~600 KB each for
    /// Qwen3's 151k vocabulary. On an RTX 5070 Ti with Qwen3-Embedding-0.6B,
    /// a request with one input of 1,966 tokens took 1,001 ms, and one with
    /// eight of them 1,682 ms: 97 ms an input, 904 ms building and dropping
    /// the context around them. 62 tokens: 40 ms a request, 7 ms an input.
    /// What is computed does not change: each input is still decoded whole
    /// in one batch, and llama.cpp attends over the same 256-cell-rounded
    /// window in a kept context, whatever its size, as in one sized to the
    /// input.
    ///
    /// Requests are not run side by side: each input waits for those before
    /// it, from any request, so a request of many inputs does not hold back
    /// a short one for all of them, only for the input being embedded.
    pub fn embed(&self, texts: &[String]) -> Result<Embedded, String> {
        let mut inputs = Vec::with_capacity(texts.len());
        for text in texts {
            let mut tokens = self
                .model
                .str_to_token(text, AddBos::Always)
                .map_err(|e| format!("Tokenization failed: {e}"))?;

            if tokens.len() as u32 > self.n_ctx {
                tracing::warn!(
                    "Embedding input truncated: {} tokens exceeds the {} the embedder was \
                     loaded with",
                    tokens.len(),
                    self.n_ctx
                );
                tokens.truncate(self.n_ctx as usize);
            }
            inputs.push(tokens);
        }
        let prompt_tokens = inputs.iter().map(Vec::len).sum();
        let n_embd = self.n_embd();

        let mut vectors = Vec::with_capacity(inputs.len());
        for tokens in inputs {
            if tokens.is_empty() {
                vectors.push(vec![0.0; n_embd]);
                continue;
            }
            // L2-normalize so callers can compare with a plain dot product —
            // the convention both the OpenAI and Ollama embedding endpoints
            // follow. Reranker (RANK-pooling) models are out of scope here:
            // normalizing their single relevance scalar would collapse it to
            // +-1 and lose the score entirely, so this endpoint is for
            // embedding models, not rerankers.
            let mut vector = self.worker.ask(|reply| Job::Embed(tokens, reply))?;
            normalize_l2(&mut vector);
            vectors.push(vector);
        }

        Ok(Embedded {
            vectors,
            prompt_tokens,
        })
    }
}

/// What the worker is asked, with where to send the answer.
enum Job {
    /// One input's tokens: its pooled embedding, not yet normalized.
    Embed(Vec<LlamaToken>, mpsc::SyncSender<Result<Vec<f32>, String>>),
    /// Build the kept context at this many cells now.
    Reserve(u32, mpsc::SyncSender<Result<(), String>>),
}

/// The handle `EmbeddingModel` keeps: dropping it stops the worker and frees
/// the context before returning.
struct Worker {
    tx: Option<mpsc::Sender<Job>>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn start(
        model: Arc<LlamaModel>,
        backend: Arc<LlamaBackend>,
        threads: u32,
        largest: u32,
    ) -> std::io::Result<Self> {
        let (tx, rx) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("eullm-embedding".into())
            .spawn(move || Kept::new(&model, &backend, threads, largest).serve(&rx))?;
        Ok(Self {
            tx: Some(tx),
            thread: Some(thread),
        })
    }

    /// Send the job `job` builds around its reply channel, and wait for the
    /// answer: after every job asked for before it, from any request.
    fn ask<T>(
        &self,
        job: impl FnOnce(mpsc::SyncSender<Result<T, String>>) -> Job,
    ) -> Result<T, String> {
        let stopped = || "the embedding worker has stopped".to_string();
        let (reply, answer) = mpsc::sync_channel(1);
        self.tx
            .as_ref()
            .ok_or_else(stopped)?
            .send(job(reply))
            .map_err(|_| stopped())?;
        answer.recv().map_err(|_| stopped())?
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The worker's side: the model and the context it keeps.
struct Kept<'m> {
    model: &'m LlamaModel,
    backend: &'m LlamaBackend,
    threads: u32,
    largest: u32,
    /// The kept context and its cells.
    ctx: Option<(LlamaContext<'m>, u32)>,
}

impl<'m> Kept<'m> {
    fn new(model: &'m LlamaModel, backend: &'m LlamaBackend, threads: u32, largest: u32) -> Self {
        Self {
            model,
            backend,
            threads,
            largest,
            ctx: None,
        }
    }

    fn serve(mut self, rx: &mpsc::Receiver<Job>) {
        while let Ok(job) = rx.recv() {
            match job {
                Job::Reserve(cells, reply) => {
                    let _ = reply.send(self.context(cells).map(|_| ()));
                }
                Job::Embed(tokens, reply) => {
                    let result = catch_unwind(AssertUnwindSafe(|| self.embed(&tokens)))
                        .unwrap_or_else(|panic| {
                            // Whatever the context holds now is not known.
                            self.ctx = None;
                            let what = panic
                                .downcast_ref::<String>()
                                .map(String::as_str)
                                .or_else(|| panic.downcast_ref::<&str>().copied())
                                .unwrap_or("unknown panic");
                            Err(format!("embedding failed: {what}"))
                        });
                    let _ = reply.send(result);
                }
            }
        }
    }

    /// The kept context, built or grown first when it has fewer than
    /// `cells`.
    fn context(&mut self, cells: u32) -> Result<&mut LlamaContext<'m>, String> {
        let kept = self.ctx.as_ref().map(|(_, cells)| *cells);
        if let Some(cells) = grown_context(kept, cells, self.largest) {
            // The old one goes first: both at once would need the memory of
            // both.
            self.ctx = None;
            let params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(cells))
                .with_n_batch(cells)
                .with_n_ubatch(cells)
                .with_n_threads(self.threads as i32)
                .with_n_threads_batch(self.threads as i32)
                .with_n_seq_max(1)
                .with_embeddings(true)
                .with_pooling_type(LlamaPoolingType::Unspecified);
            let ctx = self
                .model
                .new_context(self.backend, params)
                .map_err(|e| format!("Failed to create embedding context: {e}"))?;
            self.ctx = Some((ctx, cells));
        }
        match self.ctx.as_mut() {
            Some((ctx, _)) => Ok(ctx),
            None => Err("no embedding context".into()),
        }
    }

    fn embed(&mut self, tokens: &[LlamaToken]) -> Result<Vec<f32>, String> {
        let n_embd = usize::try_from(self.model.n_embd()).unwrap_or(0);
        let ctx = self.context(cells_for(tokens.len(), self.largest))?;
        ctx.clear_kv_cache();
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            // logits=true on every token, not just the last: mean/None
            // pooling need every token's embedding, and this matches
            // upstream's own embedding example (`batch_add_seq` in
            // examples/embedding/embedding.cpp) rather than the
            // last-token-only shape generation uses.
            batch
                .add(*token, i as i32, &[0], true)
                .map_err(|e| format!("Failed to build embedding batch: {e}"))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| format!("Embedding decode failed: {e}"))?;

        // Ask for the pooled sequence embedding first; the only way the
        // safe wrapper reports "this model's pooling resolved to NONE"
        // is that specific error coming back from the call itself (it
        // does not expose `llama_pooling_type(ctx)` to read up front —
        // see `EmbeddingsError::NonePoolType`), so that is the branch
        // this falls back on rather than predicting it beforehand.
        match ctx.embeddings_seq_ith(0) {
            Ok(v) => Ok(v.to_vec()),
            Err(EmbeddingsError::NonePoolType) => mean_pool(ctx, tokens.len(), n_embd),
            Err(e) => Err(format!("Failed to read embedding: {e}")),
        }
    }
}

/// Mean-pool per-token embeddings when the model declares no pooling type of
/// its own. `embeddings_ith` returns a reference into the context that
/// borrows `ctx` immutably, so this takes `&LlamaContext` rather than being a
/// method that could conflict with the `&mut ctx` calls around it.
fn mean_pool(ctx: &LlamaContext, n_tokens: usize, n_embd: usize) -> Result<Vec<f32>, String> {
    let mut sum = vec![0.0f32; n_embd];
    for i in 0..n_tokens {
        let token_embd = ctx
            .embeddings_ith(i as i32)
            .map_err(|e| format!("Failed to read per-token embedding {i}: {e}"))?;
        for (s, v) in sum.iter_mut().zip(token_embd) {
            *s += v;
        }
    }
    let n = n_tokens as f32;
    for s in &mut sum {
        *s /= n;
    }
    Ok(sum)
}

fn normalize_l2(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vector.iter_mut() {
            *v /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // An input gets the cells of one whole batch, in steps of 256.
    #[test]
    fn an_input_gets_the_cells_of_one_batch() {
        assert_eq!(cells_for(2, 2048), 256);
        assert_eq!(cells_for(256, 2048), 256);
        assert_eq!(cells_for(257, 2048), 512);
        assert_eq!(cells_for(1966, 2048), 2048);
        assert_eq!(cells_for(0, 2048), 256);
    }

    // The largest context holds every input the model takes, in one batch.
    #[test]
    fn the_largest_context_holds_every_input() {
        for max in [1, 16, 100, 512, 1000, DEFAULT_EMBEDDING_CTX, 8192] {
            let largest = largest_context(max);
            assert!(
                largest >= max && largest.is_multiple_of(KV_CELL_STEP),
                "max {max}"
            );
            for tokens in [1, 2, 31, 255, 256, 257, 999, 1000, 2047, 2048, 8192] {
                let tokens = tokens.min(max as usize);
                let cells = cells_for(tokens, largest);
                assert!(cells as usize >= tokens, "{tokens} tokens, max {max}");
                assert!(cells <= largest, "{tokens} tokens, max {max}");
            }
        }
        assert_eq!(largest_context(DEFAULT_EMBEDDING_CTX), 2048);
        assert_eq!(largest_context(1000), 1024);
    }

    #[test]
    fn the_kept_context_grows_only_when_an_input_needs_more() {
        // None kept: built at what the input needs.
        assert_eq!(grown_context(None, 256, 2048), Some(256));
        // Large enough: kept.
        assert_eq!(grown_context(Some(512), 256, 2048), None);
        assert_eq!(grown_context(Some(512), 512, 2048), None);
        // Too small: at least doubled, never past the largest.
        assert_eq!(grown_context(Some(256), 512, 2048), Some(512));
        assert_eq!(grown_context(Some(256), 768, 2048), Some(768));
        assert_eq!(grown_context(Some(512), 768, 2048), Some(1024));
        assert_eq!(grown_context(Some(1536), 1792, 2048), Some(2048));
    }

    /// The GGUF in `EULLM_EMBEDDING_TEST_MODEL`, as a server loads one, on
    /// the one backend every test here shares: llama.cpp's can be
    /// initialized only once at a time in a process.
    fn load_test_model() -> EmbeddingModel {
        let path = std::env::var("EULLM_EMBEDDING_TEST_MODEL")
            .expect("set EULLM_EMBEDDING_TEST_MODEL to an embedding GGUF");
        let backend = crate::inference::test_backend();
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as u32);
        EmbeddingModel::load(Path::new(&path), threads, DEFAULT_EMBEDDING_CTX, backend)
            .expect("load the model")
    }

    /// An input gets the same vector in the context kept after a longer one
    /// as in one built for it alone, and from one request of many inputs as
    /// from a request each — what a context built per request gave.
    ///
    /// ```text
    /// EULLM_EMBEDDING_TEST_MODEL=/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf \
    ///     cargo test --release --bin eullm embedding::tests::real_ -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs an embedding GGUF in EULLM_EMBEDDING_TEST_MODEL"]
    fn real_model_a_kept_context_gives_the_same_vectors() {
        let sentence = "La legge disciplina i contratti e le obbligazioni delle parti. ";
        let short = "Che cosa prevede la legge sui contratti?".to_string();
        let long = sentence.repeat(40);

        let alone = load_test_model()
            .embed(std::slice::from_ref(&short))
            .expect("embed");
        let model = load_test_model();
        let first = model.embed(std::slice::from_ref(&long)).expect("embed");
        let after = model.embed(std::slice::from_ref(&short)).expect("embed");
        let together = model.embed(&[long, short]).expect("embed");

        assert_eq!(after.vectors[0], alone.vectors[0]);
        assert_eq!(together.vectors[0], first.vectors[0]);
        assert_eq!(together.vectors[1], alone.vectors[0]);
        assert_eq!(
            together.prompt_tokens,
            first.prompt_tokens + alone.prompt_tokens
        );
    }

    /// `reserve_context` builds the largest context at once, and inputs are
    /// then embedded in it without building another.
    #[test]
    #[ignore = "needs an embedding GGUF in EULLM_EMBEDDING_TEST_MODEL"]
    fn real_model_embeds_in_the_reserved_context() {
        let model = load_test_model();
        model.reserve_context().expect("reserve");
        let started = std::time::Instant::now();
        let embedded = model
            .embed(&[
                "uno".to_string(),
                "due tre".to_string(),
                "quattro".to_string(),
            ])
            .expect("embed");
        eprintln!(
            "three inputs in the reserved context: {:?}",
            started.elapsed()
        );
        assert_eq!(embedded.vectors.len(), 3);
        let norm: f32 = embedded.vectors[0]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "{norm}");
    }
}
